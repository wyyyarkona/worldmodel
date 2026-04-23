from __future__ import annotations

import torch
import torch.nn as nn

from .embeddings import (
    Learned3DPositionEmbedding,
    QueryEmbedding,
    SegmentEmbedding,
    StageEmbedding,
)
from .projectors import ContextProjector, VideoProjector
from .qwen_comparator import QwenComparator
from .text_label_embedder import TextLabelEmbedder


class ScoreModelV2(nn.Module):
    # Full pairwise score model:
    # f1/f2 latent -> compressed video tokens
    # text/image embeddings -> separate image/text context tokens
    # Each content segment is preceded by a Qwen-tokenized text label so the
    # comparator sees explicit "视频 1:" / "视频 2:" / "参考图像:" / "提示词:" /
    # "阶段:" cues plus a trailing task prompt before the query tokens.
    # pooled query states -> scalar score
    def __init__(
        self,
        qwen_model_path: str,
        latent_dim: int = 16,
        patch_dim: int = 1536,
        hidden_dim: int = 2048,
        text_dim: int = 4096,
        image_dim: int = 1280,
        frames: int = 11,
        height: int = 15,
        width: int = 26,
        context_max_text_tokens: int = 512,
        context_max_image_tokens: int = 64,
        num_query_tokens: int = 4,
        num_qwen_layers: int = 6,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        freeze_qwen_backbone: bool = True,
        gradient_checkpointing: bool = True,
        bidirectional_attention: bool = False,
        readout_mode: str = "query",
        task_prompt: str = "",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout_mode = str(readout_mode)
        if self.readout_mode not in {"query", "h1_h2", "hybrid"}:
            raise ValueError(
                f"Unsupported readout_mode='{self.readout_mode}'. "
                "Expected one of: query, h1_h2, hybrid."
            )
        self.video_projector = VideoProjector(
            latent_dim=latent_dim,
            patch_dim=patch_dim,
            hidden_dim=hidden_dim,
        )
        self.video_pos_embed = Learned3DPositionEmbedding(
            frames=frames,
            height=height,
            width=width,
            dim=hidden_dim,
        )
        self.context_projector = ContextProjector(
            text_dim=text_dim,
            image_dim=image_dim,
            hidden_dim=hidden_dim,
            max_text_tokens=context_max_text_tokens,
            max_image_tokens=context_max_image_tokens,
        )
        self.segment_embed = SegmentEmbedding(dim=hidden_dim)
        self.stage_embed = StageEmbedding(num_stages=3, dim=hidden_dim)
        self.query_embed = QueryEmbedding(num_queries=num_query_tokens, dim=hidden_dim)
        self.comparator = QwenComparator(
            qwen_model_path=qwen_model_path,
            hidden_size=hidden_dim,
            num_layers=num_qwen_layers,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            freeze_backbone=freeze_qwen_backbone,
            gradient_checkpointing=gradient_checkpointing,
            bidirectional_attention=bidirectional_attention,
        )
        # Static Qwen-tokenized cues for section labels + task prompt. The
        # task_prompt kwarg stays for config compatibility: non-empty strings
        # override the default prompt baked into TextLabelEmbedder.
        self.text_labels = TextLabelEmbedder(
            qwen_model_path=qwen_model_path,
            hidden_dim=hidden_dim,
            task_prompt_override=task_prompt,
        )
        if self.readout_mode != "query":
            # Labels are fused into the h1/h2/context segments, so the old
            # h1_h2 / hybrid slicing would mix content with label tokens.
            # Fail fast until a label-aware readout is implemented.
            raise NotImplementedError(
                "readout_mode='h1_h2' and 'hybrid' are temporarily disabled after "
                "the text-label refactor. Use readout_mode='query' for now."
            )
        readout_input_dim = hidden_dim
        self.score_head = nn.Sequential(
            nn.Linear(readout_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 1),
        )

    def encode_video(self, latent):
        # Compress the 5D latent into a flat token sequence with learned 3D positions.
        tokens_3d = self.video_projector(latent)
        flat_tokens = self.video_pos_embed(tokens_3d)
        return flat_tokens

    def _fetch_labels(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        # One pass to gather all six label/prompt tensors on the right device/dtype.
        labels = {}
        for key in TextLabelEmbedder.LABEL_KEYS:
            tensor = self.text_labels.get(key, batch_size)
            labels[key] = tensor.to(device=device, dtype=dtype)
        return labels

    def build_sequence(self, h1, h2, image_tokens, text_tokens, stage_id):
        # New order:
        # ["视频 1:" + h1 ; "视频 2:" + h2 ; "参考图像:" + image + "提示词:" + text ;
        #  "阶段:" + stage ; task_prompt ; query]
        # Each label shares the segment embedding of the section it introduces.
        # task_prompt gets the dedicated "prompt" segment. The query remains at
        # the tail so causal attention lets it summarize everything before it.
        batch_size = h1.size(0)
        device = h1.device
        dtype = h1.dtype

        labels = self._fetch_labels(batch_size=batch_size, device=device, dtype=dtype)
        stage_token = self.stage_embed(stage_id)
        query = self.query_embed(batch_size)

        h1_segment = torch.cat([labels["h1_label"], h1], dim=1)
        h1_segment = self.segment_embed.add(h1_segment, "h1")

        h2_segment = torch.cat([labels["h2_label"], h2], dim=1)
        h2_segment = self.segment_embed.add(h2_segment, "h2")

        context_segment = torch.cat(
            [labels["image_label"], image_tokens, labels["text_label"], text_tokens],
            dim=1,
        )
        context_segment = self.segment_embed.add(context_segment, "context")

        stage_segment = torch.cat([labels["stage_label"], stage_token], dim=1)
        stage_segment = self.segment_embed.add(stage_segment, "stage")

        prompt_segment = self.segment_embed.add(labels["task_prompt"], "prompt")

        query_segment = self.segment_embed.add(query, "query")

        sequence = torch.cat(
            [h1_segment, h2_segment, context_segment, stage_segment, prompt_segment, query_segment],
            dim=1,
        )
        attention_mask = torch.ones(
            sequence.size(0),
            sequence.size(1),
            device=sequence.device,
            dtype=torch.long,
        )
        return sequence, attention_mask

    def forward(self, f1, f2, text_emb, image_emb, stage_id, return_aux_stats=False):
        # Encode both candidate latents with shared projector weights, then
        # project text/image separately so the comparator sees each conditioning
        # stream behind its own label token.
        h1 = self.encode_video(f1)
        h2 = self.encode_video(f2)
        image_tokens = self.context_projector.project_image(image_emb)
        text_tokens = self.context_projector.project_text(text_emb)

        sequence, attention_mask = self.build_sequence(h1, h2, image_tokens, text_tokens, stage_id)
        comparator_param = next(self.comparator.parameters())
        sequence = sequence.to(device=comparator_param.device, dtype=comparator_param.dtype)
        attention_mask = attention_mask.to(device=comparator_param.device)
        hidden_states = self.comparator(sequence, attention_mask=attention_mask)

        # Query tokens are always the final num_query_tokens positions.
        num_queries = self.query_embed.query.size(1)
        query_hidden = hidden_states[:, -num_queries:]
        readout_features = query_hidden.mean(dim=1)

        head_param = next(self.score_head.parameters())
        readout_features = readout_features.to(device=head_param.device, dtype=head_param.dtype)
        logit = self.score_head(readout_features).squeeze(-1)
        score = torch.sigmoid(logit)

        outputs = {
            "score": score,
            "logit": logit,
        }
        if return_aux_stats:
            # Warmup uses the projector output statistics as an auxiliary alignment target.
            # We keep the same content that feeds the comparator (image + text combined)
            # so the alignment target matches what the downstream layers actually see.
            projector_tokens = torch.cat([h1, h2, image_tokens, text_tokens], dim=1)
            outputs["proj_mean"] = projector_tokens.mean(dim=(0, 1))
            outputs["proj_std"] = projector_tokens.std(dim=(0, 1), unbiased=False)
        return outputs
