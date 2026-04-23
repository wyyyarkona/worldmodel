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


class ScoreModelV2(nn.Module):
    # Full pairwise score model:
    # f1/f2 latent -> compressed video tokens
    # text/image embeddings -> context tokens
    # query + h1 + h2 + context + stage -> Qwen comparator
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
        self.task_prompt = str(task_prompt or "").strip()
        self._cached_prompt_tokens = None
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
        readout_input_dim = hidden_dim if self.readout_mode == "query" else hidden_dim * 4
        if self.readout_mode == "h1_h2":
            readout_input_dim = hidden_dim * 3
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

    def get_prompt_tokens(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if not self.task_prompt:
            return None
        if self._cached_prompt_tokens is None:
            self._cached_prompt_tokens = self.comparator.encode_text_prompt(self.task_prompt).detach()
        prompt_tokens = self._cached_prompt_tokens
        prompt_tokens = prompt_tokens.to(device=device, dtype=dtype)
        if prompt_tokens.size(0) != batch_size:
            prompt_tokens = prompt_tokens.expand(batch_size, -1, -1).contiguous()
        return prompt_tokens

    def build_sequence(self, h1, h2, context, stage_id):
        # Build the exact comparator order from the plan:
        # [h1 ; h2 ; context ; stage ; prompt ; query]
        # With causal attention enabled, placing query tokens at the tail lets
        # them attend to the full comparison context before readout.
        batch_size = h1.size(0)
        query = self.query_embed(batch_size)
        stage = self.stage_embed(stage_id)
        prompt = self.get_prompt_tokens(batch_size=batch_size, device=h1.device, dtype=h1.dtype)

        # Segment embeddings tell the comparator what each token group represents.
        query = self.segment_embed.add(query, "query")
        h1 = self.segment_embed.add(h1, "h1")
        h2 = self.segment_embed.add(h2, "h2")
        context = self.segment_embed.add(context, "context")
        stage = self.segment_embed.add(stage, "stage")
        prompt_parts = [h1, h2, context, stage]
        if prompt is not None:
            prompt = self.segment_embed.add(prompt, "prompt")
            prompt_parts.append(prompt)
        prompt_parts.append(query)

        sequence = torch.cat(prompt_parts, dim=1)
        attention_mask = torch.ones(
            sequence.size(0),
            sequence.size(1),
            device=sequence.device,
            dtype=torch.long,
        )
        return sequence, attention_mask

    def split_hidden_states(self, hidden_states, h1_len, h2_len, context_len, prompt_len=0):
        query_len = self.query_embed.query.size(1)
        h1_hidden = hidden_states[:, :h1_len]
        h2_hidden = hidden_states[:, h1_len:h1_len + h2_len]
        context_hidden = hidden_states[:, h1_len + h2_len:h1_len + h2_len + context_len]
        stage_hidden = hidden_states[:, h1_len + h2_len + context_len:h1_len + h2_len + context_len + 1]
        prompt_start = h1_len + h2_len + context_len + 1
        prompt_hidden = hidden_states[:, prompt_start:prompt_start + prompt_len]
        query_hidden = hidden_states[:, -query_len:]
        return {
            "h1": h1_hidden,
            "h2": h2_hidden,
            "context": context_hidden,
            "stage": stage_hidden,
            "prompt": prompt_hidden,
            "query": query_hidden,
        }

    def build_readout_features(self, hidden_parts):
        query_pooled = hidden_parts["query"].mean(dim=1)
        h1_pooled = hidden_parts["h1"].mean(dim=1)
        h2_pooled = hidden_parts["h2"].mean(dim=1)
        if self.readout_mode == "query":
            return query_pooled
        if self.readout_mode == "h1_h2":
            return torch.cat(
                [h1_pooled, h2_pooled, h1_pooled - h2_pooled],
                dim=-1,
            )
        return torch.cat(
            [query_pooled, h1_pooled, h2_pooled, h1_pooled - h2_pooled],
            dim=-1,
        )

    def forward(self, f1, f2, text_emb, image_emb, stage_id, return_aux_stats=False):
        # Encode both candidate latents with shared projector weights.
        h1 = self.encode_video(f1)
        h2 = self.encode_video(f2)
        context = self.context_projector(text_emb, image_emb)
        prompt_len = 0 if not self.task_prompt else self.get_prompt_tokens(1, h1.device, h1.dtype).size(1)

        # Only the trailing query slice is used for the final decision head.
        sequence, attention_mask = self.build_sequence(h1, h2, context, stage_id)
        comparator_param = next(self.comparator.parameters())
        sequence = sequence.to(device=comparator_param.device, dtype=comparator_param.dtype)
        attention_mask = attention_mask.to(device=comparator_param.device)
        hidden_states = self.comparator(sequence, attention_mask=attention_mask)
        hidden_parts = self.split_hidden_states(
            hidden_states,
            h1_len=h1.size(1),
            h2_len=h2.size(1),
            context_len=context.size(1),
            prompt_len=prompt_len,
        )
        readout_features = self.build_readout_features(hidden_parts)
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
            projector_tokens = torch.cat([h1, h2, context], dim=1)
            outputs["proj_mean"] = projector_tokens.mean(dim=(0, 1))
            outputs["proj_std"] = projector_tokens.std(dim=(0, 1), unbiased=False)
        return outputs
