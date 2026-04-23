from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .qwen_comparator import _find_first_attr


DEFAULT_TASK_PROMPT = (
    "任务:你是视频质量评估专家。上面给出了视频 1、视频 2、参考图像和提示词。"
    "请综合评估两个视频的质量,从以下四个维度判断视频 1 是否优于视频 2:\n"
    "1. 与提示词的语义一致性\n"
    "2. 与参考图像的视觉风格匹配度\n"
    "3. 画面清晰度与细节还原\n"
    "4. 时序连贯性与运动自然度\n"
    "接下来的 query 将聚合你的判断结果。"
)


DEFAULT_LABELS = {
    "h1_label": "视频 1:",
    "h2_label": "视频 2:",
    "image_label": "参考图像:",
    "text_label": "提示词:",
    "stage_label": "阶段:",
}


def _load_qwen_for_embed(qwen_model_path: str, torch_dtype: torch.dtype):
    # Mirror the loading fallback used by QwenComparator so that whichever
    # transformers/Qwen variant works there also works here. We do not need
    # flash-attn here since the model is only touched once to extract embed_tokens.
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_path,
            torch_dtype=torch_dtype,
        )
    except Exception:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            qwen_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )


def _extract_embed_tokens(model: nn.Module) -> nn.Embedding:
    # Walk the same attribute chain QwenComparator uses.
    language_model = _find_first_attr(model, ["language_model", "model"])
    if language_model is None:
        raise AttributeError("Could not locate the Qwen language model on the loaded checkpoint.")
    model_body = _find_first_attr(language_model, ["model"])
    layer_source = model_body if model_body is not None else language_model
    embed_tokens = _find_first_attr(layer_source, ["embed_tokens", "wte"])
    if embed_tokens is None:
        raise AttributeError("Could not locate embed_tokens on the Qwen language model.")
    return embed_tokens


class TextLabelEmbedder(nn.Module):
    # Pre-computes embeddings for static textual cues (段标签 + 任务 prompt) via
    # Qwen's own tokenizer + embed_tokens, then caches them as frozen buffers.
    # Runtime cost per forward = one expand + copy of cached tensors.
    LABEL_KEYS = ("h1_label", "h2_label", "image_label", "text_label", "stage_label", "task_prompt")

    def __init__(
        self,
        qwen_model_path: str,
        hidden_dim: int,
        task_prompt_override: str = "",
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)

        # Borrow embed_tokens once to compute the static embeddings, then drop
        # the rest of the backbone so we don't hold two copies of Qwen in VRAM.
        temp_model = _load_qwen_for_embed(qwen_model_path, torch_dtype=torch_dtype)
        embed_tokens = _extract_embed_tokens(temp_model)

        qwen_embed_dim = embed_tokens.weight.size(-1)
        if qwen_embed_dim != hidden_dim:
            raise ValueError(
                "TextLabelEmbedder requires Qwen embed_tokens dim to match hidden_dim. "
                f"Got qwen_embed_dim={qwen_embed_dim}, hidden_dim={hidden_dim}. "
                "Use a Qwen checkpoint whose hidden_size matches model.hidden_dim."
            )

        # Build the text table. The caller can override only the task prompt;
        # section labels are fixed because their wording is part of the model design.
        prompt = (task_prompt_override or "").strip() or DEFAULT_TASK_PROMPT
        texts = {**DEFAULT_LABELS, "task_prompt": prompt}
        self._texts: dict[str, str] = dict(texts)
        self._token_counts: dict[str, int] = {}

        # Freeze embed_tokens hard: it is only referenced during init, but we
        # keep a detached copy around for debugging / introspection.
        for parameter in embed_tokens.parameters():
            parameter.requires_grad = False
        embed_tokens.eval()

        with torch.no_grad():
            for key in self.LABEL_KEYS:
                text = self._texts[key]
                if not text:
                    raise ValueError(f"TextLabelEmbedder received an empty text for key '{key}'")
                # add_special_tokens=False because these strings are inserted
                # mid-sequence; we do not want BOS/EOS markers inside.
                input_ids = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"]
                embedding = embed_tokens(input_ids).detach().to(dtype=torch_dtype)
                # Shape is [1, L, D] so that `get(key, B)` only needs an expand.
                self.register_buffer(f"emb_{key}", embedding.contiguous(), persistent=False)
                self._token_counts[key] = int(input_ids.size(1))

        # Drop the heavyweight backbone but keep a frozen embed_tokens reference
        # so the module is self-contained if we ever want to embed new text.
        self.embed_tokens = embed_tokens
        for parameter in self.embed_tokens.parameters():
            parameter.requires_grad = False
        del temp_model

    @torch.no_grad()
    def get(self, key: str, batch_size: int) -> torch.Tensor:
        if key not in self.LABEL_KEYS:
            raise KeyError(f"Unknown label key '{key}'. Expected one of: {self.LABEL_KEYS}")
        emb = getattr(self, f"emb_{key}")
        if emb.size(0) != batch_size:
            emb = emb.expand(batch_size, -1, -1).contiguous()
        return emb

    def token_count(self, key: str) -> int:
        return self._token_counts[key]

    def print_label_info(self) -> None:
        # Quick verification helper — confirms tokenization is as expected.
        print("TextLabelEmbedder cached entries:")
        for key in self.LABEL_KEYS:
            text = self._texts[key]
            n_tok = self._token_counts[key]
            preview = text if len(text) <= 60 else text[:57] + "..."
            print(f"  {key:<14s} | tokens={n_tok:4d} | text={preview!r}")
