from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer


def _find_first_attr(module: nn.Module, attr_names: list[str]) -> Any:
    # Small helper because the exact Qwen wrapper path can differ across
    # transformers versions and checkpoints.
    for name in attr_names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return None


class FrontQwenBackbone(nn.Module):
    # Comparator-only wrapper that keeps exactly the front language-model blocks
    # we will execute at training/inference time. LoRA should attach here so the
    # trainable parameter set matches the actual forward path.
    def __init__(
        self,
        layers: list[nn.Module],
        final_norm: nn.Module | None,
        rotary_emb: nn.Module | None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(list(layers))
        self.final_norm = final_norm if final_norm is not None else nn.Identity()
        self.rotary_emb = rotary_emb

    def _build_position_embeddings(self, hidden_states, position_ids):
        if self.rotary_emb is None:
            return None
        try:
            return self.rotary_emb(hidden_states, position_ids)
        except TypeError:
            pass
        try:
            return self.rotary_emb(hidden_states, seq_len=hidden_states.size(1))
        except TypeError:
            pass
        try:
            return self.rotary_emb(position_ids)
        except TypeError:
            return None

    def forward(self, hidden_states, attention_mask=None):
        # The comparator receives already-projected custom embeddings rather than
        # token ids, so we directly feed hidden states through the selected layers.
        ref_param = next(self.parameters())
        hidden_states = hidden_states.to(device=ref_param.device, dtype=ref_param.dtype)
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        # Qwen2.5-VL rotary embeddings expect 3-axis multimodal position ids with
        # shape [3, batch, seq]. For our comparator-only sequence there is no
        # real temporal/spatial structure, so we reuse the same monotonic index
        # on all three axes.
        base_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_ids = base_position_ids.unsqueeze(0).expand(3, -1, -1)
        cache_position = torch.arange(seq_len, device=device)
        position_embeddings = self._build_position_embeddings(hidden_states, position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(device=device)

        for layer in self.layers:
            # Different transformers/Qwen versions accept slightly different forward
            # signatures, so we progressively fall back to smaller argument sets.
            kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
                "use_cache": False,
                "output_attentions": False,
                "cache_position": cache_position,
            }
            try:
                layer_outputs = layer(**kwargs)
            except TypeError:
                kwargs.pop("cache_position", None)
                try:
                    layer_outputs = layer(**kwargs)
                except TypeError:
                    kwargs.pop("position_ids", None)
                    layer_outputs = layer(**kwargs)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        return self.final_norm(hidden_states)


class QwenComparator(nn.Module):
    # Wrapper around the first few Qwen2.5-VL language-model blocks.
    # The wrapper intentionally:
    # 1) keeps only the first `num_layers` decoder blocks
    # 2) freezes the original backbone by default
    # 3) tries to add LoRA to attention/MLP projections
    # 4) can optionally disable the causal mask so comparison becomes bidirectional
    def __init__(
        self,
        qwen_model_path: str,
        hidden_size: int = 2048,
        num_layers: int = 6,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        freeze_backbone: bool = True,
        gradient_checkpointing: bool = True,
        bidirectional_attention: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.supports_lora = False
        self.lora_attach_error = None
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": attn_implementation,
        }
        if attn_implementation == "flash_attention_2" and torch.cuda.is_available():
            # Flash Attention 2 is happiest when the model is instantiated on GPU
            # directly, which also avoids the noisy CPU-init warning from transformers.
            load_kwargs["device_map"] = {"": torch.cuda.current_device()}

        try:
            # LoRA is optional so the model can still be inspected without PEFT.
            from peft import LoraConfig, get_peft_model
            self.supports_lora = True
        except ImportError:
            LoraConfig = None
            get_peft_model = None

        try:
            # Preferred path for recent transformers releases with native Qwen2.5-VL.
            from transformers import Qwen2_5_VLForConditionalGeneration

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_model_path,
                **load_kwargs,
            )
        except Exception:
            # Fallback path for environments exposing Qwen VL through the generic auto class.
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                qwen_model_path,
                **load_kwargs,
                trust_remote_code=True,
            )

        language_model, layer_source, layers, final_norm, rotary_emb, embed_tokens = self._resolve_qwen_components(model)

        if freeze_backbone:
            # Freeze all pretrained Qwen weights first, then selectively re-enable
            # LoRA parameters later if LoRA attaches successfully.
            for parameter in model.parameters():
                parameter.requires_grad = False

        self.backbone = FrontQwenBackbone(
            layers=list(layers[: self.num_layers]),
            final_norm=final_norm,
            rotary_emb=rotary_emb,
        )
        self.embed_tokens = embed_tokens

        if self.supports_lora:
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            try:
                self.backbone = get_peft_model(self.backbone, lora_cfg)
            except Exception as exc:
                self.lora_attach_error = exc
                self.supports_lora = False

        if gradient_checkpointing:
            self._enable_gradient_checkpointing(model, language_model)
        if bidirectional_attention:
            self._patch_bidirectional_mask(model, language_model)

    def _resolve_qwen_components(self, model: nn.Module):
        language_model = _find_first_attr(model, ["language_model", "model"])
        if language_model is None:
            raise AttributeError("Could not locate the Qwen language model on the loaded checkpoint.")

        model_body = _find_first_attr(language_model, ["model"])
        layer_source = model_body if model_body is not None else language_model
        layers = _find_first_attr(layer_source, ["layers", "h", "blocks"])
        if layers is None:
            raise AttributeError("Could not find transformer layers on the Qwen language model.")
        final_norm = _find_first_attr(layer_source, ["norm", "ln_f"])
        rotary_emb = _find_first_attr(layer_source, ["rotary_emb"])
        embed_tokens = _find_first_attr(layer_source, ["embed_tokens", "wte"])
        if embed_tokens is None:
            raise AttributeError("Could not locate token embedding layer on the Qwen language model.")
        return language_model, layer_source, layers, final_norm, rotary_emb, embed_tokens

    def _enable_gradient_checkpointing(self, model: nn.Module, language_model: nn.Module):
        # Enable checkpointing wherever the source checkpoint exposes the hook.
        # The front comparator backbone reuses these exact block modules.
        for module in (model, language_model):
            fn = getattr(module, "gradient_checkpointing_enable", None)
            if callable(fn):
                fn()

    def _patch_bidirectional_mask(self, model: nn.Module, language_model: nn.Module):
        # The Qwen LM is decoder-style, but for pairwise comparison we need
        # every token to see the full concatenated sequence.
        candidates = [model, language_model, getattr(language_model, "model", None)]
        for module in candidates:
            if module is None:
                continue
            if hasattr(module, "_update_causal_mask"):
                def _no_causal_mask(this, attention_mask, input_tensor, cache_position=None, past_key_values=None, output_attentions=False):
                    return attention_mask

                module._update_causal_mask = types.MethodType(_no_causal_mask, module)

    def forward(self, hidden_states, attention_mask=None):
        return self.backbone(hidden_states, attention_mask=attention_mask)

    def encode_text_prompt(self, text: str) -> torch.Tensor:
        tokenized = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device=next(self.embed_tokens.parameters()).device)
        prompt_tokens = self.embed_tokens(input_ids)
        return prompt_tokens

    def print_trainable_parameters(self):
        # Handy utility when checking the LoRA trainable ratio on the server.
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        ratio = 100.0 * trainable / max(total, 1)
        print(f"trainable params: {trainable} || all params: {total} || trainable%: {ratio:.4f}")

    def ensure_lora_attached(self):
        if not self.supports_lora:
            if self.lora_attach_error is not None:
                raise RuntimeError(
                    "PEFT is installed, but attaching LoRA to the Qwen backbone failed."
                ) from self.lora_attach_error
            raise RuntimeError(
                "LoRA support is unavailable. Ensure `peft` is installed and compatible with the current transformers version."
            )
        has_lora = any("lora_" in name for name, _ in self.backbone.named_parameters())
        if not has_lora:
            raise RuntimeError(
                "LoRA attachment completed without creating any `lora_` parameters on the front comparator backbone. "
                "Check the target module names against the loaded Qwen layers."
            )
