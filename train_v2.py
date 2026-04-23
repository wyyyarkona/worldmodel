from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from score_model_v2.losses import PairwiseScoreLoss, WarmupAlignmentLoss
from score_model_v2.models.score_model_v2 import ScoreModelV2

REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_GRAD_NORM = 1.0
STAGE_ID_TO_NAME = {
    0: "early",
    1: "middle",
    2: "late",
}
MARGIN_BUCKETS = [
    ("small", 0.0, 0.1),
    ("medium", 0.1, 0.3),
    ("large", 0.3, float("inf")),
]


def load_config(path: str) -> dict[str, Any]:
    # Keep config loading minimal: JSON works directly, YAML works when PyYAML is available.
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML configs for train_v2.py.") from exc
    config = yaml.safe_load(text)

    def expand_env(obj: Any) -> Any:
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, list):
            return [expand_env(x) for x in obj]
        if isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        return obj

    return expand_env(config)


def resolve_repo_relative_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def parse_args():
    # CLI overrides are intentionally small; the YAML file remains the main source
    # of truth for the experiment layout.
    parser = argparse.ArgumentParser(description="Train the Qwen-based score model v2.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ddp_backend", type=str, default="nccl")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--no_auto_resume", action="store_true")
    parser.add_argument("--disable_curriculum", action="store_true")
    return parser.parse_args()


def bucket_stage_name(value: Any):
    # If the manifest already carries an explicit stage label, trust it directly.
    # Supported forms are: "early"/"middle"/"late" or integer ids 0/1/2.
    if value is None:
        return None
    if isinstance(value, str):
        value_lower = value.lower()
        mapping = {"early": 0, "middle": 1, "mid": 1, "late": 2}
        return mapping.get(value_lower)
    if isinstance(value, int) and value in {0, 1, 2}:
        return value
    return None


def bucket_stage_from_step(step_index: int, min_step: int, max_step: int):
    # Stage bucketing is defined by denoising step index, not by raw timestep.
    # We split the observed step-index range into three equal bands:
    # small step index -> early, middle band -> middle, large step index -> late.
    if max_step <= min_step:
        return 1
    normalized = (step_index - min_step) / max(max_step - min_step, 1)
    if normalized < (1.0 / 3.0):
        return 0
    if normalized < (2.0 / 3.0):
        return 1
    return 2


def load_tensor(path):
    # Support both plain tensors and small dict payloads written by earlier tools.
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("latent", "context", "clip_fea", "tensor"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(payload).__name__}")
    return payload.float()


class PairwiseLatentDatasetV2(Dataset):
    # Dataset that reads the current pairwise manifest format while staying
    # compatible with both older `context_path/clip_fea_path` fields and the
    # clearer `text_emb_path/image_emb_path` naming.
    def __init__(self, manifest_path, curriculum_stages=None, max_samples=None):
        self.manifest_path = Path(manifest_path)
        self.items = self._load_items()
        # Compute the step-index range once from the whole manifest so stage
        # bucketing stays stable even when curriculum later filters the dataset.
        self.min_step_index, self.max_step_index = self._compute_step_range(self.items)
        if curriculum_stages is not None:
            allowed = set(curriculum_stages)
            # Curriculum mode filters by the resolved stage bucket, which is
            # derived from step index rather than raw timestep values.
            self.items = [
                item for item in self.items
                if self._resolve_stage_id(item) in allowed
            ]
        if max_samples is not None:
            # Smoke tests and early pipeline validation can cap the dataset size.
            self.items = self.items[:max_samples]

    def _load_items(self):
        text = self.manifest_path.read_text(encoding="utf-8").strip()
        if self.manifest_path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("items") or data.get("samples") or data.get("data") or []
        return data

    @staticmethod
    def _extract_step_index(item):
        # Try the common step-index field names produced by the current
        # pair-building pipeline.
        for key in ("t_idx", "step_idx", "step", "step_id"):
            value = item.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _compute_step_range(self, items):
        # Stage bucketing depends on the minimum/maximum denoising step index
        # observed in the manifest.
        step_indices = []
        for item in items:
            step_index = self._extract_step_index(item)
            if step_index is not None:
                step_indices.append(step_index)
        if not step_indices:
            return 0, 0
        return min(step_indices), max(step_indices)

    def _resolve_stage_id(self, item):
        explicit_stage = bucket_stage_name(item.get("stage_id"))
        if explicit_stage is not None:
            return explicit_stage
        step_index = self._extract_step_index(item)
        if step_index is not None:
            # Prefer denoising step index over timestep for stage assignment.
            return bucket_stage_from_step(
                step_index,
                min_step=self.min_step_index,
                max_step=self.max_step_index,
            )
        # Keep a middle-stage fallback only when step-index metadata is absent.
        return 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        # The current local pipeline stores precomputed text embeddings at
        # `context_path` and visual embeddings at `clip_fea_path`.
        text_path = item.get("text_emb_path") or item.get("context_path")
        image_path = item.get("image_emb_path") or item.get("clip_fea_path")
        f1_path = item.get("f1_path") or item.get("z_t_a_path")
        f2_path = item.get("f2_path") or item.get("z_t_b_path")
        if text_path is None or image_path is None:
            raise ValueError("Each sample must provide text/context and image/clip feature paths.")
        if f1_path is None or f2_path is None:
            raise ValueError("Each sample must provide f1/f2 latent paths or z_t_a/z_t_b latent paths.")
        return {
            "sample_index": int(index),
            "f1": load_tensor(f1_path),
            "f2": load_tensor(f2_path),
            "text_emb": load_tensor(text_path),
            "image_emb": load_tensor(image_path),
            "step_index": self._extract_step_index(item),
            "stage_id": self._resolve_stage_id(item),
            "teacher_score_a": float(item.get("score_a", item.get("teacher_score_a", item.get("label", 0.5)))),
            "teacher_score_b": float(item.get("score_b", item.get("teacher_score_b", 0.0))),
            "sample_weight": float(item.get("weight", 1.0)),
        }


def pad_token_batch(tensors):
    # Text/image token counts vary sample by sample, so we right-pad the batch
    # before stacking into [B, L, D].
    max_len = max(tensor.size(0) for tensor in tensors)
    feat_dim = tensors[0].size(-1)
    padded = []
    for tensor in tensors:
        if tensor.dim() != 2:
            raise ValueError(f"Expected token tensor with shape [L, D], got {tuple(tensor.shape)}")
        if tensor.size(1) != feat_dim:
            raise ValueError(f"Mismatched feature dimensions in batch: {tensor.size(1)} vs {feat_dim}")
        if tensor.size(0) < max_len:
            pad = tensor.new_zeros(max_len - tensor.size(0), feat_dim)
            tensor = torch.cat([tensor, pad], dim=0)
        padded.append(tensor)
    return torch.stack(padded, dim=0)


def pad_video_latent_batch(tensors):
    # Some precomputed z_t latents can differ by one frame in the temporal
    # dimension (for example 20 vs 21). Pad within the batch so dataloader
    # collation stays robust instead of assuming perfectly identical shapes.
    if not tensors:
        raise ValueError("Expected at least one latent tensor to collate.")
    if any(tensor.dim() != 4 for tensor in tensors):
        raise ValueError(f"Expected latent tensors with shape [C, T, H, W], got {[tuple(t.shape) for t in tensors]}")

    max_c = max(tensor.size(0) for tensor in tensors)
    max_t = max(tensor.size(1) for tensor in tensors)
    max_h = max(tensor.size(2) for tensor in tensors)
    max_w = max(tensor.size(3) for tensor in tensors)

    padded = []
    for tensor in tensors:
        if tensor.size(0) != max_c:
            raise ValueError(
                f"Mismatched latent channel dimensions in batch: {tensor.size(0)} vs {max_c}"
            )
        pad_t = max_t - tensor.size(1)
        pad_h = max_h - tensor.size(2)
        pad_w = max_w - tensor.size(3)
        if pad_t or pad_h or pad_w:
            pad = (0, pad_w, 0, pad_h, 0, pad_t)
            tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)
        padded.append(tensor)
    return torch.stack(padded, dim=0)


def collate_batch(samples):
    # Video latents are expected to have identical shapes here, while token
    # sequences are padded dynamically. In practice some latent clips differ by
    # one frame, so latents are also padded batch-locally.
    return {
        "sample_index": torch.tensor([sample["sample_index"] for sample in samples], dtype=torch.long),
        "f1": pad_video_latent_batch([sample["f1"] for sample in samples]),
        "f2": pad_video_latent_batch([sample["f2"] for sample in samples]),
        "text_emb": pad_token_batch([sample["text_emb"] for sample in samples]),
        "image_emb": pad_token_batch([sample["image_emb"] for sample in samples]),
        "step_index": torch.tensor(
            [(-1 if sample["step_index"] is None else int(sample["step_index"])) for sample in samples],
            dtype=torch.long,
        ),
        "stage_id": torch.tensor([sample["stage_id"] for sample in samples], dtype=torch.long),
        "teacher_score_a": torch.tensor([sample["teacher_score_a"] for sample in samples], dtype=torch.float32),
        "teacher_score_b": torch.tensor([sample["teacher_score_b"] for sample in samples], dtype=torch.float32),
        "sample_weight": torch.tensor([sample["sample_weight"] for sample in samples], dtype=torch.float32),
    }


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def init_distributed(args):
    # Support both single-process training and torchrun multi-process launches.
    if not is_distributed():
        return torch.device(args.device), 0, 1, False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ["WORLD_SIZE"])
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA training requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = args.ddp_backend
    else:
        device = torch.device(args.device)
        backend = "gloo"
    dist.init_process_group(backend=backend)
    return device, rank, world_size, True


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def normalize_stage_name(stage_name):
    if stage_name is None:
        return None
    for prefix in ("warmup", "lora", "curriculum"):
        if stage_name == prefix or stage_name.startswith(f"{prefix}_phase"):
            return prefix
    return stage_name


def stage_sort_key(stage_name):
    order = {
        "warmup": 0,
        "lora": 1,
        "curriculum": 2,
    }
    normalized = normalize_stage_name(stage_name)
    return order[normalized]


def resolve_resume_path(args, output_dir):
    # Prefer an explicit checkpoint path, otherwise fall back to the rolling latest checkpoint.
    if args.resume_from:
        return Path(args.resume_from)
    if args.no_auto_resume:
        return None
    latest_path = output_dir / "latest_checkpoint.pt"
    if latest_path.exists():
        return latest_path
    return None


def build_dataloader(dataset, batch_size, num_workers, sampler, shuffle, pin_memory, persistent_workers):
    effective_persistent_workers = persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=effective_persistent_workers,
        collate_fn=collate_batch,
    )


def set_stage_trainability(model, stage_name):
    # Training policy:
    # warmup -> only new projector/embedding/head modules
    # lora/curriculum -> new modules + LoRA
    for parameter in model.parameters():
        parameter.requires_grad = False

    always_train = [
        model.video_projector,
        model.video_pos_embed,
        model.context_projector,
        model.segment_embed,
        model.stage_embed,
        model.query_embed,
        model.score_head,
    ]
    for module in always_train:
        for parameter in module.parameters():
            parameter.requires_grad = True

    if stage_name in {"lora", "curriculum"}:
        model.comparator.ensure_lora_attached()
        found_lora = False
        for name, parameter in model.comparator.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
                found_lora = True
        if not found_lora:
            raise RuntimeError(
                f"Stage '{stage_name}' requires LoRA parameters, but none were found on the comparator. "
                "Check that PEFT is installed and that LoRA attached to the Qwen backbone successfully."
            )


def build_optimizer(model, stage_name, config):
    # Warmup uses one LR for all trainable new modules. Later stages split LoRA
    # and non-LoRA parameters into separate groups.
    if stage_name == "warmup":
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=float(config["training"]["warmup_lr"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )

    lora_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(parameter)
        else:
            other_params.append(parameter)
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": float(config["training"]["lora_lr"])})
    if other_params:
        param_groups.append({"params": other_params, "lr": float(config["training"]["base_lr"])})
    return torch.optim.AdamW(
        param_groups,
        weight_decay=float(config["training"]["weight_decay"]),
    )


def _cosine_with_warmup_lambda(current_step: int, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1) -> float:
    # The first optimizer update should still have a small positive LR instead of zero,
    # so the warmup branch uses (current_step + 1).
    if current_step < num_warmup_steps:
        return float(current_step + 1) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def _linear_warmup_only_lambda(current_step: int, num_warmup_steps: int) -> float:
    if current_step < num_warmup_steps:
        return float(current_step + 1) / float(max(1, num_warmup_steps))
    return 1.0


def build_scheduler(
    optimizer: Optimizer,
    stage_name: str,
    num_training_steps: int,
    *,
    min_lr_ratio: float = 0.1,
    warmup_ratio: float = 0.05,
    warmup_floor: int = 50,
    warmup_ceiling: int = 500,
) -> LambdaLR:
    # Each phase gets its own scheduler. Warmup stage only ramps up and then holds.
    # LoRA/curriculum phases ramp up and then decay with cosine.
    num_training_steps = max(int(num_training_steps), 1)
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    num_warmup_steps = max(warmup_floor, min(num_warmup_steps, warmup_ceiling))
    num_warmup_steps = min(num_warmup_steps, max(num_training_steps - 1, 1))

    if stage_name == "warmup":
        def lr_lambda(current_step: int) -> float:
            return _linear_warmup_only_lambda(current_step, num_warmup_steps)
    else:
        def lr_lambda(current_step: int) -> float:
            return _cosine_with_warmup_lambda(
                current_step,
                num_warmup_steps,
                num_training_steps,
                min_lr_ratio=min_lr_ratio,
            )

    return LambdaLR(optimizer, lr_lambda)


def move_batch(batch, device):
    # Move every tensor field to the chosen device and leave Python metadata untouched.
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def clip_gradients_from_optimizer(optimizer, max_grad_norm):
    # Clip exactly the parameters owned by the current optimizer rather than
    # rescanning the entire model each step.
    parameters = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter is None or (not parameter.requires_grad) or parameter.grad is None:
                continue
            identifier = id(parameter)
            if identifier in seen:
                continue
            seen.add(identifier)
            parameters.append(parameter)
    if parameters:
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_grad_norm)


def gather_variable_1d_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Gather a 1D tensor from every rank even when the local lengths differ.
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensor = tensor.to(device=device)
    local_length = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    world_size = dist.get_world_size()
    gathered_lengths = [torch.zeros_like(local_length) for _ in range(world_size)]
    dist.all_gather(gathered_lengths, local_length)
    lengths = [int(item.item()) for item in gathered_lengths]
    max_length = max(lengths, default=0)
    if tensor.numel() < max_length:
        pad = torch.zeros(max_length - tensor.numel(), device=device, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad], dim=0)
    gathered = [torch.zeros(max_length, device=device, dtype=tensor.dtype) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    slices = [
        rank_tensor[:rank_length]
        for rank_tensor, rank_length in zip(gathered, lengths)
        if rank_length > 0
    ]
    if not slices:
        return torch.zeros(0, device=device, dtype=tensor.dtype)
    return torch.cat(slices, dim=0)


def safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    weights = weights.to(dtype=torch.float32)
    denom = weights.sum().item()
    if denom <= 0:
        return None
    return float((values.to(dtype=torch.float32) * weights).sum().item() / denom)


def compute_weighted_auc(
    labels: torch.Tensor,
    scores: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> float | None:
    # Weighted ROC-AUC via the weighted Mann-Whitney statistic. Returns None
    # when the validation split only contains one class.
    labels = labels.to(dtype=torch.long)
    scores = scores.to(dtype=torch.float32)
    if sample_weight is None:
        sample_weight = torch.ones_like(scores, dtype=torch.float32)
    else:
        sample_weight = sample_weight.to(dtype=torch.float32)
    positive_mask = labels == 1
    negative_mask = labels == 0
    total_pos = float(sample_weight[positive_mask].sum().item())
    total_neg = float(sample_weight[negative_mask].sum().item())
    if total_pos <= 0 or total_neg <= 0:
        return None

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_weights = sample_weight[order]

    auc_contrib = 0.0
    cumulative_negative_weight = 0.0
    start = 0
    num_items = sorted_scores.numel()
    while start < num_items:
        end = start + 1
        while end < num_items and float(sorted_scores[end].item()) == float(sorted_scores[start].item()):
            end += 1
        group_labels = sorted_labels[start:end]
        group_weights = sorted_weights[start:end]
        group_positive_weight = float(group_weights[group_labels == 1].sum().item())
        group_negative_weight = float(group_weights[group_labels == 0].sum().item())
        auc_contrib += group_positive_weight * cumulative_negative_weight
        auc_contrib += 0.5 * group_positive_weight * group_negative_weight
        cumulative_negative_weight += group_negative_weight
        start = end
    return float(auc_contrib / (total_pos * total_neg))


def build_bucket_accuracy(mask: torch.Tensor, correct: torch.Tensor, weights: torch.Tensor) -> dict[str, float | int | None]:
    count = int(mask.sum().item())
    if count <= 0:
        return {
            "count": 0,
            "accuracy": None,
            "weighted_accuracy": None,
        }
    masked_correct = correct[mask].to(dtype=torch.float32)
    masked_weights = weights[mask].to(dtype=torch.float32)
    return {
        "count": count,
        "accuracy": float(masked_correct.mean().item()),
        "weighted_accuracy": safe_weighted_mean(masked_correct, masked_weights),
    }


def compute_eval_metrics(
    logits: torch.Tensor,
    teacher_score_a: torch.Tensor,
    teacher_score_b: torch.Tensor,
    sample_weight: torch.Tensor,
    stage_id: torch.Tensor,
    step_index: torch.Tensor,
    sample_index: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pred_prob = torch.sigmoid(logits.to(dtype=torch.float32))
    pred_label = (logits > 0).to(dtype=torch.long)
    delta = teacher_score_a.to(dtype=torch.float32) - teacher_score_b.to(dtype=torch.float32)
    target_label = (delta > 0).to(dtype=torch.long)
    margin = delta.abs()
    correct = (pred_label == target_label)
    correct_float = correct.to(dtype=torch.float32)
    sample_weight = sample_weight.to(dtype=torch.float32)

    metrics: dict[str, Any] = {
        "num_eval_samples": int(logits.numel()),
        "pairwise_accuracy": float(correct_float.mean().item()) if correct_float.numel() > 0 else None,
        "weighted_pairwise_accuracy": safe_weighted_mean(correct_float, sample_weight),
        "auc": compute_weighted_auc(target_label, pred_prob, sample_weight),
        "mean_pred_prob": float(pred_prob.mean().item()) if pred_prob.numel() > 0 else None,
        "mean_margin": float(margin.mean().item()) if margin.numel() > 0 else None,
        "brier_score": float(((pred_prob - target_label.to(dtype=torch.float32)) ** 2).mean().item()) if pred_prob.numel() > 0 else None,
    }

    by_stage: dict[str, Any] = {}
    for stage_value, stage_name in STAGE_ID_TO_NAME.items():
        stage_mask = stage_id == stage_value
        by_stage[stage_name] = build_bucket_accuracy(stage_mask, correct, sample_weight)
    metrics["accuracy_by_stage"] = by_stage

    valid_steps = torch.unique(step_index[step_index >= 0]).tolist()
    by_step: dict[str, Any] = {}
    for step_value in sorted(int(step) for step in valid_steps):
        step_mask = step_index == step_value
        by_step[str(step_value)] = build_bucket_accuracy(step_mask, correct, sample_weight)
    metrics["accuracy_by_step"] = by_step

    by_margin: dict[str, Any] = {}
    for bucket_name, low, high in MARGIN_BUCKETS:
        margin_mask = margin >= low
        if math.isfinite(high):
            margin_mask = margin_mask & (margin < high)
        by_margin[bucket_name] = build_bucket_accuracy(margin_mask, correct, sample_weight)
    metrics["accuracy_by_margin_bucket"] = by_margin

    prediction_rows: list[dict[str, Any]] = []
    for i in range(logits.numel()):
        stage_value = int(stage_id[i].item())
        prediction_rows.append(
            {
                "sample_index": int(sample_index[i].item()),
                "step_index": int(step_index[i].item()),
                "stage_id": stage_value,
                "stage_name": STAGE_ID_TO_NAME.get(stage_value, "unknown"),
                "teacher_score_a": float(teacher_score_a[i].item()),
                "teacher_score_b": float(teacher_score_b[i].item()),
                "teacher_delta": float(delta[i].item()),
                "margin": float(margin[i].item()),
                "sample_weight": float(sample_weight[i].item()),
                "pred_logit": float(logits[i].item()),
                "pred_prob": float(pred_prob[i].item()),
                "pred_label": int(pred_label[i].item()),
                "target_label": int(target_label[i].item()),
                "correct": bool(correct[i].item()),
            }
        )
    return metrics, prediction_rows


def save_prediction_rows(
    output_dir: Path,
    stage_name: str,
    phase_index: int,
    epoch: int,
    prediction_rows: list[dict[str, Any]],
):
    prediction_dir = Path(output_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{stage_name}_phase{phase_index}_epoch_{epoch:03d}_val_predictions.jsonl"
    lines = [json.dumps(row, ensure_ascii=False) for row in prediction_rows]
    prediction_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return prediction_path


def save_metrics_json(output_dir: Path, metrics: list[dict[str, Any]]):
    # Flush the running metrics list to disk after every epoch so monitoring
    # scripts can read the latest state without waiting for training to finish.
    metrics_path = Path(output_dir) / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics_path


def run_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    pairwise_loss,
    warmup_loss,
    stage_name,
    phase_index,
    epoch,
    max_steps=None,
):
    # Single training epoch shared by warmup, lora, and curriculum stages.
    model.train()
    total_loss = 0.0
    total_steps = 0
    progress_total = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    progress = tqdm(
        dataloader,
        total=progress_total,
        disable=not is_main_process(),
        leave=False,
        dynamic_ncols=True,
        desc=f"{stage_name}:p{phase_index}:e{epoch}:train",
    )
    for step_index, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["f1"],
            batch["f2"],
            batch["text_emb"],
            batch["image_emb"],
            batch["stage_id"],
            return_aux_stats=(stage_name == "warmup"),
        )
        loss = pairwise_loss(
            outputs["logit"],
            batch["teacher_score_a"],
            batch["teacher_score_b"],
            batch["sample_weight"],
        )
        if stage_name == "warmup":
            # Only warmup includes the auxiliary projector-distribution penalty.
            loss = loss + warmup_loss(outputs["proj_mean"], outputs["proj_std"])
        loss.backward()
        clip_gradients_from_optimizer(optimizer, MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        loss_value = float(loss.item())
        total_loss += loss_value
        total_steps += 1
        if is_main_process():
            current_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{total_loss / max(total_steps, 1):.4f}",
                lr=f"{current_lr:.2e}",
            )
        if max_steps is not None and step_index >= max_steps:
            break
    progress.close()
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([total_loss, total_steps], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_steps = int(stats[1].item())
    return total_loss / max(total_steps, 1)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    pairwise_loss,
    stage_name,
    phase_index,
    epoch,
    max_steps=None,
):
    # Validation mirrors the pairwise ranking objective without the warmup-only
    # auxiliary statistics loss.
    model.eval()
    total_loss = 0.0
    total_steps = 0
    logits_chunks = []
    teacher_a_chunks = []
    teacher_b_chunks = []
    weight_chunks = []
    stage_chunks = []
    step_chunks = []
    sample_index_chunks = []
    progress_total = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    progress = tqdm(
        dataloader,
        total=progress_total,
        disable=not is_main_process(),
        leave=False,
        dynamic_ncols=True,
        desc=f"{stage_name}:p{phase_index}:e{epoch}:eval",
    )
    for step_index, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        outputs = model(
            batch["f1"],
            batch["f2"],
            batch["text_emb"],
            batch["image_emb"],
            batch["stage_id"],
            return_aux_stats=False,
        )
        loss = pairwise_loss(
            outputs["logit"],
            batch["teacher_score_a"],
            batch["teacher_score_b"],
            batch["sample_weight"],
        )
        logits_chunks.append(outputs["logit"].detach().to(dtype=torch.float32, device=device))
        teacher_a_chunks.append(batch["teacher_score_a"].detach().to(dtype=torch.float32, device=device))
        teacher_b_chunks.append(batch["teacher_score_b"].detach().to(dtype=torch.float32, device=device))
        weight_chunks.append(batch["sample_weight"].detach().to(dtype=torch.float32, device=device))
        stage_chunks.append(batch["stage_id"].detach().to(dtype=torch.long, device=device))
        step_chunks.append(batch["step_index"].detach().to(dtype=torch.long, device=device))
        sample_index_chunks.append(batch["sample_index"].detach().to(dtype=torch.long, device=device))
        loss_value = float(loss.item())
        total_loss += loss_value
        total_steps += 1
        if is_main_process():
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                avg=f"{total_loss / max(total_steps, 1):.4f}",
            )
        if max_steps is not None and step_index >= max_steps:
            break
    progress.close()
    if dist.is_available() and dist.is_initialized():
        stats = torch.tensor([total_loss, total_steps], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_steps = int(stats[1].item())
    average_loss = total_loss / max(total_steps, 1)

    local_logits = torch.cat(logits_chunks, dim=0) if logits_chunks else torch.zeros(0, device=device, dtype=torch.float32)
    local_teacher_a = torch.cat(teacher_a_chunks, dim=0) if teacher_a_chunks else torch.zeros(0, device=device, dtype=torch.float32)
    local_teacher_b = torch.cat(teacher_b_chunks, dim=0) if teacher_b_chunks else torch.zeros(0, device=device, dtype=torch.float32)
    local_weight = torch.cat(weight_chunks, dim=0) if weight_chunks else torch.zeros(0, device=device, dtype=torch.float32)
    local_stage = torch.cat(stage_chunks, dim=0) if stage_chunks else torch.zeros(0, device=device, dtype=torch.long)
    local_step = torch.cat(step_chunks, dim=0) if step_chunks else torch.zeros(0, device=device, dtype=torch.long)
    local_sample_index = torch.cat(sample_index_chunks, dim=0) if sample_index_chunks else torch.zeros(0, device=device, dtype=torch.long)

    gathered_logits = gather_variable_1d_tensor(local_logits, device).cpu()
    gathered_teacher_a = gather_variable_1d_tensor(local_teacher_a, device).cpu()
    gathered_teacher_b = gather_variable_1d_tensor(local_teacher_b, device).cpu()
    gathered_weight = gather_variable_1d_tensor(local_weight, device).cpu()
    gathered_stage = gather_variable_1d_tensor(local_stage, device).cpu()
    gathered_step = gather_variable_1d_tensor(local_step, device).cpu()
    gathered_sample_index = gather_variable_1d_tensor(local_sample_index, device).cpu()

    eval_metrics, prediction_rows = compute_eval_metrics(
        gathered_logits,
        gathered_teacher_a,
        gathered_teacher_b,
        gathered_weight,
        gathered_stage,
        gathered_step,
        gathered_sample_index,
    )
    eval_metrics["loss"] = average_loss
    return eval_metrics, prediction_rows


def should_save_epoch_checkpoint(epoch, interval):
    if interval is None:
        return False
    interval = int(interval)
    if interval <= 0:
        return False
    return epoch % interval == 0


def save_checkpoint(model, optimizer, scheduler, output_dir, stage_name, epoch):
    # Save a periodic per-epoch checkpoint directory. Rolling resume state is
    # handled separately by save_resume_checkpoint().
    checkpoint_dir = Path(output_dir) / f"{stage_name}_epoch_{epoch:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    payload = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "stage": stage_name,
        "epoch": epoch,
    }
    torch.save(payload, checkpoint_dir / "checkpoint.pt")


def save_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    output_dir,
    stage_name,
    phase_index,
    allowed_stages,
    epoch,
    metrics,
):
    # Persist enough trainer state to continue the staged training loop without
    # manually tracking which phase was already finished.
    model_to_save = unwrap_model(model)
    payload = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "stage": stage_name,
        "phase_index": phase_index,
        "allowed_stages": allowed_stages,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, Path(output_dir) / "latest_checkpoint.pt")


def load_resume_checkpoint(resume_path, device):
    return torch.load(resume_path, map_location=device)


def count_trainable_ratio(model):
    # Useful for checking the design target that LoRA-stage trainable params stay small.
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total, trainable / max(total, 1)


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(resolve_repo_relative_path(args.output_dir or config["paths"]["output_dir"]))
    device, rank, world_size, distributed = init_distributed(args)
    try:
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
        ddp_barrier()

        train_manifest = resolve_repo_relative_path(args.train_manifest or config["paths"]["train_manifest"])
        val_manifest = resolve_repo_relative_path(args.val_manifest or config["paths"].get("val_manifest"))
        training_cfg = config["training"]
        if args.disable_curriculum:
            training_cfg = dict(training_cfg)
            epochs_cfg = dict(training_cfg.get("epochs", {}))
            epochs_cfg["curriculum"] = 0
            training_cfg["epochs"] = epochs_cfg
        checkpoint_interval_epochs = training_cfg.get("checkpoint_interval_epochs", 1)
        smoke_cfg = training_cfg.get("smoke_test", {})
        max_train_samples = args.max_train_samples
        max_val_samples = args.max_val_samples
        max_train_steps_per_epoch = args.max_train_steps_per_epoch
        max_eval_steps = args.max_eval_steps
        if args.smoke_test:
            # Smoke-test mode forces a small end-to-end run so data loading,
            # forward, backward, DDP, and checkpoint writing all get exercised.
            max_train_samples = smoke_cfg.get("max_train_samples", max_train_samples)
            max_val_samples = smoke_cfg.get("max_val_samples", max_val_samples)
            max_train_steps_per_epoch = smoke_cfg.get("max_train_steps_per_epoch", max_train_steps_per_epoch)
            max_eval_steps = smoke_cfg.get("max_eval_steps", max_eval_steps)

        model_cfg = config["model"]
        model = ScoreModelV2(
            qwen_model_path=resolve_repo_relative_path(model_cfg["qwen_model_path"]),
            latent_dim=int(model_cfg["latent_dim"]),
            patch_dim=int(model_cfg["patch_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            text_dim=int(model_cfg["text_dim"]),
            image_dim=int(model_cfg["image_dim"]),
            frames=int(model_cfg["frames"]),
            height=int(model_cfg["height"]),
            width=int(model_cfg["width"]),
            context_max_text_tokens=int(model_cfg["context_max_text_tokens"]),
            context_max_image_tokens=int(model_cfg["context_max_image_tokens"]),
            num_query_tokens=int(model_cfg["num_query_tokens"]),
            num_qwen_layers=int(model_cfg["num_qwen_layers"]),
            lora_r=int(model_cfg["lora_r"]),
            lora_alpha=int(model_cfg["lora_alpha"]),
            lora_dropout=float(model_cfg["lora_dropout"]),
            torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
            attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
            bidirectional_attention=bool(model_cfg.get("bidirectional_attention", False)),
            readout_mode=model_cfg.get("readout_mode", "query"),
            task_prompt=model_cfg.get("task_prompt", ""),
            freeze_qwen_backbone=bool(model_cfg.get("freeze_qwen_backbone", True)),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        ).to(device)
        resume_path = resolve_resume_path(args, output_dir)
        resume_state = None
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            resume_state = load_resume_checkpoint(resume_path, device)
            model.load_state_dict(resume_state["model"], strict=False)
            ddp_print(f"[resume] loaded model weights from {resume_path}")

        pairwise_loss = PairwiseScoreLoss(
            tau=float(config["loss"]["tau"]),
            margin=float(config["loss"]["margin"]),
        )
        warmup_loss = WarmupAlignmentLoss(weight=float(config["loss"]["warmup_alignment_weight"]))

        # Each tuple is (stage_name, stage_selector). Curriculum is expanded into
        # multiple phases later so we can expose only late -> middle -> early data.
        stage_plan = [
            ("warmup", list(config["training"]["curriculum"].get("warmup_stages", [0, 1, 2]))),
            ("lora", list(config["training"]["curriculum"].get("lora_stages", [0, 1, 2]))),
            ("curriculum", list(config["training"]["curriculum"].get("curriculum_stage_ids", [2, 1, 0]))),
        ]

        metrics = list(resume_state.get("metrics", [])) if resume_state is not None else []
        resume_stage = resume_state.get("stage") if resume_state is not None else None
        normalized_resume_stage = normalize_stage_name(resume_stage)
        resume_phase_index = int(resume_state.get("phase_index", 1)) if resume_state is not None else 1
        resume_epoch = int(resume_state.get("epoch", 0)) if resume_state is not None else 0
        batch_size = int(training_cfg["batch_size"])
        num_workers = int(training_cfg["num_workers"])
        ddp_print(
            f"[dist] distributed={distributed} rank={rank} world_size={world_size} "
            f"device={device} batch_size={batch_size} num_workers={num_workers} "
            f"smoke_test={args.smoke_test}"
        )
        if resume_state is not None:
            ddp_print(
                f"[resume] stage={resume_stage} phase_index={resume_phase_index} "
                f"epoch={resume_epoch}"
            )

        for stage_name, stage_selector in stage_plan:
            epochs = int(training_cfg["epochs"].get(stage_name, 0))
            if epochs <= 0:
                continue
            if args.smoke_test:
                # In smoke-test mode we only need a single epoch per active phase.
                epochs = 1

            set_stage_trainability(model, stage_name)
            trainable, total, ratio = count_trainable_ratio(model)
            ddp_print(f"[stage={stage_name}] trainable={trainable} total={total} ratio={ratio:.4f}")

            if distributed:
                ddp_kwargs = {"find_unused_parameters": args.ddp_find_unused_parameters}
                if device.type == "cuda":
                    ddp_kwargs["device_ids"] = [device.index]
                    ddp_kwargs["output_device"] = device.index
                stage_model = DDP(model, **ddp_kwargs)
            else:
                stage_model = model

            if stage_name == "curriculum":
                # Curriculum runs one phase per stage bucket, from easier to harder.
                phase_schedule = [[stage] for stage in stage_selector]
            else:
                phase_schedule = [stage_selector]

            for phase_index, allowed_stages in enumerate(phase_schedule, start=1):
                if resume_state is not None:
                    if stage_sort_key(stage_name) < stage_sort_key(normalized_resume_stage):
                        continue
                    if stage_name == normalized_resume_stage and phase_index < resume_phase_index:
                        continue
                # Rebuild datasets per phase so filtering happens at the manifest level.
                train_dataset = PairwiseLatentDatasetV2(
                    train_manifest,
                    curriculum_stages=allowed_stages,
                    max_samples=max_train_samples,
                )
                val_dataset = (
                    PairwiseLatentDatasetV2(
                        val_manifest,
                        curriculum_stages=allowed_stages,
                        max_samples=max_val_samples,
                    )
                    if val_manifest else None
                )
                train_sampler = (
                    DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                    if distributed else None
                )
                val_sampler = (
                    DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
                    if (distributed and val_dataset is not None) else None
                )
                train_loader = build_dataloader(
                    train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    sampler=train_sampler,
                    shuffle=train_sampler is None,
                    pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers,
                )
                val_loader = None
                if val_dataset is not None:
                    val_loader = build_dataloader(
                        val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        sampler=val_sampler,
                        shuffle=False,
                        pin_memory=args.pin_memory,
                        persistent_workers=args.persistent_workers,
                    )

                optimizer = build_optimizer(unwrap_model(stage_model), stage_name, config)
                steps_per_epoch = len(train_loader) if max_train_steps_per_epoch is None else min(len(train_loader), max_train_steps_per_epoch)
                num_training_steps = max(steps_per_epoch * epochs, 1)
                scheduler = build_scheduler(optimizer, stage_name, num_training_steps)
                if (
                    resume_state is not None
                    and stage_name == normalized_resume_stage
                    and phase_index == resume_phase_index
                ):
                    optimizer.load_state_dict(resume_state["optimizer"])
                    if resume_state.get("scheduler") is not None:
                        scheduler.load_state_dict(resume_state["scheduler"])
                    ddp_print(
                        f"[resume] loaded optimizer/scheduler state for stage={stage_name} phase={phase_index}"
                    )
                for epoch in range(1, epochs + 1):
                    if (
                        resume_state is not None
                        and stage_name == normalized_resume_stage
                        and phase_index == resume_phase_index
                        and epoch <= resume_epoch
                    ):
                        continue
                    if train_sampler is not None:
                        # Keep shuffling consistent but different across epochs.
                        train_sampler.set_epoch(epoch + phase_index * 1000)
                    train_loss = run_epoch(
                        stage_model,
                        train_loader,
                        optimizer,
                        scheduler,
                        device,
                        pairwise_loss,
                        warmup_loss,
                        stage_name,
                        phase_index,
                        epoch,
                        max_steps=max_train_steps_per_epoch,
                    )
                    val_loss = None
                    val_metrics = None
                    if val_loader is not None:
                        val_metrics, prediction_rows = evaluate(
                            stage_model,
                            val_loader,
                            device,
                            pairwise_loss,
                            stage_name,
                            phase_index,
                            epoch,
                            max_steps=max_eval_steps,
                        )
                        val_loss = val_metrics["loss"]
                    else:
                        prediction_rows = []
                    record = {
                        "stage": stage_name,
                        "phase": phase_index,
                        "allowed_stages": allowed_stages,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                    if val_metrics is not None:
                        record.update(
                            {
                                "val_pairwise_accuracy": val_metrics["pairwise_accuracy"],
                                "val_weighted_pairwise_accuracy": val_metrics["weighted_pairwise_accuracy"],
                                "val_auc": val_metrics["auc"],
                                "val_mean_pred_prob": val_metrics["mean_pred_prob"],
                                "val_mean_margin": val_metrics["mean_margin"],
                                "val_brier_score": val_metrics["brier_score"],
                                "val_num_eval_samples": val_metrics["num_eval_samples"],
                                "val_accuracy_by_stage": val_metrics["accuracy_by_stage"],
                                "val_accuracy_by_step": val_metrics["accuracy_by_step"],
                                "val_accuracy_by_margin_bucket": val_metrics["accuracy_by_margin_bucket"],
                            }
                        )
                    metrics.append(record)
                    ddp_print(json.dumps(record, ensure_ascii=False))
                    if is_main_process():
                        if prediction_rows:
                            prediction_path = save_prediction_rows(
                                output_dir,
                                stage_name,
                                phase_index,
                                epoch,
                                prediction_rows,
                            )
                            record["val_predictions_path"] = str(prediction_path)
                        save_metrics_json(output_dir, metrics)
                        save_resume_checkpoint(
                            stage_model,
                            optimizer,
                            scheduler,
                            output_dir,
                            stage_name,
                            phase_index,
                            allowed_stages,
                            epoch,
                            metrics,
                        )
                        if should_save_epoch_checkpoint(epoch, checkpoint_interval_epochs):
                            save_checkpoint(
                                stage_model,
                                optimizer,
                                scheduler,
                                output_dir,
                                f"{stage_name}_phase{phase_index}",
                                epoch,
                            )
                    ddp_barrier()

                if (
                    resume_state is not None
                    and stage_name == normalized_resume_stage
                    and phase_index == resume_phase_index
                ):
                    # Resume state only applies to the first unfinished phase.
                    resume_state = None

            if isinstance(stage_model, DDP):
                model = unwrap_model(stage_model)

        # Keep one compact metrics file for plotting or later inspection.
        if is_main_process():
            metadata = {
                "args": vars(args),
                "smoke_test": {
                    "enabled": args.smoke_test,
                    "max_train_samples": max_train_samples,
                    "max_val_samples": max_val_samples,
                    "max_train_steps_per_epoch": max_train_steps_per_epoch,
                    "max_eval_steps": max_eval_steps,
                },
                "rank": rank,
                "world_size": world_size,
                "distributed": distributed,
                "device": str(device),
                "ddp_backend": args.ddp_backend if device.type == "cuda" else "gloo",
            }
            (output_dir / "train_config.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        ddp_barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
