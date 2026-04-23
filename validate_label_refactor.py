"""Validation script for the text-label refactor of ScoreModelV2.

Run this on the server where Qwen2.5-VL + torch + transformers + peft are
installed. It performs the two checks the refactor plan called for:

1. Prints the tokenization info for every static label / task prompt.
2. Runs a swap test: forward(f1, f2, ...) vs forward(f2, f1, ...) and reports
   |out1 - out2|.mean (expected to be > 0 — asymmetric) and |out1 + out2|.mean
   (no expectation, just sanity print).

Usage:
    python validate_label_refactor.py \
        --config configs/v2_default.yaml \
        --val_manifest /path/to/val_manifest.jsonl \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys

import torch

from score_model_v2.train_v2 import (
    PairwiseLatentDatasetV2,
    build_dataloader,
    load_config,
    move_batch,
    resolve_repo_relative_path,
)
from score_model_v2.eval_manifest import build_model_from_config


def parse_args():
    parser = argparse.ArgumentParser(description="Validate the text-label refactor.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    model = build_model_from_config(config, device)
    model.eval()

    # Validation 1: label tokenization info
    print("=" * 72)
    print("Validation 1: TextLabelEmbedder tokenization")
    print("=" * 72)
    model.text_labels.print_label_info()

    # Validation 2: swap test
    print()
    print("=" * 72)
    print("Validation 2: swap test (forward asymmetry)")
    print("=" * 72)
    dataset = PairwiseLatentDatasetV2(
        resolve_repo_relative_path(args.val_manifest),
        curriculum_stages=None,
        max_samples=args.max_samples,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=None,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )
    batch = next(iter(dataloader))
    batch = move_batch(batch, device)

    with torch.no_grad():
        out1 = model(
            batch["f1"], batch["f2"], batch["text_emb"], batch["image_emb"], batch["stage_id"]
        )["logit"]
        out2 = model(
            batch["f2"], batch["f1"], batch["text_emb"], batch["image_emb"], batch["stage_id"]
        )["logit"]
    out1 = out1.detach().to(dtype=torch.float32)
    out2 = out2.detach().to(dtype=torch.float32)

    abs_diff_mean = (out1 - out2).abs().mean().item()
    abs_sum_mean = (out1 + out2).abs().mean().item()
    print(f"|out1 - out2|.mean = {abs_diff_mean:.4f}")
    print(f"|out1 + out2|.mean = {abs_sum_mean:.4f}")
    print()
    print("Interpretation:")
    print("  - |out1 - out2| should be noticeably > 0: the model produces")
    print("    different logits when f1 and f2 are swapped, which is required")
    print("    for pairwise ranking to be learnable.")
    print("  - |out1 + out2| near 0 would mean logit(f1,f2) ≈ -logit(f2,f1),")
    print("    i.e. the head is already anti-symmetric (not guaranteed here).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
