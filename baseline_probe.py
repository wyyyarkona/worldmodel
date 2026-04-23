from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import random_split

from score_model_v2.losses import PairwiseScoreLoss
from score_model_v2.models.projectors import ContextProjector, VideoProjector
from score_model_v2.train_v2 import (
    PairwiseLatentDatasetV2,
    build_dataloader,
    collate_batch,
    load_config,
    move_batch,
    resolve_repo_relative_path,
)


class MeanDifferenceProbe(nn.Module):
    # Simple sanity-check baseline: compare mean pooled h1/h2 features directly,
    # without using the Qwen comparator. If this probe can learn but the full
    # model cannot, the issue is likely inside the comparator path.
    def __init__(self, latent_dim=16, patch_dim=1536, hidden_dim=2048):
        super().__init__()
        self.video_projector = VideoProjector(
            latent_dim=latent_dim,
            patch_dim=patch_dim,
            hidden_dim=hidden_dim,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, f1, f2):
        h1 = self.video_projector(f1).reshape(f1.size(0), -1, self.head.in_features).mean(dim=1)
        h2 = self.video_projector(f2).reshape(f2.size(0), -1, self.head.in_features).mean(dim=1)
        diff = self.norm(h1 - h2)
        logit = self.head(diff).squeeze(-1)
        score = torch.sigmoid(logit)
        return {
            "logit": logit,
            "score": score,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny mean-difference baseline probe for score_model_v2 data sanity checking."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate_probe(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    pred_probs = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        outputs = model(batch["f1"], batch["f2"])
        pred = (outputs["logit"] > 0).long()
        target = (batch["teacher_score_a"] - batch["teacher_score_b"] > 0).long()
        correct += int((pred == target).sum().item())
        total += int(target.numel())
        pred_probs.extend(float(x.item()) for x in outputs["score"])
    summary = {
        "accuracy": (correct / total) if total > 0 else None,
        "pred_prob_mean": sum(pred_probs) / len(pred_probs) if pred_probs else None,
        "pred_prob_min": min(pred_probs) if pred_probs else None,
        "pred_prob_max": max(pred_probs) if pred_probs else None,
    }
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    model_cfg = config["model"]
    if args.train_manifest and args.val_manifest:
        train_dataset = PairwiseLatentDatasetV2(
            resolve_repo_relative_path(args.train_manifest),
            curriculum_stages=None,
            max_samples=args.max_samples,
        )
        val_dataset = PairwiseLatentDatasetV2(
            resolve_repo_relative_path(args.val_manifest),
            curriculum_stages=None,
            max_samples=args.max_samples,
        )
    elif args.manifest:
        dataset = PairwiseLatentDatasetV2(
            resolve_repo_relative_path(args.manifest),
            curriculum_stages=None,
            max_samples=args.max_samples,
        )
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        raise ValueError("Provide either --manifest, or both --train_manifest and --val_manifest.")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=None,
        shuffle=True,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=None,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    model = MeanDifferenceProbe(
        latent_dim=int(model_cfg["latent_dim"]),
        patch_dim=int(model_cfg["patch_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(config["training"]["weight_decay"]))
    loss_fn = PairwiseScoreLoss(
        tau=float(config["loss"]["tau"]),
        margin=float(config["loss"]["margin"]),
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["f1"], batch["f2"])
            loss = loss_fn(
                outputs["logit"],
                batch["teacher_score_a"],
                batch["teacher_score_b"],
                batch["sample_weight"],
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            steps += 1

        train_loss = running_loss / max(steps, 1)
        val_summary = evaluate_probe(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "val_accuracy": val_summary["accuracy"],
            "val_pred_prob_mean": val_summary["pred_prob_mean"],
            "val_pred_prob_min": val_summary["pred_prob_min"],
            "val_pred_prob_max": val_summary["pred_prob_max"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] baseline history saved to {output_path}")


if __name__ == "__main__":
    main()
