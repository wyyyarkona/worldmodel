from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from score_model_v2.baseline_train_ddp import PairwiseBaselineScorerA, PairwiseBaselineScorerB
from score_model_v2.losses import PairwiseScoreLoss
from score_model_v2.train_v2 import (
    PairwiseLatentDatasetV2,
    build_dataloader,
    evaluate,
    load_config,
    resolve_repo_relative_path,
    save_prediction_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained baseline checkpoint on a specified manifest."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_variant", type=str, default=None, choices=["a", "b", None])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--name", type=str, default="baseline_eval")
    return parser.parse_args()


def infer_variant(args, checkpoint):
    if args.model_variant is not None:
        return args.model_variant
    variant = checkpoint.get("model_variant")
    if variant in {"a", "b"}:
        return variant
    stage_name = str(checkpoint.get("stage_name", ""))
    if "baseline_b" in stage_name:
        return "b"
    return "a"


def build_baseline_from_config(config: dict, device: torch.device, variant: str):
    model_cfg = config["model"]
    if variant == "a":
        model = PairwiseBaselineScorerA(
            latent_dim=int(model_cfg["latent_dim"]),
            patch_dim=int(model_cfg["patch_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            head_hidden_dim=int(model_cfg.get("hidden_dim", 2048)),
        )
    else:
        model = PairwiseBaselineScorerB(
            latent_dim=int(model_cfg["latent_dim"]),
            patch_dim=int(model_cfg["patch_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
        )
    model.variant_name = variant
    return model.to(device)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    variant = infer_variant(args, checkpoint)
    model = build_baseline_from_config(config, device, variant)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    training_cfg = config["training"]
    batch_size = int(args.batch_size or training_cfg["batch_size"])
    num_workers = int(args.num_workers or training_cfg["num_workers"])
    dataset = PairwiseLatentDatasetV2(
        resolve_repo_relative_path(args.manifest),
        curriculum_stages=None,
        max_samples=args.max_samples,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=None,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )
    pairwise_loss = PairwiseScoreLoss(
        tau=float(config["loss"]["tau"]),
        margin=float(config["loss"]["margin"]),
    )
    metrics, prediction_rows = evaluate(
        model,
        dataloader,
        device,
        pairwise_loss,
        stage_name=f"{args.name}_{variant}",
        phase_index=1,
        epoch=1,
        max_steps=args.max_eval_steps,
    )
    prediction_path = save_prediction_rows(
        output_dir=output_dir,
        stage_name=args.name,
        phase_index=1,
        epoch=1,
        prediction_rows=prediction_rows,
    )
    metrics["model_variant"] = variant
    metrics["predictions_path"] = str(prediction_path)
    metrics_path = output_dir / f"{args.name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] baseline variant: {variant}")
    print(f"[done] metrics saved to {metrics_path}")
    print(f"[done] predictions saved to {prediction_path}")


if __name__ == "__main__":
    main()
