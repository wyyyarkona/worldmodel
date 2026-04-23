from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from score_model_v2.losses import PairwiseScoreLoss
from score_model_v2.models.score_model_v2 import ScoreModelV2
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
        description="Evaluate a trained score_model_v2 checkpoint on a specified manifest."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--name", type=str, default="eval")
    return parser.parse_args()


def build_model_from_config(config: dict, device: torch.device):
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
    return model


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
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
        stage_name=args.name,
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
    metrics["predictions_path"] = str(prediction_path)
    metrics_path = output_dir / f"{args.name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[done] metrics saved to {metrics_path}")
    print(f"[done] predictions saved to {prediction_path}")


if __name__ == "__main__":
    main()
