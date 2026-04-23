from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from score_model_v2.losses import PairwiseScoreLoss
from score_model_v2.train_v2 import (
    PairwiseLatentDatasetV2,
    build_dataloader,
    build_scheduler,
    cleanup_distributed,
    count_trainable_ratio,
    ddp_barrier,
    ddp_print,
    init_distributed,
    is_main_process,
    load_config,
    move_batch,
    resolve_repo_relative_path,
    save_checkpoint,
    save_prediction_rows,
    save_resume_checkpoint,
    should_save_epoch_checkpoint,
    unwrap_model,
    evaluate,
    clip_gradients_from_optimizer,
)
from score_model_v2.models.projectors import VideoProjector


class PairwiseBaselineScorerA(nn.Module):
    # Structure A:
    # [h1, h2, h1-h2, h1*h2] -> MLP head
    def __init__(self, latent_dim=16, patch_dim=1536, hidden_dim=2048, head_hidden_dim=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.video_projector = VideoProjector(
            latent_dim=latent_dim,
            patch_dim=patch_dim,
            hidden_dim=hidden_dim,
        )
        self.compare_norm = nn.LayerNorm(hidden_dim * 4)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(head_hidden_dim, 1),
        )

    def encode_video(self, latent):
        tokens = self.video_projector(latent)
        return tokens.reshape(tokens.size(0), -1, self.hidden_dim).mean(dim=1)

    def forward(
        self,
        f1,
        f2,
        text_emb=None,
        image_emb=None,
        stage_id=None,
        return_aux_stats=False,
    ):
        h1 = self.encode_video(f1)
        h2 = self.encode_video(f2)
        features = torch.cat([h1, h2, h1 - h2, h1 * h2], dim=-1)
        features = self.compare_norm(features)
        logit = self.score_head(features).squeeze(-1)
        score = torch.sigmoid(logit)
        return {
            "logit": logit,
            "score": score,
        }


class PairwiseBaselineScorerB(nn.Module):
    # Structure B:
    # (h1-h2) -> LayerNorm -> Linear
    def __init__(self, latent_dim=16, patch_dim=1536, hidden_dim=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.video_projector = VideoProjector(
            latent_dim=latent_dim,
            patch_dim=patch_dim,
            hidden_dim=hidden_dim,
        )
        self.compare_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)

    def encode_video(self, latent):
        tokens = self.video_projector(latent)
        return tokens.reshape(tokens.size(0), -1, self.hidden_dim).mean(dim=1)

    def forward(
        self,
        f1,
        f2,
        text_emb=None,
        image_emb=None,
        stage_id=None,
        return_aux_stats=False,
    ):
        h1 = self.encode_video(f1)
        h2 = self.encode_video(f2)
        features = self.compare_norm(h1 - h2)
        logit = self.score_head(features).squeeze(-1)
        score = torch.sigmoid(logit)
        return {
            "logit": logit,
            "score": score,
        }


def should_use_tqdm():
    return is_main_process() and sys.stdout.isatty()


def parse_args():
    parser = argparse.ArgumentParser(description="DDP baseline trainer for score_model_v2 pairwise data.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ddp_backend", type=str, default="nccl")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--no_auto_resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_hidden_dim", type=int, default=2048)
    parser.add_argument(
        "--model_variant",
        type=str,
        default="a",
        choices=["a", "b"],
        help="Baseline structure variant: a=[h1,h2,h1-h2,h1*h2]->MLP, b=(h1-h2)->LN->Linear",
    )
    return parser.parse_args()


def resolve_resume_path(args, output_dir: Path):
    if args.resume_from:
        return Path(args.resume_from)
    if args.no_auto_resume:
        return None
    latest_path = output_dir / "latest_checkpoint.pt"
    if latest_path.exists():
        return latest_path
    return None


def run_epoch_baseline(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    pairwise_loss,
    epoch,
    max_steps=None,
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    progress_total = min(len(dataloader), max_steps) if max_steps is not None else len(dataloader)
    use_tqdm = should_use_tqdm()
    progress = tqdm(
        dataloader,
        total=progress_total,
        disable=not use_tqdm,
        leave=False,
        dynamic_ncols=True,
        desc=f"baseline-{getattr(model.module if isinstance(model, DDP) else model, 'variant_name', 'a')}:p1:e{epoch}:train",
    )
    for step_index, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["f1"], batch["f2"])
        loss = pairwise_loss(
            outputs["logit"],
            batch["teacher_score_a"],
            batch["teacher_score_b"],
            batch["sample_weight"],
        )
        loss.backward()
        clip_gradients_from_optimizer(optimizer, max_grad_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
        total_steps += 1
        if use_tqdm:
            current_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(
                loss=f"{float(loss.item()):.4f}",
                avg=f"{total_loss / max(total_steps, 1):.4f}",
                lr=f"{current_lr:.2e}",
            )
        if max_steps is not None and step_index >= max_steps:
            break
    progress.close()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.tensor([total_loss, total_steps], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_loss = float(stats[0].item())
        total_steps = int(stats[1].item())
    return total_loss / max(total_steps, 1)


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(resolve_repo_relative_path(args.output_dir))
    device, rank, world_size, distributed = init_distributed(args)
    try:
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
        ddp_barrier()

        model_cfg = config["model"]
        if args.model_variant == "a":
            model = PairwiseBaselineScorerA(
                latent_dim=int(model_cfg["latent_dim"]),
                patch_dim=int(model_cfg["patch_dim"]),
                hidden_dim=int(model_cfg["hidden_dim"]),
                head_hidden_dim=int(args.head_hidden_dim),
            ).to(device)
        else:
            model = PairwiseBaselineScorerB(
                latent_dim=int(model_cfg["latent_dim"]),
                patch_dim=int(model_cfg["patch_dim"]),
                hidden_dim=int(model_cfg["hidden_dim"]),
            ).to(device)
        model.variant_name = args.model_variant

        resume_path = resolve_resume_path(args, output_dir)
        resume_state = None
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            resume_state = torch.load(resume_path, map_location=device)
            model.load_state_dict(resume_state["model"], strict=False)
            ddp_print(f"[baseline resume] loaded model weights from {resume_path}")

        train_dataset = PairwiseLatentDatasetV2(
            resolve_repo_relative_path(args.train_manifest),
            curriculum_stages=None,
            max_samples=args.max_train_samples,
        )
        val_dataset = (
            PairwiseLatentDatasetV2(
                resolve_repo_relative_path(args.val_manifest),
                curriculum_stages=None,
                max_samples=args.max_val_samples,
            )
            if args.val_manifest
            else None
        )

        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if distributed
            else None
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if (distributed and val_dataset is not None)
            else None
        )

        batch_size = int(config["training"]["batch_size"])
        num_workers = int(config["training"]["num_workers"])
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

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        steps_per_epoch = len(train_loader) if args.max_train_steps_per_epoch is None else min(len(train_loader), args.max_train_steps_per_epoch)
        num_training_steps = max(steps_per_epoch * int(args.epochs), 1)
        scheduler = build_scheduler(optimizer, "lora", num_training_steps)
        checkpoint_interval_epochs = int(config["training"].get("checkpoint_interval_epochs", 5))
        metrics = list(resume_state.get("metrics", [])) if resume_state is not None else []
        resume_epoch = int(resume_state.get("epoch", 0)) if resume_state is not None else 0
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            if resume_state.get("scheduler") is not None:
                scheduler.load_state_dict(resume_state["scheduler"])

        ddp_print(
            f"[baseline] distributed={distributed} rank={rank} world_size={world_size} "
            f"variant={args.model_variant} device={device} batch_size={batch_size} train_samples={len(train_dataset)} "
            f"val_samples={(len(val_dataset) if val_dataset is not None else 0)}"
        )
        trainable, total, ratio = count_trainable_ratio(model)
        ddp_print(f"[baseline] trainable={trainable} total={total} ratio={ratio:.4f}")

        stage_model = model
        if distributed:
            ddp_kwargs = {}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [device.index]
                ddp_kwargs["output_device"] = device.index
            stage_model = DDP(model, **ddp_kwargs)

        pairwise_loss = PairwiseScoreLoss(
            tau=float(config["loss"]["tau"]),
            margin=float(config["loss"]["margin"]),
        )

        for epoch in range(1, int(args.epochs) + 1):
            if resume_state is not None and epoch <= resume_epoch:
                continue
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = run_epoch_baseline(
                stage_model,
                train_loader,
                optimizer,
                scheduler,
                device,
                pairwise_loss,
                epoch=epoch,
                max_steps=args.max_train_steps_per_epoch,
            )
            val_metrics = None
            prediction_rows = []
            val_loss = None
            if val_loader is not None:
                val_metrics, prediction_rows = evaluate(
                    stage_model,
                    val_loader,
                    device,
                    pairwise_loss,
                    stage_name=f"baseline_{args.model_variant}",
                    phase_index=1,
                    epoch=epoch,
                    max_steps=args.max_eval_steps,
                )
                val_loss = val_metrics["loss"]

            record: dict[str, Any] = {
                "stage": f"baseline_{args.model_variant}",
                "phase": 1,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_variant": args.model_variant,
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
                        stage_name=f"baseline_{args.model_variant}",
                        phase_index=1,
                        epoch=epoch,
                        prediction_rows=prediction_rows,
                    )
                    record["val_predictions_path"] = str(prediction_path)
                metrics_path = output_dir / "metrics.json"
                metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                save_resume_checkpoint(
                    stage_model,
                    optimizer,
                    scheduler,
                    output_dir,
                    stage_name=f"baseline_{args.model_variant}",
                    phase_index=1,
                    allowed_stages=[0, 1, 2],
                    epoch=epoch,
                    metrics=metrics,
                )
                if should_save_epoch_checkpoint(epoch, checkpoint_interval_epochs):
                    save_checkpoint(
                        stage_model,
                        optimizer,
                        scheduler,
                        output_dir,
                        f"baseline_{args.model_variant}_phase1",
                        epoch,
                    )
            ddp_barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
