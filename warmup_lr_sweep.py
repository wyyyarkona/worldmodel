from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from score_model_v2.train_v2 import load_config, resolve_repo_relative_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run score_model_v2 warmup-only training with multiple warmup_lr values."
    )
    parser.add_argument("--config", type=str, required=True, help="Base YAML/JSON config.")
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True, help="Root dir containing one subdir per lr run.")
    parser.add_argument(
        "--warmup_lrs",
        type=float,
        nargs="+",
        required=True,
        help="List of warmup_lr values to sweep.",
    )
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--torchrun_bin", type=str, default="torchrun")
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--master_port_base", type=int, default=29600)
    parser.add_argument("--use_torchrun", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_train_steps_per_epoch", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--readout_mode", type=str, default=None)
    parser.add_argument("--bidirectional_attention", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--base_lr", type=float, default=None, help="Optional override for training.base_lr.")
    parser.add_argument("--lora_lr", type=float, default=None, help="Optional override for training.lora_lr.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Optional override for training.weight_decay.")
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--no_auto_resume", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def ensure_yaml_available():
    try:
        import yaml  # noqa: F401
    except ImportError as exc:
        raise ImportError("PyYAML is required for warmup_lr_sweep.py.") from exc


def format_lr_tag(value: float) -> str:
    text = f"{value:.12g}"
    return text.replace("-", "m").replace(".", "p")


def dump_config(path: Path, config: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_sweep_config(base_config: dict[str, Any], args: argparse.Namespace, lr: float) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    training_cfg = config.setdefault("training", {})
    epochs_cfg = dict(training_cfg.get("epochs", {}))
    epochs_cfg["warmup"] = int(args.warmup_epochs)
    epochs_cfg["lora"] = 0
    epochs_cfg["curriculum"] = 0
    training_cfg["epochs"] = epochs_cfg
    training_cfg["warmup_lr"] = float(lr)
    if args.base_lr is not None:
        training_cfg["base_lr"] = float(args.base_lr)
    if args.lora_lr is not None:
        training_cfg["lora_lr"] = float(args.lora_lr)
    if args.weight_decay is not None:
        training_cfg["weight_decay"] = float(args.weight_decay)
    if args.batch_size is not None:
        training_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        training_cfg["num_workers"] = int(args.num_workers)

    model_cfg = config.setdefault("model", {})
    if args.readout_mode is not None:
        model_cfg["readout_mode"] = args.readout_mode
    if args.bidirectional_attention is not None:
        model_cfg["bidirectional_attention"] = args.bidirectional_attention == "true"
    if args.attn_implementation is not None:
        model_cfg["attn_implementation"] = args.attn_implementation
    return config


def build_train_command(
    args: argparse.Namespace,
    config_path: Path,
    output_dir: Path,
    train_manifest: str,
    val_manifest: str | None,
    master_port: int,
) -> tuple[list[str], dict[str, str]]:
    base_cmd = [
        args.python_bin,
        "-m",
        "score_model_v2.train_v2",
        "--config",
        str(config_path),
        "--train_manifest",
        train_manifest,
        "--output_dir",
        str(output_dir),
        "--disable_curriculum",
        "--no_auto_resume",
    ]
    if val_manifest:
        base_cmd.extend(["--val_manifest", val_manifest])
    if args.device:
        base_cmd.extend(["--device", args.device])
    if args.ddp_find_unused_parameters:
        base_cmd.append("--ddp_find_unused_parameters")
    if args.pin_memory:
        base_cmd.append("--pin_memory")
    if args.persistent_workers:
        base_cmd.append("--persistent_workers")
    if args.max_train_samples is not None:
        base_cmd.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.max_val_samples is not None:
        base_cmd.extend(["--max_val_samples", str(args.max_val_samples)])
    if args.max_train_steps_per_epoch is not None:
        base_cmd.extend(["--max_train_steps_per_epoch", str(args.max_train_steps_per_epoch)])
    if args.max_eval_steps is not None:
        base_cmd.extend(["--max_eval_steps", str(args.max_eval_steps)])
    if args.smoke_test:
        base_cmd.append("--smoke_test")

    env = os.environ.copy()
    env["MASTER_PORT"] = str(master_port)

    if args.use_torchrun or args.nproc_per_node != 1:
        command = [
            args.torchrun_bin,
            "--nproc_per_node",
            str(args.nproc_per_node),
            "--master_port",
            str(master_port),
            *base_cmd[1:],
        ]
    else:
        command = base_cmd
    return command, env


def main() -> None:
    args = parse_args()
    ensure_yaml_available()

    base_config_path = Path(resolve_repo_relative_path(args.config))
    train_manifest = resolve_repo_relative_path(args.train_manifest)
    val_manifest = resolve_repo_relative_path(args.val_manifest) if args.val_manifest else None
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_config(str(base_config_path))

    for index, lr in enumerate(args.warmup_lrs):
        lr_tag = format_lr_tag(lr)
        run_name = f"warmup_lr_{lr_tag}"
        run_dir = output_root / run_name
        config_dir = output_root / "_configs"
        config_path = config_dir / f"{run_name}.yaml"
        run_config = build_sweep_config(base_config, args, lr)
        dump_config(config_path, run_config)

        command, env = build_train_command(
            args=args,
            config_path=config_path,
            output_dir=run_dir,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            master_port=args.master_port_base + index,
        )

        print(f"[warmup_lr_sweep] lr={lr} run_dir={run_dir}")
        print("[warmup_lr_sweep] command=" + " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
