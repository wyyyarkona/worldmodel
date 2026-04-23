from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample smaller train/val manifests for score_model_v2 baseline experiments."
    )
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_samples", type=int, default=4500)
    parser.add_argument("--val_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_output_name", type=str, default="pairs_train.baseline5k.jsonl")
    parser.add_argument("--val_output_name", type=str, default="pairs_val.baseline5k.jsonl")
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def symmetric_group_key(row: dict) -> tuple:
    # Keep a pair and its reversed counterpart together by building a canonical
    # key that ignores candidate ordering but preserves the rest of the sample
    # identity (condition and step metadata).
    f1_path = row.get("f1_path") or row.get("z_t_a_path")
    f2_path = row.get("f2_path") or row.get("z_t_b_path")
    if f1_path is None or f2_path is None:
        raise ValueError("Each sampled row must provide candidate latent paths.")

    ordered_pair = tuple(sorted([str(f1_path), str(f2_path)]))
    context_path = row.get("context_path") or row.get("text_emb_path")
    image_path = row.get("clip_fea_path") or row.get("image_emb_path")
    stage_id = row.get("stage_id")
    step_key = None
    for key in ("t_idx", "step_idx", "step", "step_id", "timestep"):
        if key in row:
            step_key = (key, row.get(key))
            break
    return (
        ordered_pair,
        str(context_path),
        str(image_path),
        str(stage_id),
        step_key,
    )


def sample_grouped_rows(rows, sample_size: int, rng: random.Random):
    groups = defaultdict(list)
    for row in rows:
        groups[symmetric_group_key(row)].append(row)

    grouped_rows = list(groups.values())
    if sample_size >= sum(len(group) for group in grouped_rows):
        flattened = []
        for group in grouped_rows:
            flattened.extend(group)
        return flattened, grouped_rows

    rng.shuffle(grouped_rows)
    sampled_rows = []
    sampled_groups = []
    for group in grouped_rows:
        if sampled_rows and len(sampled_rows) + len(group) > sample_size:
            continue
        sampled_groups.append(group)
        sampled_rows.extend(group)
        if len(sampled_rows) >= sample_size:
            break

    if not sampled_rows:
        raise RuntimeError("Unable to sample any grouped rows. Check manifest contents.")
    return sampled_rows, sampled_groups


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    train_rows = load_jsonl(Path(args.train_manifest))
    val_rows = load_jsonl(Path(args.val_manifest))

    sampled_train, sampled_train_groups = sample_grouped_rows(train_rows, args.train_samples, rng)
    sampled_val, sampled_val_groups = sample_grouped_rows(val_rows, args.val_samples, rng)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_output = output_dir / args.train_output_name
    val_output = output_dir / args.val_output_name

    write_jsonl(train_output, sampled_train)
    write_jsonl(val_output, sampled_val)

    summary = {
        "seed": args.seed,
        "train_manifest": str(Path(args.train_manifest)),
        "val_manifest": str(Path(args.val_manifest)),
        "train_total_rows": len(train_rows),
        "val_total_rows": len(val_rows),
        "train_sampled_rows": len(sampled_train),
        "val_sampled_rows": len(sampled_val),
        "train_sampled_groups": len(sampled_train_groups),
        "val_sampled_groups": len(sampled_val_groups),
        "train_output": str(train_output),
        "val_output": str(val_output),
    }
    summary_path = output_dir / "sampling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
