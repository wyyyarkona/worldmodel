from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze score_model_v2 validation prediction jsonl files."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to a *_val_predictions.jsonl file.",
    )
    return parser.parse_args()


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def safe_min(values):
    if not values:
        return None
    return min(values)


def safe_max(values):
    if not values:
        return None
    return max(values)


def quantile(sorted_values, q: float):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    alpha = position - left
    return sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha


def summarize_numeric(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "mean": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "min": safe_min(values),
        "p25": quantile(sorted_values, 0.25),
        "mean": safe_mean(values),
        "median": quantile(sorted_values, 0.5),
        "p75": quantile(sorted_values, 0.75),
        "max": safe_max(values),
    }


def bucket_margin(margin: float):
    if margin < 0.1:
        return "small"
    if margin < 0.3:
        return "medium"
    return "large"


def main():
    args = parse_args()
    path = Path(args.predictions)
    rows = load_rows(path)
    if not rows:
        raise ValueError(f"no rows found in {path}")

    pred_probs = [float(row["pred_prob"]) for row in rows]
    pred_logits = [float(row["pred_logit"]) for row in rows]
    margins = [float(row["margin"]) for row in rows]
    deltas = [float(row["teacher_delta"]) for row in rows]
    correct_values = [1.0 if bool(row["correct"]) else 0.0 for row in rows]
    target_counter = Counter(int(row["target_label"]) for row in rows)
    pred_counter = Counter(int(row["pred_label"]) for row in rows)

    by_step = defaultdict(list)
    by_stage = defaultdict(list)
    by_margin_bucket = defaultdict(list)
    for row in rows:
        by_step[str(row["step_index"])].append(1.0 if bool(row["correct"]) else 0.0)
        by_stage[str(row["stage_name"])].append(1.0 if bool(row["correct"]) else 0.0)
        by_margin_bucket[bucket_margin(float(row["margin"]))].append(1.0 if bool(row["correct"]) else 0.0)

    summary = {
        "path": str(path),
        "num_rows": len(rows),
        "accuracy": safe_mean(correct_values),
        "target_label_counts": dict(target_counter),
        "pred_label_counts": dict(pred_counter),
        "pred_prob": summarize_numeric(pred_probs),
        "pred_logit": summarize_numeric(pred_logits),
        "teacher_delta": summarize_numeric(deltas),
        "margin": summarize_numeric(margins),
        "accuracy_by_step": {
            key: {
                "count": len(values),
                "accuracy": safe_mean(values),
            }
            for key, values in sorted(by_step.items(), key=lambda item: item[0])
        },
        "accuracy_by_stage": {
            key: {
                "count": len(values),
                "accuracy": safe_mean(values),
            }
            for key, values in sorted(by_stage.items(), key=lambda item: item[0])
        },
        "accuracy_by_margin_bucket": {
            key: {
                "count": len(values),
                "accuracy": safe_mean(values),
            }
            for key, values in sorted(by_margin_bucket.items(), key=lambda item: item[0])
        },
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
