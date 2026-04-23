from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGE_ID_TO_NAME = {
    0: "early",
    1: "middle",
    2: "late",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect pairwise manifest distributions for score_model_v2."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        nargs="+",
        required=True,
        help="One or more pair manifest paths to inspect.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    left = int(position)
    right = min(left + 1, len(sorted_values) - 1)
    alpha = position - left
    return sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
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
        "min": sorted_values[0],
        "p25": quantile(sorted_values, 0.25),
        "mean": safe_mean(values),
        "median": quantile(sorted_values, 0.5),
        "p75": quantile(sorted_values, 0.75),
        "max": sorted_values[-1],
    }


def resolve_step_index(row: dict[str, Any]) -> int | None:
    for key in ("t_idx", "step_idx", "step", "step_id", "timestep"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def bucket_stage_name(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        mapping = {"early": 0, "middle": 1, "mid": 1, "late": 2}
        return mapping.get(value.lower())
    if isinstance(value, int) and value in {0, 1, 2}:
        return value
    return None


def bucket_stage_from_step(step_index: int, min_step: int, max_step: int) -> int:
    if max_step <= min_step:
        return 1
    normalized = (step_index - min_step) / max(max_step - min_step, 1)
    if normalized < (1.0 / 3.0):
        return 0
    if normalized < (2.0 / 3.0):
        return 1
    return 2


def resolve_stage_id(row: dict[str, Any], min_step: int, max_step: int) -> int:
    explicit_stage = bucket_stage_name(row.get("stage_id"))
    if explicit_stage is not None:
        return explicit_stage
    step_index = resolve_step_index(row)
    if step_index is not None:
        return bucket_stage_from_step(step_index, min_step=min_step, max_step=max_step)
    return 1


def margin_bucket(margin: float) -> str:
    if margin < 0.1:
        return "small"
    if margin < 0.3:
        return "medium"
    return "large"


def symmetric_group_key(row: dict[str, Any]) -> tuple:
    f1_path = row.get("f1_path") or row.get("z_t_a_path")
    f2_path = row.get("f2_path") or row.get("z_t_b_path")
    ordered_pair = tuple(sorted([str(f1_path), str(f2_path)]))
    context_path = row.get("context_path") or row.get("text_emb_path")
    image_path = row.get("clip_fea_path") or row.get("image_emb_path")
    stage_value = row.get("stage_id")
    step_index = resolve_step_index(row)
    label_score_key = row.get("label_score_key")
    return (
        ordered_pair,
        str(context_path),
        str(image_path),
        str(stage_value),
        step_index,
        str(label_score_key),
    )


def summarize_manifest(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"no rows found in {path}")

    step_indices = [step for step in (resolve_step_index(row) for row in rows) if step is not None]
    min_step = min(step_indices) if step_indices else 0
    max_step = max(step_indices) if step_indices else 0

    clip_counter: Counter[str] = Counter()
    clip_step_counter: Counter[str] = Counter()
    step_counter: Counter[str] = Counter()
    stage_counter: Counter[str] = Counter()
    label_score_key_counter: Counter[str] = Counter()
    symmetric_group_sizes: Counter[int] = Counter()
    margin_bucket_counter: Counter[str] = Counter()
    direction_counter: Counter[str] = Counter()

    score_a_values: list[float] = []
    score_b_values: list[float] = []
    delta_values: list[float] = []
    abs_delta_values: list[float] = []
    label_values: list[float] = []
    weight_values: list[float] = []

    missing_field_counter: Counter[str] = Counter()
    grouped_rows: defaultdict[tuple, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        clip_id = row.get("clip_id")
        if clip_id is not None:
            clip_counter[str(clip_id)] += 1

        step_index = resolve_step_index(row)
        if step_index is not None:
            step_counter[str(step_index)] += 1
            if clip_id is not None:
                clip_step_counter[f"{clip_id}|{step_index}"] += 1

        stage_id = resolve_stage_id(row, min_step=min_step, max_step=max_step)
        stage_counter[STAGE_ID_TO_NAME[stage_id]] += 1

        label_score_key_counter[str(row.get("label_score_key", "missing"))] += 1

        score_a = row.get("score_a")
        score_b = row.get("score_b")
        delta = row.get("delta")
        label = row.get("label")
        weight = row.get("weight")

        if score_a is None:
            missing_field_counter["score_a"] += 1
        else:
            score_a = float(score_a)
            score_a_values.append(score_a)

        if score_b is None:
            missing_field_counter["score_b"] += 1
        else:
            score_b = float(score_b)
            score_b_values.append(score_b)

        if delta is None and score_a is not None and score_b is not None:
            delta = score_a - score_b
        if delta is None:
            missing_field_counter["delta"] += 1
        else:
            delta = float(delta)
            delta_values.append(delta)
            abs_delta_values.append(abs(delta))
            margin_bucket_counter[margin_bucket(abs(delta))] += 1
            direction_counter["a_better" if delta > 0 else "b_better" if delta < 0 else "tie"] += 1

        if label is None:
            missing_field_counter["label"] += 1
        else:
            label_values.append(float(label))

        if weight is None:
            missing_field_counter["weight"] += 1
        else:
            weight_values.append(float(weight))

        grouped_rows[symmetric_group_key(row)].append(row)

    complete_symmetric_groups = 0
    reverse_pair_issues = 0
    for group_rows in grouped_rows.values():
        symmetric_group_sizes[len(group_rows)] += 1
        if len(group_rows) == 2:
            complete_symmetric_groups += 1
        else:
            reverse_pair_issues += 1

    group_size_values = [len(group_rows) for group_rows in grouped_rows.values()]
    clip_step_group_size_values = list(clip_step_counter.values())

    summary = {
        "path": str(path),
        "num_rows": len(rows),
        "num_unique_clips": len(clip_counter),
        "num_unique_clip_steps": len(clip_step_counter),
        "num_symmetric_groups": len(grouped_rows),
        "complete_symmetric_groups": complete_symmetric_groups,
        "symmetric_group_size_counts": {str(k): v for k, v in sorted(symmetric_group_sizes.items())},
        "reverse_pair_issue_groups": reverse_pair_issues,
        "step_index_counts": dict(sorted(step_counter.items(), key=lambda item: item[0])),
        "resolved_stage_counts": dict(sorted(stage_counter.items(), key=lambda item: item[0])),
        "label_score_key_counts": dict(sorted(label_score_key_counter.items(), key=lambda item: item[0])),
        "margin_bucket_counts": dict(sorted(margin_bucket_counter.items(), key=lambda item: item[0])),
        "direction_counts": dict(sorted(direction_counter.items(), key=lambda item: item[0])),
        "missing_field_counts": dict(sorted(missing_field_counter.items(), key=lambda item: item[0])),
        "score_a": summarize_numeric(score_a_values),
        "score_b": summarize_numeric(score_b_values),
        "delta": summarize_numeric(delta_values),
        "abs_delta": summarize_numeric(abs_delta_values),
        "label": summarize_numeric(label_values),
        "weight": summarize_numeric(weight_values),
        "rows_per_clip_step": summarize_numeric([float(v) for v in clip_step_group_size_values]),
        "rows_per_symmetric_group": summarize_numeric([float(v) for v in group_size_values]),
        "top_clip_ids_by_rows": [
            {"clip_id": clip_id, "count": count}
            for clip_id, count in clip_counter.most_common(10)
        ],
    }
    return summary


def main() -> None:
    args = parse_args()
    summaries = [summarize_manifest(Path(path)) for path in args.manifest]
    output: dict[str, Any]
    if len(summaries) == 1:
        output = summaries[0]
    else:
        output = {"manifests": summaries}
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
