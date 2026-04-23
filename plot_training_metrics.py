import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot score_model_v2 training metrics from metrics.json."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Training output directory containing metrics.json.",
    )
    parser.add_argument(
        "--metrics_json",
        type=str,
        default=None,
        help="Explicit path to metrics.json. Overrides --run_dir when provided.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for plots and summary. Defaults to the metrics file directory.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Score Model V2 Training Metrics",
        help="Figure title.",
    )
    return parser.parse_args()


def resolve_paths(args):
    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
    elif args.run_dir:
        metrics_path = Path(args.run_dir) / "metrics.json"
    else:
        raise ValueError("Either --run_dir or --metrics_json must be provided.")

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return metrics_path, output_dir


def load_metrics(metrics_path):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Expected metrics.json to be a JSON list.")
    return payload


def stage_phase_label(record):
    return f"{record.get('stage', 'unknown')}:p{record.get('phase', '?')}"


def save_summary(output_dir, metrics):
    by_stage_phase = defaultdict(list)
    best_val = None
    best_val_record = None

    for index, record in enumerate(metrics, start=1):
        enriched = dict(record)
        enriched["index"] = index
        by_stage_phase[stage_phase_label(record)].append(enriched)

        val_loss = record.get("val_loss")
        if val_loss is not None and (best_val is None or float(val_loss) < best_val):
            best_val = float(val_loss)
            best_val_record = enriched

    summary = {
        "num_records": len(metrics),
        "stage_phase_counts": {key: len(value) for key, value in by_stage_phase.items()},
        "best_val_record": best_val_record,
        "metrics": metrics,
    }
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def plot_metrics(output_dir, title, metrics):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib`."
        ) from exc

    if not metrics:
        raise ValueError("metrics.json is empty.")

    indices = list(range(1, len(metrics) + 1))
    train_losses = [float(record["train_loss"]) for record in metrics]
    val_losses = [
        float(record["val_loss"]) if record.get("val_loss") is not None else None
        for record in metrics
    ]
    val_pairwise_accuracy = [
        float(record["val_pairwise_accuracy"]) if record.get("val_pairwise_accuracy") is not None else None
        for record in metrics
    ]
    val_weighted_pairwise_accuracy = [
        float(record["val_weighted_pairwise_accuracy"]) if record.get("val_weighted_pairwise_accuracy") is not None else None
        for record in metrics
    ]
    val_auc = [
        float(record["val_auc"]) if record.get("val_auc") is not None else None
        for record in metrics
    ]
    val_brier = [
        float(record["val_brier_score"]) if record.get("val_brier_score") is not None else None
        for record in metrics
    ]
    val_mean_pred_prob = [
        float(record["val_mean_pred_prob"]) if record.get("val_mean_pred_prob") is not None else None
        for record in metrics
    ]
    labels = [stage_phase_label(record) for record in metrics]

    by_stage_phase = defaultdict(list)
    for index, record in zip(indices, metrics):
        by_stage_phase[stage_phase_label(record)].append((index, record))

    def series_points(values):
        return [(idx, value) for idx, value in zip(indices, values) if value is not None]

    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    fig.suptitle(title)

    axes[0, 0].plot(indices, train_losses, marker="o", linewidth=2, color="#1f77b4")
    axes[0, 0].set_title("Train Loss By Record")
    axes[0, 0].set_xlabel("Record Index")
    axes[0, 0].set_ylabel("Train Loss")
    axes[0, 0].grid(True, alpha=0.3)

    valid_val = [(idx, value) for idx, value in zip(indices, val_losses) if value is not None]
    if valid_val:
        axes[0, 1].plot(
            [item[0] for item in valid_val],
            [item[1] for item in valid_val],
            marker="o",
            linewidth=2,
            color="#2ca02c",
        )
        axes[0, 1].set_title("Validation Loss By Record")
        axes[0, 1].set_xlabel("Record Index")
        axes[0, 1].set_ylabel("Val Loss")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].set_title("Validation Loss By Record")
        axes[0, 1].text(0.5, 0.5, "No validation records", ha="center", va="center")
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])

    for stage_phase, rows in by_stage_phase.items():
        xs = [index for index, _ in rows]
        ys = [float(record["train_loss"]) for _, record in rows]
        axes[1, 0].plot(xs, ys, marker="o", linewidth=2, label=stage_phase)
    axes[1, 0].set_title("Train Loss By Stage/Phase")
    axes[1, 0].set_xlabel("Record Index")
    axes[1, 0].set_ylabel("Train Loss")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    stage_labels = [f"{index}\n{label}\ne{metrics[index - 1].get('epoch', '?')}" for index, label in zip(indices, labels)]
    axes[1, 1].plot(indices, train_losses, marker="o", linewidth=2, color="#ff7f0e")
    if valid_val:
        axes[1, 1].plot(
            [item[0] for item in valid_val],
            [item[1] for item in valid_val],
            marker="s",
            linewidth=2,
            color="#2ca02c",
            label="val",
        )
        axes[1, 1].legend()
    axes[1, 1].set_title("Stage / Phase / Epoch Timeline")
    axes[1, 1].set_xlabel("Record Index")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_xticks(indices)
    axes[1, 1].set_xticklabels(stage_labels, rotation=45, ha="right", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    valid_accuracy = series_points(val_pairwise_accuracy)
    valid_weighted_accuracy = series_points(val_weighted_pairwise_accuracy)
    valid_auc = series_points(val_auc)
    if valid_accuracy:
        axes[2, 0].plot(
            [item[0] for item in valid_accuracy],
            [item[1] for item in valid_accuracy],
            marker="o",
            linewidth=2,
            label="pairwise acc",
        )
    if valid_weighted_accuracy:
        axes[2, 0].plot(
            [item[0] for item in valid_weighted_accuracy],
            [item[1] for item in valid_weighted_accuracy],
            marker="s",
            linewidth=2,
            label="weighted acc",
        )
    if valid_auc:
        axes[2, 0].plot(
            [item[0] for item in valid_auc],
            [item[1] for item in valid_auc],
            marker="^",
            linewidth=2,
            label="auc",
        )
    axes[2, 0].set_title("Validation Ranking Metrics")
    axes[2, 0].set_xlabel("Record Index")
    axes[2, 0].set_ylabel("Metric Value")
    axes[2, 0].set_ylim(0.0, 1.05)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(fontsize=8)

    valid_brier = series_points(val_brier)
    valid_mean_pred_prob = series_points(val_mean_pred_prob)
    if valid_brier:
        axes[2, 1].plot(
            [item[0] for item in valid_brier],
            [item[1] for item in valid_brier],
            marker="o",
            linewidth=2,
            label="brier score",
            color="#8c564b",
        )
    if valid_mean_pred_prob:
        axes[2, 1].plot(
            [item[0] for item in valid_mean_pred_prob],
            [item[1] for item in valid_mean_pred_prob],
            marker="s",
            linewidth=2,
            label="mean pred prob",
            color="#17becf",
        )
    axes[2, 1].set_title("Validation Probability Metrics")
    axes[2, 1].set_xlabel("Record Index")
    axes[2, 1].set_ylabel("Metric Value")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend(fontsize=8)

    def extract_bucket_series(record_key, child_key="accuracy"):
        keys = set()
        for record in metrics:
            payload = record.get(record_key) or {}
            if isinstance(payload, dict):
                keys.update(payload.keys())
        series = {}
        for key in sorted(keys, key=str):
            xs = []
            ys = []
            for idx, record in zip(indices, metrics):
                payload = record.get(record_key) or {}
                bucket = payload.get(key)
                if not isinstance(bucket, dict):
                    continue
                value = bucket.get(child_key)
                if value is None:
                    continue
                xs.append(idx)
                ys.append(float(value))
            if xs:
                series[str(key)] = (xs, ys)
        return series

    stage_series = extract_bucket_series("val_accuracy_by_stage")
    if stage_series:
        for name, (xs, ys) in stage_series.items():
            axes[3, 0].plot(xs, ys, marker="o", linewidth=2, label=name)
        axes[3, 0].set_ylim(0.0, 1.05)
        axes[3, 0].set_title("Validation Accuracy By Stage")
        axes[3, 0].set_xlabel("Record Index")
        axes[3, 0].set_ylabel("Accuracy")
        axes[3, 0].grid(True, alpha=0.3)
        axes[3, 0].legend(fontsize=8)
    else:
        axes[3, 0].set_title("Validation Accuracy By Stage")
        axes[3, 0].text(0.5, 0.5, "No stage metrics found", ha="center", va="center")
        axes[3, 0].set_xticks([])
        axes[3, 0].set_yticks([])

    step_series = extract_bucket_series("val_accuracy_by_step")
    if step_series:
        for name, (xs, ys) in step_series.items():
            axes[3, 1].plot(xs, ys, marker="o", linewidth=2, label=f"step {name}")
        axes[3, 1].set_ylim(0.0, 1.05)
        axes[3, 1].set_title("Validation Accuracy By Step")
        axes[3, 1].set_xlabel("Record Index")
        axes[3, 1].set_ylabel("Accuracy")
        axes[3, 1].grid(True, alpha=0.3)
        axes[3, 1].legend(fontsize=8)
    else:
        axes[3, 1].set_title("Validation Accuracy By Step")
        axes[3, 1].text(0.5, 0.5, "No step metrics found", ha="center", va="center")
        axes[3, 1].set_xticks([])
        axes[3, 1].set_yticks([])

    margin_series = extract_bucket_series("val_accuracy_by_margin_bucket")
    if margin_series:
        extra_fig, extra_ax = plt.subplots(1, 1, figsize=(10, 5))
        extra_fig.suptitle(f"{title} - Margin Bucket Accuracy")
        for name, (xs, ys) in margin_series.items():
            extra_ax.plot(xs, ys, marker="o", linewidth=2, label=name)
        extra_ax.set_ylim(0.0, 1.05)
        extra_ax.set_xlabel("Record Index")
        extra_ax.set_ylabel("Accuracy")
        extra_ax.set_title("Validation Accuracy By Margin Bucket")
        extra_ax.grid(True, alpha=0.3)
        extra_ax.legend(fontsize=8)
        extra_fig.tight_layout()
        margin_plot_path = output_dir / "training_metrics_margin_buckets.png"
        extra_fig.savefig(margin_plot_path, dpi=160, bbox_inches="tight")
        plt.close(extra_fig)

    fig.tight_layout()
    plot_path = output_dir / "training_metrics_v2.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main():
    args = parse_args()
    metrics_path, output_dir = resolve_paths(args)
    metrics = load_metrics(metrics_path)
    summary_path = save_summary(output_dir, metrics)
    plot_path = plot_metrics(output_dir, args.title, metrics)
    print(f"[done] saved summary to {summary_path}")
    print(f"[done] saved plot to {plot_path}")


if __name__ == "__main__":
    main()
