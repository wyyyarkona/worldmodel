#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_train.baseline5k.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_val.baseline5k.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/zhulanyun/lihailong/outputs/score_model_v2_full_lr_sweep}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
LORA_EPOCHS="${LORA_EPOCHS:-10}"
CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-0}"
BASE_CONFIG_PATH="${BASE_CONFIG_PATH:-${REPO_DIR}/score_model_v2/configs/v2_default.yaml}"

READOUT_MODES=(
  "query"
  "h1_h2"
)

LR_VALUES=(
  "1e-5"
  "5e-5"
  "1e-4"
  "5e-4"
  "1e-3"
  "5e-3"
  "1e-2"
  "5e-2"
)

mkdir -p "${OUTPUT_ROOT}"

echo "[sweep] repo=${REPO_DIR}"
echo "[sweep] base_config=${BASE_CONFIG_PATH}"
echo "[sweep] train_manifest=${TRAIN_MANIFEST}"
echo "[sweep] val_manifest=${VAL_MANIFEST}"
echo "[sweep] output_root=${OUTPUT_ROOT}"
echo "[sweep] nproc_per_node=${NPROC_PER_NODE}"
echo "[sweep] cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}"
echo "[sweep] warmup_epochs=${WARMUP_EPOCHS} lora_epochs=${LORA_EPOCHS} curriculum_epochs=${CURRICULUM_EPOCHS}"

for READOUT_MODE in "${READOUT_MODES[@]}"; do
  for LR_VALUE in "${LR_VALUES[@]}"; do
    SAFE_LR="${LR_VALUE//./p}"
    RUN_NAME="${READOUT_MODE}_lr${SAFE_LR}"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    CONFIG_PATH="${RUN_DIR}/config.yaml"
    mkdir -p "${RUN_DIR}"

    echo "[sweep] preparing ${RUN_NAME}"
    "${PYTHON_BIN}" - <<PY
import copy
from pathlib import Path
import yaml

base_config_path = Path(r"${BASE_CONFIG_PATH}")
run_config_path = Path(r"${CONFIG_PATH}")
cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
cfg = copy.deepcopy(cfg)
cfg["model"]["readout_mode"] = "${READOUT_MODE}"
cfg["training"]["warmup_lr"] = float("${LR_VALUE}")
cfg["training"]["base_lr"] = float("${LR_VALUE}")
cfg["training"]["lora_lr"] = float("${LR_VALUE}")
cfg["training"]["epochs"]["warmup"] = int("${WARMUP_EPOCHS}")
cfg["training"]["epochs"]["lora"] = int("${LORA_EPOCHS}")
cfg["training"]["epochs"]["curriculum"] = int("${CURRICULUM_EPOCHS}")
run_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(run_config_path)
PY

    MASTER_PORT=$(python - <<PY
seed = abs(hash("${RUN_NAME}")) % 1000
print(30000 + seed)
PY
)

    echo "[sweep] starting ${RUN_NAME} master_port=${MASTER_PORT}"
    (
      cd "${REPO_DIR}"
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
      export MASTER_PORT
      "${TORCHRUN_BIN}" --nproc_per_node "${NPROC_PER_NODE}" -m score_model_v2.train_v2 \
        --config "${CONFIG_PATH}" \
        --train_manifest "${TRAIN_MANIFEST}" \
        --val_manifest "${VAL_MANIFEST}" \
        --output_dir "${RUN_DIR}" \
        --no_auto_resume \
        --ddp_find_unused_parameters \
        --disable_curriculum \
        2>&1 | tee "${RUN_DIR}/train.log"
    )

    echo "[sweep] plotting ${RUN_NAME}"
    (
      cd "${REPO_DIR}"
      "${PYTHON_BIN}" -m score_model_v2.plot_training_metrics --run_dir "${RUN_DIR}" \
        > "${RUN_DIR}/plot.log" 2>&1
    )

    echo "[sweep] analyzing final predictions for ${RUN_NAME}"
    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import subprocess
import sys

run_dir = Path(r"${RUN_DIR}")
metrics_path = run_dir / "metrics.json"
if not metrics_path.exists():
    raise FileNotFoundError(metrics_path)

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if not metrics:
    raise ValueError(f"metrics.json is empty: {metrics_path}")

last = metrics[-1]
stage = last["stage"]
phase = int(last["phase"])
epoch = int(last["epoch"])
prediction_path = run_dir / "predictions" / f"{stage}_phase{phase}_epoch_{epoch:03d}_val_predictions.jsonl"
if not prediction_path.exists():
    raise FileNotFoundError(prediction_path)

analysis_path = run_dir / "final_prediction_analysis.json"
summary_path = run_dir / "final_result_summary.json"

proc = subprocess.run(
    [sys.executable, "-m", "score_model_v2.analyze_predictions", "--predictions", str(prediction_path)],
    check=True,
    text=True,
    capture_output=True,
)
analysis_path.write_text(proc.stdout, encoding="utf-8")
analysis = json.loads(proc.stdout)

summary = {
    "run_name": "${RUN_NAME}",
    "readout_mode": "${READOUT_MODE}",
    "lr": "${LR_VALUE}",
    "final_stage": stage,
    "final_phase": phase,
    "final_epoch": epoch,
    "metrics": {
        "val_loss": last.get("val_loss"),
        "val_pairwise_accuracy": last.get("val_pairwise_accuracy"),
        "val_weighted_pairwise_accuracy": last.get("val_weighted_pairwise_accuracy"),
        "val_auc": last.get("val_auc"),
        "val_mean_pred_prob": last.get("val_mean_pred_prob"),
        "val_brier_score": last.get("val_brier_score"),
    },
    "prediction_analysis": {
        "accuracy": analysis.get("accuracy"),
        "pred_label_counts": analysis.get("pred_label_counts"),
        "pred_prob_mean": analysis.get("pred_prob", {}).get("mean"),
        "pred_prob_min": analysis.get("pred_prob", {}).get("min"),
        "pred_prob_max": analysis.get("pred_prob", {}).get("max"),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
    echo "[sweep] finished ${RUN_NAME}"
  done
done

echo "[sweep] building global summary"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path(r"${OUTPUT_ROOT}")
rows = []
for run_dir in sorted(root.iterdir()):
    if not run_dir.is_dir():
        continue
    summary_path = run_dir / "final_result_summary.json"
    if not summary_path.exists():
        continue
    rows.append(json.loads(summary_path.read_text(encoding="utf-8")))

summary_path = root / "all_results_summary.json"
summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(rows, indent=2, ensure_ascii=False))
print(f"[done] wrote {summary_path}")
PY

echo "[sweep] all experiments finished"
