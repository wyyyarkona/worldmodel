#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/score_model_v2/configs/v2_default.yaml}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
VAL_MANIFEST="${VAL_MANIFEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
USE_TORCHRUN="${USE_TORCHRUN:-0}"
RESUME_FROM="${RESUME_FROM:-}"
NO_AUTO_RESUME="${NO_AUTO_RESUME:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-2048}"
MODEL_VARIANT="${MODEL_VARIANT:-a}"

CMD=(
  --config "${CONFIG_PATH}"
  --train_manifest "${TRAIN_MANIFEST}"
  --output_dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --head_hidden_dim "${HEAD_HIDDEN_DIM}"
  --model_variant "${MODEL_VARIANT}"
)

if [[ -n "${VAL_MANIFEST}" ]]; then
  CMD+=(--val_manifest "${VAL_MANIFEST}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  CMD+=(--resume_from "${RESUME_FROM}")
fi
if [[ "${NO_AUTO_RESUME}" == "1" ]]; then
  CMD+=(--no_auto_resume)
fi
if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
  CMD+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
  CMD+=(--max_val_samples "${MAX_VAL_SAMPLES}")
fi

echo "[baseline] repo=${REPO_DIR}"
echo "[baseline] config=${CONFIG_PATH}"
echo "[baseline] train_manifest=${TRAIN_MANIFEST}"
echo "[baseline] val_manifest=${VAL_MANIFEST}"
echo "[baseline] output_dir=${OUTPUT_DIR}"
echo "[baseline] model_variant=${MODEL_VARIANT}"

if [[ "${USE_TORCHRUN}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
  "${TORCHRUN_BIN}" --nproc_per_node "${NPROC_PER_NODE}" -m score_model_v2.baseline_train_ddp "${CMD[@]}"
else
  "${PYTHON_BIN}" -m score_model_v2.baseline_train_ddp "${CMD[@]}"
fi
