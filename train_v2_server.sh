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
SMOKE_TEST="${SMOKE_TEST:-0}"
RESUME_FROM="${RESUME_FROM:-}"
NO_AUTO_RESUME="${NO_AUTO_RESUME:-0}"

CMD=(
  --config
  "${CONFIG_PATH}"
)

if [[ -n "${TRAIN_MANIFEST}" ]]; then
  CMD+=(--train_manifest "${TRAIN_MANIFEST}")
fi
if [[ -n "${VAL_MANIFEST}" ]]; then
  CMD+=(--val_manifest "${VAL_MANIFEST}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ "${SMOKE_TEST}" == "1" ]]; then
  CMD+=(--smoke_test)
fi
if [[ -n "${RESUME_FROM}" ]]; then
  CMD+=(--resume_from "${RESUME_FROM}")
fi
if [[ "${NO_AUTO_RESUME}" == "1" ]]; then
  CMD+=(--no_auto_resume)
fi

echo "[score_model_v2] repo=${REPO_DIR}"
echo "[score_model_v2] config=${CONFIG_PATH}"
echo "[score_model_v2] use_torchrun=${USE_TORCHRUN} nproc_per_node=${NPROC_PER_NODE} smoke_test=${SMOKE_TEST} no_auto_resume=${NO_AUTO_RESUME}"

if [[ "${USE_TORCHRUN}" == "1" || "${NPROC_PER_NODE}" != "1" ]]; then
  "${TORCHRUN_BIN}" --nproc_per_node "${NPROC_PER_NODE}" -m score_model_v2.train_v2 "${CMD[@]}"
else
  "${PYTHON_BIN}" -m score_model_v2.train_v2 "${CMD[@]}"
fi
