#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_train.baseline5k.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-/data/zhulanyun/lihailong/data/score_model_data/openvid_i2v_81f/manifests/baseline_5k_split_v2/pairs_val.baseline5k.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/zhulanyun/lihailong/outputs/score_model_v2_readout_lr_sweep}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

EXPERIMENTS=(
  "query_blr1e4 score_model_v2/configs/v2_query_blr1e4_w10_l10.yaml 29571"
  "query_blr5e5 score_model_v2/configs/v2_query_blr5e5_w10_l10.yaml 29572"
  "h1h2_blr1e4 score_model_v2/configs/v2_h1h2_blr1e4_w10_l10.yaml 29573"
  "h1h2_blr5e5 score_model_v2/configs/v2_h1h2_blr5e5_w10_l10.yaml 29574"
)

mkdir -p "${OUTPUT_ROOT}"

echo "[sweep] repo=${REPO_DIR}"
echo "[sweep] train_manifest=${TRAIN_MANIFEST}"
echo "[sweep] val_manifest=${VAL_MANIFEST}"
echo "[sweep] output_root=${OUTPUT_ROOT}"
echo "[sweep] nproc_per_node=${NPROC_PER_NODE}"
echo "[sweep] cuda_visible_devices=${CUDA_VISIBLE_DEVICES_VALUE}"

for experiment in "${EXPERIMENTS[@]}"; do
  read -r NAME CONFIG_PATH MASTER_PORT <<<"${experiment}"
  RUN_DIR="${OUTPUT_ROOT}/${NAME}"
  mkdir -p "${RUN_DIR}"
  echo "[sweep] starting ${NAME}"
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
  echo "[sweep] finished ${NAME}"
done

echo "[sweep] all experiments finished"
