#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/gciql_chunk/tworoom}"
POLICY_PATH="${POLICY_PATH:-${RUN_ROOT}/checkpoints/gciql_chunk_tworoom_dino_bs256_e10_policy/weights_epoch_10.pt}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/tworoom.h5}"
GPU_ID="${GPU_ID:-4}"
SEED="${SEED:-777}"
NUM_EVAL="${NUM_EVAL:-50}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-gciql_chunk_tworoom_offset${GOAL_OFFSET_STEPS}_budget${EVAL_BUDGET}_seed${SEED}_results.txt}"

[[ -f "${POLICY_PATH}" ]] || { echo "Missing policy: ${POLICY_PATH}" >&2; exit 1; }
[[ -f "${DATASET_PATH}" ]] || { echo "Missing dataset: ${DATASET_PATH}" >&2; exit 1; }

mkdir -p "${RUN_ROOT}/eval" "${RUN_ROOT}/tmp"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export STABLEWM_HOME="${RUN_ROOT}"
export SPT_CACHE_DIR="${RUN_ROOT}"
export TMPDIR="${RUN_ROOT}/tmp"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${REPO_ROOT}/.venv/bin/python" scripts/plan/eval_chunk.py \
  --config-name=tworoom \
  "policy=${POLICY_PATH}" \
  "eval.dataset_name=${DATASET_PATH}" \
  "eval.num_eval=${NUM_EVAL}" \
  "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}" \
  "eval.eval_budget=${EVAL_BUDGET}" \
  "output.filename=${OUTPUT_FILENAME}" \
  "seed=${SEED}"
