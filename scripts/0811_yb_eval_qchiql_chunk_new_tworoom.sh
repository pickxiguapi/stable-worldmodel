#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/qchiql_chunk_new/tworoom}"
POLICY_DIR="${POLICY_DIR:-qchiql_chunk_new_tworoom_vittiny_bs192_e10}"
POLICY_PATH="${POLICY_PATH:-${RUN_ROOT}/checkpoints/${POLICY_DIR}/weights_epoch_10.pt}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/tworoom.h5}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench}"
OGBENCH_SITE_PACKAGES="${OGBENCH_SITE_PACKAGES:-${OGBENCH_ROOT}/.venv/lib/python3.10/site-packages}"
EGL_RUNTIME_ROOT="${EGL_RUNTIME_ROOT:-/root/data/yyf/egl-runtime/root}"
GPU_ID="${GPU_ID:-4}"
SEED="${SEED:-42}"
NUM_EVAL="${NUM_EVAL:-50}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/eval/offset${GOAL_OFFSET_STEPS}_budget${EVAL_BUDGET}_seed${SEED}}"
OUTPUT_FILENAME="${OUTPUT_FILENAME:-qchiql_chunk_new_tworoom_offset${GOAL_OFFSET_STEPS}_budget${EVAL_BUDGET}_seed${SEED}_results.txt}"

[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || {
  echo "ERROR: Python executable not found: ${REPO_ROOT}/.venv/bin/python" >&2
  exit 1
}
[[ -s "${POLICY_PATH}" ]] || { echo "ERROR: policy not found: ${POLICY_PATH}" >&2; exit 1; }
[[ -s "${POLICY_PATH%/*}/config.json" ]] || { echo "ERROR: config not found beside policy" >&2; exit 1; }
[[ -s "${DATASET_PATH}" ]] || { echo "ERROR: dataset not found: ${DATASET_PATH}" >&2; exit 1; }
[[ -d "${OGBENCH_SITE_PACKAGES}" ]] || {
  echo "ERROR: evaluation packages not found: ${OGBENCH_SITE_PACKAGES}" >&2
  exit 1
}

mkdir -p "${OUTPUT_ROOT}" "${RUN_ROOT}/tmp"
cd "${REPO_ROOT}"

set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  STABLEWM_HOME="${RUN_ROOT}" \
  SPT_CACHE_DIR="${RUN_ROOT}" \
  TMPDIR="${RUN_ROOT}/tmp" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  STABLEWM_EXTRA_SITE_PACKAGES="${OGBENCH_ROOT}:${OGBENCH_SITE_PACKAGES}" \
  LD_LIBRARY_PATH="${EGL_RUNTIME_ROOT}/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${REPO_ROOT}/.venv/bin/python" scripts/plan/eval_chunk.py \
    --config-name=tworoom \
    "policy=${POLICY_PATH}" \
    "eval.dataset_name=${DATASET_PATH}" \
    "eval.num_eval=${NUM_EVAL}" \
    "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}" \
    "eval.eval_budget=${EVAL_BUDGET}" \
    "output.filename=${OUTPUT_FILENAME}" \
    "seed=${SEED}" \
    2>&1 | tee "${OUTPUT_ROOT}/eval.log"
status="${PIPESTATUS[0]}"
set -e

printf '%s\n' "${status}" >"${OUTPUT_ROOT}/exit_status.txt"
exit "${status}"
