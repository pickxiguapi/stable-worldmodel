#!/usr/bin/env bash
set -euo pipefail

OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench}"
LEWM_RUNS_ROOT="${LEWM_RUNS_ROOT:-/root/data/yyf/lewm-runs/OGBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/yyf/lewm-runs/evals/gpu567_20260810}"
GPU_ID="${GPU_ID:-7}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LEWM_RUNS_ROOT}/lewm-cube-visual-gciql-bs256-100k/sd000_20260809_015848}"
RUN_DIR="${OUTPUT_ROOT}/cube_gciql"

[[ -s "${CHECKPOINT_DIR}/params_100000.pkl" ]] || {
  echo "ERROR: checkpoint not found: ${CHECKPOINT_DIR}/params_100000.pkl" >&2
  exit 1
}
[[ -f "${OGBENCH_ROOT}/scripts/eval_lewm.sh" ]] || {
  echo "ERROR: evaluator Bash not found: ${OGBENCH_ROOT}/scripts/eval_lewm.sh" >&2
  exit 1
}

mkdir -p "${RUN_DIR}"
set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  bash "${OGBENCH_ROOT}/scripts/eval_lewm.sh" cube gciql "${CHECKPOINT_DIR}" \
    2>&1 | tee "${RUN_DIR}/eval.log"
status="${PIPESTATUS[0]}"
set -e
printf '%s\n' "${status}" >"${RUN_DIR}/exit_status.txt"
exit "${status}"
