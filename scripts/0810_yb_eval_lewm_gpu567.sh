#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench}"
LEWM_RUNS_ROOT="${LEWM_RUNS_ROOT:-/root/data/yyf/lewm-runs/OGBench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/yyf/lewm-runs/evals/gpu567_20260810}"

names=(pusht_gciql cube_hiql pusht_hiql)
tasks=(pusht cube pusht)
methods=(gciql hiql hiql)
gpus=(5 6 7)
checkpoint_dirs=(
  "${LEWM_RUNS_ROOT}/lewm-pusht-visual-gciql-bs256-100k/sd000_20260809_021452"
  "${LEWM_RUNS_ROOT}/lewm-cube-visual-hiql-bs256-100k/sd000_20260809_015848"
  "${LEWM_RUNS_ROOT}/lewm-pusht-visual-hiql-bs256-100k/sd000_20260809_021452"
)

run_worker() {
  local index="$1"
  local name="${names[$index]}"
  local task="${tasks[$index]}"
  local method="${methods[$index]}"
  local gpu="${gpus[$index]}"
  local checkpoint_dir="${checkpoint_dirs[$index]}"
  local run_dir="${OUTPUT_ROOT}/${name}"
  local status

  mkdir -p "${run_dir}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
    bash "${OGBENCH_ROOT}/scripts/eval_lewm.sh" \
      "${task}" "${method}" "${checkpoint_dir}" \
      2>&1 | tee "${run_dir}/eval.log"
  status="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "${status}" >"${run_dir}/exit_status.txt"
  return "${status}"
}

if [[ "${1:-}" == "--worker" ]]; then
  [[ $# -eq 2 ]] || { echo "Usage: $0 --worker <index>" >&2; exit 2; }
  run_worker "$2"
  exit $?
fi

[[ -x "${OGBENCH_ROOT}/.venv/bin/python" ]] || {
  echo "ERROR: OGBench Python not found: ${OGBENCH_ROOT}/.venv/bin/python" >&2
  exit 1
}
[[ -f "${OGBENCH_ROOT}/scripts/eval_lewm.sh" ]] || {
  echo "ERROR: evaluator Bash not found: ${OGBENCH_ROOT}/scripts/eval_lewm.sh" >&2
  exit 1
}
command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux is not installed." >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
printf 'name\ttask\tmethod\tgpu\tcheckpoint\n' >"${OUTPUT_ROOT}/manifest.tsv"

for i in "${!names[@]}"; do
  name="${names[$i]}"
  checkpoint_dir="${checkpoint_dirs[$i]}"
  session="eval-lewm-${name}-100k"

  [[ -s "${checkpoint_dir}/params_100000.pkl" ]] || {
    echo "ERROR: checkpoint not found: ${checkpoint_dir}/params_100000.pkl" >&2
    exit 1
  }
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${tasks[$i]}" "${methods[$i]}" "${gpus[$i]}" "${checkpoint_dir}" \
    >>"${OUTPUT_ROOT}/manifest.tsv"
done

for i in "${!names[@]}"; do
  name="${names[$i]}"
  session="eval-lewm-${name}-100k"
  tmux new-session -d -s "${session}" -c "${OGBENCH_ROOT}" \
    "bash '${SCRIPT_PATH}' --worker '${i}'"
  echo "Started ${name} on physical GPU ${gpus[$i]}: ${session}"
done

echo "Manifest: ${OUTPUT_ROOT}/manifest.tsv"
