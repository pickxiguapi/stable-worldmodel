#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-/root/data/yyf/stable-worldmodel}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs/gciql_chunk/eval_gpu567_20260810}"
OGBENCH_ROOT="${OGBENCH_ROOT:-/root/data/yyf/ogbench}"
OGBENCH_SITE_PACKAGES="${OGBENCH_SITE_PACKAGES:-${OGBENCH_ROOT}/.venv/lib/python3.10/site-packages}"
EGL_RUNTIME_ROOT="${EGL_RUNTIME_ROOT:-/root/data/yyf/egl-runtime/root}"
SEED="${SEED:-777}"
NUM_EVAL="${NUM_EVAL:-50}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"

names=(reacher pusht ogbench_cube)
configs=(reacher pusht cube)
datasets=(reacher.h5 pusht_expert_train.h5 cube_single_expert.h5)
gpus=(5 6 7)
policy_dirs=(
  gciql_chunk_reacher_dino_bs256_e10_policy
  gciql_chunk_pusht_dino_bs256_e10_policy
  gciql_chunk_ogbench_cube_dino_bs256_e10_policy
)

run_worker() {
  local index="$1"
  local name="${names[$index]}"
  local config="${configs[$index]}"
  local gpu="${gpus[$index]}"
  local run_root="${REPO_ROOT}/runs/gciql_chunk/${name}"
  local policy="${run_root}/checkpoints/${policy_dirs[$index]}/weights_epoch_10.pt"
  local dataset="${REPO_ROOT}/datasets/${datasets[$index]}"
  local run_dir="${OUTPUT_ROOT}/${name}/seed_${SEED}"
  local output_filename="gciql_chunk_${name}_offset${GOAL_OFFSET_STEPS}_budget${EVAL_BUDGET}_seed${SEED}_results.txt"
  local status

  mkdir -p "${run_dir}" "${run_root}/tmp"
  cd "${REPO_ROOT}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" \
    STABLEWM_HOME="${run_root}" \
    SPT_CACHE_DIR="${run_root}" \
    TMPDIR="${run_root}/tmp" \
    PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    STABLEWM_EXTRA_SITE_PACKAGES="${OGBENCH_ROOT}:${OGBENCH_SITE_PACKAGES}" \
    LD_LIBRARY_PATH="${EGL_RUNTIME_ROOT}/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${REPO_ROOT}/.venv/bin/python" scripts/plan/eval_chunk.py \
      "--config-name=${config}" \
      "policy=${policy}" \
      "eval.dataset_name=${dataset}" \
      "eval.num_eval=${NUM_EVAL}" \
      "eval.goal_offset_steps=${GOAL_OFFSET_STEPS}" \
      "eval.eval_budget=${EVAL_BUDGET}" \
      "output.filename=${output_filename}" \
      "seed=${SEED}" \
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

[[ -x "${REPO_ROOT}/.venv/bin/python" ]] || {
  echo "ERROR: Python executable not found: ${REPO_ROOT}/.venv/bin/python" >&2
  exit 1
}
[[ -f "${REPO_ROOT}/scripts/plan/eval_chunk.py" ]] || {
  echo "ERROR: chunk evaluator not found." >&2
  exit 1
}
[[ -d "${OGBENCH_SITE_PACKAGES}" ]] || {
  echo "ERROR: offline evaluation packages not found: ${OGBENCH_SITE_PACKAGES}" >&2
  exit 1
}
command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux is not installed." >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}"
printf 'name\tconfig\tgpu\tpolicy\tdataset\n' >"${OUTPUT_ROOT}/manifest.tsv"

for i in "${!names[@]}"; do
  name="${names[$i]}"
  run_root="${REPO_ROOT}/runs/gciql_chunk/${name}"
  policy="${run_root}/checkpoints/${policy_dirs[$i]}/weights_epoch_10.pt"
  config_json="${run_root}/checkpoints/${policy_dirs[$i]}/config.json"
  dataset="${REPO_ROOT}/datasets/${datasets[$i]}"
  session="eval-gciql-chunk-${name}-e10"

  [[ -s "${policy}" ]] || { echo "ERROR: policy not found: ${policy}" >&2; exit 1; }
  [[ -s "${config_json}" ]] || { echo "ERROR: config not found: ${config_json}" >&2; exit 1; }
  [[ -s "${dataset}" ]] || { echo "ERROR: dataset not found: ${dataset}" >&2; exit 1; }
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${configs[$i]}" "${gpus[$i]}" "${policy}" "${dataset}" \
    >>"${OUTPUT_ROOT}/manifest.tsv"
done

for i in "${!names[@]}"; do
  name="${names[$i]}"
  session="eval-gciql-chunk-${name}-e10"
  tmux new-session -d -s "${session}" -c "${REPO_ROOT}" \
    "bash '${SCRIPT_PATH}' --worker '${i}'"
  echo "Started ${name} on physical GPU ${gpus[$i]}: ${session}"
done

echo "Manifest: ${OUTPUT_ROOT}/manifest.tsv"
