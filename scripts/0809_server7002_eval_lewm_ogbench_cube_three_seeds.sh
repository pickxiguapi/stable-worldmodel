#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/yyf/yyf/stable-worldmodel}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
UV_BIN="${UV_BIN:-/home/yyf/.local/bin/uv}"
DATA_ROOT="${DATA_ROOT:-/mnt/18T/yyf/stablewm-data}"
DATASET="${DATASET:-${DATA_ROOT}/datasets/cube_single_expert.h5}"
CHECKPOINT="${CHECKPOINT:-${DATA_ROOT}/checkpoints/lewm_ogbench_cube/weights_epoch_10.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/evals/lewm_ogbench_cube_epoch10}"

seeds=(1 42 666)
gpus=(0 1 2)

[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
[[ -s "${DATASET}" ]] || { echo "ERROR: Dataset not found: ${DATASET}" >&2; exit 1; }
[[ -s "${CHECKPOINT}" ]] || { echo "ERROR: Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
[[ -s "$(dirname "${CHECKPOINT}")/config.json" ]] || {
  echo "ERROR: Checkpoint config not found: $(dirname "${CHECKPOINT}")/config.json" >&2
  exit 1
}
command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux is not installed." >&2; exit 1; }

if ! "${PYTHON_BIN}" -c 'import ogbench' >/dev/null 2>&1; then
  [[ -x "${UV_BIN}" ]] || { echo "ERROR: uv executable not found: ${UV_BIN}" >&2; exit 1; }
  echo "Installing locked OGBench evaluation dependency into ${PYTHON_BIN}."
  "${UV_BIN}" pip install --python "${PYTHON_BIN}" 'ogbench==1.2.1'
fi
"${PYTHON_BIN}" -c 'import ogbench'

for seed in "${seeds[@]}"; do
  session="eval-lewm-ogbench-cube-e10-seed${seed}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_DIR}"

for i in "${!seeds[@]}"; do
  seed="${seeds[$i]}"
  gpu="${gpus[$i]}"
  session="eval-lewm-ogbench-cube-e10-seed${seed}"
  seed_dir="${OUTPUT_ROOT}/seed_${seed}"
  model_dir="${seed_dir}/model"
  hydra_dir="${seed_dir}/hydra"
  log_file="${seed_dir}/eval.log"
  policy="${model_dir}/weights_epoch_10.pt"
  command=""

  mkdir -p "${model_dir}" "${hydra_dir}"
  ln -sfn "${CHECKPOINT}" "${policy}"
  ln -sfn "$(dirname "${CHECKPOINT}")/config.json" "${model_dir}/config.json"

  printf -v command '%q ' \
    env \
    "CUDA_VISIBLE_DEVICES=${gpu}" \
    "STABLEWM_HOME=${DATA_ROOT}" \
    "PYTHONPATH=${REPO_DIR}" \
    MUJOCO_GL=egl \
    "${PYTHON_BIN}" scripts/plan/eval_wm.py \
    --config-name=cube \
    "policy=${policy}" \
    "eval.dataset_name=${DATASET}" \
    "+plan_config.history_len=3" \
    "seed=${seed}" \
    "output.filename=ogb_cube_epoch10_seed${seed}_results.txt" \
    "hydra.run.dir=${hydra_dir}"
  printf -v command '%s2>&1 | tee -a %q' "${command}" "${log_file}"

  tmux new-session -d -s "${session}" -c "${REPO_DIR}" "${command}"
  echo "Started seed ${seed} on physical GPU ${gpu}: ${session}"
  echo "Output: ${model_dir}"
  echo "Log:    ${log_file}"
done
