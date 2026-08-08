#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Server 11 defaults. Environment variables and command-line options can
# still override these values through launch_four_init.
DATASETS_DIR="${DATASETS_DIR:-/data/yyf/H-LeWM}"
RUNS_DIR="${RUNS_DIR:-/data/yyf/H-LeWM/runs}"
LOGS_DIR="${LOGS_DIR:-${SCRIPT_DIR}/../logs}"
VENV_DIR="${VENV_DIR:-/data/yyf/H-LeWM/envs/stable-worldmodel}"

# shellcheck source=launch_four_tasks_common.sh
source "${SCRIPT_DIR}/launch_four_tasks_common.sh"
launch_four_init "0,1,2,3" "$@"

tasks=(tworoom reacher pusht ogbench_cube)
datasets=(tworoom.h5 reacher.h5 pusht_expert_train.h5 cube_single_expert.h5)
action_dims=(2 2 2 5)
script_name=$(basename "$0" .sh)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  if [[ -n "$ONLY_TASK" && "$task" != "$ONLY_TASK" ]]; then
    continue
  fi

  exp_id="${task}_${script_name}_vit_tiny_bs256_e10"
  launch_four_run "${exp_id}" "${GPU_IDS[$i]}" gchiql "$task" scripts/train/gchiql.py \
    "dataset_name=${DATASETS_DIR}/${datasets[$i]}" \
    "output_model_name=${exp_id}" \
    "+subdir=${exp_id}" \
    trainer.max_epochs=10 \
    batch_size=256 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=vit_tiny \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    "dinowm.action_dim=${action_dims[$i]}" \
    frameskip=1 \
    goal_gamma=0.99 \
    seed=42 \
    wandb.enabled=true \
    wandb.config.entity=xiguapi \
    wandb.config.project=stable-wm \
    "hydra.run.dir=${LOGS_DIR}/gchiql/${task}/hydra"
done

launch_four_summary gchiql
