#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="${DATASETS_DIR:-/mnt/18T/yyf/stablewm-data/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/18T/yyf/stablewm-data/runs}"
RUNS_DIR="${RUNS_DIR:-${OUTPUT_DIR}}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_DIR}}"
VENV_DIR="${VENV_DIR:-/home/yyf/yyf/stable-worldmodel}"

# This server runs only the two pixel manipulation tasks.
tasks=(pusht ogbench_cube)
datasets=(pusht_expert_train.h5 cube_single_expert.h5)
action_dims=(2 5)
subgoal_steps=(10 10)
LAUNCH_GPU_COUNT=2
LAUNCH_REQUIRED_DATASETS=("${datasets[@]}")

# shellcheck source=launch_four_tasks_common.sh
source "${SCRIPT_DIR}/launch_four_tasks_common.sh"
launch_four_init "1,2" "$@"

script_name=$(basename "$0" .sh)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  exp_id="${task}_${script_name}_vit_tiny_bs128_e10"
  launch_four_run "${exp_id}" "${GPU_IDS[$i]}" gchiql_chunk "$task" scripts/train/gchiql_chunk.py \
    "dataset_name=${DATASETS_DIR}/${datasets[$i]}" \
    "output_model_name=${exp_id}" \
    "+subdir=${exp_id}" \
    trainer.max_epochs=10 \
    batch_size=128 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=vit_tiny \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    "dinowm.action_dim=${action_dims[$i]}" \
    frameskip=5 \
    "subgoal_steps=${subgoal_steps[$i]}" \
    low_actor_rep_grad=true \
    goal_gamma=0.95099 \
    seed=42 \
    wandb.enabled=true \
    wandb.config.entity=xiguapi \
    wandb.config.project=stable-wm \
    "hydra.run.dir=${LOGS_DIR}/gchiql_chunk/${task}/hydra"
done

launch_four_summary gchiql_chunk
