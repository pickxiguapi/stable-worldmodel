#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launch_four_tasks_common.sh
source "${SCRIPT_DIR}/launch_four_tasks_common.sh"
launch_four_init "0,1,2,3" "$@"

tasks=(tworoom reacher pusht ogbench_cube)
datasets=(tworoom.h5 reacher.h5 pusht_expert_train.h5 cube_single_expert.h5)
action_dims=(2 2 2 5)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  launch_four_run "gciql-${task}" "${GPU_IDS[$i]}" gciql "$task" scripts/train/gciql.py \
    "dataset_name=${DATASETS_DIR}/${datasets[$i]}" \
    "output_model_name=gciql_${task}_dino_bs128_e10" \
    "+subdir=${task}" \
    trainer.max_epochs=10 \
    batch_size=128 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    "dinowm.action_dim=${action_dims[$i]}" \
    frameskip=1 \
    seed=42 \
    wandb.enabled=true \
    wandb.config.entity=xiguapi \
    wandb.config.project=stable-wm \
    "hydra.run.dir=${LOGS_DIR}/gciql/${task}/hydra"
done

launch_four_summary gciql
