#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launch_four_tasks_common.sh
source "${SCRIPT_DIR}/launch_four_tasks_common.sh"
launch_four_init "0,1,2,3" "$@"

tasks=(tworoom reacher pusht ogbench_cube)
datasets=(tworoom.h5 reacher.h5 pusht_expert_train.h5 cube_single_expert.h5)
data_configs=(tworoom dmc pusht ogb)

for i in "${!tasks[@]}"; do
  task="${tasks[$i]}"
  launch_four_run "lewm-${task}" "${GPU_IDS[$i]}" lewm "$task" scripts/train/lewm.py \
    "data=${data_configs[$i]}" \
    "data.dataset.name=${DATASETS_DIR}/${datasets[$i]}" \
    "output_model_name=lewm_${task}" \
    "subdir=${task}" \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    loader.batch_size=128 \
    loader.num_workers=4 \
    wandb.enabled=true \
    wandb.config.entity=xiguapi \
    wandb.config.project=stable-wm \
    "hydra.run.dir=${LOGS_DIR}/lewm/${task}/hydra"
done

launch_four_summary lewm
