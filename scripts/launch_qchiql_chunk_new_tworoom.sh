#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_GPU_COUNT=1
LAUNCH_REQUIRED_DATASETS=(tworoom.h5)

# shellcheck source=launch_four_tasks_common.sh
source "${SCRIPT_DIR}/launch_four_tasks_common.sh"
launch_four_init "4" "$@"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-192}"
run_id="qchiql_chunk_new_tworoom_vittiny_bs${BATCH_SIZE}_s42_${RUN_TAG}"

launch_four_run "qchiql-chunk-new-tworoom" "${GPU_IDS[0]}" qchiql_chunk_new tworoom scripts/train/qchiql_chunk_new.py \
  "dataset_name=${DATASETS_DIR}/tworoom.h5" \
  "output_model_name=qchiql_chunk_new_tworoom_vittiny_bs${BATCH_SIZE}_e10" \
  "subdir=${run_id}" \
  trainer.max_epochs=10 \
  "batch_size=${BATCH_SIZE}" \
  num_workers=8 \
  train_subset_fraction=1.0 \
  encoder=vit_tiny \
  dinowm.history_size=3 \
  dinowm.td_offset=1 \
  dinowm.use_proprio_encoder=false \
  dinowm.action_dim=2 \
  frameskip=5 \
  subgoal_steps=10 \
  low_actor_rep_grad=true \
  seed=42 \
  wandb.enabled=true \
  wandb.config.entity=xiguapi \
  wandb.config.project=stable-wm \
  "hydra.run.dir=${LOGS_DIR}/qchiql_chunk_new/tworoom/hydra"

launch_four_summary qchiql_chunk_new
