#!/usr/bin/env bash
set -e

# Run from the repository root. This script lives in ./scripts.
cd "$(dirname "$0")/.."

# Expected datasets:
#   data/tworoom.h5
#   data/reacher.h5
#   data/pusht_expert_train.h5
#   data/cube_single_expert.h5

# GPU 0: TwoRoom
mkdir -p runs/gcivl/tworoom logs/gcivl/tworoom
tmux new-session -d -s gcivl-tworoom -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=0 STABLEWM_HOME=./runs/gcivl/tworoom SPT_CACHE_DIR=./runs/gcivl/tworoom PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gcivl.py \
    dataset_name=./data/tworoom.h5 \
    output_model_name=gcivl_tworoom_dino_bs128_e10 \
    +subdir=tworoom \
    trainer.max_epochs=10 \
    batch_size=128 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=2 \
    frameskip=1 \
    seed=42 \
    hydra.run.dir=./logs/gcivl/tworoom/hydra \
    2>&1 | tee ./logs/gcivl/tworoom/train.log"

# GPU 1: Reacher
mkdir -p runs/gcivl/reacher logs/gcivl/reacher
tmux new-session -d -s gcivl-reacher -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=1 STABLEWM_HOME=./runs/gcivl/reacher SPT_CACHE_DIR=./runs/gcivl/reacher PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gcivl.py \
    dataset_name=./data/reacher.h5 \
    output_model_name=gcivl_reacher_dino_bs128_e10 \
    +subdir=reacher \
    trainer.max_epochs=10 \
    batch_size=128 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=2 \
    frameskip=1 \
    seed=42 \
    hydra.run.dir=./logs/gcivl/reacher/hydra \
    2>&1 | tee ./logs/gcivl/reacher/train.log"

# GPU 2: Push-T
mkdir -p runs/gcivl/pusht logs/gcivl/pusht
tmux new-session -d -s gcivl-pusht -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=2 STABLEWM_HOME=./runs/gcivl/pusht SPT_CACHE_DIR=./runs/gcivl/pusht PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gcivl.py \
    dataset_name=./data/pusht_expert_train.h5 \
    output_model_name=gcivl_pusht_dino_bs128_e10 \
    +subdir=pusht \
    trainer.max_epochs=10 \
    batch_size=128 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=2 \
    frameskip=1 \
    seed=42 \
    hydra.run.dir=./logs/gcivl/pusht/hydra \
    2>&1 | tee ./logs/gcivl/pusht/train.log"

# GPU 3: OGBench Cube
mkdir -p runs/gcivl/ogbench_cube logs/gcivl/ogbench_cube
tmux new-session -d -s gcivl-ogbench_cube -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=3 STABLEWM_HOME=./runs/gcivl/ogbench_cube SPT_CACHE_DIR=./runs/gcivl/ogbench_cube PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gcivl.py \
    dataset_name=./data/cube_single_expert.h5 \
    output_model_name=gcivl_ogbench_cube_dino_bs128_e10 \
    +subdir=ogbench_cube \
    trainer.max_epochs=10 \
    batch_size=128 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=5 \
    frameskip=1 \
    seed=42 \
    hydra.run.dir=./logs/gcivl/ogbench_cube/hydra \
    2>&1 | tee ./logs/gcivl/ogbench_cube/train.log"

echo "Started four GCIVL runs. Logs are under ./logs/gcivl/."
