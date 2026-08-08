#!/usr/bin/env bash
set -e

# Run from the repository root. This script lives in ./scripts.
cd "$(dirname "$0")/.."

# Expected datasets:
#   datasets/tworoom.h5
#   datasets/reacher.h5
#   datasets/pusht_expert_train.h5
#   datasets/cube_single_expert.h5

# GPU 0: TwoRoom
mkdir -p runs/gciql/tworoom logs/gciql/tworoom
tmux new-session -d -s gciql-tworoom -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=0 STABLEWM_HOME=./runs/gciql/tworoom SPT_CACHE_DIR=./runs/gciql/tworoom PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql.py \
    dataset_name=./datasets/tworoom.h5 \
    output_model_name=gciql_tworoom_dino_bs128_e10 \
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
    hydra.run.dir=./logs/gciql/tworoom/hydra \
    2>&1 | tee ./logs/gciql/tworoom/train.log"

# GPU 1: Reacher
mkdir -p runs/gciql/reacher logs/gciql/reacher
tmux new-session -d -s gciql-reacher -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=1 STABLEWM_HOME=./runs/gciql/reacher SPT_CACHE_DIR=./runs/gciql/reacher PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql.py \
    dataset_name=./datasets/reacher.h5 \
    output_model_name=gciql_reacher_dino_bs128_e10 \
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
    hydra.run.dir=./logs/gciql/reacher/hydra \
    2>&1 | tee ./logs/gciql/reacher/train.log"

# GPU 2: Push-T
mkdir -p runs/gciql/pusht logs/gciql/pusht
tmux new-session -d -s gciql-pusht -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=2 STABLEWM_HOME=./runs/gciql/pusht SPT_CACHE_DIR=./runs/gciql/pusht PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql.py \
    dataset_name=./datasets/pusht_expert_train.h5 \
    output_model_name=gciql_pusht_dino_bs128_e10 \
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
    hydra.run.dir=./logs/gciql/pusht/hydra \
    2>&1 | tee ./logs/gciql/pusht/train.log"

# GPU 3: OGBench Cube
mkdir -p runs/gciql/ogbench_cube logs/gciql/ogbench_cube
tmux new-session -d -s gciql-ogbench_cube -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=3 STABLEWM_HOME=./runs/gciql/ogbench_cube SPT_CACHE_DIR=./runs/gciql/ogbench_cube PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql.py \
    dataset_name=./datasets/cube_single_expert.h5 \
    output_model_name=gciql_ogbench_cube_dino_bs128_e10 \
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
    hydra.run.dir=./logs/gciql/ogbench_cube/hydra \
    2>&1 | tee ./logs/gciql/ogbench_cube/train.log"

echo "Started four GCIQL runs. Logs are under ./logs/gciql/."
