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
mkdir -p runs/lewm/tworoom logs/lewm/tworoom
tmux new-session -d -s lewm-tworoom -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=0 STABLEWM_HOME=./runs/lewm/tworoom SPT_CACHE_DIR=./runs/lewm/tworoom PYTHONPATH=. \
  ./.venv/bin/python scripts/train/lewm.py \
    data=tworoom \
    data.dataset.name=$PWD/datasets/tworoom.h5 \
    output_model_name=lewm_tworoom \
    subdir=tworoom \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    loader.batch_size=128 \
    loader.num_workers=4 \
    hydra.run.dir=./logs/lewm/tworoom/hydra \
    2>&1 | tee ./logs/lewm/tworoom/train.log"

# GPU 1: Reacher
mkdir -p runs/lewm/reacher logs/lewm/reacher
tmux new-session -d -s lewm-reacher -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=1 STABLEWM_HOME=./runs/lewm/reacher SPT_CACHE_DIR=./runs/lewm/reacher PYTHONPATH=. \
  ./.venv/bin/python scripts/train/lewm.py \
    data=dmc \
    data.dataset.name=$PWD/datasets/reacher.h5 \
    output_model_name=lewm_reacher \
    subdir=reacher \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    loader.batch_size=128 \
    loader.num_workers=4 \
    hydra.run.dir=./logs/lewm/reacher/hydra \
    2>&1 | tee ./logs/lewm/reacher/train.log"

# GPU 2: Push-T
mkdir -p runs/lewm/pusht logs/lewm/pusht
tmux new-session -d -s lewm-pusht -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=2 STABLEWM_HOME=./runs/lewm/pusht SPT_CACHE_DIR=./runs/lewm/pusht PYTHONPATH=. \
  ./.venv/bin/python scripts/train/lewm.py \
    data=pusht \
    data.dataset.name=$PWD/datasets/pusht_expert_train.h5 \
    output_model_name=lewm_pusht \
    subdir=pusht \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    loader.batch_size=128 \
    loader.num_workers=4 \
    hydra.run.dir=./logs/lewm/pusht/hydra \
    2>&1 | tee ./logs/lewm/pusht/train.log"

# GPU 3: OGBench Cube
mkdir -p runs/lewm/ogbench_cube logs/lewm/ogbench_cube
tmux new-session -d -s lewm-ogbench_cube -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=3 STABLEWM_HOME=./runs/lewm/ogbench_cube SPT_CACHE_DIR=./runs/lewm/ogbench_cube PYTHONPATH=. \
  ./.venv/bin/python scripts/train/lewm.py \
    data=ogb \
    data.dataset.name=$PWD/datasets/cube_single_expert.h5 \
    output_model_name=lewm_ogbench_cube \
    subdir=ogbench_cube \
    trainer.max_epochs=10 \
    trainer.devices=1 \
    loader.batch_size=128 \
    loader.num_workers=4 \
    hydra.run.dir=./logs/lewm/ogbench_cube/hydra \
    2>&1 | tee ./logs/lewm/ogbench_cube/train.log"

echo "Started four LeWM runs. Logs are under ./logs/lewm/."
