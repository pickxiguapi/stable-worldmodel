#!/usr/bin/env bash
set -e

# Run from the repository root. This script lives in ./scripts.
cd "$(dirname "$0")/.."

# Expected datasets:
#   datasets/tworoom.h5
#   datasets/reacher.h5
#   datasets/pusht_expert_train.h5
#   datasets/cube_single_expert.h5
#
# GCIQL-Chunk: flat Q-Chunking with frameskip=5 (HiQC arXiv:2607.20834
# low-level design; no hierarchy, no flow matching). Runs on GPUs 4-7.

# GPU 4: TwoRoom
mkdir -p runs/gciql_chunk/tworoom logs/gciql_chunk/tworoom
tmux new-session -d -s gciql-chunk-tworoom -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=4 STABLEWM_HOME=./runs/gciql_chunk/tworoom SPT_CACHE_DIR=./runs/gciql_chunk/tworoom PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql_chunk.py \
    dataset_name=./datasets/tworoom.h5 \
    output_model_name=gciql_chunk_tworoom_dino_bs256_e10 \
    +subdir=tworoom \
    trainer.max_epochs=10 \
    batch_size=256 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=2 \
    frameskip=5 \
    goal_gamma=0.95099 \
    seed=42 \
    hydra.run.dir=./logs/gciql_chunk/tworoom/hydra \
    2>&1 | tee ./logs/gciql_chunk/tworoom/train.log"

# GPU 5: Reacher
mkdir -p runs/gciql_chunk/reacher logs/gciql_chunk/reacher
tmux new-session -d -s gciql-chunk-reacher -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=5 STABLEWM_HOME=./runs/gciql_chunk/reacher SPT_CACHE_DIR=./runs/gciql_chunk/reacher PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql_chunk.py \
    dataset_name=./datasets/reacher.h5 \
    output_model_name=gciql_chunk_reacher_dino_bs256_e10 \
    +subdir=reacher \
    trainer.max_epochs=10 \
    batch_size=256 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=2 \
    frameskip=5 \
    goal_gamma=0.95099 \
    seed=42 \
    hydra.run.dir=./logs/gciql_chunk/reacher/hydra \
    2>&1 | tee ./logs/gciql_chunk/reacher/train.log"

# GPU 6: Push-T
mkdir -p runs/gciql_chunk/pusht logs/gciql_chunk/pusht
tmux new-session -d -s gciql-chunk-pusht -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=6 STABLEWM_HOME=./runs/gciql_chunk/pusht SPT_CACHE_DIR=./runs/gciql_chunk/pusht PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql_chunk.py \
    dataset_name=./datasets/pusht_expert_train.h5 \
    output_model_name=gciql_chunk_pusht_dino_bs256_e10 \
    +subdir=pusht \
    trainer.max_epochs=10 \
    batch_size=256 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=2 \
    frameskip=5 \
    goal_gamma=0.95099 \
    seed=42 \
    hydra.run.dir=./logs/gciql_chunk/pusht/hydra \
    2>&1 | tee ./logs/gciql_chunk/pusht/train.log"

# GPU 7: OGBench Cube
mkdir -p runs/gciql_chunk/ogbench_cube logs/gciql_chunk/ogbench_cube
tmux new-session -d -s gciql-chunk-ogbench_cube -c "$PWD" \
  "CUDA_VISIBLE_DEVICES=7 STABLEWM_HOME=./runs/gciql_chunk/ogbench_cube SPT_CACHE_DIR=./runs/gciql_chunk/ogbench_cube PYTHONPATH=. \
  ./.venv/bin/python scripts/train/gciql_chunk.py \
    dataset_name=./datasets/cube_single_expert.h5 \
    output_model_name=gciql_chunk_ogbench_cube_dino_bs256_e10 \
    +subdir=ogbench_cube \
    trainer.max_epochs=10 \
    batch_size=256 \
    num_workers=8 \
    train_subset_fraction=1.0 \
    encoder_type=dino \
    dinowm.history_size=3 \
    dinowm.td_offset=1 \
    dinowm.use_proprio_encoder=false \
    dinowm.action_dim=5 \
    frameskip=5 \
    goal_gamma=0.95099 \
    seed=42 \
    hydra.run.dir=./logs/gciql_chunk/ogbench_cube/hydra \
    2>&1 | tee ./logs/gciql_chunk/ogbench_cube/train.log"

echo "Started four GCIQL-Chunk runs on GPUs 4-7. Logs are under ./logs/gciql_chunk/."
