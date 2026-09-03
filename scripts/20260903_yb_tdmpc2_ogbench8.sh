#!/usr/bin/env bash
set -euo pipefail

# Yingbo Cloud: prepare and launch pixels-only TD-MPC2 on OGBench-8Tasks.
# Every operation, including a one-task smoke test, is routed through this
# workspace-controlled Bash as required by the project experiment policy.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# The clean checkout reuses an existing virtualenv whose editable install may
# point at an older working tree.  Put this checkout first so every Python
# entry point imports the exact GitHub-main source paired with this Bash.
export PYTHONPATH="$STABLEWM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE=${MODE:-status}
PYTHON_BIN=${PYTHON_BIN:-$STABLEWM_ROOT/.venv/bin/python}
SOURCE_ROOT=${SOURCE_ROOT:-/root/data/yyf/ogbench-cache/data}
DATASET_ROOT=${DATASET_ROOT:-/root/data/yyf/stablewm-data/datasets/ogbench8-tdmpc2-pixels-gc-h50}
RUN_ROOT=${RUN_ROOT:-/root/data/yyf/tdmpc2-ogbench8-runs}
SEGMENT_TRANSITIONS=${SEGMENT_TRANSITIONS:-50}
MIN_TRANSITIONS=${MIN_TRANSITIONS:-5}
MAX_STEPS=${MAX_STEPS:-100000}
SEED=${SEED:-1}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-60}
SMOKE_STEPS=${SMOKE_STEPS:-2}

envs=(
  visual-cube-single-play-v0
  visual-cube-double-play-v0
  visual-cube-triple-play-v0
  visual-scene-play-v0
  visual-cube-single-noisy-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0
  visual-scene-noisy-v0
)
tags=(
  cs_play
  cd_play
  ct_play
  scene_play
  cs_noisy
  cd_noisy
  ct_noisy
  scene_noisy
)
read -r -a gpus <<<"$GPU_IDS"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if (( ${#gpus[@]} != ${#envs[@]} )); then
  echo "GPU_IDS must contain exactly ${#envs[@]} GPU IDs" >&2
  exit 2
fi

converter="$STABLEWM_ROOT/scripts/data/convert_ogbench_npz_tdmpc2.py"

dataset_path() {
  local index=$1
  printf '%s/%s.h5\n' "$DATASET_ROOT" "${envs[$index]}"
}

run_name() {
  local index=$1
  printf 'tdmpc2_ogbench8_pixels_%s_gc%s_ms%s_s%s\n' \
    "${tags[$index]}" "$SEGMENT_TRANSITIONS" "$MAX_STEPS" "$SEED"
}

prepare_data() {
  mkdir -p "$DATASET_ROOT"
  for i in "${!envs[@]}"; do
    local source="$SOURCE_ROOT/${envs[$i]}.npz"
    local dest
    dest=$(dataset_path "$i")
    if [[ ! -f "$source" ]]; then
      echo "Missing OGBench source dataset: $source" >&2
      exit 2
    fi
    if [[ ! -f "$dest" ]]; then
      "$PYTHON_BIN" "$converter" convert \
        --source "$source" \
        --dest "$dest" \
        --segment-transitions "$SEGMENT_TRANSITIONS" \
        --min-transitions "$MIN_TRANSITIONS"
    fi
    "$PYTHON_BIN" "$converter" validate \
      --segment-transitions "$SEGMENT_TRANSITIONS" "$dest"
  done
}

gpu_memory_mib() {
  local gpu=$1
  nvidia-smi --id="$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]'
}

all_gpus_free() {
  local busy=0
  for gpu in "${gpus[@]}"; do
    local used
    used=$(gpu_memory_mib "$gpu")
    if (( used >= GPU_MEMORY_LIMIT_MIB )); then
      echo "GPU $gpu busy: ${used} MiB used" >&2
      busy=1
    fi
  done
  (( busy == 0 ))
}

run_one() {
  : "${TASK_INDEX:?TASK_INDEX is required for MODE=run-one}"
  : "${GPU_ID:?GPU_ID is required for MODE=run-one}"
  local dataset
  local name
  dataset=$(dataset_path "$TASK_INDEX")
  name=$(run_name "$TASK_INDEX")
  local run_dir="$RUN_ROOT/$name"
  local stablewm_home="$run_dir/stablewm-home"
  mkdir -p "$run_dir" "$stablewm_home"

  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export STABLEWM_HOME="$stablewm_home"
  export PYTHONUNBUFFERED=1
  exec "$PYTHON_BIN" scripts/train/tdmpc2.py \
    "dataset_name=$dataset" \
    "output_model_name=$name" \
    "subdir=$name" \
    "seed=$SEED" \
    "wandb.enable=false" \
    "goal_obs_key=pixels" \
    "~model.cfg.wm.encoding.state" \
    "+model.cfg.wm.encoding.pixels=128" \
    "model.cfg.image_channels=6" \
    "+trainer.max_steps=$MAX_STEPS" \
    2>&1 | tee "$run_dir/train.log"
}

launch_all() {
  prepare_data
  if ! all_gpus_free; then
    echo "Launch aborted: all eight GPUs must be below ${GPU_MEMORY_LIMIT_MIB} MiB." >&2
    echo "Use MODE=wait-launch to wait without sharing GPUs." >&2
    exit 3
  fi

  mkdir -p "$RUN_ROOT"
  for i in "${!envs[@]}"; do
    local pending_name
    pending_name=$(run_name "$i")
    if [[ -e "$RUN_ROOT/$pending_name" ]]; then
      echo "Refusing to reuse existing run directory: $RUN_ROOT/$pending_name" >&2
      exit 4
    fi
  done
  for i in "${!envs[@]}"; do
    local name
    local session
    name=$(run_name "$i")
    session="${name:0:70}"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Refusing duplicate tmux session: $session" >&2
      exit 4
    fi
    tmux new-session -d -s "$session" \
      "cd '$STABLEWM_ROOT' && MODE=run-one TASK_INDEX='$i' GPU_ID='${gpus[$i]}' PYTHON_BIN='$PYTHON_BIN' SOURCE_ROOT='$SOURCE_ROOT' DATASET_ROOT='$DATASET_ROOT' RUN_ROOT='$RUN_ROOT' SEGMENT_TRANSITIONS='$SEGMENT_TRANSITIONS' MIN_TRANSITIONS='$MIN_TRANSITIONS' MAX_STEPS='$MAX_STEPS' SEED='$SEED' bash '$STABLEWM_ROOT/scripts/20260903_yb_tdmpc2_ogbench8.sh'"
    echo "Launched $name on physical GPU ${gpus[$i]} in tmux $session"
  done
}

wait_and_launch() {
  prepare_data
  while ! all_gpus_free; do
    echo "$(date '+%F %T %Z') waiting ${WAIT_INTERVAL_SECONDS}s for all GPUs"
    sleep "$WAIT_INTERVAL_SECONDS"
  done
  launch_all
}

smoke_test() {
  prepare_data
  local dataset
  dataset=$(dataset_path 0)
  local stamp
  stamp=$(date +%Y%m%dT%H%M%S)
  local run_dir="$RUN_ROOT/smoke_cpu_$stamp"
  mkdir -p "$run_dir"
  cd "$STABLEWM_ROOT"
  CUDA_VISIBLE_DEVICES='' STABLEWM_HOME="$run_dir/stablewm-home" \
    "$PYTHON_BIN" scripts/train/tdmpc2.py \
      "dataset_name=$dataset" \
      "output_model_name=tdmpc2_ogbench8_smoke" \
      "subdir=smoke_cpu_$stamp" \
      "wandb.enable=false" \
      "goal_obs_key=pixels" \
      "~model.cfg.wm.encoding.state" \
      "+model.cfg.wm.encoding.pixels=128" \
      "model.cfg.image_channels=6" \
      "batch_size=8" \
      "num_workers=2" \
      "trainer.accelerator=cpu" \
      "trainer.devices=1" \
      "trainer.precision=32-true" \
      "+trainer.max_steps=$SMOKE_STEPS" \
      "+trainer.limit_val_batches=1" \
      2>&1 | tee "$run_dir/train.log"
}

show_status() {
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
  echo "TD-MPC2 tmux sessions:"
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep '^tdmpc2_ogbench8_' || true
}

case "$MODE" in
  prepare) prepare_data ;;
  launch) launch_all ;;
  wait-launch) wait_and_launch ;;
  run-one) run_one ;;
  smoke) smoke_test ;;
  status) show_status ;;
  *)
    echo "MODE must be prepare, launch, wait-launch, run-one, smoke, or status" >&2
    exit 2
    ;;
esac
