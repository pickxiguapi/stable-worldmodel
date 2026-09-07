#!/usr/bin/env bash
set -euo pipefail

# Yingbo Cloud: official TD-MPC2 core + goal-conditioned offline OGBench.
# Every validation, smoke, training, and wait operation is routed here.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MODE=${MODE:-status}
PYTHON_BIN=${PYTHON_BIN:-$STABLEWM_ROOT/.venv/bin/python}
DATASET_ROOT=${DATASET_ROOT:-/root/data/yyf/stablewm-data/datasets/ogbench8-tdmpc2-pixels-gc-h50}
RUN_ROOT=${RUN_ROOT:-/root/data/yyf/tdmpc2-official-gc-ogbench8-runs}
RUN_LABEL=${RUN_LABEL:-official_e9f5932_gc_h50_ms100000_s1}
OFFICIAL_COMMIT=${OFFICIAL_COMMIT:-e9f59321933cbc8e11a002b842adc7d4ffae8ff1}
SEGMENT_TRANSITIONS=${SEGMENT_TRANSITIONS:-50}
MAX_STEPS=${MAX_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-256}
HORIZON=${HORIZON:-3}
MODEL_SIZE=${MODEL_SIZE:-5}
SEED=${SEED:-1}
LOG_INTERVAL=${LOG_INTERVAL:-100}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10000}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
WAIT_INTERVAL_SECONDS=${WAIT_INTERVAL_SECONDS:-60}
COMPILE=${COMPILE:-1}

export PYTHONPATH="$STABLEWM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}

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

check_official_source() {
  local source="$STABLEWM_ROOT/third_party/tdmpc2"
  if [[ ! -d "$source/.git" && ! -f "$source/.git" ]]; then
    echo "Official TD-MPC2 submodule is not initialized: $source" >&2
    exit 2
  fi
  local actual
  actual=$(git -C "$source" rev-parse HEAD)
  if [[ "$actual" != "$OFFICIAL_COMMIT" ]]; then
    echo "Official TD-MPC2 commit mismatch: expected $OFFICIAL_COMMIT, found $actual" >&2
    exit 2
  fi
  if [[ -n $(git -C "$source" status --short) ]]; then
    echo "Official TD-MPC2 submodule has local modifications" >&2
    git -C "$source" status --short >&2
    exit 2
  fi
}

dataset_path() {
  local index=$1
  printf '%s/%s.h5\n' "$DATASET_ROOT" "${envs[$index]}"
}

run_name() {
  local index=$1
  printf 'tdmpc2_official_%s_gc_h%s_m%s_s%s_%s\n' \
    "${tags[$index]}" "$SEGMENT_TRANSITIONS" "$MAX_STEPS" "$SEED" "$RUN_LABEL"
}

validate_data() {
  check_official_source
  local converter="$STABLEWM_ROOT/scripts/data/convert_ogbench_npz_tdmpc2.py"
  for i in "${!envs[@]}"; do
    local dataset
    dataset=$(dataset_path "$i")
    if [[ ! -f "$dataset" ]]; then
      echo "Missing converted dataset: $dataset" >&2
      exit 2
    fi
    "$PYTHON_BIN" "$converter" validate \
      --segment-transitions "$SEGMENT_TRANSITIONS" "$dataset"
  done
}

audit_data() {
  check_official_source
  local converter="$STABLEWM_ROOT/scripts/data/convert_ogbench_npz_tdmpc2.py"
  for i in "${!envs[@]}"; do
    "$PYTHON_BIN" "$converter" validate \
      --segment-transitions "$SEGMENT_TRANSITIONS" \
      --verify-source "$(dataset_path "$i")"
  done
}

unit_test() {
  check_official_source
  cd "$STABLEWM_ROOT"
  "$PYTHON_BIN" -m pytest -q tests/test_tdmpc2_official_gc.py
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
  check_official_source

  local dataset
  local name
  dataset=$(dataset_path "$TASK_INDEX")
  name=$(run_name "$TASK_INDEX")
  local run_dir="$RUN_ROOT/$name"
  local compile_arg=--compile
  if [[ "$COMPILE" == 0 ]]; then
    compile_arg=--no-compile
  fi
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse run directory: $run_dir" >&2
    exit 4
  fi
  mkdir -p "$RUN_ROOT"

  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  exec "$PYTHON_BIN" scripts/train/tdmpc2_official_gc.py \
    --dataset "$dataset" \
    --output-dir "$run_dir" \
    --task "${envs[$TASK_INDEX]}" \
    --steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --horizon "$HORIZON" \
    --segment-transitions "$SEGMENT_TRANSITIONS" \
    --model-size "$MODEL_SIZE" \
    --seed "$SEED" \
    --log-interval "$LOG_INTERVAL" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --episodic \
    "$compile_arg" \
    2>&1 | tee "$RUN_ROOT/$name.log"
}

smoke() {
  validate_data
  if ! all_gpus_free; then
    echo "Smoke requires all configured GPUs free; set GPU_IDS to eight copies is not supported." >&2
    exit 3
  fi
  local stamp
  stamp=$(date +%Y%m%dT%H%M%S)
  MODE=run-one TASK_INDEX=0 GPU_ID="${gpus[0]}" \
    PYTHON_BIN="$PYTHON_BIN" DATASET_ROOT="$DATASET_ROOT" RUN_ROOT="$RUN_ROOT" \
    OFFICIAL_COMMIT="$OFFICIAL_COMMIT" SEGMENT_TRANSITIONS="$SEGMENT_TRANSITIONS" \
    RUN_LABEL="smoke_${stamp}" MAX_STEPS=2 BATCH_SIZE=8 HORIZON="$HORIZON" \
    MODEL_SIZE="$MODEL_SIZE" SEED="$SEED" LOG_INTERVAL=1 CHECKPOINT_INTERVAL=2 \
    COMPILE="$COMPILE" \
    bash "$STABLEWM_ROOT/scripts/20260907_yb_tdmpc2_official_gc_ogbench8.sh"
}

launch_all() {
  validate_data
  if ! all_gpus_free; then
    echo "Launch aborted: all eight GPUs must be below ${GPU_MEMORY_LIMIT_MIB} MiB." >&2
    echo "Use MODE=wait-launch to queue without sharing GPUs." >&2
    exit 3
  fi

  mkdir -p "$RUN_ROOT"
  for i in "${!envs[@]}"; do
    local name
    name=$(run_name "$i")
    if [[ -e "$RUN_ROOT/$name" || -e "$RUN_ROOT/$name.log" ]]; then
      echo "Refusing to reuse run artifacts for $name" >&2
      exit 4
    fi
  done

  for i in "${!envs[@]}"; do
    local name
    local session
    name=$(run_name "$i")
    session="${name:0:90}"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Refusing duplicate tmux session: $session" >&2
      exit 4
    fi
    tmux new-session -d -s "$session" \
      "cd '$STABLEWM_ROOT' && MODE=run-one TASK_INDEX='$i' GPU_ID='${gpus[$i]}' PYTHON_BIN='$PYTHON_BIN' DATASET_ROOT='$DATASET_ROOT' RUN_ROOT='$RUN_ROOT' RUN_LABEL='$RUN_LABEL' OFFICIAL_COMMIT='$OFFICIAL_COMMIT' SEGMENT_TRANSITIONS='$SEGMENT_TRANSITIONS' MAX_STEPS='$MAX_STEPS' BATCH_SIZE='$BATCH_SIZE' HORIZON='$HORIZON' MODEL_SIZE='$MODEL_SIZE' SEED='$SEED' LOG_INTERVAL='$LOG_INTERVAL' CHECKPOINT_INTERVAL='$CHECKPOINT_INTERVAL' COMPILE='$COMPILE' bash '$STABLEWM_ROOT/scripts/20260907_yb_tdmpc2_official_gc_ogbench8.sh'"
    echo "Launched $name on physical GPU ${gpus[$i]} in tmux $session"
  done
}

wait_and_launch() {
  validate_data
  while ! all_gpus_free; do
    echo "$(date '+%F %T %Z') waiting ${WAIT_INTERVAL_SECONDS}s for all GPUs"
    sleep "$WAIT_INTERVAL_SECONDS"
  done
  MODE=smoke GPU_IDS="$GPU_IDS" \
    bash "$STABLEWM_ROOT/scripts/20260907_yb_tdmpc2_official_gc_ogbench8.sh"
  launch_all
}

queue_waiter() {
  validate_data
  mkdir -p "$RUN_ROOT"
  local session="wait_tdmpc2_official_gc_${RUN_LABEL:0:55}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Waiter already exists: $session" >&2
    exit 4
  fi
  tmux new-session -d -s "$session" \
    "cd '$STABLEWM_ROOT' && MODE=wait-launch PYTHON_BIN='$PYTHON_BIN' DATASET_ROOT='$DATASET_ROOT' RUN_ROOT='$RUN_ROOT' RUN_LABEL='$RUN_LABEL' OFFICIAL_COMMIT='$OFFICIAL_COMMIT' SEGMENT_TRANSITIONS='$SEGMENT_TRANSITIONS' MAX_STEPS='$MAX_STEPS' BATCH_SIZE='$BATCH_SIZE' HORIZON='$HORIZON' MODEL_SIZE='$MODEL_SIZE' SEED='$SEED' LOG_INTERVAL='$LOG_INTERVAL' CHECKPOINT_INTERVAL='$CHECKPOINT_INTERVAL' GPU_IDS='$GPU_IDS' GPU_MEMORY_LIMIT_MIB='$GPU_MEMORY_LIMIT_MIB' WAIT_INTERVAL_SECONDS='$WAIT_INTERVAL_SECONDS' COMPILE='$COMPILE' bash '$STABLEWM_ROOT/scripts/20260907_yb_tdmpc2_official_gc_ogbench8.sh' 2>&1 | tee '$RUN_ROOT/${session}.log'"
  echo "Queued waiter in tmux $session"
}

status() {
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
  echo "Official GC TD-MPC2 tmux sessions:"
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep -E '^(tdmpc2_official_|wait_tdmpc2_official_)' || true
  if [[ -d "$RUN_ROOT" ]]; then
    find "$RUN_ROOT" -name 'weights_step_*.pt' -type f -print \
      | sort | tail -n 20
  fi
}

case "$MODE" in
  test) unit_test ;;
  validate) validate_data ;;
  audit-data) audit_data ;;
  smoke) smoke ;;
  run-one) run_one ;;
  launch) launch_all ;;
  wait-launch) wait_and_launch ;;
  queue-waiter) queue_waiter ;;
  status) status ;;
  *)
    echo "MODE must be test, validate, audit-data, smoke, run-one, launch, wait-launch, queue-waiter, or status" >&2
    exit 2
    ;;
esac
