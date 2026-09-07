#!/usr/bin/env bash
set -euo pipefail

# Yingbo Cloud: 10-episode closed-loop evaluation for the two simple GC tasks.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MODE=${MODE:-status}
PYTHON_BIN=${PYTHON_BIN:-$STABLEWM_ROOT/.venv/bin/python}
RUN_ROOT=${RUN_ROOT:-/root/data/yyf/tdmpc2-official-gc-ogbench8-runs}
EVAL_ROOT=${EVAL_ROOT:-/root/data/yyf/tdmpc2-official-gc-simple-eval}
OGBENCH_ROOT=${OGBENCH_ROOT:-/root/data/yyf/ogbench-eval-main-20260830}
OGBENCH_SITE_PACKAGES=${OGBENCH_SITE_PACKAGES:-$OGBENCH_ROOT/.venv/lib/python3.10/site-packages}
EGL_RUNTIME_ROOT=${EGL_RUNTIME_ROOT:-$EVAL_ROOT/.runtime/egl}
TRAIN_LABEL=${TRAIN_LABEL:-neg1zero_actorbc1_mppi_original}
EVAL_LABEL=${EVAL_LABEL:-eval10_s42}
MAX_STEPS=${MAX_STEPS:-100000}
SEGMENT_TRANSITIONS=${SEGMENT_TRANSITIONS:-50}
TRAIN_SEED=${TRAIN_SEED:-1}
EPISODES=${EPISODES:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
EVAL_SEED=${EVAL_SEED:-42}
VISUALIZE_INFO=${VISUALIZE_INFO:-0}
GPU_IDS=${GPU_IDS:-"0 1"}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONUNBUFFERED=1
export PYTHONPATH="$STABLEWM_ROOT:$OGBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OGBENCH_SITE_PACKAGES

tags=(cs_play cs_noisy)
labels=(cube_single_play cube_single_noisy)
read -r -a gpus <<<"$GPU_IDS"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$OGBENCH_ROOT/ogbench/__init__.py" ]]; then
  echo "Clean OGBench source not found: $OGBENCH_ROOT" >&2
  exit 2
fi
if (( ${#gpus[@]} != ${#tags[@]} )); then
  echo "GPU_IDS must contain exactly two GPU IDs" >&2
  exit 2
fi

prepare_egl_runtime() {
  if ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; then
    return
  fi
  local lib_dir="$EGL_RUNTIME_ROOT/root/usr/lib/x86_64-linux-gnu"
  if [[ ! -f "$lib_dir/libEGL.so.1" || ! -f "$lib_dir/libGL.so.1" ]]; then
    local deb_dir="$EGL_RUNTIME_ROOT/debs"
    mkdir -p "$deb_dir" "$EGL_RUNTIME_ROOT/root"
    (
      cd "$deb_dir"
      apt-get download -qq libegl1 libglvnd0 libgl1 libglx0 libopengl0
      for deb in ./*.deb; do
        dpkg-deb -x "$deb" "$EGL_RUNTIME_ROOT/root"
      done
    )
  fi
  export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

run_name() {
  local index=$1
  printf 'tdmpc2_official_%s_gc_h%s_m%s_s%s_%s\n' \
    "${tags[$index]}" "$SEGMENT_TRANSITIONS" "$MAX_STEPS" \
    "$TRAIN_SEED" "$TRAIN_LABEL"
}

checkpoint_path() {
  local index=$1
  local run_dir="$RUN_ROOT/$(run_name "$index")"
  local checkpoint="$run_dir/weights_step_${MAX_STEPS}.pt"
  if [[ ! -f "$checkpoint" || ! -f "$run_dir/official_config.json" ]]; then
    echo "Missing final checkpoint/config under $run_dir" >&2
    return 2
  fi
  printf '%s\n' "$checkpoint"
}

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]'
}

run_one() {
  : "${TASK_INDEX:?TASK_INDEX is required for MODE=run-one}"
  : "${GPU_ID:?GPU_ID is required for MODE=run-one}"
  prepare_egl_runtime
  local run_dir="$RUN_ROOT/$(run_name "$TASK_INDEX")"
  local checkpoint
  checkpoint=$(checkpoint_path "$TASK_INDEX")
  local output_dir="$EVAL_ROOT/$EVAL_LABEL/${labels[$TASK_INDEX]}"
  local visualize_arg=--visualize-info
  if [[ "$VISUALIZE_INFO" == 0 ]]; then
    visualize_arg=--no-visualize-info
  fi
  mkdir -p "$EVAL_ROOT/$EVAL_LABEL"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to reuse evaluation directory: $output_dir" >&2
    exit 4
  fi
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  exec "$PYTHON_BIN" scripts/plan/eval_tdmpc2_official_gc_cube.py \
    --checkpoint "$checkpoint" \
    --config "$run_dir/official_config.json" \
    --output-dir "$output_dir" \
    --label "${labels[$TASK_INDEX]}" \
    --episodes "$EPISODES" \
    --seed "$EVAL_SEED" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    "$visualize_arg" \
    2>&1 | tee "$EVAL_ROOT/$EVAL_LABEL/${labels[$TASK_INDEX]}.log"
}

launch() {
  mkdir -p "$EVAL_ROOT/$EVAL_LABEL"
  for i in "${!tags[@]}"; do
    checkpoint_path "$i" >/dev/null
    local used
    used=$(gpu_memory_mib "${gpus[$i]}")
    if (( used >= GPU_MEMORY_LIMIT_MIB )); then
      echo "GPU ${gpus[$i]} busy: ${used} MiB used" >&2
      exit 3
    fi
    if [[ -e "$EVAL_ROOT/$EVAL_LABEL/${labels[$i]}" ]]; then
      echo "Refusing existing evaluation output for ${labels[$i]}" >&2
      exit 4
    fi
  done
  for i in "${!tags[@]}"; do
    local session="eval_official_tdmpc2_${labels[$i]}_${EVAL_LABEL}"
    tmux new-session -d -s "${session:0:90}" \
      "cd '$STABLEWM_ROOT' && MODE=run-one TASK_INDEX='$i' GPU_ID='${gpus[$i]}' PYTHON_BIN='$PYTHON_BIN' RUN_ROOT='$RUN_ROOT' EVAL_ROOT='$EVAL_ROOT' OGBENCH_ROOT='$OGBENCH_ROOT' OGBENCH_SITE_PACKAGES='$OGBENCH_SITE_PACKAGES' EGL_RUNTIME_ROOT='$EGL_RUNTIME_ROOT' TRAIN_LABEL='$TRAIN_LABEL' EVAL_LABEL='$EVAL_LABEL' MAX_STEPS='$MAX_STEPS' SEGMENT_TRANSITIONS='$SEGMENT_TRANSITIONS' TRAIN_SEED='$TRAIN_SEED' EPISODES='$EPISODES' MAX_EPISODE_STEPS='$MAX_EPISODE_STEPS' EVAL_SEED='$EVAL_SEED' VISUALIZE_INFO='$VISUALIZE_INFO' bash '$STABLEWM_ROOT/scripts/20260907_yb_eval_tdmpc2_official_gc_simple.sh'"
    echo "Launched ${labels[$i]} on physical GPU ${gpus[$i]}"
  done
}

status() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep '^eval_official_tdmpc2_' || true
  find "$EVAL_ROOT" -name results.json -type f -print 2>/dev/null \
    | sort | tail -n 10 || true
}

case "$MODE" in
  launch) launch ;;
  run-one) run_one ;;
  status) status ;;
  *)
    echo "MODE must be launch, run-one, or status" >&2
    exit 2
    ;;
esac
