#!/usr/bin/env bash
set -euo pipefail

# Yingbo Cloud: evaluate the two single-cube TD-MPC2 checkpoints. All remote
# execution is routed through this workspace-controlled Bash.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONUNBUFFERED=1

MODE=${MODE:-status}
PYTHON_BIN=${PYTHON_BIN:-$STABLEWM_ROOT/.venv/bin/python}
RUN_ROOT=${RUN_ROOT:-/root/data/yyf/tdmpc2-ogbench8-runs}
EVAL_ROOT=${EVAL_ROOT:-/root/data/yyf/tdmpc2-ogbench8-eval}
OGBENCH_ROOT=${OGBENCH_ROOT:-/root/data/yyf/ogbench-eval-main-20260830}
OGBENCH_SITE_PACKAGES=${OGBENCH_SITE_PACKAGES:-$OGBENCH_ROOT/.venv/lib/python3.10/site-packages}
EGL_RUNTIME_ROOT=${EGL_RUNTIME_ROOT:-$EVAL_ROOT/.runtime/egl}
RUN_LABEL=${RUN_LABEL:-$(date +%Y%m%dT%H%M%S)}
EPISODES=${EPISODES:-10}
NUM_ENVS=${NUM_ENVS:-10}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
SEED=${SEED:-42}
REWARD_TASK_ID=${REWARD_TASK_ID:-2}
GPU_IDS=${GPU_IDS:-"0 1"}
GPU_MEMORY_LIMIT_MIB=${GPU_MEMORY_LIMIT_MIB:-500}
NO_VIDEO=${NO_VIDEO:-0}

export PYTHONPATH="$STABLEWM_ROOT:$OGBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OGBENCH_SITE_PACKAGES

names=(
  tdmpc2_ogbench8_pixels_cs_play_gc50_ms100000_s1
  tdmpc2_ogbench8_pixels_cs_noisy_gc50_ms100000_s1
)
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
if [[ ! -f "$OGBENCH_SITE_PACKAGES/pygame/__init__.py" ]]; then
  echo "OGBench runtime extras not found: $OGBENCH_SITE_PACKAGES" >&2
  exit 2
fi
if (( ${#gpus[@]} != ${#names[@]} )); then
  echo "GPU_IDS must contain exactly ${#names[@]} GPU IDs" >&2
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

prepare_egl_runtime

checkpoint_path() {
  local index=$1
  local checkpoint_dir="$RUN_ROOT/${names[$index]}/stablewm-home/checkpoints/${names[$index]}"
  local checkpoint="$checkpoint_dir/weights_epoch_31.pt"
  if [[ ! -f "$checkpoint" || ! -f "$checkpoint_dir/config.json" ]]; then
    echo "Missing final checkpoint or config under $checkpoint_dir" >&2
    return 2
  fi
  printf '%s\n' "$checkpoint"
}

gpu_memory_mib() {
  local gpu=$1
  nvidia-smi --id="$gpu" --query-gpu=memory.used \
    --format=csv,noheader,nounits | tr -d '[:space:]'
}

run_one() {
  : "${TASK_INDEX:?TASK_INDEX is required for MODE=run-one}"
  : "${GPU_ID:?GPU_ID is required for MODE=run-one}"
  local checkpoint
  checkpoint=$(checkpoint_path "$TASK_INDEX")
  local output_dir="$EVAL_ROOT/$RUN_LABEL/${labels[$TASK_INDEX]}"
  local video_arg=()
  mkdir -p "$EVAL_ROOT/$RUN_LABEL"
  if [[ "$NO_VIDEO" == 1 ]]; then
    video_arg=(--no-video)
  fi

  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  exec "$PYTHON_BIN" scripts/plan/eval_tdmpc2_ogbench_cube.py \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --label "${labels[$TASK_INDEX]}" \
    --episodes "$EPISODES" \
    --num-envs "$NUM_ENVS" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --seed "$SEED" \
    --reward-task-id "$REWARD_TASK_ID" \
    "${video_arg[@]}" \
    2>&1 | tee "$EVAL_ROOT/$RUN_LABEL/${labels[$TASK_INDEX]}.log"
}

launch() {
  mkdir -p "$EVAL_ROOT/$RUN_LABEL"
  for i in "${!names[@]}"; do
    checkpoint_path "$i" >/dev/null
    local used
    used=$(gpu_memory_mib "${gpus[$i]}")
    if (( used >= GPU_MEMORY_LIMIT_MIB )); then
      echo "GPU ${gpus[$i]} busy: ${used} MiB used" >&2
      exit 3
    fi
    local output_dir="$EVAL_ROOT/$RUN_LABEL/${labels[$i]}"
    if [[ -e "$output_dir" ]]; then
      echo "Refusing to reuse evaluation directory: $output_dir" >&2
      exit 4
    fi
  done

  for i in "${!names[@]}"; do
    local session="eval_tdmpc2_${labels[$i]}_${RUN_LABEL}"
    tmux new-session -d -s "${session:0:90}" \
      "cd '$STABLEWM_ROOT' && MODE=run-one TASK_INDEX='$i' GPU_ID='${gpus[$i]}' PYTHON_BIN='$PYTHON_BIN' RUN_ROOT='$RUN_ROOT' EVAL_ROOT='$EVAL_ROOT' RUN_LABEL='$RUN_LABEL' EPISODES='$EPISODES' NUM_ENVS='$NUM_ENVS' MAX_EPISODE_STEPS='$MAX_EPISODE_STEPS' SEED='$SEED' REWARD_TASK_ID='$REWARD_TASK_ID' NO_VIDEO='$NO_VIDEO' bash '$STABLEWM_ROOT/scripts/20260904_yb_eval_tdmpc2_cube.sh'"
    echo "Launched ${labels[$i]} on physical GPU ${gpus[$i]} in tmux ${session:0:90}"
  done
  echo "RUN_LABEL=$RUN_LABEL"
}

status() {
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits
  echo "Evaluation tmux sessions:"
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep '^eval_tdmpc2_' || true
  if [[ -d "$EVAL_ROOT" ]]; then
    find "$EVAL_ROOT" -name results.json -type f -print | sort | tail -n 10
  fi
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
