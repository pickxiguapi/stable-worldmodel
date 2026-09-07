#!/usr/bin/env bash
set -euo pipefail

# Yingbo Cloud: G3 GCIQL-guided TD-MPC2 on cube-single-play.
# All validation, smoke, training, evaluation, and status operations route here.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MODE=${MODE:-status}
PYTHON_BIN=${PYTHON_BIN:-$STABLEWM_ROOT/.venv/bin/python}
DATASET=${DATASET:-/root/data/yyf/stablewm-data/datasets/ogbench8-tdmpc2-pixels-gc-h50/visual-cube-single-play-v0.h5}
RUN_ROOT=${RUN_ROOT:-/root/data/yyf/tdmpc2-gciql-g3-runs}
EVAL_ROOT=${EVAL_ROOT:-/root/data/yyf/tdmpc2-gciql-g3-eval}
PYTHON_DEV_ROOT=${PYTHON_DEV_ROOT:-$RUN_ROOT/.runtime/python-dev}
OGBENCH_ROOT=${OGBENCH_ROOT:-/root/data/yyf/ogbench-eval-main-20260830}
OGBENCH_SITE_PACKAGES=${OGBENCH_SITE_PACKAGES:-$OGBENCH_ROOT/.venv/lib/python3.10/site-packages}
EGL_RUNTIME_ROOT=${EGL_RUNTIME_ROOT:-$EVAL_ROOT/.runtime/egl}
RUN_LABEL=${RUN_LABEL:-g3_iqlv_awr3_goal25_policycenter}
OFFICIAL_COMMIT=${OFFICIAL_COMMIT:-e9f59321933cbc8e11a002b842adc7d4ffae8ff1}
TASK=${TASK:-visual-cube-single-play-v0}
SEGMENT_TRANSITIONS=${SEGMENT_TRANSITIONS:-50}
MAX_STEPS=${MAX_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-256}
HORIZON=${HORIZON:-3}
MODEL_SIZE=${MODEL_SIZE:-5}
EXPECTILE=${EXPECTILE:-0.9}
AWR_BETA=${AWR_BETA:-3.0}
AWR_CLIP=${AWR_CLIP:-100.0}
VALUE_TAU=${VALUE_TAU:-0.005}
GOAL_SEQUENCE_FRACTION=${GOAL_SEQUENCE_FRACTION:-0.25}
SEED=${SEED:-1}
LOG_INTERVAL=${LOG_INTERVAL:-100}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-10000}
GPU_ID=${GPU_ID:-6}
MIN_FREE_MEMORY_MIB=${MIN_FREE_MEMORY_MIB:-1800}
COMPILE=${COMPILE:-1}
EPISODES=${EPISODES:-10}
EVAL_SEED=${EVAL_SEED:-42}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
VISUALIZE_INFO=${VISUALIZE_INFO:-0}

export PYTHONPATH="$STABLEWM_ROOT:$OGBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OGBENCH_SITE_PACKAGES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

run_name() {
  printf 'tdmpc2_g3_cs_play_gc_h%s_m%s_s%s_%s\n' \
    "$SEGMENT_TRANSITIONS" "$MAX_STEPS" "$SEED" "$RUN_LABEL"
}

check_official_source() {
  local source="$STABLEWM_ROOT/third_party/tdmpc2"
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

prepare_python_headers() {
  if [[ -f /usr/include/python3.10/Python.h ]]; then
    return
  fi
  local include_root="$PYTHON_DEV_ROOT/usr/include"
  local include_dir="$include_root/python3.10"
  local platform_dir="$PYTHON_DEV_ROOT/usr/include/x86_64-linux-gnu/python3.10"
  if [[ ! -f "$include_dir/Python.h" || ! -f "$platform_dir/pyconfig.h" ]]; then
    local deb_dir="$PYTHON_DEV_ROOT/debs"
    mkdir -p "$deb_dir" "$PYTHON_DEV_ROOT"
    (
      cd "$deb_dir"
      apt-get download -qq libpython3.10-dev python3.10-dev
      for deb in ./*.deb; do
        dpkg-deb -x "$deb" "$PYTHON_DEV_ROOT"
      done
    )
  fi
  export CPATH="$include_dir:$include_root${CPATH:+:$CPATH}"
}

prepare_egl_runtime() {
  if ldconfig -p 2>/dev/null | grep -q 'libEGL\.so\.1'; then
    return
  fi
  local lib_dir="$EGL_RUNTIME_ROOT/root/usr/lib/x86_64-linux-gnu"
  if [[ ! -f "$lib_dir/libEGL.so.1" ]]; then
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

preflight_gpu() {
  local free
  free=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free \
    --format=csv,noheader,nounits | tr -d '[:space:]')
  if (( free < MIN_FREE_MEMORY_MIB )); then
    echo "GPU $GPU_ID has only ${free} MiB free; need ${MIN_FREE_MEMORY_MIB} MiB" >&2
    exit 3
  fi
  echo "GPU_PREFLIGHT gpu=$GPU_ID free_mib=$free required_mib=$MIN_FREE_MEMORY_MIB"
}

audit_data() {
  check_official_source
  if [[ ! -f "$DATASET" ]]; then
    echo "Missing dataset: $DATASET" >&2
    exit 2
  fi
  "$PYTHON_BIN" "$STABLEWM_ROOT/scripts/data/convert_ogbench_npz_tdmpc2.py" \
    validate --segment-transitions "$SEGMENT_TRANSITIONS" \
    --verify-source "$DATASET"
}

unit_test() {
  check_official_source
  cd "$STABLEWM_ROOT"
  "$PYTHON_BIN" -m pytest -q \
    tests/test_tdmpc2_official_gc.py tests/test_tdmpc2_gciql.py
}

run_one() {
  check_official_source
  prepare_python_headers
  preflight_gpu
  local name
  name=$(run_name)
  local run_dir="$RUN_ROOT/$name"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse run directory: $run_dir" >&2
    exit 4
  fi
  mkdir -p "$RUN_ROOT"
  local compile_arg=--compile
  if [[ "$COMPILE" == 0 ]]; then
    compile_arg=--no-compile
  fi
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  exec "$PYTHON_BIN" scripts/train/tdmpc2_gciql.py \
    --dataset "$DATASET" --output-dir "$run_dir" --task "$TASK" \
    --steps "$MAX_STEPS" --batch-size "$BATCH_SIZE" --horizon "$HORIZON" \
    --segment-transitions "$SEGMENT_TRANSITIONS" --model-size "$MODEL_SIZE" \
    --expectile "$EXPECTILE" --awr-beta "$AWR_BETA" --awr-clip "$AWR_CLIP" \
    --value-tau "$VALUE_TAU" \
    --goal-sequence-fraction "$GOAL_SEQUENCE_FRACTION" \
    --seed "$SEED" --log-interval "$LOG_INTERVAL" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" --episodic "$compile_arg" \
    2>&1 | tee "$RUN_ROOT/$name.log"
}

smoke() {
  audit_data
  unit_test
  local stamp
  stamp=$(date +%Y%m%dT%H%M%S)
  MODE=run-one RUN_LABEL="smoke_${stamp}" MAX_STEPS=2 BATCH_SIZE=8 \
    LOG_INTERVAL=1 CHECKPOINT_INTERVAL=2 COMPILE=0 \
    bash "$STABLEWM_ROOT/scripts/20260907_yb_tdmpc2_gciql_g3_single.sh"
}

launch_train() {
  audit_data
  unit_test
  preflight_gpu
  local name
  name=$(run_name)
  local session="${name:0:90}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing duplicate tmux session: $session" >&2
    exit 4
  fi
  tmux new-session -d -s "$session" \
    "cd '$STABLEWM_ROOT' && MODE=run-one PYTHON_BIN='$PYTHON_BIN' DATASET='$DATASET' RUN_ROOT='$RUN_ROOT' PYTHON_DEV_ROOT='$PYTHON_DEV_ROOT' RUN_LABEL='$RUN_LABEL' OFFICIAL_COMMIT='$OFFICIAL_COMMIT' TASK='$TASK' SEGMENT_TRANSITIONS='$SEGMENT_TRANSITIONS' MAX_STEPS='$MAX_STEPS' BATCH_SIZE='$BATCH_SIZE' HORIZON='$HORIZON' MODEL_SIZE='$MODEL_SIZE' EXPECTILE='$EXPECTILE' AWR_BETA='$AWR_BETA' AWR_CLIP='$AWR_CLIP' VALUE_TAU='$VALUE_TAU' GOAL_SEQUENCE_FRACTION='$GOAL_SEQUENCE_FRACTION' SEED='$SEED' LOG_INTERVAL='$LOG_INTERVAL' CHECKPOINT_INTERVAL='$CHECKPOINT_INTERVAL' GPU_ID='$GPU_ID' MIN_FREE_MEMORY_MIB='$MIN_FREE_MEMORY_MIB' COMPILE='$COMPILE' bash '$STABLEWM_ROOT/scripts/20260907_yb_tdmpc2_gciql_g3_single.sh'"
  echo "Launched $name on physical GPU $GPU_ID in tmux $session"
}

run_eval() {
  check_official_source
  prepare_egl_runtime
  preflight_gpu
  local name
  name=$(run_name)
  local run_dir="$RUN_ROOT/$name"
  local checkpoint="$run_dir/weights_step_${MAX_STEPS}.pt"
  local output_dir="$EVAL_ROOT/${name}_eval${EPISODES}_s${EVAL_SEED}"
  if [[ ! -f "$checkpoint" || ! -f "$run_dir/official_config.json" ]]; then
    echo "Missing final checkpoint/config under $run_dir" >&2
    exit 2
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to reuse evaluation directory: $output_dir" >&2
    exit 4
  fi
  local visualize_arg=--visualize-info
  if [[ "$VISUALIZE_INFO" == 0 ]]; then
    visualize_arg=--no-visualize-info
  fi
  mkdir -p "$EVAL_ROOT"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  exec "$PYTHON_BIN" scripts/plan/eval_tdmpc2_gciql_cube.py \
    --checkpoint "$checkpoint" --config "$run_dir/official_config.json" \
    --output-dir "$output_dir" --label cube_single_play \
    --episodes "$EPISODES" --seed "$EVAL_SEED" \
    --max-episode-steps "$MAX_EPISODE_STEPS" "$visualize_arg" \
    2>&1 | tee "$EVAL_ROOT/${name}_eval${EPISODES}_s${EVAL_SEED}.log"
}

status() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total \
    --format=csv,noheader,nounits
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep '^tdmpc2_g3_' || true
  find "$RUN_ROOT" -name 'weights_step_*.pt' -type f -print 2>/dev/null \
    | sort | tail -n 10 || true
  find "$EVAL_ROOT" -name results.json -type f -print 2>/dev/null \
    | sort | tail -n 10 || true
}

case "$MODE" in
  test) unit_test ;;
  audit-data) audit_data ;;
  smoke) smoke ;;
  run-one) run_one ;;
  launch-train) launch_train ;;
  eval) run_eval ;;
  status) status ;;
  *)
    echo "MODE must be test, audit-data, smoke, run-one, launch-train, eval, or status" >&2
    exit 2
    ;;
esac
