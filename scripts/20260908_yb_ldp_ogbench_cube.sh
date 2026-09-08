#!/usr/bin/env bash
set -euo pipefail

# Yingbo Cloud: goal-conditioned LDP on an official visual OGBench task.
# Every environment, validation, smoke, training, and evaluation action routes here.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MODE=${MODE:-status}
PYTHON_BIN=${PYTHON_BIN:-/root/data/yyf/ogbench-new/.venv/bin/python}
TEST_PYTHON_BIN=${TEST_PYTHON_BIN:-$STABLEWM_ROOT/.venv/bin/python}
OVERLAY_INSTALLER_BIN=${OVERLAY_INSTALLER_BIN:-/usr/bin/python3.10}
ENV_SPEC=${ENV_SPEC:-$STABLEWM_ROOT/scripts/config/ldp_ogbench_env_spec.json}
DATASET_ROOT=${DATASET_ROOT:-/root/data/yyf/stablewm-data/datasets/ogbench8-tdmpc2-pixels-gc-h50}
DATASET_ID=${DATASET_ID:-visual-cube-single-play-v0}
SOURCE_DATASET=${SOURCE_DATASET:-$DATASET_ROOT/$DATASET_ID.h5}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/data/yyf/ldp-ogbench}
LDP_RUNTIME_ROOT=${LDP_RUNTIME_ROOT:-$ARTIFACT_ROOT/runtime-v2}
LDP_OVERLAY=${LDP_OVERLAY:-$LDP_RUNTIME_ROOT/site-packages}
WHEELHOUSE=${WHEELHOUSE:-/root/data/yyf/ldp-wheelhouse}
RUN_LABEL=${RUN_LABEL:-gc_finalgoal_h8_a4_ds100}
OGBENCH_ROOT=${OGBENCH_ROOT:-/root/data/yyf/ogbench-official-1d414099}
OGBENCH_REPOSITORY=${OGBENCH_REPOSITORY:-https://github.com/seohongpark/ogbench.git}
OGBENCH_COMMIT=${OGBENCH_COMMIT:-1d4140997f60c52c6fb0702ec100dc988b18c548}
OGBENCH_SITE_PACKAGES=${OGBENCH_SITE_PACKAGES:-$OGBENCH_ROOT/.venv/lib/python3.10/site-packages}
EGL_RUNTIME_ROOT=${EGL_RUNTIME_ROOT:-$ARTIFACT_ROOT/.runtime/egl}
UPSTREAM_COMMIT=${UPSTREAM_COMMIT:-a26cbf1d2c0aec7adc5d9746f47831b162a41c0c}
GPU_ID=${GPU_ID:-6}
MIN_FREE_MEMORY_MIB=${MIN_FREE_MEMORY_MIB:-17000}
SEED=${SEED:-1}
VAE_STEPS=${VAE_STEPS:-100000}
VAE_BATCH_SIZE=${VAE_BATCH_SIZE:-128}
VAE_LOG_EVERY=${VAE_LOG_EVERY:-100}
VAE_SAVE_EVERY=${VAE_SAVE_EVERY:-10000}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-512}
VAE_VALIDATION_SAMPLES=${VAE_VALIDATION_SAMPLES:-128}
MAX_VAE_VALIDATION_MSE=${MAX_VAE_VALIDATION_MSE:-0.1}
LDP_STEPS=${LDP_STEPS:-100000}
LDP_BATCH_SIZE=${LDP_BATCH_SIZE:-128}
LDP_LOG_EVERY=${LDP_LOG_EVERY:-100}
LDP_SAVE_EVERY=${LDP_SAVE_EVERY:-10000}
LDP_VALIDATION_BATCHES=${LDP_VALIDATION_BATCHES:-4}
PRED_HORIZON=${PRED_HORIZON:-8}
ACTION_HORIZON=${ACTION_HORIZON:-4}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-100}
EPISODES=${EPISODES:-10}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_TASK_IDS=${EVAL_TASK_IDS:-1 2 3 4 5}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
DEFAULT_ENV_ID=${DATASET_ID/-play/}
DEFAULT_ENV_ID=${DEFAULT_ENV_ID/-noisy/}
ENV_ID=${ENV_ID:-$DEFAULT_ENV_ID}
DEFAULT_TASK_TAG=${DATASET_ID#visual-}
DEFAULT_TASK_TAG=${DEFAULT_TASK_TAG%-v0}
DEFAULT_TASK_TAG=${DEFAULT_TASK_TAG//-/_}
TASK_TAG=${TASK_TAG:-$DEFAULT_TASK_TAG}
RESUME=${RESUME:-0}

RUN_NAME="ldp_${TASK_TAG}_${RUN_LABEL}_s${SEED}"
VAE_DIR=${VAE_DIR:-$ARTIFACT_ROOT/runs/${RUN_NAME}_vae}
LATENT_FILE=${LATENT_FILE:-$ARTIFACT_ROOT/data/${RUN_NAME}_latents.h5}
LDP_DIR=${LDP_DIR:-$ARTIFACT_ROOT/runs/${RUN_NAME}_ldp}
EVAL_DIR=${EVAL_DIR:-$ARTIFACT_ROOT/evals/${RUN_NAME}_eval${EPISODES}_s${EVAL_SEED}}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$LDP_OVERLAY:$STABLEWM_ROOT:$STABLEWM_ROOT/third_party/latent_diffusion_planning:$OGBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OGBENCH_ROOT OGBENCH_SITE_PACKAGES
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-8}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.18}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

check_upstream() {
  local source="$STABLEWM_ROOT/third_party/latent_diffusion_planning"
  if [[ ! -d "$source" ]]; then
    echo "LDP submodule is missing: $source" >&2
    exit 2
  fi
  local actual
  actual=$(git -C "$source" rev-parse HEAD)
  if [[ "$actual" != "$UPSTREAM_COMMIT" ]]; then
    echo "LDP commit mismatch: expected $UPSTREAM_COMMIT, found $actual" >&2
    exit 2
  fi
  if [[ -n $(git -C "$source" status --porcelain) ]]; then
    echo "LDP submodule has local modifications" >&2
    git -C "$source" status --short >&2
    exit 2
  fi
}

setup_ogbench() {
  local temporary="${OGBENCH_ROOT}.building"
  if [[ ! -d "$OGBENCH_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$OGBENCH_ROOT")"
    if [[ -e "$temporary" ]]; then
      echo "Incomplete OGBench checkout already exists: $temporary" >&2
      exit 4
    fi
    git clone --filter=blob:none "$OGBENCH_REPOSITORY" "$temporary"
    git -C "$temporary" checkout --detach "$OGBENCH_COMMIT"
    mv "$temporary" "$OGBENCH_ROOT"
  fi
  local actual origin
  actual=$(git -C "$OGBENCH_ROOT" rev-parse HEAD)
  origin=$(git -C "$OGBENCH_ROOT" remote get-url origin)
  if [[ "$actual" != "$OGBENCH_COMMIT" ]]; then
    echo "OGBench commit mismatch: expected $OGBENCH_COMMIT, found $actual" >&2
    exit 2
  fi
  if [[ "${origin%.git}" != "${OGBENCH_REPOSITORY%.git}" ]]; then
    echo "OGBench origin mismatch: expected $OGBENCH_REPOSITORY, found $origin" >&2
    exit 2
  fi
  if [[ -n $(git -C "$OGBENCH_ROOT" status --porcelain --untracked-files=no) ]]; then
    echo "Official OGBench checkout has tracked modifications" >&2
    exit 2
  fi
  echo "OGBENCH_READY commit=$actual origin=$origin root=$OGBENCH_ROOT"
}

require_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Read-only OGBench JAX environment not found: $PYTHON_BIN" >&2
    exit 2
  fi
  local marker="$LDP_RUNTIME_ROOT/.ldp_env_spec_sha256"
  if [[ ! -f "$marker" ]]; then
    echo "LDP dependency overlay is not initialized. Run MODE=setup-env first." >&2
    exit 2
  fi
  local expected actual
  expected=$(python3 -c 'import hashlib,json,sys; print(hashlib.sha256(json.dumps(json.load(open(sys.argv[1])),sort_keys=True,separators=(",",":")).encode()).hexdigest())' "$ENV_SPEC")
  actual=$(<"$marker")
  if [[ "$actual" != "$expected" ]]; then
    echo "LDP dependency overlay spec mismatch: expected $expected, found $actual" >&2
    exit 4
  fi
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

setup_env() {
  check_upstream
  setup_ogbench
  if [[ ! -f "$ENV_SPEC" ]]; then
    echo "Missing environment spec: $ENV_SPEC" >&2
    exit 2
  fi
  local spec_hash marker temporary
  spec_hash=$(python3 -c 'import hashlib,json,sys; print(hashlib.sha256(json.dumps(json.load(open(sys.argv[1])),sort_keys=True,separators=(",",":")).encode()).hexdigest())' "$ENV_SPEC")
  marker="$LDP_RUNTIME_ROOT/.ldp_env_spec_sha256"
  if [[ -f "$marker" && $(<"$marker") == "$spec_hash" ]]; then
    echo "ENV_REUSE spec_sha256=$spec_hash python=$PYTHON_BIN overlay=$LDP_OVERLAY"
    return
  fi
  if [[ -f "$marker" ]]; then
    echo "Existing LDP overlay does not match the tracked spec; refusing to overwrite it." >&2
    exit 4
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Read-only OGBench JAX environment is missing: $PYTHON_BIN" >&2
    exit 2
  fi
  if [[ ! -x "$OVERLAY_INSTALLER_BIN" ]]; then
    echo "Overlay installer Python is missing: $OVERLAY_INSTALLER_BIN" >&2
    exit 2
  fi
  if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "Offline LDP wheelhouse is missing: $WHEELHOUSE" >&2
    exit 2
  fi
  if [[ -d "$LDP_RUNTIME_ROOT" ]]; then
    echo "Existing unmarked LDP runtime found; refusing to overwrite it: $LDP_RUNTIME_ROOT" >&2
    exit 4
  fi
  temporary="${LDP_RUNTIME_ROOT}.building"
  if [[ -d "$temporary" ]]; then
    echo "Removing incomplete overlay build: $temporary"
    rm -rf "$temporary"
  fi
  mkdir -p "$temporary/site-packages"
  "$OVERLAY_INSTALLER_BIN" -m pip install --no-index --no-deps \
    --find-links "$WHEELHOUSE" --target "$temporary/site-packages" \
    'diffusers==0.27.2' 'huggingface-hub==0.23.1' \
    'filelock==3.19.1' 'importlib-metadata==8.7.0' \
    'regex==2025.7.34' 'safetensors==0.6.2' 'zipp==3.23.0'
  printf '%s\n' "$spec_hash" > "$temporary/.ldp_env_spec_sha256"
  mv "$temporary" "$LDP_RUNTIME_ROOT"
  if [[ -d "$STABLEWM_ROOT/.venv-ldp" && -f "$STABLEWM_ROOT/.venv-ldp/pyvenv.cfg" ]]; then
    echo "Removing incomplete environment from the earlier failed bootstrap: $STABLEWM_ROOT/.venv-ldp"
    rm -rf "$STABLEWM_ROOT/.venv-ldp"
  fi
  echo "ENV_CREATED spec_sha256=$spec_hash python=$PYTHON_BIN overlay=$LDP_OVERLAY"
}

env_witness() {
  require_python
  check_upstream
  preflight_gpu
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  "$PYTHON_BIN" -c 'import diffusers,flax,h5py,jax,jax.numpy as jnp,optax,orbax.checkpoint; from diffusers import FlaxAutoencoderKL; k=jax.random.PRNGKey(0); x=jax.random.normal(k,(8,8)); y=(x@x).block_until_ready(); print("LDP_JAX_WITNESS", y.shape, jax.devices()[0].device_kind, float(y.sum())); print("VERSIONS",jax.__version__,flax.__version__,optax.__version__,diffusers.__version__,h5py.__version__)'
}

unit_test() {
  require_python
  check_upstream
  if [[ ! -x "$TEST_PYTHON_BIN" ]]; then
    echo "Test environment not found: $TEST_PYTHON_BIN" >&2
    exit 2
  fi
  cd "$STABLEWM_ROOT"
  "$TEST_PYTHON_BIN" -m pytest -q tests/test_ldp_ogbench_data.py
  "$TEST_PYTHON_BIN" -m py_compile \
    scripts/data/ldp_ogbench_data.py scripts/train/ldp_ogbench.py
}

audit_data() {
  require_python
  check_upstream
  if [[ ! -f "$SOURCE_DATASET" ]]; then
    echo "Missing source dataset: $SOURCE_DATASET" >&2
    exit 2
  fi
  cd "$STABLEWM_ROOT"
  "$PYTHON_BIN" scripts/data/convert_ogbench_npz_tdmpc2.py \
    validate --segment-transitions 50 --verify-source "$SOURCE_DATASET"
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py audit --source "$SOURCE_DATASET"
}

audit_loader() {
  require_python
  check_upstream
  if [[ ! -f "$SOURCE_DATASET" ]]; then
    echo "Missing source dataset: $SOURCE_DATASET" >&2
    exit 2
  fi
  cd "$STABLEWM_ROOT"
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py audit --source "$SOURCE_DATASET"
}

train_vae() {
  require_python
  check_upstream
  preflight_gpu
  mkdir -p "$(dirname "$VAE_DIR")"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  local resume_args=()
  if [[ "$RESUME" == 1 ]]; then
    resume_args=(--resume)
  fi
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py train-vae \
    --source "$SOURCE_DATASET" --output-dir "$VAE_DIR" \
    --steps "$VAE_STEPS" --batch-size "$VAE_BATCH_SIZE" --seed "$SEED" \
    --log-every "$VAE_LOG_EVERY" --save-every "$VAE_SAVE_EVERY" \
    "${resume_args[@]}"
}

encode_data() {
  require_python
  check_upstream
  preflight_gpu
  mkdir -p "$(dirname "$LATENT_FILE")"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  local max_args=()
  if [[ -n ${MAX_ENCODE_EPISODES:-} ]]; then
    max_args=(--max-episodes "$MAX_ENCODE_EPISODES")
  fi
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py encode \
    --source "$SOURCE_DATASET" --vae-dir "$VAE_DIR" \
    --output "$LATENT_FILE" --batch-size "$ENCODE_BATCH_SIZE" \
    --validation-samples "$VAE_VALIDATION_SAMPLES" \
    --max-validation-mse "$MAX_VAE_VALIDATION_MSE" \
    "${max_args[@]}"
}

train_ldp() {
  require_python
  check_upstream
  preflight_gpu
  mkdir -p "$(dirname "$LDP_DIR")"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  local resume_args=()
  if [[ "$RESUME" == 1 ]]; then
    resume_args=(--resume)
  fi
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py train-ldp \
    --source "$SOURCE_DATASET" --latents "$LATENT_FILE" \
    --output-dir "$LDP_DIR" --steps "$LDP_STEPS" \
    --batch-size "$LDP_BATCH_SIZE" --pred-horizon "$PRED_HORIZON" \
    --action-horizon "$ACTION_HORIZON" --diffusion-steps "$DIFFUSION_STEPS" \
    --seed "$SEED" --log-every "$LDP_LOG_EVERY" --save-every "$LDP_SAVE_EVERY" \
    --validation-batches "$LDP_VALIDATION_BATCHES" "${resume_args[@]}"
}

run_eval() {
  require_python
  check_upstream
  preflight_gpu
  prepare_egl_runtime
  mkdir -p "$(dirname "$EVAL_DIR")"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  local horizon_args=()
  if [[ -n "$MAX_EPISODE_STEPS" ]]; then
    horizon_args=(--max-episode-steps "$MAX_EPISODE_STEPS")
  fi
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py eval \
    --run-dir "$LDP_DIR" --vae-dir "$VAE_DIR" --output-dir "$EVAL_DIR" \
    --dataset-id "$DATASET_ID" --env-id "$ENV_ID" \
    --episodes "$EPISODES" --seed "$EVAL_SEED" \
    --task-ids $EVAL_TASK_IDS "${horizon_args[@]}"
}

audit_environment() {
  require_python
  check_upstream
  prepare_egl_runtime
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  local horizon_args=()
  if [[ -n "$MAX_EPISODE_STEPS" ]]; then
    horizon_args=(--max-episode-steps "$MAX_EPISODE_STEPS")
  fi
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py audit-environment \
    --dataset-id "$DATASET_ID" --env-id "$ENV_ID" --seed "$EVAL_SEED" \
    "${horizon_args[@]}"
}

pipeline() {
  audit_data
  unit_test
  train_vae
  encode_data
  train_ldp
  run_eval
}

smoke() {
  audit_data
  unit_test
  env_witness
  local stamp smoke_root
  stamp=$(date +%Y%m%dT%H%M%S)
  smoke_root="$ARTIFACT_ROOT/smoke/$stamp"
  VAE_DIR="$smoke_root/vae" LATENT_FILE="$smoke_root/latents.h5" \
    LDP_DIR="$smoke_root/ldp" EVAL_DIR="$smoke_root/eval" \
    VAE_STEPS=2 VAE_BATCH_SIZE=2 VAE_LOG_EVERY=1 VAE_SAVE_EVERY=2 \
    ENCODE_BATCH_SIZE=64 MAX_ENCODE_EPISODES=40 MAX_VAE_VALIDATION_MSE=1.0 \
    LDP_STEPS=2 LDP_BATCH_SIZE=2 LDP_LOG_EVERY=1 LDP_SAVE_EVERY=2 \
    EPISODES=1 EVAL_TASK_IDS='1 2 3 4 5' MAX_EPISODE_STEPS=1 pipeline_smoke
  echo "SMOKE_COMPLETE=$smoke_root"
}

pipeline_smoke() {
  train_vae
  encode_data
  train_ldp
  run_eval
}

launch() {
  require_python
  check_upstream
  audit_data
  unit_test
  preflight_gpu
  local session="ldp_${RUN_NAME:0:70}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing duplicate tmux session: $session" >&2
    exit 4
  fi
  if [[ -e "$VAE_DIR" || -e "$LATENT_FILE" || -e "$LDP_DIR" || -e "$EVAL_DIR" ]]; then
    echo "Refusing to reuse an existing formal artifact path" >&2
    exit 4
  fi
  tmux new-session -d -s "$session" \
    "cd '$STABLEWM_ROOT' && MODE=pipeline GPU_ID='$GPU_ID' MIN_FREE_MEMORY_MIB='$MIN_FREE_MEMORY_MIB' DATASET_ROOT='$DATASET_ROOT' DATASET_ID='$DATASET_ID' SOURCE_DATASET='$SOURCE_DATASET' ARTIFACT_ROOT='$ARTIFACT_ROOT' TASK_TAG='$TASK_TAG' RUN_LABEL='$RUN_LABEL' SEED='$SEED' VAE_STEPS='$VAE_STEPS' VAE_BATCH_SIZE='$VAE_BATCH_SIZE' VAE_LOG_EVERY='$VAE_LOG_EVERY' VAE_SAVE_EVERY='$VAE_SAVE_EVERY' ENCODE_BATCH_SIZE='$ENCODE_BATCH_SIZE' VAE_VALIDATION_SAMPLES='$VAE_VALIDATION_SAMPLES' MAX_VAE_VALIDATION_MSE='$MAX_VAE_VALIDATION_MSE' LDP_STEPS='$LDP_STEPS' LDP_BATCH_SIZE='$LDP_BATCH_SIZE' LDP_LOG_EVERY='$LDP_LOG_EVERY' LDP_SAVE_EVERY='$LDP_SAVE_EVERY' LDP_VALIDATION_BATCHES='$LDP_VALIDATION_BATCHES' PRED_HORIZON='$PRED_HORIZON' ACTION_HORIZON='$ACTION_HORIZON' DIFFUSION_STEPS='$DIFFUSION_STEPS' ENV_ID='$ENV_ID' EVAL_TASK_IDS='$EVAL_TASK_IDS' EPISODES='$EPISODES' EVAL_SEED='$EVAL_SEED' MAX_EPISODE_STEPS='$MAX_EPISODE_STEPS' RESUME='$RESUME' bash '$STABLEWM_ROOT/scripts/20260908_yb_ldp_ogbench_cube.sh' 2>&1 | tee '$ARTIFACT_ROOT/${RUN_NAME}.log'"
  echo "LAUNCHED session=$session gpu=$GPU_ID log=$ARTIFACT_ROOT/${RUN_NAME}.log"
}

status() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total \
    --format=csv,noheader,nounits
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^ldp_' || true
  find "$ARTIFACT_ROOT" -maxdepth 5 \
    \( -name checkpoint.msgpack -o -name results.json -o -name metrics.jsonl \) \
    -type f -print 2>/dev/null | sort | tail -n 30 || true
  if [[ -f "$ARTIFACT_ROOT/${RUN_NAME}.log" ]]; then
    tail -n 30 "$ARTIFACT_ROOT/${RUN_NAME}.log"
  fi
}

case "$MODE" in
  setup-ogbench) setup_ogbench ;;
  setup-env) setup_env ;;
  env-witness) env_witness ;;
  test) unit_test ;;
  audit-data|data-test) audit_data ;;
  audit-loader) audit_loader ;;
  train-vae) train_vae ;;
  encode) encode_data ;;
  train-ldp) train_ldp ;;
  audit-environment) audit_environment ;;
  eval) run_eval ;;
  pipeline) pipeline ;;
  smoke) smoke ;;
  launch) launch ;;
  status) status ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
