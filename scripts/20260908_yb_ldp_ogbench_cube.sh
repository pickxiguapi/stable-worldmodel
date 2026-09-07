#!/usr/bin/env bash
set -euo pipefail

# Yingbo Cloud: goal-conditioned LDP on official OGBench cube-single.
# Every environment, validation, smoke, training, and evaluation action routes here.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MODE=${MODE:-status}
PYTHON_BIN=${PYTHON_BIN:-$STABLEWM_ROOT/.venv-ldp/bin/python}
ENV_SPEC=${ENV_SPEC:-$STABLEWM_ROOT/scripts/config/ldp_ogbench_env_spec.json}
SOURCE_DATASET=${SOURCE_DATASET:-/root/data/yyf/stablewm-data/datasets/ogbench8-tdmpc2-pixels-gc-h50/visual-cube-single-play-v0.h5}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/data/yyf/ldp-ogbench}
RUN_LABEL=${RUN_LABEL:-gc_finalgoal_h8_a4_ds100}
OGBENCH_ROOT=${OGBENCH_ROOT:-/root/data/yyf/ogbench-eval-main-20260830}
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
LDP_STEPS=${LDP_STEPS:-100000}
LDP_BATCH_SIZE=${LDP_BATCH_SIZE:-128}
LDP_LOG_EVERY=${LDP_LOG_EVERY:-100}
LDP_SAVE_EVERY=${LDP_SAVE_EVERY:-10000}
PRED_HORIZON=${PRED_HORIZON:-8}
ACTION_HORIZON=${ACTION_HORIZON:-4}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-100}
EPISODES=${EPISODES:-10}
EVAL_SEED=${EVAL_SEED:-42}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
ENV_ID=${ENV_ID:-visual-cube-single-v0}

RUN_NAME="ldp_cube_single_${RUN_LABEL}_s${SEED}"
VAE_DIR=${VAE_DIR:-$ARTIFACT_ROOT/runs/${RUN_NAME}_vae}
LATENT_FILE=${LATENT_FILE:-$ARTIFACT_ROOT/data/${RUN_NAME}_latents.h5}
LDP_DIR=${LDP_DIR:-$ARTIFACT_ROOT/runs/${RUN_NAME}_ldp}
EVAL_DIR=${EVAL_DIR:-$ARTIFACT_ROOT/evals/${RUN_NAME}_eval${EPISODES}_s${EVAL_SEED}}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$STABLEWM_ROOT:$STABLEWM_ROOT/third_party/latent_diffusion_planning:$OGBENCH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OGBENCH_SITE_PACKAGES
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

require_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    echo "Run MODE=setup-env first." >&2
    exit 2
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
  if [[ ! -f "$ENV_SPEC" ]]; then
    echo "Missing environment spec: $ENV_SPEC" >&2
    exit 2
  fi
  local spec_hash marker
  spec_hash=$(python3 -c 'import hashlib,json,sys; print(hashlib.sha256(json.dumps(json.load(open(sys.argv[1])),sort_keys=True,separators=(",",":")).encode()).hexdigest())' "$ENV_SPEC")
  marker="$STABLEWM_ROOT/.venv-ldp/.ldp_env_spec_sha256"
  if [[ -x "$PYTHON_BIN" ]]; then
    if [[ -f "$marker" && $(<"$marker") == "$spec_hash" ]]; then
      echo "ENV_REUSE spec_sha256=$spec_hash python=$PYTHON_BIN"
      return
    fi
    if [[ -f "$marker" ]]; then
      echo "Existing .venv-ldp does not match the tracked spec; refusing to overwrite it." >&2
      exit 4
    fi
  fi
  if [[ -d "$STABLEWM_ROOT/.venv-ldp" ]]; then
    if [[ ! -f "$STABLEWM_ROOT/.venv-ldp/pyvenv.cfg" ]]; then
      echo "Refusing to remove an unrecognized partial environment" >&2
      exit 4
    fi
    echo "Removing incomplete environment from a failed bootstrap: $STABLEWM_ROOT/.venv-ldp"
    rm -rf "$STABLEWM_ROOT/.venv-ldp"
  fi
  if [[ ! -x "$STABLEWM_ROOT/.venv/bin/python" ]]; then
    echo "Bootstrap environment is missing: $STABLEWM_ROOT/.venv" >&2
    exit 2
  fi
  "$STABLEWM_ROOT/.venv/bin/python" -m virtualenv \
    --python "$(command -v python3.10)" "$STABLEWM_ROOT/.venv-ldp"
  "$PYTHON_BIN" -m pip install --upgrade 'pip==24.0'
  "$PYTHON_BIN" -m pip install \
    'numpy==1.26.4' 'scipy==1.13.1' 'h5py==3.11.0' 'pytest==8.3.5'
  "$PYTHON_BIN" -m pip install \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    'jax[cuda12_pip]==0.4.26'
  "$PYTHON_BIN" -m pip install \
    'jax==0.4.26' 'flax==0.8.4' 'optax==0.2.2' \
    'orbax-checkpoint==0.5.14' 'diffusers==0.27.2' \
    'huggingface-hub==0.23.1' 'hydra-core==1.2.0' 'omegaconf==2.3.0'
  printf '%s\n' "$spec_hash" > "$marker"
  "$PYTHON_BIN" -m pip freeze > "$STABLEWM_ROOT/.venv-ldp/requirements.freeze.txt"
  echo "ENV_CREATED spec_sha256=$spec_hash python=$PYTHON_BIN"
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
  cd "$STABLEWM_ROOT"
  "$PYTHON_BIN" -m pytest -q tests/test_ldp_ogbench_data.py
  "$PYTHON_BIN" -m py_compile \
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

train_vae() {
  require_python
  check_upstream
  preflight_gpu
  mkdir -p "$(dirname "$VAE_DIR")"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py train-vae \
    --source "$SOURCE_DATASET" --output-dir "$VAE_DIR" \
    --steps "$VAE_STEPS" --batch-size "$VAE_BATCH_SIZE" --seed "$SEED" \
    --log-every "$VAE_LOG_EVERY" --save-every "$VAE_SAVE_EVERY"
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
    "${max_args[@]}"
}

train_ldp() {
  require_python
  check_upstream
  preflight_gpu
  mkdir -p "$(dirname "$LDP_DIR")"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py train-ldp \
    --source "$SOURCE_DATASET" --latents "$LATENT_FILE" \
    --output-dir "$LDP_DIR" --steps "$LDP_STEPS" \
    --batch-size "$LDP_BATCH_SIZE" --pred-horizon "$PRED_HORIZON" \
    --action-horizon "$ACTION_HORIZON" --diffusion-steps "$DIFFUSION_STEPS" \
    --seed "$SEED" --log-every "$LDP_LOG_EVERY" --save-every "$LDP_SAVE_EVERY"
}

run_eval() {
  require_python
  check_upstream
  preflight_gpu
  prepare_egl_runtime
  mkdir -p "$(dirname "$EVAL_DIR")"
  cd "$STABLEWM_ROOT"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  "$PYTHON_BIN" scripts/train/ldp_ogbench.py eval \
    --run-dir "$LDP_DIR" --vae-dir "$VAE_DIR" --output-dir "$EVAL_DIR" \
    --env-id "$ENV_ID" --episodes "$EPISODES" --seed "$EVAL_SEED" \
    --max-episode-steps "$MAX_EPISODE_STEPS"
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
    ENCODE_BATCH_SIZE=64 MAX_ENCODE_EPISODES=2 \
    LDP_STEPS=2 LDP_BATCH_SIZE=2 LDP_LOG_EVERY=1 LDP_SAVE_EVERY=2 \
    EPISODES=1 MAX_EPISODE_STEPS=1 pipeline_smoke
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
    "cd '$STABLEWM_ROOT' && MODE=pipeline GPU_ID='$GPU_ID' SOURCE_DATASET='$SOURCE_DATASET' ARTIFACT_ROOT='$ARTIFACT_ROOT' RUN_LABEL='$RUN_LABEL' SEED='$SEED' VAE_STEPS='$VAE_STEPS' VAE_BATCH_SIZE='$VAE_BATCH_SIZE' LDP_STEPS='$LDP_STEPS' LDP_BATCH_SIZE='$LDP_BATCH_SIZE' PRED_HORIZON='$PRED_HORIZON' ACTION_HORIZON='$ACTION_HORIZON' DIFFUSION_STEPS='$DIFFUSION_STEPS' EPISODES='$EPISODES' EVAL_SEED='$EVAL_SEED' bash '$STABLEWM_ROOT/scripts/20260908_yb_ldp_ogbench_cube.sh' 2>&1 | tee '$ARTIFACT_ROOT/${RUN_NAME}.log'"
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
  setup-env) setup_env ;;
  env-witness) env_witness ;;
  test) unit_test ;;
  audit-data|data-test) audit_data ;;
  train-vae) train_vae ;;
  encode) encode_data ;;
  train-ldp) train_ldp ;;
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
