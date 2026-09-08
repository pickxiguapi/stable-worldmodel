#!/usr/bin/env bash
set -euo pipefail

# Eight-task Yingbo Cloud matrix for goal-conditioned LDP.
# The matrix is the four official visual manipulation environments crossed with
# the play/noisy offline datasets.  Every experiment action delegates to the
# tracked single-task bash so configuration and safety checks stay identical.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STABLEWM_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TASK_SCRIPT="$SCRIPT_DIR/20260908_yb_ldp_ogbench_cube.sh"

MODE=${MODE:-status}
TASK_INDEX=${TASK_INDEX:-}
DATASET_ROOT=${DATASET_ROOT:-/root/data/yyf/stablewm-data/datasets/ogbench8-tdmpc2-pixels-gc-h50}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/data/yyf/ldp-ogbench}
LDP_RUNTIME_ROOT=${LDP_RUNTIME_ROOT:-/root/data/yyf/ldp-ogbench/runtime}
LDP_OVERLAY=${LDP_OVERLAY:-$LDP_RUNTIME_ROOT/site-packages}
OGBENCH_ROOT=${OGBENCH_ROOT:-/root/data/yyf/ogbench-eval-main-20260830}
CORE_LABEL=${CORE_LABEL:-gc_finalgoal_h8_a4_ds100_v300k_p500k_b128}
SEED=${SEED:-1}
VAE_STEPS=${VAE_STEPS:-300000}
VAE_BATCH_SIZE=${VAE_BATCH_SIZE:-128}
VAE_LOG_EVERY=${VAE_LOG_EVERY:-100}
VAE_SAVE_EVERY=${VAE_SAVE_EVERY:-50000}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-512}
VAE_VALIDATION_SAMPLES=${VAE_VALIDATION_SAMPLES:-128}
MAX_VAE_VALIDATION_MSE=${MAX_VAE_VALIDATION_MSE:-0.1}
LDP_STEPS=${LDP_STEPS:-500000}
LDP_BATCH_SIZE=${LDP_BATCH_SIZE:-128}
LDP_LOG_EVERY=${LDP_LOG_EVERY:-100}
LDP_SAVE_EVERY=${LDP_SAVE_EVERY:-100000}
LDP_VALIDATION_BATCHES=${LDP_VALIDATION_BATCHES:-4}
PRED_HORIZON=${PRED_HORIZON:-8}
ACTION_HORIZON=${ACTION_HORIZON:-4}
DIFFUSION_STEPS=${DIFFUSION_STEPS:-100}
EPISODES=${EPISODES:-10}
EVAL_SEED=${EVAL_SEED:-42}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
REWARD_TASK_ID=${REWARD_TASK_ID:-2}
MIN_FREE_MEMORY_MIB=${MIN_FREE_MEMORY_MIB:-17000}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.18}

dataset_ids=(
  visual-cube-single-play-v0
  visual-cube-double-play-v0
  visual-cube-triple-play-v0
  visual-scene-play-v0
  visual-cube-single-noisy-v0
  visual-cube-double-noisy-v0
  visual-cube-triple-noisy-v0
  visual-scene-noisy-v0
)
env_ids=(
  visual-cube-single-v0
  visual-cube-double-v0
  visual-cube-triple-v0
  visual-scene-v0
  visual-cube-single-v0
  visual-cube-double-v0
  visual-cube-triple-v0
  visual-scene-v0
)
task_tags=(
  cube_single
  cube_double_play
  cube_triple_play
  scene_play
  cube_single_noisy
  cube_double_noisy
  cube_triple_noisy
  scene_noisy
)
# cube-single-play was already launched on GPU 6 before this matrix existed.
gpu_ids=(6 0 1 2 3 4 5 7)

validate_index() {
  local index=$1
  if [[ ! "$index" =~ ^[0-7]$ ]]; then
    echo "TASK_INDEX must be an integer in [0, 7], got: $index" >&2
    exit 2
  fi
}

run_name() {
  local index=$1
  printf 'ldp_%s_%s_s%s\n' "${task_tags[$index]}" "$CORE_LABEL" "$SEED"
}

run_base() {
  local index=$1
  local child_mode=$2
  validate_index "$index"
  DATASET_ROOT="$DATASET_ROOT" \
  DATASET_ID="${dataset_ids[$index]}" \
  SOURCE_DATASET="$DATASET_ROOT/${dataset_ids[$index]}.h5" \
  ARTIFACT_ROOT="$ARTIFACT_ROOT" \
  LDP_RUNTIME_ROOT="$LDP_RUNTIME_ROOT" LDP_OVERLAY="$LDP_OVERLAY" \
  OGBENCH_ROOT="$OGBENCH_ROOT" \
  TASK_TAG="${task_tags[$index]}" \
  RUN_LABEL="$CORE_LABEL" \
  GPU_ID="${gpu_ids[$index]}" \
  MIN_FREE_MEMORY_MIB="$MIN_FREE_MEMORY_MIB" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION" \
  ENV_ID="${env_ids[$index]}" \
  REWARD_TASK_ID="$REWARD_TASK_ID" \
  SEED="$SEED" \
  VAE_STEPS="$VAE_STEPS" VAE_BATCH_SIZE="$VAE_BATCH_SIZE" \
  VAE_LOG_EVERY="$VAE_LOG_EVERY" VAE_SAVE_EVERY="$VAE_SAVE_EVERY" \
  ENCODE_BATCH_SIZE="$ENCODE_BATCH_SIZE" \
  VAE_VALIDATION_SAMPLES="$VAE_VALIDATION_SAMPLES" \
  MAX_VAE_VALIDATION_MSE="$MAX_VAE_VALIDATION_MSE" \
  LDP_STEPS="$LDP_STEPS" LDP_BATCH_SIZE="$LDP_BATCH_SIZE" \
  LDP_LOG_EVERY="$LDP_LOG_EVERY" LDP_SAVE_EVERY="$LDP_SAVE_EVERY" \
  LDP_VALIDATION_BATCHES="$LDP_VALIDATION_BATCHES" \
  PRED_HORIZON="$PRED_HORIZON" ACTION_HORIZON="$ACTION_HORIZON" \
  DIFFUSION_STEPS="$DIFFUSION_STEPS" EPISODES="$EPISODES" \
  EVAL_SEED="$EVAL_SEED" MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" \
  MODE="$child_mode" bash "$TASK_SCRIPT"
}

plan() {
  local index name
  for index in "${!dataset_ids[@]}"; do
    name=$(run_name "$index")
    printf 'TASK index=%s gpu=%s dataset=%s env=%s tag=%s run=%s\n' \
      "$index" "${gpu_ids[$index]}" "${dataset_ids[$index]}" \
      "${env_ids[$index]}" "${task_tags[$index]}" "$name"
  done
}

audit_all() {
  local index
  for index in "${!dataset_ids[@]}"; do
    echo "AUDIT_TASK index=$index dataset=${dataset_ids[$index]}"
    run_base "$index" audit-data
  done
  echo "AUDIT_MATRIX_COMPLETE=8"
}

smoke_one() {
  : "${TASK_INDEX:?TASK_INDEX is required for MODE=smoke-one}"
  validate_index "$TASK_INDEX"
  local smoke_root="$ARTIFACT_ROOT/smoke_matrix/${task_tags[$TASK_INDEX]}"
  ARTIFACT_ROOT="$smoke_root" \
  MIN_FREE_MEMORY_MIB=1000 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.08 \
  run_base "$TASK_INDEX" smoke
}

smoke_missing() {
  local stamp index session log
  stamp=$(date +%Y%m%dT%H%M%S)
  mkdir -p "$ARTIFACT_ROOT/smoke_matrix/logs"
  # Index 0 already passed an end-to-end smoke and is in formal training.
  for index in 1 2 3 4 5 6 7; do
    session="ldp_smoke_${task_tags[$index]}_$stamp"
    log="$ARTIFACT_ROOT/smoke_matrix/logs/${task_tags[$index]}_$stamp.log"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "Refusing duplicate smoke session: $session" >&2
      exit 4
    fi
    tmux new-session -d -s "$session" \
      "cd '$STABLEWM_ROOT' && MODE=smoke-one TASK_INDEX='$index' DATASET_ROOT='$DATASET_ROOT' ARTIFACT_ROOT='$ARTIFACT_ROOT' CORE_LABEL='$CORE_LABEL' SEED='$SEED' bash '$0' 2>&1 | tee '$log'"
    echo "SMOKE_LAUNCHED index=$index gpu=${gpu_ids[$index]} session=$session log=$log"
  done
}

launch_one() {
  : "${TASK_INDEX:?TASK_INDEX is required for MODE=launch-one}"
  run_base "$TASK_INDEX" launch
}

launch_missing() {
  local index
  # Index 0 is the existing cube-single-play formal run on GPU 6.
  for index in 1 2 3 4 5 6 7; do
    echo "FORMAL_LAUNCH index=$index gpu=${gpu_ids[$index]} dataset=${dataset_ids[$index]}"
    TASK_INDEX="$index" MODE=launch-one DATASET_ROOT="$DATASET_ROOT" \
      ARTIFACT_ROOT="$ARTIFACT_ROOT" CORE_LABEL="$CORE_LABEL" SEED="$SEED" \
      bash "$0"
  done
}

status() {
  local index name session log vae latent ldp eval
  local phase session_state vae_step ldp_step anomalies
  date -Iseconds
  nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  printf 'FORMAL_TMUX_COUNT='
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep -c '^ldp_ldp_' || true
  for index in "${!dataset_ids[@]}"; do
    name=$(run_name "$index")
    session="ldp_${name:0:70}"
    log="$ARTIFACT_ROOT/$name.log"
    vae="$ARTIFACT_ROOT/runs/${name}_vae/checkpoint.msgpack"
    latent="$ARTIFACT_ROOT/data/${name}_latents.h5"
    ldp="$ARTIFACT_ROOT/runs/${name}_ldp/checkpoint.msgpack"
    eval="$ARTIFACT_ROOT/evals/${name}_eval${EPISODES}_s${EVAL_SEED}/results.json"

    vae_step=0
    ldp_step=0
    anomalies=0
    phase=starting
    if [[ -f "$log" ]]; then
      vae_step=$(grep 'VAE_METRICS=' "$log" | tail -1 \
        | sed -n 's/.*"step": \([0-9][0-9]*\).*/\1/p' || true)
      ldp_step=$(grep 'LDP_METRICS=' "$log" | tail -1 \
        | sed -n 's/.*"step": \([0-9][0-9]*\).*/\1/p' || true)
      vae_step=${vae_step:-0}
      ldp_step=${ldp_step:-0}
      anomalies=$(grep -Ec \
        'Traceback|CUDA out of memory|NaN|nan|Fatal|ERROR' "$log" || true)
      if grep -q 'VAE_METRICS=' "$log"; then
        phase=vae
      fi
      if grep -q 'VAE_COMPLETE=' "$log"; then
        phase=encode
      fi
      if grep -q 'LATENT_COMPLETE=' "$log"; then
        phase=ldp
      fi
      if grep -q 'LDP_COMPLETE=' "$log"; then
        phase=eval
      fi
    fi
    if [[ -s "$eval" ]]; then
      phase=complete
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      session_state=live
    else
      session_state=missing
    fi

    printf 'STATUS index=%s gpu=%s dataset=%s run=%s session=%s phase=%s vae_step=%s vae_ckpt=%s latent=%s ldp_step=%s ldp_ckpt=%s eval=%s anomalies=%s\n' \
      "$index" "${gpu_ids[$index]}" "${dataset_ids[$index]}" "$name" \
      "$session_state" \
      "$phase" "$vae_step" \
      "$([[ -s "$vae" ]] && echo yes || echo no)" \
      "$([[ -s "$latent" ]] && echo yes || echo no)" \
      "$ldp_step" \
      "$([[ -s "$ldp" ]] && echo yes || echo no)" \
      "$([[ -s "$eval" ]] && echo yes || echo no)" "$anomalies"
  done
}

case "$MODE" in
  plan) plan ;;
  audit) audit_all ;;
  smoke-one) smoke_one ;;
  smoke-missing) smoke_missing ;;
  launch-one) launch_one ;;
  launch-missing) launch_missing ;;
  status) status ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
