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
PYTHON_BIN=${PYTHON_BIN:-/root/data/yyf/ogbench-new/.venv/bin/python}
DATASET_ROOT=${DATASET_ROOT:-/root/data/yyf/stablewm-data/datasets/ogbench8-tdmpc2-pixels-gc-h50}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/root/data/yyf/ldp-ogbench}
LDP_RUNTIME_ROOT=${LDP_RUNTIME_ROOT:-/root/data/yyf/ldp-ogbench/runtime-v2}
LDP_OVERLAY=${LDP_OVERLAY:-$LDP_RUNTIME_ROOT/site-packages}
OGBENCH_ROOT=${OGBENCH_ROOT:-/root/data/yyf/ogbench-official-1d414099}
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
EPISODES=${EPISODES:-50}
EVAL_SEED=${EVAL_SEED:-42}
EVAL_TASK_IDS=${EVAL_TASK_IDS:-1 2 3 4 5}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
ALLOW_NONSTANDARD_HORIZON=${ALLOW_NONSTANDARD_HORIZON:-0}
FORMAL_EPISODES=50
FORMAL_TASK_IDS='1 2 3 4 5'
MIN_FREE_MEMORY_MIB=${MIN_FREE_MEMORY_MIB:-17000}
MIN_FREE_DISK_GIB=${MIN_FREE_DISK_GIB:-32}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.18}
COMPLETION_AUDIT_OUTPUT=${COMPLETION_AUDIT_OUTPUT:-$ARTIFACT_ROOT/audits/ldp_ogbench8_completion_eval${FORMAL_EPISODES}_s${EVAL_SEED}.json}

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
native_horizons=(200 500 1000 750 200 500 1000 750)
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

require_formal_eval_contract() {
  local normalized_task_ids
  local -a requested_task_ids=()
  read -r -a requested_task_ids <<< "$EVAL_TASK_IDS"
  normalized_task_ids="${requested_task_ids[*]}"
  if [[ "$EPISODES" != "$FORMAL_EPISODES" ]]; then
    echo "Formal evaluation requires EPISODES=$FORMAL_EPISODES, got: $EPISODES" >&2
    exit 2
  fi
  if [[ "$normalized_task_ids" != "$FORMAL_TASK_IDS" ]]; then
    echo "Formal evaluation requires EVAL_TASK_IDS='$FORMAL_TASK_IDS', got: '$normalized_task_ids'" >&2
    exit 2
  fi
  if [[ -n "$MAX_EPISODE_STEPS" ]]; then
    echo "Formal evaluation must use the registered OGBench horizon" >&2
    exit 2
  fi
  if [[ "$ALLOW_NONSTANDARD_HORIZON" != 0 ]]; then
    echo "ALLOW_NONSTANDARD_HORIZON is smoke-only" >&2
    exit 2
  fi
}

require_formal_pipeline_contract() {
  require_formal_eval_contract
  local mismatches=()
  [[ "$VAE_STEPS" == 300000 ]] || mismatches+=("VAE_STEPS=$VAE_STEPS")
  [[ "$VAE_BATCH_SIZE" == 128 ]] || mismatches+=("VAE_BATCH_SIZE=$VAE_BATCH_SIZE")
  [[ "$LDP_STEPS" == 500000 ]] || mismatches+=("LDP_STEPS=$LDP_STEPS")
  [[ "$LDP_BATCH_SIZE" == 128 ]] || mismatches+=("LDP_BATCH_SIZE=$LDP_BATCH_SIZE")
  [[ "$PRED_HORIZON" == 8 ]] || mismatches+=("PRED_HORIZON=$PRED_HORIZON")
  [[ "$ACTION_HORIZON" == 4 ]] || mismatches+=("ACTION_HORIZON=$ACTION_HORIZON")
  [[ "$DIFFUSION_STEPS" == 100 ]] || mismatches+=("DIFFUSION_STEPS=$DIFFUSION_STEPS")
  [[ -z ${MAX_ENCODE_EPISODES:-} ]] || mismatches+=("MAX_ENCODE_EPISODES=${MAX_ENCODE_EPISODES}")
  if (( ${#mismatches[@]} )); then
    echo "Formal pipeline contract mismatch: ${mismatches[*]}" >&2
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
  EVAL_TASK_IDS="$EVAL_TASK_IDS" \
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
  ALLOW_NONSTANDARD_HORIZON="$ALLOW_NONSTANDARD_HORIZON" \
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

audit_environments() {
  local index
  for index in "${!dataset_ids[@]}"; do
    echo "ENV_AUDIT index=$index dataset=${dataset_ids[$index]}"
    run_base "$index" audit-environment
  done
  echo "ENV_AUDIT_MATRIX_COMPLETE=8"
}

audit_loaders() {
  local index
  for index in "${!dataset_ids[@]}"; do
    echo "LOADER_AUDIT index=$index dataset=${dataset_ids[$index]}"
    run_base "$index" audit-loader
  done
  echo "LOADER_AUDIT_MATRIX_COMPLETE=8"
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
  require_formal_pipeline_contract
  run_base "$TASK_INDEX" launch
}

launch_missing() {
  local index
  require_formal_pipeline_contract
  # Index 0 is the existing cube-single-play formal run on GPU 6.
  for index in 1 2 3 4 5 6 7; do
    echo "FORMAL_LAUNCH index=$index gpu=${gpu_ids[$index]} dataset=${dataset_ids[$index]}"
    TASK_INDEX="$index" MODE=launch-one DATASET_ROOT="$DATASET_ROOT" \
      ARTIFACT_ROOT="$ARTIFACT_ROOT" CORE_LABEL="$CORE_LABEL" SEED="$SEED" \
      bash "$0"
  done
}

eval_one() {
  : "${TASK_INDEX:?TASK_INDEX is required for MODE=eval-one}"
  require_formal_eval_contract
  run_base "$TASK_INDEX" eval
}

eval_result_valid() {
  local index=$1
  local result=$2
  "$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1])); e=r.get("environment"); tasks=r.get("tasks"); n=int(sys.argv[3]); valid=isinstance(e,dict) and isinstance(tasks,list) and r.get("dataset_id")==sys.argv[2] and r.get("task_ids")==[1,2,3,4,5] and r.get("episodes_per_task")==n and r.get("total_episodes")==5*n and r.get("seed")==int(sys.argv[4]) and len(tasks)==5 and all(isinstance(t,dict) for t in tasks) and [t.get("task_id") for t in tasks]==[1,2,3,4,5] and all(t.get("episodes")==n for t in tasks) and e.get("id")==sys.argv[5] and e.get("uses_registered_horizon") is True and e.get("max_episode_steps")==int(sys.argv[6]); raise SystemExit(0 if valid else 1)' \
    "$result" "${dataset_ids[$index]}" "$FORMAL_EPISODES" "$EVAL_SEED" \
    "${env_ids[$index]}" "${native_horizons[$index]}" 2>/dev/null
}

eval_result_fully_valid() {
  local index=$1
  cd "$STABLEWM_ROOT"
  PYTHONPATH="$LDP_OVERLAY:$STABLEWM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" scripts/audit_ldp_ogbench8.py \
      --dataset-root "$DATASET_ROOT" \
      --artifact-root "$ARTIFACT_ROOT" \
      --label "$CORE_LABEL" --seed "$SEED" \
      --episodes "$FORMAL_EPISODES" --eval-seed "$EVAL_SEED" \
      --task-index "$index"
}

launch_eval_ready() {
  require_formal_eval_contract
  local index name formal_session eval_session formal_log eval_log ldp ldp_config ldp_state eval
  local eval_dir quarantine
  for index in "${!dataset_ids[@]}"; do
    name=$(run_name "$index")
    formal_session="ldp_${name:0:70}"
    eval_session="ldp_eval${FORMAL_EPISODES}_${name:0:60}"
    formal_log="$ARTIFACT_ROOT/$name.log"
    eval_log="$ARTIFACT_ROOT/${name}_official_eval${FORMAL_EPISODES}_s${EVAL_SEED}.log"
    ldp="$ARTIFACT_ROOT/runs/${name}_ldp/checkpoint.msgpack"
    ldp_config="$ARTIFACT_ROOT/runs/${name}_ldp/config.json"
    ldp_state="$ARTIFACT_ROOT/runs/${name}_ldp/resume_state.json"
    eval="$ARTIFACT_ROOT/evals/${name}_eval${FORMAL_EPISODES}_s${EVAL_SEED}/results.json"
    eval_dir=${eval%/results.json}
    if [[ -s "$eval" ]]; then
      if eval_result_fully_valid "$index" >/dev/null; then
        echo "EVAL_ALREADY_COMPLETE index=$index result=$eval"
        continue
      fi
      if tmux has-session -t "$formal_session" 2>/dev/null; then
        echo "EVAL_INVALID_WAIT_FORMAL index=$index result=$eval" >&2
        continue
      fi
      if tmux has-session -t "$eval_session" 2>/dev/null; then
        echo "EVAL_INVALID_WAIT_EVAL_SESSION index=$index result=$eval" >&2
        continue
      fi
      quarantine="${eval_dir}_invalid_$(date +%Y%m%dT%H%M%S)"
      mv "$eval_dir" "$quarantine"
      echo "EVAL_INVALID_QUARANTINED index=$index from=$eval_dir to=$quarantine" >&2
    fi
    if tmux has-session -t "$formal_session" 2>/dev/null; then
      echo "EVAL_WAIT_FORMAL index=$index session=$formal_session"
      continue
    fi
    if tmux has-session -t "$eval_session" 2>/dev/null; then
      echo "EVAL_ALREADY_RUNNING index=$index session=$eval_session"
      continue
    fi
    if [[ -d "$eval_dir" && ! -s "$eval" ]]; then
      quarantine="${eval_dir}_incomplete_$(date +%Y%m%dT%H%M%S)"
      mv "$eval_dir" "$quarantine"
      echo "EVAL_INCOMPLETE_QUARANTINED index=$index from=$eval_dir to=$quarantine" >&2
    fi
    if [[ ! -s "$ldp" ]] || [[ ! -s "$ldp_config" ]] || [[ ! -s "$ldp_state" ]] \
      || [[ ! -f "$formal_log" ]] || ! grep -q 'LDP_COMPLETE=' "$formal_log" \
      || ! "$PYTHON_BIN" -c 'import json,sys; c=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); raise SystemExit(0 if c.get("steps")==500000 and s.get("step")==500000 else 1)' "$ldp_config" "$ldp_state" 2>/dev/null; then
      echo "EVAL_NOT_READY index=$index"
      continue
    fi
    tmux new-session -d -s "$eval_session" \
      "cd '$STABLEWM_ROOT' && MODE=eval-one TASK_INDEX='$index' DATASET_ROOT='$DATASET_ROOT' ARTIFACT_ROOT='$ARTIFACT_ROOT' CORE_LABEL='$CORE_LABEL' SEED='$SEED' EPISODES='$FORMAL_EPISODES' EVAL_SEED='$EVAL_SEED' EVAL_TASK_IDS='$FORMAL_TASK_IDS' MAX_EPISODE_STEPS='' ALLOW_NONSTANDARD_HORIZON='0' LDP_RUNTIME_ROOT='$LDP_RUNTIME_ROOT' OGBENCH_ROOT='$OGBENCH_ROOT' MIN_FREE_MEMORY_MIB='$MIN_FREE_MEMORY_MIB' XLA_PYTHON_CLIENT_MEM_FRACTION='$XLA_PYTHON_CLIENT_MEM_FRACTION' bash '$0' 2>&1 | tee '$eval_log'"
    echo "EVAL_LAUNCHED index=$index gpu=${gpu_ids[$index]} session=$eval_session log=$eval_log"
  done
}

metric_progress() {
  local log=$1
  local marker=$2
  local target=$3
  "$PYTHON_BIN" -c '
import json, os, sys, time
path, marker, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
record = None
with open(path, errors="replace") as stream:
    for line in stream:
        if marker in line:
            record = json.loads(line.split(marker, 1)[1])
if record is None:
    print("-1 0 -1")
else:
    step = int(record["step"])
    elapsed = float(record["elapsed_seconds"])
    speed = step / elapsed if elapsed > 0 else 0.0
    eta = (target - step) / speed / 3600 if speed > 0 else -1.0
    age = max(0, int(time.time() - os.path.getmtime(path)))
    print(f"{age} {speed:.4f} {max(0.0, eta):.2f}")
' "$log" "$marker" "$target"
}

status() {
  local index name session eval_session log eval_log vae latent ldp eval
  local phase session_state eval_session_state vae_step ldp_step anomalies eval_state
  local progress metric_age_s steps_per_s eta_h
  local disk_available_kib disk_available_gib disk_state
  date -Iseconds
  disk_available_kib=$(df -Pk "$ARTIFACT_ROOT" | awk 'NR == 2 {print $4}')
  disk_available_gib=$((disk_available_kib / 1024 / 1024))
  if (( disk_available_gib < MIN_FREE_DISK_GIB )); then
    disk_state=warn
  else
    disk_state=ok
  fi
  printf 'DISK artifact_root=%s available_gib=%s minimum_gib=%s state=%s\n' \
    "$ARTIFACT_ROOT" "$disk_available_gib" "$MIN_FREE_DISK_GIB" "$disk_state"
  nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  printf 'FORMAL_TMUX_COUNT='
  tmux list-sessions -F '#{session_name}' 2>/dev/null \
    | grep -c '^ldp_ldp_' || true
  for index in "${!dataset_ids[@]}"; do
    name=$(run_name "$index")
    session="ldp_${name:0:70}"
    eval_session="ldp_eval${FORMAL_EPISODES}_${name:0:60}"
    log="$ARTIFACT_ROOT/$name.log"
    eval_log="$ARTIFACT_ROOT/${name}_official_eval${FORMAL_EPISODES}_s${EVAL_SEED}.log"
    vae="$ARTIFACT_ROOT/runs/${name}_vae/checkpoint.msgpack"
    latent="$ARTIFACT_ROOT/data/${name}_latents.h5"
    ldp="$ARTIFACT_ROOT/runs/${name}_ldp/checkpoint.msgpack"
    eval="$ARTIFACT_ROOT/evals/${name}_eval${FORMAL_EPISODES}_s${EVAL_SEED}/results.json"

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
        'Traceback \(most recent call last\):|CUDA out of memory|(^|[^[:alpha:]])(NaN|nan)([^[:alpha:]]|$)|(^|[^[:alpha:]])FATAL([^[:alpha:]]|$)|(^|[^[:alpha:]])ERROR([^[:alpha:]]|$)' \
        "$log" || true)
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
    if [[ -f "$eval_log" ]]; then
      anomalies=$((anomalies + $(grep -Ec \
        'Traceback \(most recent call last\):|CUDA out of memory|(^|[^[:alpha:]])(NaN|nan)([^[:alpha:]]|$)|(^|[^[:alpha:]])FATAL([^[:alpha:]]|$)|(^|[^[:alpha:]])ERROR([^[:alpha:]]|$)' \
        "$eval_log" || true)))
    fi
    eval_state=no
    if [[ -s "$eval" ]]; then
      if eval_result_valid "$index" "$eval"; then
        phase=eval_result_pending_audit
        eval_state=candidate
      else
        phase=invalid_eval
        eval_state=invalid
      fi
    fi
    if tmux has-session -t "$session" 2>/dev/null; then
      session_state=live
    else
      session_state=missing
    fi
    if tmux has-session -t "$eval_session" 2>/dev/null; then
      eval_session_state=live
    else
      eval_session_state=missing
    fi
    progress='-1 0 -1'
    if [[ "$phase" == vae ]]; then
      progress=$(metric_progress "$log" 'VAE_METRICS=' "$VAE_STEPS")
    elif [[ "$phase" == ldp ]]; then
      progress=$(metric_progress "$log" 'LDP_METRICS=' "$LDP_STEPS")
    fi
    read -r metric_age_s steps_per_s eta_h <<< "$progress"

    printf 'STATUS index=%s gpu=%s dataset=%s run=%s session=%s eval_session=%s phase=%s vae_step=%s vae_ckpt=%s latent=%s ldp_step=%s ldp_ckpt=%s eval=%s anomalies=%s metric_age_s=%s steps_per_s=%s phase_eta_h=%s\n' \
      "$index" "${gpu_ids[$index]}" "${dataset_ids[$index]}" "$name" \
      "$session_state" "$eval_session_state" \
      "$phase" "$vae_step" \
      "$([[ -s "$vae" ]] && echo yes || echo no)" \
      "$([[ -s "$latent" ]] && echo yes || echo no)" \
      "$ldp_step" \
      "$([[ -s "$ldp" ]] && echo yes || echo no)" \
      "$eval_state" "$anomalies" "$metric_age_s" "$steps_per_s" "$eta_h"
  done
}

audit_completion() {
  require_formal_eval_contract
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
  fi
  cd "$STABLEWM_ROOT"
  PYTHONPATH="$LDP_OVERLAY:$STABLEWM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" scripts/audit_ldp_ogbench8.py \
      --dataset-root "$DATASET_ROOT" \
      --artifact-root "$ARTIFACT_ROOT" \
      --label "$CORE_LABEL" \
      --seed "$SEED" --episodes "$FORMAL_EPISODES" --eval-seed "$EVAL_SEED" \
      --output "$COMPLETION_AUDIT_OUTPUT"
}

case "$MODE" in
  plan) plan ;;
  audit) audit_all ;;
  audit-environments) audit_environments ;;
  audit-loaders) audit_loaders ;;
  smoke-one) smoke_one ;;
  smoke-missing) smoke_missing ;;
  launch-one) launch_one ;;
  launch-missing) launch_missing ;;
  eval-one) eval_one ;;
  launch-eval-ready) launch_eval_ready ;;
  status) status ;;
  audit-completion) audit_completion ;;
  *)
    echo "Unknown MODE=$MODE" >&2
    exit 2
    ;;
esac
