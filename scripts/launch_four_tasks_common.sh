#!/usr/bin/env bash

# Shared argument parsing and tmux launching for the four-task experiment scripts.
# This file is meant to be sourced, not executed directly.

launch_four_usage() {
  local expected_gpu_count="${LAUNCH_GPU_COUNT:-4}"
  cat <<EOF
Usage: bash scripts/$(basename "$0") [options]

Options:
  --datasets-dir DIR  Directory containing the required .h5 datasets
  --runs-dir DIR      Root directory for checkpoints/runs
  --logs-dir DIR      Root directory for logs (default: <repo>/logs)
  --venv-dir DIR      Virtual environment directory (default: <repo>/.venv)
  --gpus LIST         ${expected_gpu_count} comma-separated GPU IDs
  --dry-run           Validate inputs and print commands without starting tmux
  -h, --help          Show this help

The same values can be supplied through DATASETS_DIR, RUNS_DIR, LOGS_DIR,
VENV_DIR, and GPUS. Command-line options take precedence. For compatibility,
a single positional argument is treated as --datasets-dir.
EOF
}

launch_four_init() {
  local repo_default_gpus="$1"
  local expected_gpu_count="${LAUNCH_GPU_COUNT:-4}"
  shift

  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
  DATASETS_DIR="${DATASETS_DIR:-${REPO_ROOT}/datasets}"
  RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs}"
  LOGS_DIR="${LOGS_DIR:-${REPO_ROOT}/logs}"
  VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
  GPUS="${GPUS:-${repo_default_gpus}}"
  ONLY_TASK="${ONLY_TASK:-}"
  DRY_RUN="${DRY_RUN:-0}"

  while (($#)); do
    case "$1" in
      --datasets-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: --datasets-dir requires a value." >&2; return 2; }
        DATASETS_DIR="$2"
        shift 2
        ;;
      --runs-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: --runs-dir requires a value." >&2; return 2; }
        RUNS_DIR="$2"
        shift 2
        ;;
      --logs-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: --logs-dir requires a value." >&2; return 2; }
        LOGS_DIR="$2"
        shift 2
        ;;
      --venv-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: --venv-dir requires a value." >&2; return 2; }
        VENV_DIR="$2"
        shift 2
        ;;
      --gpus)
        [[ $# -ge 2 ]] || { echo "ERROR: --gpus requires a value." >&2; return 2; }
        GPUS="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        launch_four_usage
        exit 0
        ;;
      --*)
        echo "ERROR: unknown option: $1" >&2
        launch_four_usage >&2
        return 2
        ;;
      *)
        if [[ "${positional_dataset_seen:-0}" == 1 ]]; then
          echo "ERROR: unexpected positional argument: $1" >&2
          return 2
        fi
        DATASETS_DIR="$1"
        positional_dataset_seen=1
        shift
        ;;
    esac
  done

  IFS=',' read -r -a GPU_IDS <<< "$GPUS"
  if [[ ${#GPU_IDS[@]} -ne $expected_gpu_count ]]; then
    echo "ERROR: --gpus/GPUS must contain exactly ${expected_gpu_count} comma-separated GPU IDs; got: ${GPUS}" >&2
    return 2
  fi
  local gpu_id
  for gpu_id in "${GPU_IDS[@]}"; do
    if [[ -z "$gpu_id" ]]; then
      echo "ERROR: --gpus/GPUS contains an empty GPU ID: ${GPUS}" >&2
      return 2
    fi
  done

  local filename
  local required_datasets=(tworoom.h5 reacher.h5 pusht_expert_train.h5 cube_single_expert.h5)
  if declare -p LAUNCH_REQUIRED_DATASETS >/dev/null 2>&1; then
    required_datasets=("${LAUNCH_REQUIRED_DATASETS[@]}")
  fi
  case "$ONLY_TASK" in
    '') ;;
    tworoom) required_datasets=(tworoom.h5) ;;
    reacher) required_datasets=(reacher.h5) ;;
    pusht) required_datasets=(pusht_expert_train.h5) ;;
    ogbench_cube) required_datasets=(cube_single_expert.h5) ;;
    *) echo "ERROR: invalid ONLY_TASK: ${ONLY_TASK}" >&2; return 2 ;;
  esac
  for filename in "${required_datasets[@]}"; do
    if [[ ! -f "${DATASETS_DIR}/${filename}" ]]; then
      echo "ERROR: dataset not found: ${DATASETS_DIR}/${filename}" >&2
      return 1
    fi
  done

  PYTHON_BIN="${VENV_DIR}/bin/python"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    return 1
  fi
  if [[ "$DRY_RUN" != 1 ]] && ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed or not on PATH." >&2
    return 1
  fi

  cd "$REPO_ROOT"
}

launch_four_run() {
  local session_name="$1"
  local gpu_id="$2"
  local experiment="$3"
  local task="$4"
  local train_script="$5"
  shift 5

  local run_dir="${RUNS_DIR}/${experiment}/${task}"
  local log_dir="${LOGS_DIR}/${experiment}/${task}"
  local tmp_dir="${run_dir}/tmp"
  local train_log="${log_dir}/train.log"
  if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$run_dir" "$log_dir" "$tmp_dir"
  fi

  local command=''
  local proxy_var proxy_value
  for proxy_var in http_proxy https_proxy all_proxy no_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY; do
    proxy_value="${!proxy_var:-}"
    printf -v command '%s%s=%q ' "$command" "$proxy_var" "$proxy_value"
  done
  printf -v command '%sTMPDIR=%q CUDA_VISIBLE_DEVICES=%q STABLEWM_HOME=%q SPT_CACHE_DIR=%q PYTHONPATH=%q %q %q ' \
    "$command" \
    "$tmp_dir" "$gpu_id" "$run_dir" "$run_dir" "$REPO_ROOT" "$PYTHON_BIN" "$train_script"

  local argument
  for argument in "$@"; do
    printf -v command '%s%q ' "$command" "$argument"
  done
  printf -v command '%s2>&1 | tee %q' "$command" "$train_log"

  if [[ "$DRY_RUN" == 1 ]]; then
    printf '[dry-run] tmux session %s (GPU %s):\n%s\n' "$session_name" "$gpu_id" "$command"
  else
    tmux new-session -d -s "$session_name" -c "$REPO_ROOT" "$command"
  fi
}

launch_four_summary() {
  local experiment="$1"
  local launch_count="${LAUNCH_GPU_COUNT:-4}"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run completed for ${experiment}; no tmux sessions were started."
  else
    if [[ -n "$ONLY_TASK" ]]; then
      echo "Started ${experiment}/${ONLY_TASK} on its configured GPU."
    else
      echo "Started ${launch_count} ${experiment} runs on GPUs ${GPUS}."
    fi
  fi
  echo "Datasets: ${DATASETS_DIR}"
  echo "Runs:    ${RUNS_DIR}/${experiment}"
  echo "Logs:    ${LOGS_DIR}/${experiment}"
}
