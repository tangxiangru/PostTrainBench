#!/bin/bash

set -uo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <num-gpus> <eval> <agent> <base-model>" >&2
    exit 2
fi

NUM_GPUS="$1"
EVALUATION_TASK="$2"
AGENT="$3"
MODEL_TO_TRAIN="$4"
FAILURES=0

fail() {
    echo "PREFLIGHT ERROR: $*" >&2
    FAILURES=$((FAILURES + 1))
}

ok() {
    echo "PREFLIGHT OK: $*"
}

check_command() {
    local command_name="$1"
    if command -v "$command_name" >/dev/null 2>&1; then
        ok "$command_name=$(command -v "$command_name")"
    else
        fail "required command not found: $command_name"
    fi
}

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    fail "num-gpus must be a positive integer, got: $NUM_GPUS"
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
    fail "not running inside a Slurm allocation"
else
    ok "Slurm job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-$(hostname)}"
fi

if [ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]; then
    fail "PostTrainBench requires exactly one node per run"
fi

for command_name in apptainer flock fuse-overlayfs fusermount uuidgen nvidia-smi python3; do
    check_command "$command_name"
done
if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-0}" = "1" ]; then
    check_command nvidia-container-cli
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    GPU_COUNT="$(printf '%s\n' "$GPU_NAMES" | awk 'NF {count++} END {print count+0}')"
    H100_COUNT="$(printf '%s\n' "$GPU_NAMES" | awk '/H100/ {count++} END {print count+0}')"
    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        fail "nvidia-smi found ${GPU_COUNT} GPU(s), need ${NUM_GPUS}"
    elif [ "$H100_COUNT" -lt "$NUM_GPUS" ]; then
        fail "nvidia-smi found ${H100_COUNT} H100 GPU(s), need ${NUM_GPUS}"
    else
        ok "nvidia-smi found ${GPU_COUNT} GPU(s), including ${H100_COUNT} H100(s)"
    fi
fi

if [ "${POST_TRAIN_BENCH_SLURM_GPU_MODE:-gres}" = "gres" ]; then
    if [ -z "${SLURM_JOB_GPUS:-}" ]; then
        fail "Slurm GRES allocation did not set SLURM_JOB_GPUS"
    else
        ALLOCATED_GPU_COUNT="$(printf '%s' "$SLURM_JOB_GPUS" | awk -F, '{print NF}')"
        if [ "$ALLOCATED_GPU_COUNT" -ne "$NUM_GPUS" ]; then
            fail "SLURM_JOB_GPUS=${SLURM_JOB_GPUS} exposes ${ALLOCATED_GPU_COUNT} GPU(s), expected ${NUM_GPUS}"
        else
            ok "Slurm allocated GPU(s): ${SLURM_JOB_GPUS}"
        fi
    fi
fi

SCRATCH_DIR="${POST_TRAIN_BENCH_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
MIN_SCRATCH_GB="${POST_TRAIN_BENCH_SLURM_MIN_SCRATCH_GB:-400}"
if [ ! -d "$SCRATCH_DIR" ] || [ ! -w "$SCRATCH_DIR" ]; then
    fail "scratch directory is not writable: $SCRATCH_DIR"
elif ! [[ "$MIN_SCRATCH_GB" =~ ^[0-9]+$ ]]; then
    fail "POST_TRAIN_BENCH_SLURM_MIN_SCRATCH_GB must be an integer"
else
    AVAILABLE_KB="$(df -Pk "$SCRATCH_DIR" | awk 'NR == 2 {print $4}')"
    REQUIRED_KB=$((MIN_SCRATCH_GB * 1024 * 1024))
    if [ "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]; then
        fail "scratch has $((AVAILABLE_KB / 1024 / 1024)) GiB free; need ${MIN_SCRATCH_GB} GiB"
    else
        ok "scratch=$SCRATCH_DIR ($((AVAILABLE_KB / 1024 / 1024)) GiB free)"
    fi
fi

if [ ! -d "${HF_HOME:-}" ] || [ ! -r "${HF_HOME:-}" ]; then
    fail "HF_HOME is missing or unreadable: ${HF_HOME:-<unset>}"
else
    ok "HF_HOME=${HF_HOME}"
fi

BASE_MODEL_REVISION="${POST_TRAIN_BENCH_BASE_MODEL_REVISION:-}"
if [ -n "$BASE_MODEL_REVISION" ]; then
    if ! [[ "$BASE_MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
        fail "POST_TRAIN_BENCH_BASE_MODEL_REVISION must be a 40-character commit"
    else
        BASE_MODEL_CACHE_KEY="models--${MODEL_TO_TRAIN//\//--}"
        BASE_MODEL_SNAPSHOT="${HF_HOME}/hub/${BASE_MODEL_CACHE_KEY}/snapshots/${BASE_MODEL_REVISION}"
        if python3 src/utils/validate_model_snapshot.py "$BASE_MODEL_SNAPSHOT"
        then
            ok "base model=${MODEL_TO_TRAIN}@${BASE_MODEL_REVISION} snapshot=${BASE_MODEL_SNAPSHOT}"
        else
            fail "pinned base-model snapshot is missing or incomplete: ${MODEL_TO_TRAIN}@${BASE_MODEL_REVISION}"
        fi
    fi
fi

if [ -z "${POST_TRAIN_BENCH_RESULTS_DIR:-}" ]; then
    fail "POST_TRAIN_BENCH_RESULTS_DIR is unset"
elif ! mkdir -p "$POST_TRAIN_BENCH_RESULTS_DIR" || [ ! -w "$POST_TRAIN_BENCH_RESULTS_DIR" ]; then
    fail "results directory is not writable: $POST_TRAIN_BENCH_RESULTS_DIR"
else
    ok "results=$POST_TRAIN_BENCH_RESULTS_DIR"
fi

CONTAINERS_DIR="${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}"
MAIN_CONTAINER="${POST_TRAIN_BENCH_CONTAINER_NAME:-standard}"
for image in "${MAIN_CONTAINER}.sif" vllm_debug.sif; do
    if [ -r "${CONTAINERS_DIR}/${image}" ]; then
        ok "container=${CONTAINERS_DIR}/${image}"
    else
        fail "container not found: ${CONTAINERS_DIR}/${image}"
    fi
done

source src/judges/judge_lib.sh
if ! configure_judge_profile "${POST_TRAIN_BENCH_JUDGE_PROFILE:-official}"; then
    fail "invalid POST_TRAIN_BENCH_JUDGE_PROFILE=${POST_TRAIN_BENCH_JUDGE_PROFILE:-official}"
fi
if ! resolve_judge_auth_mode; then
    fail "could not resolve judge auth mode"
fi
case "$JUDGE_AUTH_MODE" in
    chatgpt)
        if [ "$JUDGE_PROFILE" != "official" ]; then
            fail "chatgpt auth is valid only with POST_TRAIN_BENCH_JUDGE_PROFILE=official"
        fi
        if [ -r "${CONTAINERS_DIR}/${JUDGE_CONTAINER}" ]; then
            ok "container=${CONTAINERS_DIR}/${JUDGE_CONTAINER}"
        else
            fail "judge container not found: ${CONTAINERS_DIR}/${JUDGE_CONTAINER}"
        fi
        CODEX_JUDGE_AUTH="${POST_TRAIN_BENCH_CODEX_JUDGE_AUTH_FILE:-agents/codex_non_api/auth.json}"
        if [ -r "$CODEX_JUDGE_AUTH" ]; then
            ok "judge auth=$CODEX_JUDGE_AUTH"
        else
            fail "judge auth not found: $CODEX_JUDGE_AUTH"
        fi
        JUDGE_LOCK_FILE="${POST_TRAIN_BENCH_JUDGE_LOCK_FILE:-}"
        if [ -z "$JUDGE_LOCK_FILE" ]; then
            fail "POST_TRAIN_BENCH_JUDGE_LOCK_FILE is required for concurrent official judges"
        elif ! mkdir -p "$(dirname "$JUDGE_LOCK_FILE")" || ! touch "$JUDGE_LOCK_FILE"; then
            fail "official judge lock file is not writable: $JUDGE_LOCK_FILE"
        else
            ok "official judge lock=$JUDGE_LOCK_FILE"
        fi
        ;;
    claude_oauth|vertex)
        if [ "$JUDGE_PROFILE" != "claude" ]; then
            fail "$JUDGE_AUTH_MODE is valid only with POST_TRAIN_BENCH_JUDGE_PROFILE=claude"
        fi
        if [ -r "${CONTAINERS_DIR}/${JUDGE_CONTAINER}" ]; then
            ok "container=${CONTAINERS_DIR}/${JUDGE_CONTAINER}"
            if CLAUDE_VERSION="$(apptainer exec --containall "${CONTAINERS_DIR}/${JUDGE_CONTAINER}" claude --version 2>&1)"; then
                ok "Claude judge CLI=${CLAUDE_VERSION%%$'\n'*}; auth=${JUDGE_AUTH_MODE}; requested model=${JUDGE_DEFAULT_MODEL}; effort=xhigh"
            else
                fail "Claude Code CLI is not runnable in ${CONTAINERS_DIR}/${JUDGE_CONTAINER}"
            fi
            if CLAUDE_HELP="$(apptainer exec --containall "${CONTAINERS_DIR}/${JUDGE_CONTAINER}" claude --help 2>&1)"; then
                if grep -q -- '--effort' <<< "$CLAUDE_HELP" \
                    && grep -q -- '--setting-sources' <<< "$CLAUDE_HELP" \
                    && grep -q -- '--safe-mode' <<< "$CLAUDE_HELP"; then
                    ok "Claude judge CLI supports --effort, isolated settings, and safe mode"
                else
                    fail "Claude judge CLI lacks --effort, --setting-sources, or --safe-mode; rebuild/update ${JUDGE_CONTAINER}"
                fi
            else
                fail "Claude Code help check failed in ${CONTAINERS_DIR}/${JUDGE_CONTAINER}"
            fi
        else
            fail "judge container not found: ${CONTAINERS_DIR}/${JUDGE_CONTAINER}"
        fi
        if [ "$JUDGE_AUTH_MODE" = "claude_oauth" ]; then
            CLAUDE_AUTH_FILE="${POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE:-}"
            if [ -n "$CLAUDE_AUTH_FILE" ] && [ -r "$CLAUDE_AUTH_FILE" ] && [ -s "$CLAUDE_AUTH_FILE" ]; then
                ok "dedicated Claude judge OAuth token file is readable"
            else
                fail "POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE must name a readable, non-empty file"
            fi
        else
            if setup_judge_vertex_auth "${POST_TRAIN_BENCH_SCRATCH_DIR:-${TMPDIR:-/tmp}}"; then
                if [ -n "${JUDGE_VERTEX_ADC_SRC:-}" ]; then
                    ok "Vertex ADC file is readable; project=${JUDGE_VERTEX_PROJECT}; region=${JUDGE_VERTEX_REGION}"
                else
                    ok "GCE metadata ADC is reachable; project=${JUDGE_VERTEX_PROJECT}; region=${JUDGE_VERTEX_REGION}"
                fi
            else
                fail "Vertex ADC setup failed"
            fi
        fi
        ;;
    skip)
        ok "judges skipped for non-official smoke run"
        ;;
    apikey)
        fail "apikey judge mode is not wired through judge_lib.sh in this fork"
        ;;
    *)
        fail "unsupported POST_TRAIN_BENCH_JUDGE_AUTH_MODE=$JUDGE_AUTH_MODE (want chatgpt|claude_oauth|vertex|skip)"
        ;;
esac

for required_file in \
    "src/eval/tasks/${EVALUATION_TASK}/evaluate.py" \
    "src/eval/tasks/${EVALUATION_TASK}/test_data.json" \
    "agents/${AGENT}/solve.sh" \
    "agents/${AGENT}/api_keys.json"; do
    if [ -r "$required_file" ]; then
        ok "file=$required_file"
    else
        fail "required file not found: $required_file"
    fi
done

if [ "$FAILURES" -ne 0 ]; then
    echo "PREFLIGHT FAILED: ${FAILURES} problem(s)" >&2
    exit 1
fi

echo "PREFLIGHT PASSED"
