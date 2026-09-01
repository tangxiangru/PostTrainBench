#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage:
  bash src/commit_utils/slurm/submit.sh \
    --eval <task> --agent <agent> --model <base-model> \
    --hours <n> --agent-config <config> [options]

Options:
  --gpus <n>                GPUs exposed to the run (default: 1)
  --run-branch <name>        Owning top-level Git branch (required)
  --job-name <name>          Unique Slurm name beginning with <run-branch>. (required)
  --experiment-name <name>  Result suffix beginning with _<run-branch>_ (required)
  --judge-profile <name>    official or claude (default: .env or official)
  --walltime <slurm-time>   Override computed Slurm walltime
  --preflight-only          Check the selected node without running PTB
  --runtime-smoke           Run a short GPU check in the configured SIF
  --hold                    Submit the job held; release it with scontrol
  --dry-run                 Print the sbatch command without submitting
  -h, --help                Show this help
EOF
}

EVALUATION_TASK=""
AGENT=""
MODEL_TO_TRAIN=""
NUM_HOURS=""
AGENT_CONFIG=""
NUM_GPUS="1"
EXPERIMENT_NAME=""
RUN_BRANCH=""
REQUESTED_JOB_NAME=""
REQUESTED_JUDGE_PROFILE=""
WALLTIME=""
PREFLIGHT_ONLY=0
RUNTIME_SMOKE=0
DRY_RUN=0
HOLD=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --eval) EVALUATION_TASK="${2:?missing value for --eval}"; shift 2 ;;
        --agent) AGENT="${2:?missing value for --agent}"; shift 2 ;;
        --model) MODEL_TO_TRAIN="${2:?missing value for --model}"; shift 2 ;;
        --hours) NUM_HOURS="${2:?missing value for --hours}"; shift 2 ;;
        --agent-config) AGENT_CONFIG="${2:?missing value for --agent-config}"; shift 2 ;;
        --gpus) NUM_GPUS="${2:?missing value for --gpus}"; shift 2 ;;
        --run-branch) RUN_BRANCH="${2:?missing value for --run-branch}"; shift 2 ;;
        --job-name) REQUESTED_JOB_NAME="${2:?missing value for --job-name}"; shift 2 ;;
        --experiment-name) EXPERIMENT_NAME="${2:?missing value for --experiment-name}"; shift 2 ;;
        --judge-profile) REQUESTED_JUDGE_PROFILE="${2:?missing value for --judge-profile}"; shift 2 ;;
        --walltime) WALLTIME="${2:?missing value for --walltime}"; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --runtime-smoke) RUNTIME_SMOKE=1; shift ;;
        --hold) HOLD=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

RUN_BRANCH_SAFE="$(printf '%s' "$RUN_BRANCH" | tr -c 'A-Za-z0-9_.-' '-')"
JOB_NAME_SAFE="$(printf '%s' "$REQUESTED_JOB_NAME" | tr -c 'A-Za-z0-9_.-' '-')"
if [ -z "$RUN_BRANCH" ] || [ "$RUN_BRANCH_SAFE" != "$RUN_BRANCH" ]; then
    echo "ERROR: --run-branch must be a non-empty Slurm-safe branch name" >&2
    exit 2
fi
if [ -z "$REQUESTED_JOB_NAME" ] || [ "$JOB_NAME_SAFE" != "$REQUESTED_JOB_NAME" ] \
    || [[ "$JOB_NAME_SAFE" != "${RUN_BRANCH_SAFE}."* ]]; then
    echo "ERROR: --job-name must be Slurm-safe and begin with ${RUN_BRANCH_SAFE}." >&2
    exit 2
fi
if [ "${#JOB_NAME_SAFE}" -gt 128 ]; then
    echo "ERROR: --job-name exceeds 128 characters" >&2
    exit 2
fi
if [ -z "$EXPERIMENT_NAME" ] || [[ "$EXPERIMENT_NAME" != "_${RUN_BRANCH_SAFE}_"* ]]; then
    echo "ERROR: --experiment-name must begin with _${RUN_BRANCH_SAFE}_" >&2
    exit 2
fi

SUPERPROJECT_ROOT="$(git -C "$REPO_ROOT" rev-parse --show-superproject-working-tree 2>/dev/null || true)"
OWNERSHIP_ROOT="${SUPERPROJECT_ROOT:-$REPO_ROOT}"
CURRENT_BRANCH="$(git -C "$OWNERSHIP_ROOT" branch --show-current 2>/dev/null || true)"
if [ -z "$CURRENT_BRANCH" ] || [ "$CURRENT_BRANCH" != "$RUN_BRANCH" ]; then
    echo "ERROR: current ownership branch ${CURRENT_BRANCH:-<detached>} does not match --run-branch $RUN_BRANCH" >&2
    exit 1
fi
export POST_TRAIN_BENCH_RUN_BRANCH="$RUN_BRANCH"
export POST_TRAIN_BENCH_SLURM_JOB_NAME="$JOB_NAME_SAFE"

for value_name in EVALUATION_TASK AGENT MODEL_TO_TRAIN NUM_HOURS AGENT_CONFIG; do
    if [ -z "${!value_name}" ]; then
        echo "ERROR: missing required option for ${value_name}" >&2
        usage >&2
        exit 2
    fi
done

if ! [[ "$NUM_HOURS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --hours must be a positive integer" >&2
    exit 2
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --gpus must be a positive integer" >&2
    exit 2
fi
if [ "$PREFLIGHT_ONLY" = "1" ] && [ "$RUNTIME_SMOKE" = "1" ]; then
    echo "ERROR: --preflight-only and --runtime-smoke are mutually exclusive" >&2
    exit 2
fi

source src/commit_utils/set_env_vars.sh

SUBMIT_AS_ROOT="${POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT:-0}"
RUN_AS_USER="${POST_TRAIN_BENCH_SLURM_RUN_AS_USER:-}"
case "$SUBMIT_AS_ROOT" in
    0) ;;
    1)
        if [ -z "$RUN_AS_USER" ]; then
            echo "ERROR: root Slurm submission requires POST_TRAIN_BENCH_SLURM_RUN_AS_USER" >&2
            exit 1
        fi
        if ! RUN_AS_UID="$(id -u "$RUN_AS_USER" 2>/dev/null)"; then
            echo "ERROR: unknown POST_TRAIN_BENCH_SLURM_RUN_AS_USER=$RUN_AS_USER" >&2
            exit 1
        fi
        if [ "$RUN_AS_UID" = "0" ] || [ "$RUN_AS_UID" != "$(id -u)" ]; then
            echo "ERROR: root Slurm allocation must drop back to the invoking non-root user" >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: POST_TRAIN_BENCH_SLURM_SUBMIT_AS_ROOT must be 0 or 1" >&2
        exit 1
        ;;
esac

if [ -n "$REQUESTED_JUDGE_PROFILE" ]; then
    export POST_TRAIN_BENCH_JUDGE_PROFILE="$REQUESTED_JUDGE_PROFILE"
fi
case "${POST_TRAIN_BENCH_JUDGE_PROFILE:-official}" in
    official|claude) ;;
    *) echo "ERROR: --judge-profile/POST_TRAIN_BENCH_JUDGE_PROFILE must be official or claude" >&2; exit 1 ;;
esac

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER:-}" != "slurm" ]; then
    echo "ERROR: set POST_TRAIN_BENCH_JOB_SCHEDULER=slurm in .env" >&2
    exit 1
fi

PARTITION="${POST_TRAIN_BENCH_SLURM_PARTITION:-}"
if [ -z "$PARTITION" ]; then
    echo "ERROR: POST_TRAIN_BENCH_SLURM_PARTITION is required" >&2
    exit 1
fi

CPUS="${POST_TRAIN_BENCH_SLURM_CPUS_PER_TASK:-16}"
MEMORY="${POST_TRAIN_BENCH_SLURM_MEMORY:-128G}"
GPU_MODE="${POST_TRAIN_BENCH_SLURM_GPU_MODE:-gres}"
OVERHEAD_MINUTES="${POST_TRAIN_BENCH_SLURM_TIME_OVERHEAD_MINUTES:-240}"

if ! [[ "$CPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: POST_TRAIN_BENCH_SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 1
fi
if ! [[ "$OVERHEAD_MINUTES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: POST_TRAIN_BENCH_SLURM_TIME_OVERHEAD_MINUTES must be an integer" >&2
    exit 1
fi
if [ -z "$WALLTIME" ]; then
    WALLTIME="$((NUM_HOURS * 60 + OVERHEAD_MINUTES))"
fi

mkdir -p logs/slurm

SBATCH_ARGS=(
    --parsable
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --cpus-per-task="$CPUS"
    --mem="$MEMORY"
    --time="$WALLTIME"
    --job-name="$JOB_NAME_SAFE"
    --chdir="$REPO_ROOT"
    --output="${REPO_ROOT}/logs/slurm/%x-%j.out"
    --error="${REPO_ROOT}/logs/slurm/%x-%j.err"
)

if [ "$HOLD" = "1" ]; then
    SBATCH_ARGS+=(--hold)
fi

if [ -n "${POST_TRAIN_BENCH_SLURM_NODELIST:-}" ]; then
    SBATCH_ARGS+=(--nodelist="$POST_TRAIN_BENCH_SLURM_NODELIST")
fi
if [ -n "${POST_TRAIN_BENCH_SLURM_ACCOUNT:-}" ]; then
    SBATCH_ARGS+=(--account="$POST_TRAIN_BENCH_SLURM_ACCOUNT")
fi
if [ -n "${POST_TRAIN_BENCH_SLURM_QOS:-}" ]; then
    SBATCH_ARGS+=(--qos="$POST_TRAIN_BENCH_SLURM_QOS")
fi
if [ -n "${POST_TRAIN_BENCH_SLURM_RESERVATION:-}" ]; then
    SBATCH_ARGS+=(--reservation="$POST_TRAIN_BENCH_SLURM_RESERVATION")
fi

case "$GPU_MODE" in
    gres)
        GPU_TYPE="${POST_TRAIN_BENCH_SLURM_GPU_TYPE:-}"
        if [ -n "$GPU_TYPE" ]; then
            SBATCH_ARGS+=(--gres="gpu:${GPU_TYPE}:${NUM_GPUS}")
        else
            SBATCH_ARGS+=(--gres="gpu:${NUM_GPUS}")
        fi
        ;;
    # Manual visibility has no scheduler-owned GPU lock. It is retained only
    # as an explicit legacy fallback and therefore must reserve the full node.
    manual) SBATCH_ARGS+=(--exclusive) ;;
    *) echo "ERROR: unsupported POST_TRAIN_BENCH_SLURM_GPU_MODE=$GPU_MODE" >&2; exit 1 ;;
esac

if [ "$PREFLIGHT_ONLY" = "1" ]; then
    export POST_TRAIN_BENCH_SLURM_PREFLIGHT_ONLY=1
fi
if [ "$RUNTIME_SMOKE" = "1" ]; then
    export POST_TRAIN_BENCH_SLURM_RUNTIME_SMOKE=1
fi

COMMAND=(
    sbatch
    "${SBATCH_ARGS[@]}"
    "$SCRIPT_DIR/single_task.sbatch"
    "$EVALUATION_TASK"
    "$AGENT"
    "$MODEL_TO_TRAIN"
    "$NUM_HOURS"
    "$AGENT_CONFIG"
    "$NUM_GPUS"
    "$EXPERIMENT_NAME"
)

if [ "$SUBMIT_AS_ROOT" = "1" ]; then
    # Some sites cannot mint a Slurm credential for directory-backed UIDs
    # after a controller restart.  Root owns only the allocation; the batch
    # entrypoint immediately drops to RUN_AS_USER before touching the task.
    COMMAND=(sudo --preserve-env "${COMMAND[@]}")
fi

if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

JOB_ID="$("${COMMAND[@]}")"
echo "Submitted Slurm job ${JOB_ID}"
