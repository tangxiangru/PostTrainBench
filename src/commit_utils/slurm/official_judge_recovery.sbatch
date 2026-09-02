#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

if [ "$(id -u)" = "0" ]; then
    RUN_AS_USER="${POST_TRAIN_BENCH_SLURM_RUN_AS_USER:-}"
    if [ -z "$RUN_AS_USER" ] || [ "$(id -u "$RUN_AS_USER" 2>/dev/null || true)" = "0" ]; then
        echo "ERROR: refusing to run official judge recovery as root" >&2
        exit 1
    fi
    ENTRYPOINT="${POST_TRAIN_BENCH_SLURM_ENTRYPOINT:-}"
    if [ -z "$ENTRYPOINT" ] || [ ! -r "$ENTRYPOINT" ]; then
        echo "ERROR: root-owned allocation requires a readable recovery entrypoint" >&2
        exit 1
    fi
    exec /usr/sbin/runuser --user "$RUN_AS_USER" -- /bin/bash "$ENTRYPOINT" "$@"
fi

RESULT_DIR="${1:?result directory required}"
SUBMISSION_REPO_ROOT="${POST_TRAIN_BENCH_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$SUBMISSION_REPO_ROOT"
source src/commit_utils/set_env_vars.sh

if [ -n "${POST_TRAIN_BENCH_APPTAINER_BIN:-}" ]; then
    export PATH="$(dirname "$POST_TRAIN_BENCH_APPTAINER_BIN"):$PATH"
fi
if [ -n "${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SCRATCH_BASE="${POST_TRAIN_BENCH_SLURM_SCRATCH_BASE:-/tmp/posttrainbench}"
RECOVERY_SCRATCH="${SCRATCH_BASE%/}/${USER}/official-judge-${SLURM_JOB_ID}"
EXPECTED_PREFIX="${SCRATCH_BASE%/}/${USER}/official-judge-"
if [[ "$SCRATCH_BASE" != /* ]] || [ "$SCRATCH_BASE" = "/" ] \
    || [[ "$RECOVERY_SCRATCH" != "$EXPECTED_PREFIX"* ]]; then
    echo "ERROR: refused unsafe official-judge recovery scratch path: $RECOVERY_SCRATCH" >&2
    exit 1
fi
mkdir -p "$RECOVERY_SCRATCH"
export TMPDIR="$RECOVERY_SCRATCH"
cleanup() {
    local status=$?
    rm -rf -- "$RECOVERY_SCRATCH"
    return "$status"
}
trap cleanup EXIT

FROZEN_PTB_COMMIT="${POST_TRAIN_BENCH_FROZEN_PTB_COMMIT:-}"
if ! [[ "$FROZEN_PTB_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: a frozen PTB commit is required" >&2
    exit 1
fi
FROZEN_SOURCE="$RECOVERY_SCRATCH/source"
mkdir -p "$FROZEN_SOURCE"
git -C "$SUBMISSION_REPO_ROOT" cat-file -e "${FROZEN_PTB_COMMIT}^{commit}"
git -C "$SUBMISSION_REPO_ROOT" archive "$FROZEN_PTB_COMMIT" | tar -x -C "$FROZEN_SOURCE"
export POST_TRAIN_BENCH_ENV_FILE="${POST_TRAIN_BENCH_ENV_FILE:-${SUBMISSION_REPO_ROOT}/.env}"
cd "$FROZEN_SOURCE"
echo "Frozen PTB official-judge source: $FROZEN_PTB_COMMIT"

CONTEXT_RECORD="${POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD:-}"
CONTEXT_SHA256="${POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256:-}"
if [ ! -r "$CONTEXT_RECORD" ] || ! [[ "$CONTEXT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "ERROR: frozen Opus 5 high 1M context-validation evidence is required" >&2
    exit 1
fi
ACTUAL_CONTEXT_SHA256="$(sha256sum "$CONTEXT_RECORD" | awk '{print $1}')"
if [ "$ACTUAL_CONTEXT_SHA256" != "$CONTEXT_SHA256" ]; then
    echo "ERROR: context-validation evidence changed after submission" >&2
    exit 1
fi
echo "Frozen Opus 5 high 1M evidence: ${CONTEXT_SHA256}"

export POST_TRAIN_BENCH_JUDGE_PROFILE="official"
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE="vertex"
export POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="${POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL:-claude-opus-5[1m]}"
export POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER="${POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER:-opus_5.sif}"

bash src/judges/run_judges.sh --profile official "$RESULT_DIR"
python3 src/utils/validate_completed_run.py "$RESULT_DIR" --judge-profile official
