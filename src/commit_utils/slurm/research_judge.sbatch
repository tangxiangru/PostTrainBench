#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

RESULT_DIR="${1:?result directory required}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
SUBMISSION_REPO_ROOT="$REPO_ROOT"
source src/commit_utils/set_env_vars.sh

if [ -n "${POST_TRAIN_BENCH_APPTAINER_BIN:-}" ]; then
    export PATH="$(dirname "$POST_TRAIN_BENCH_APPTAINER_BIN"):$PATH"
fi
if [ -n "${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SCRATCH_BASE="${POST_TRAIN_BENCH_SLURM_SCRATCH_BASE:-/tmp/posttrainbench}"
JUDGE_SCRATCH="${SCRATCH_BASE%/}/${USER}/research-judge-${SLURM_JOB_ID}"
EXPECTED_PREFIX="${SCRATCH_BASE%/}/${USER}/research-judge-"
if [[ "$SCRATCH_BASE" != /* ]] || [ "$SCRATCH_BASE" = "/" ] \
    || [[ "$JUDGE_SCRATCH" != "$EXPECTED_PREFIX"* ]]; then
    echo "ERROR: refused unsafe research-judge scratch path: $JUDGE_SCRATCH" >&2
    exit 1
fi
mkdir -p "$JUDGE_SCRATCH"
export TMPDIR="$JUDGE_SCRATCH"
cleanup() {
    local status=$?
    rm -rf -- "$JUDGE_SCRATCH"
    return "$status"
}
trap cleanup EXIT

FROZEN_PTB_COMMIT="${POST_TRAIN_BENCH_FROZEN_PTB_COMMIT:-}"
if [ -n "$FROZEN_PTB_COMMIT" ]; then
    [[ "$FROZEN_PTB_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
        || { echo "ERROR: invalid frozen PTB commit" >&2; exit 1; }
    FROZEN_SOURCE="$JUDGE_SCRATCH/source"
    mkdir -p "$FROZEN_SOURCE"
    git -C "$SUBMISSION_REPO_ROOT" cat-file -e "${FROZEN_PTB_COMMIT}^{commit}"
    git -C "$SUBMISSION_REPO_ROOT" archive "$FROZEN_PTB_COMMIT" | tar -x -C "$FROZEN_SOURCE"
    export POST_TRAIN_BENCH_ENV_FILE="${POST_TRAIN_BENCH_ENV_FILE:-${SUBMISSION_REPO_ROOT}/.env}"
    cd "$FROZEN_SOURCE"
    echo "Frozen PTB research-judge source: $FROZEN_PTB_COMMIT"
fi

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

export POST_TRAIN_BENCH_JUDGE_PROFILE=claude
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE=vertex
export POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="${POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL:-claude-opus-5[1m]}"
bash src/judges/run_judges.sh --profile claude "$RESULT_DIR"
