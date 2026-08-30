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

export POST_TRAIN_BENCH_JUDGE_PROFILE=claude
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE=vertex
export POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="${POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL:-claude-opus-5[1m]}"
bash src/judges/run_judges.sh --profile claude "$RESULT_DIR"
