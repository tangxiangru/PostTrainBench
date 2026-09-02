#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEST_DIR"' EXIT

cat > "$TEST_DIR/test.env" <<EOF
HF_HOME="$TEST_DIR/hf"
POST_TRAIN_BENCH_RESULTS_DIR="$TEST_DIR/results"
POST_TRAIN_BENCH_CONTAINERS_DIR="$TEST_DIR/containers"
POST_TRAIN_BENCH_CONTAINER_NAME="standard"
POST_TRAIN_BENCH_PROMPT="prompt"
POST_TRAIN_BENCH_JOB_SCHEDULER="slurm"
POST_TRAIN_BENCH_SLURM_PARTITION="test-gpu"
POST_TRAIN_BENCH_SLURM_NODELIST="test-node-[0-3]"
POST_TRAIN_BENCH_SLURM_RESERVATION="test-reservation"
POST_TRAIN_BENCH_SLURM_GPU_MODE="gres"
POST_TRAIN_BENCH_SLURM_SCRATCH_BASE="/tmp/ptb-test"
EOF

mkdir -p "$TEST_DIR/bin" "$TEST_DIR/hf" "$TEST_DIR/results" "$TEST_DIR/containers"
cat > "$TEST_DIR/bin/sbatch" <<'EOF'
#!/bin/bash
printf '%s\n' "$@" > "$PTB_TEST_SBATCH_ARGS"
printf '12345\n'
EOF
chmod +x "$TEST_DIR/bin/sbatch"

export POST_TRAIN_BENCH_ENV_FILE="$TEST_DIR/test.env"
export PTB_TEST_SBATCH_ARGS="$TEST_DIR/sbatch.args"
export PATH="$TEST_DIR/bin:$PATH"
RUN_BRANCH="$(git -C "$(git -C "$REPO_ROOT" rev-parse --show-superproject-working-tree)" branch --show-current)"
JOB_NAME="${RUN_BRANCH}.ptb.test-batch.b1.preflight.r1"
EXPERIMENT_NAME="_${RUN_BRANCH}_test_batch_b1_preflight_r1"

OUTPUT="$(bash "$SCRIPT_DIR/submit.sh" \
    --eval gsm8k \
    --agent hv_noop \
    --model google/gemma-3-4b-pt \
    --hours 10 \
    --agent-config smoke \
    --run-branch "$RUN_BRANCH" \
    --job-name "$JOB_NAME" \
    --experiment-name "$EXPERIMENT_NAME" \
    --preflight-only)"

[ "$OUTPUT" = "Submitted Slurm job 12345" ]
grep -Fx -- '--partition=test-gpu' "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- '--nodelist=test-node-[0-3]' "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- '--reservation=test-reservation' "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- '--gres=gpu:1' "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- "--job-name=$JOB_NAME" "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- "--chdir=$REPO_ROOT" "$PTB_TEST_SBATCH_ARGS" >/dev/null
if grep -Fx -- '--exclusive' "$PTB_TEST_SBATCH_ARGS" >/dev/null; then
    echo "GRES mode unexpectedly requested the whole node" >&2
    exit 1
fi

DRY_RUN="$(bash "$SCRIPT_DIR/submit.sh" \
    --eval gsm8k \
    --agent hv_noop \
    --model google/gemma-3-4b-pt \
    --hours 1 \
    --agent-config smoke \
    --run-branch "$RUN_BRANCH" \
    --job-name "${RUN_BRANCH}.ptb.test-batch.b1.dry-run.r1" \
    --experiment-name "_${RUN_BRANCH}_test_batch_b1_dry_run_r1" \
    --dry-run)"
printf '%s\n' "$DRY_RUN" | grep -F -- '--time=300' >/dev/null

POST_TRAIN_BENCH_SLURM_GPU_MODE=manual bash "$SCRIPT_DIR/submit.sh" \
    --eval gsm8k \
    --agent hv_noop \
    --model google/gemma-3-4b-pt \
    --hours 1 \
    --agent-config smoke \
    --run-branch "$RUN_BRANCH" \
    --job-name "${RUN_BRANCH}.ptb.test-batch.b1.manual.r1" \
    --experiment-name "_${RUN_BRANCH}_test_batch_b1_manual_r1" >/dev/null
grep -Fx -- '--exclusive' "$PTB_TEST_SBATCH_ARGS" >/dev/null
if grep -F -- '--gres=' "$PTB_TEST_SBATCH_ARGS" >/dev/null; then
    echo "manual GPU mode unexpectedly requested Slurm GRES" >&2
    exit 1
fi

if bash "$SCRIPT_DIR/submit.sh" \
    --eval gsm8k --agent hv_noop --model google/gemma-3-4b-pt \
    --hours 1 --agent-config smoke --experiment-name _missing_branch_test \
    --dry-run >/dev/null 2>&1; then
    echo "submission without branch ownership unexpectedly succeeded" >&2
    exit 1
fi

# The WMA policy checkout and history are mounted only into a separate sidecar
# container. The scientist profile must not receive their host paths.
grep -F -- '--bind "${WMA_CHECKOUT}:/opt/awm:ro"' "$REPO_ROOT/src/run_task.sh" >/dev/null
grep -F -- '--bind "${WMA_HISTORY}:/history:ro"' "$REPO_ROOT/src/run_task.sh" >/dev/null
grep -F -- '--bind "${JOB_DIR}/task:/session:ro"' "$REPO_ROOT/src/run_task.sh" >/dev/null
grep -F -- '"${JOB_DIR}/task/.wma/requests"' "$REPO_ROOT/src/run_task.sh" >/dev/null
grep -F -- 'python3 -m awm.wma.sidecar' "$REPO_ROOT/src/run_task.sh" >/dev/null
if grep -q 'POST_TRAIN_BENCH_WMA_' "$REPO_ROOT/agents/claude_vertex_high_awm/env_passthrough.txt"; then
    echo "private WMA sidecar settings leaked into the scientist environment" >&2
    exit 1
fi

echo "Slurm adapter tests passed."
