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

OUTPUT="$(bash "$SCRIPT_DIR/submit.sh" \
    --eval gsm8k \
    --agent hv_noop \
    --model google/gemma-3-4b-pt \
    --hours 10 \
    --agent-config smoke \
    --experiment-name _slurm_test \
    --preflight-only)"

[ "$OUTPUT" = "Submitted Slurm job 12345" ]
grep -Fx -- '--partition=test-gpu' "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- '--nodelist=test-node-[0-3]' "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- '--reservation=test-reservation' "$PTB_TEST_SBATCH_ARGS" >/dev/null
grep -Fx -- '--gres=gpu:1' "$PTB_TEST_SBATCH_ARGS" >/dev/null
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
    --dry-run)"
printf '%s\n' "$DRY_RUN" | grep -F -- '--time=300' >/dev/null

POST_TRAIN_BENCH_SLURM_GPU_MODE=manual bash "$SCRIPT_DIR/submit.sh" \
    --eval gsm8k \
    --agent hv_noop \
    --model google/gemma-3-4b-pt \
    --hours 1 \
    --agent-config smoke >/dev/null
grep -Fx -- '--exclusive' "$PTB_TEST_SBATCH_ARGS" >/dev/null
if grep -F -- '--gres=' "$PTB_TEST_SBATCH_ARGS" >/dev/null; then
    echo "manual GPU mode unexpectedly requested Slurm GRES" >&2
    exit 1
fi

echo "Slurm adapter tests passed."
