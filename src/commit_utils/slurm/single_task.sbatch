#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@120

set -euo pipefail

if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <eval> <agent> <model> <hours> <agent-config> <num-gpus> <experiment-name>" >&2
    exit 2
fi

EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
NUM_HOURS="$4"
AGENT_CONFIG="$5"
NUM_GPUS="$6"
EXPERIMENT_NAME="$7"

if ! [[ "$NUM_HOURS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: hours must be a positive integer" >&2
    exit 2
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: num-gpus must be a positive integer" >&2
    exit 2
fi

REPO_ROOT="${POST_TRAIN_BENCH_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -r "${REPO_ROOT}/src/run_task.sh" ]; then
    echo "ERROR: PostTrainBench repo not found at: $REPO_ROOT" >&2
    exit 1
fi
cd "$REPO_ROOT"

source src/commit_utils/set_env_vars.sh

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER:-}" != "slurm" ]; then
    echo "ERROR: POST_TRAIN_BENCH_JOB_SCHEDULER must be slurm" >&2
    exit 1
fi

if [ -n "${POST_TRAIN_BENCH_APPTAINER_BIN:-}" ]; then
    if [ ! -x "$POST_TRAIN_BENCH_APPTAINER_BIN" ]; then
        echo "ERROR: configured apptainer is not executable: $POST_TRAIN_BENCH_APPTAINER_BIN" >&2
        exit 1
    fi
    export PATH="$(dirname "$POST_TRAIN_BENCH_APPTAINER_BIN"):${PATH}"
fi
if [ -n "${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RUN_ID="${RUN_ID}_${SLURM_ARRAY_TASK_ID}"
fi

SCRATCH_BASE="${POST_TRAIN_BENCH_SLURM_SCRATCH_BASE:-/tmp/posttrainbench}"
if [[ "$SCRATCH_BASE" != /* ]] || [ "$SCRATCH_BASE" = "/" ]; then
    echo "ERROR: POST_TRAIN_BENCH_SLURM_SCRATCH_BASE must be a safe absolute path" >&2
    exit 1
fi

JOB_SCRATCH="${SCRATCH_BASE%/}/${USER}/${RUN_ID}"
EXPECTED_PREFIX="${SCRATCH_BASE%/}/${USER}/"
if [[ "$JOB_SCRATCH" != "$EXPECTED_PREFIX"* ]] || [ "$JOB_SCRATCH" = "${EXPECTED_PREFIX%/}" ]; then
    echo "ERROR: refused unsafe scratch path: $JOB_SCRATCH" >&2
    exit 1
fi

umask 077
mkdir -p "$JOB_SCRATCH/tmp"
export POST_TRAIN_BENCH_SCRATCH_DIR="$JOB_SCRATCH"
export TMPDIR="$JOB_SCRATCH/tmp"
export APPTAINER_CACHEDIR="$JOB_SCRATCH/apptainer-cache"
export APPTAINER_TMPDIR="$JOB_SCRATCH/apptainer-tmp"
export POST_TRAIN_BENCH_EXPERIMENT_NAME="$EXPERIMENT_NAME"

cleanup() {
    local status=$?
    if [ "${POST_TRAIN_BENCH_SLURM_KEEP_SCRATCH:-0}" = "1" ]; then
        echo "Keeping Slurm scratch: $JOB_SCRATCH" >&2
        return "$status"
    fi

    while IFS= read -r -d '' mount_dir; do
        fusermount -uz "$mount_dir" 2>/dev/null || true
    done < <(find "$JOB_SCRATCH" -type d -name merged_huggingface -print0 2>/dev/null)
    rm -rf -- "$JOB_SCRATCH"
    return "$status"
}
trap cleanup EXIT

GPU_MODE="${POST_TRAIN_BENCH_SLURM_GPU_MODE:-gres}"
case "$GPU_MODE" in
    gres)
        if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
            echo "ERROR: Slurm GRES mode did not set CUDA_VISIBLE_DEVICES" >&2
            exit 1
        fi
        # Slurm normally exposes physical indices in SLURM_JOB_GPUS. Keep the
        # value for host-side, allocation-scoped nvidia-smi cleanup; the
        # container itself inherits Slurm's device cgroup and CUDA visibility.
        export POST_TRAIN_BENCH_VISIBLE_GPUS="${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES}}"
        ;;
    manual)
        GPU_IDS="$(seq -s, 0 $((NUM_GPUS - 1)))"
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        export NVIDIA_VISIBLE_DEVICES="$GPU_IDS"
        export APPTAINERENV_CUDA_VISIBLE_DEVICES="$GPU_IDS"
        export POST_TRAIN_BENCH_VISIBLE_GPUS="$GPU_IDS"
        export POST_TRAIN_BENCH_ISOLATE_GPUS="1"
        ;;
    *)
        echo "ERROR: unsupported POST_TRAIN_BENCH_SLURM_GPU_MODE=$GPU_MODE" >&2
        exit 1
        ;;
esac

echo "Slurm job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU mode: ${GPU_MODE}; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Scratch: ${JOB_SCRATCH}"

bash src/commit_utils/slurm/preflight.sh "$NUM_GPUS" "$EVALUATION_TASK" "$AGENT"

if [ "${POST_TRAIN_BENCH_SLURM_PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "Preflight-only Slurm job completed successfully."
    exit 0
fi

if [ "${POST_TRAIN_BENCH_SLURM_RUNTIME_SMOKE:-0}" = "1" ]; then
    RUNTIME_NVCCLI_ARGS=()
    if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-0}" = "1" ]; then
        RUNTIME_NVCCLI_ARGS=(--nvccli)
    fi
    apptainer exec \
        --nv \
        "${RUNTIME_NVCCLI_ARGS[@]}" \
        -c \
        --cleanenv \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
        --env "PTB_EXPECTED_GPUS=${NUM_GPUS}" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
        python -c 'import os, torch; expected=int(os.environ["PTB_EXPECTED_GPUS"]); count=torch.cuda.device_count(); names=[torch.cuda.get_device_name(i) for i in range(count)]; print(f"container GPUs: {count} {names}"); assert count == expected; assert all("H100" in name for name in names)'
    if [ -n "${POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD:-}" ]; then
        bash src/commit_utils/slurm/context_probe.sh \
            "$AGENT" "$AGENT_CONFIG" \
            "$POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD" \
            "$JOB_SCRATCH/context-probe-home"
    fi
    echo "Runtime smoke completed successfully."
    exit 0
fi

bash src/run_task.sh \
    "$EVALUATION_TASK" \
    "$AGENT" \
    "$MODEL_TO_TRAIN" \
    "$RUN_ID" \
    "$NUM_HOURS" \
    "$AGENT_CONFIG" \
    "$NUM_GPUS"
