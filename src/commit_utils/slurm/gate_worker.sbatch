#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

MODE="${1:?mode required}"
GATE_DIR="${2:?gate output directory required}"
HOLD_SECONDS="${3:-60}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
source src/commit_utils/set_env_vars.sh

if [ -n "${POST_TRAIN_BENCH_APPTAINER_BIN:-}" ]; then
    export PATH="$(dirname "$POST_TRAIN_BENCH_APPTAINER_BIN"):$PATH"
fi
if [ -n "${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$GATE_DIR"
ALLOCATED="${SLURM_JOB_GPUS:?Slurm did not provide SLURM_JOB_GPUS}"
export POST_TRAIN_BENCH_VISIBLE_GPUS="$ALLOCATED"
export POST_TRAIN_BENCH_SLURM_GPU_MODE=gres
IMAGE="${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif"
UUID="$(nvidia-smi -i "$ALLOCATED" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
printf '%s\n' "$UUID" > "$GATE_DIR/${SLURM_JOB_ID}.uuid"
{
    echo "job_id=${SLURM_JOB_ID}"
    echo "node=${SLURMD_NODENAME:-$(hostname)}"
    echo "job_gpus=${ALLOCATED}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    echo "uuid=${UUID}"
    echo "cpus=${SLURM_CPUS_PER_TASK:-}"
    echo "mem_mb=${SLURM_MEM_PER_NODE:-}"
} > "$GATE_DIR/${SLURM_JOB_ID}.allocation"

validate_device_cgroup() {
    local allocated_id="${ALLOCATED%%,*}" device index
    for index in $(seq 0 7); do
        device="/dev/nvidia${index}"
        if [ "$index" = "$allocated_id" ]; then
            if ! python3 - "$device" <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
os.close(fd)
PY
            then
                echo "ERROR: allocated device is not openable: $device" >&2
                return 1
            fi
        elif python3 - "$device" <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
os.close(fd)
PY
        then
            echo "ERROR: unallocated device is openable: $device" >&2
            return 1
        fi
    done
}

container_cuda_check() {
    apptainer exec --nv -c --cleanenv \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
        --env "PTB_EXPECTED_UUID=${UUID}" \
        --writable-tmpfs "$IMAGE" \
        python -c 'import os, torch; assert torch.cuda.device_count() == 1; assert "H100" in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0), os.environ["PTB_EXPECTED_UUID"])'
}

case "$MODE" in
    canary)
        validate_device_cgroup
        container_cuda_check
        sleep "$HOLD_SECONDS"
        ;;
    survivor)
        apptainer exec --nv -c --cleanenv \
            --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
            --bind "$GATE_DIR:/gate" \
            --writable-tmpfs "$IMAGE" \
            python - "$HOLD_SECONDS" "$SLURM_JOB_ID" <<'PY'
import pathlib, sys, time, torch
seconds, job_id = int(sys.argv[1]), sys.argv[2]
x = torch.ones((1024, 1024), device="cuda")
heartbeat = pathlib.Path("/gate/survivor.heartbeat")
for tick in range(seconds):
    x.add_(1)
    heartbeat.write_text(f"{job_id} {tick}\n")
    time.sleep(1)
PY
        ;;
    reaper)
        for _ in $(seq 1 60); do
            [ -s "$GATE_DIR/survivor.heartbeat" ] && break
            sleep 1
        done
        [ -s "$GATE_DIR/survivor.heartbeat" ] || { echo "ERROR: survivor heartbeat absent" >&2; exit 1; }
        before="$(cat "$GATE_DIR/survivor.heartbeat")"
        apptainer exec --nv -c --cleanenv \
            --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
            --writable-tmpfs "$IMAGE" \
            python -c 'import time, torch; x=torch.ones((1024,1024), device="cuda"); time.sleep(300)' &
        own_gpu_pid=$!
        sleep 10
        source src/utils/gpu_reap.sh
        ptb_reap_allocated_gpu_processes
        wait "$own_gpu_pid" 2>/dev/null || true
        apptainer exec --nv -c --cleanenv \
            --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
            --writable-tmpfs "$IMAGE" \
            python -c 'import torch; x=torch.ones((1024,1024), device="cuda"); x.add_(1); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), float(x[0,0]))' \
            > "$GATE_DIR/reaper-eval-smoke.txt"
        sleep 15
        after="$(cat "$GATE_DIR/survivor.heartbeat")"
        if [ "$before" = "$after" ]; then
            echo "ERROR: survivor heartbeat stopped during cleanup" >&2
            exit 1
        fi
        printf '%s\n%s\n' "$before" "$after" > "$GATE_DIR/reaper-survivor-proof.txt"
        ;;
    *) echo "ERROR: unknown gate worker mode: $MODE" >&2; exit 2 ;;
esac
