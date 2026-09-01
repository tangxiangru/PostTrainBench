#!/bin/bash

# Kill stale compute processes only on GPUs assigned to this allocation and
# only when they belong to the current Unix user.
ptb_reap_allocated_gpu_processes() {
    local gpu_selector="${POST_TRAIN_BENCH_VISIBLE_GPUS:-${SLURM_JOB_GPUS:-}}"
    local query_output="" pid="" owner=""
    local -a device_args=()

    case "${POST_TRAIN_BENCH_EVAL_GPU_REAP:-own}" in
        none) return 0 ;;
        own) ;;
        *) echo "ERROR: POST_TRAIN_BENCH_EVAL_GPU_REAP must be own or none" >&2; return 1 ;;
    esac
    if [ -n "$gpu_selector" ]; then
        # Under Slurm GRES, the devices cgroup exposes only this job's physical
        # GPU and NVML remaps it to logical index 0.  SLURM_JOB_GPUS keeps the
        # physical index, so passing it to ``nvidia-smi -i`` misses processes
        # whenever the allocation is not physical GPU 0.  Query every
        # cgroup-visible GPU instead; the cgroup itself is the scope boundary.
        if [ "${POST_TRAIN_BENCH_SLURM_GPU_MODE:-}" != "gres" ]; then
            device_args=(-i "$gpu_selector")
        fi
    elif [ "${POST_TRAIN_BENCH_SLURM_GPU_MODE:-}" = "gres" ]; then
        echo "ERROR: refusing GPU cleanup without an allocation-scoped selector" >&2
        return 1
    fi

    query_output="$(nvidia-smi "${device_args[@]}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    while IFS= read -r pid; do
        pid="${pid//[[:space:]]/}"
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        owner="$(ps -o user= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
        if [ "$owner" = "${USER}" ]; then
            echo "Reaping prior evaluation GPU process pid=${pid} gpu=${gpu_selector:-allocation}"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done <<< "$query_output"
}
