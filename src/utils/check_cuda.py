#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import torch

def get_gpu_processes(gpu_index):
    """Get processes running on a specific GPU using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=" + str(gpu_index), 
             "--query-compute-apps=pid,used_memory", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                pid, mem = line.split(',')
                processes.append((int(pid.strip()), float(mem.strip())))
        return processes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def check_h100():
    expected_gpus = int(os.environ.get("NUM_GPUS", "1"))
    # hv-patches: upstream hard-codes the substring "H100" and additionally
    # requires the GPU to be idle. Neither holds on our hardware
    # (4x NVIDIA RTX 6000 Ada Generation, often partly occupied), so both are
    # now configurable and default to upstream behaviour:
    #   POST_TRAIN_BENCH_GPU_NAME_MATCH  substring the device name must contain
    #                                    (comma-separated alternatives; empty
    #                                    string disables the name check)
    #   POST_TRAIN_BENCH_ALLOW_BUSY_GPU  "1" -> warn instead of failing when
    #                                    other processes hold the GPU
    name_match_raw = os.environ.get("POST_TRAIN_BENCH_GPU_NAME_MATCH", "H100")
    wanted_names = [s.strip() for s in name_match_raw.split(",") if s.strip()]
    allow_busy = os.environ.get("POST_TRAIN_BENCH_ALLOW_BUSY_GPU", "0") == "1"

    if not torch.cuda.is_available():
        print("❌ CUDA is not available")
        return False

    device_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {device_count} device(s) (expected {expected_gpus})")
    if device_count != expected_gpus:
        print(f"❌ Expected {expected_gpus} GPU(s), got {device_count}")
        return False

    match_found = not wanted_names  # empty allowlist -> name check disabled
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {name} ({props.total_memory / 1e9:.1f} GB)")

        if wanted_names and not any(w in name for w in wanted_names):
            continue
        match_found = True

        # Check for running processes on this GPU
        processes = get_gpu_processes(i)
        if processes is None:
            print(f"  ⚠ Could not check processes on GPU {i} (nvidia-smi failed)")
        elif processes:
            print(f"  {'⚠' if allow_busy else '❌'} GPU {i} has {len(processes)} process(es) running:")
            for pid, mem in processes:
                print(f"      PID {pid}: {mem:.1f} MiB")
            if not allow_busy:
                return False
        else:
            print(f"  ✓ GPU {i} is idle")

    if match_found:
        expected_desc = "/".join(wanted_names) if wanted_names else "any GPU"
        print(f"✓ {expected_desc} detected ({device_count} GPU(s))")
    else:
        print(f"❌ No GPU matching {wanted_names} found")
        return False

    return True

if __name__ == "__main__":
    cuda_available = check_h100()
    if not cuda_available:
        Path("cuda_not_available").touch()

    import sys
    sys.exit(0 if cuda_available else 1)
