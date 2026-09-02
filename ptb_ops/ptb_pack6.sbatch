#!/bin/bash
#SBATCH --job-name=ptb-pack6
#SBATCH --partition=a3
# Superseded by ptb_pack.sbatch, but job 82648 was launched from a copy of this file, so
# it is still a live submit path. 11 of a3's 14 nodes are reserved -- see ptb_pack.sbatch.
#SBATCH --reservation=robtang-a3
#SBATCH --exclude=slurm2-a3nodeset0-[0-2]
# a3nodeset0-[0-2] are froilanchoi's. An empty `--reservation=` on the command line
# voids the line above and lands the job there -- see ptb_pack.sbatch, 89727/89809/89810.
# A command-line --exclude REPLACES this line, so it must carry these nodes too.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.out
#SBATCH --error=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.err
set -uo pipefail

# Six PostTrainBench cells on six GPUs of one exclusive a3 node.
#
# ptb_arm.py submits one whole node per cell, which is the honest shape when the
# benchmark's contract is one GPU and the node is the allocation unit. It is also
# the wrong shape here: a3 is 14 nodes, ten of them belong to other people, and a
# twelve-cell batch submitted one-per-node would run two at a time for nine hours.
# Six cells share a node instead, one GPU each. What makes that safe is device
# scoping, not luck:
#
#   - POST_TRAIN_BENCH_VISIBLE_GPUS is per cell, and run_task.sh's reap passes it
#     to `nvidia-smi -i`. Before that fix the reap killed every compute process on
#     the node, so cell 3 finishing would have shot cells 0-5. Do not run this
#     against a run_task.sh older than f4fd440.
#   - TMP_SUBDIR and JOB_DIR carry a uuidgen, so scratch does not collide.
#   - inspect_ai's vllm provider takes a free port when none is given, so six
#     concurrent servers do not fight over 8000. Cells are staggered anyway.
#   - TMP_HF_CACHE is a fixed path but it is the *inside*-container mount point,
#     and each cell has its own mount namespace.
#
# Arms are interleaved across GPUs rather than packed per node, so that "which arm"
# is not the same question as "which node".
#
# Usage:  sbatch ptb_pack6.sbatch <arm_gpu0> <arm_gpu1> ... <arm_gpu5>
ARMS=("$@")
[ "${#ARMS[@]}" -eq 6 ] || { echo "FATAL: need exactly 6 arm names, got ${#ARMS[@]}" >&2; exit 1; }

TASK="${PTB_TASK:-gsm8k}"
MODEL_TO_TRAIN="${PTB_MODEL:-Qwen/Qwen3-1.7B-Base}"
NUM_HOURS="${PTB_NUM_HOURS:-1}"
AGENT_CONFIG="${PTB_AGENT_CONFIG:-claude-opus-5}"
STAGGER="${PTB_STAGGER:-90}"

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
echo "arms=${ARMS[*]} task=${TASK} hours=${NUM_HOURS} agent_config=${AGENT_CONFIG}"

export PATH="/rmeng_data/robtang/tools/apt-root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)" >&2; exit 1; }
echo "apptainer=$(apptainer --version)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Pinned, unlike the ten-hour mainline pair: two arms that update the CLI at job
# start can end up on two different builds, and this batch exists to compare the
# payloads and nothing else.
export POST_TRAIN_BENCH_SKIP_CLI_UPDATE="1"
echo "skip_cli_update=1"

cd "/rmeng_data/robtang/PostTrainBench"
export REPO_ROOT="$(pwd)"
source src/commit_utils/set_env_vars.sh

# huggingface_hub ranks HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > "$HF_HOME/hub" and
# this cluster exports the middle one globally, so an --env HF_HOME passed to a
# non---cleanenv exec loses and the download lands on the 64 MiB writable tmpfs.
unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE
echo "hf_home=${HF_HOME:-<unset>} hub_cache_overrides=cleared"

SCRATCH="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
mkdir -p "$SCRATCH" || { echo "FATAL: cannot create $SCRATCH on $(hostname)" >&2; exit 1; }
[ -w "$SCRATCH" ] || { echo "FATAL: $SCRATCH is not writable on $(hostname)" >&2; exit 1; }
echo "scratch=$SCRATCH free=$(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"

LOGDIR="/rmeng_data/robtang/PostTrainBench/slurm_logs"
declare -a PIDS=()
for gpu in 0 1 2 3 4 5; do
    arm="${ARMS[$gpu]}"
    cell="${SLURM_JOB_ID}_g${gpu}"
    log="${LOGDIR}/ptb-pack6-${SLURM_JOB_ID}-g${gpu}-${arm}.log"
    (
        sleep $(( gpu * STAGGER ))
        export NUM_GPUS=1
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export POST_TRAIN_BENCH_VISIBLE_GPUS="${gpu}"
        echo "cell start gpu=${gpu} arm=${arm} cell=${cell} at $(date --iso-8601=seconds)"
        bash src/run_task.sh "${TASK}" "${arm}" "${MODEL_TO_TRAIN}" "${cell}" "${NUM_HOURS}" "${AGENT_CONFIG}" 1
    ) > "${log}" 2>&1 &
    PIDS+=("$!")
    echo "launched gpu=${gpu} arm=${arm} cell=${cell} pid=${PIDS[$gpu]} log=${log}"
done

rc_all=0
for gpu in 0 1 2 3 4 5; do
    wait "${PIDS[$gpu]}"
    rc=$?
    echo "cell_exit gpu=${gpu} arm=${ARMS[$gpu]} cell=${SLURM_JOB_ID}_g${gpu} rc=${rc}"
    [ "$rc" -eq 0 ] || rc_all=1
done

echo "pack_exit=${rc_all} end=$(date --iso-8601=seconds)"
exit "${rc_all}"
