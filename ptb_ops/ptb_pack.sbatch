#!/bin/bash
#SBATCH --job-name=ptb-pack
#SBATCH --partition=a3
# 11 of a3's 14 nodes are held by reservation robtang-a3, the other 3 by froilanchoi's
# UNLIMITED jobs, so a pack submitted without the reservation never starts -- it waits on
# three nodes that never free and squeue calls it (Resources). Not settable from the
# environment: SBATCH_RESERVATION applies to every partition, and an airs/eval job that
# inherits it dies at submit with "allocation failure".
#SBATCH --reservation=robtang-a3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.out
#SBATCH --error=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.err
set -uo pipefail

# N PostTrainBench cells on N GPUs of one exclusive a3 node, 1 <= N <= 8.
#
# ptb_arm.py submits a whole exclusive node per cell, which is the honest shape for
# a one-GPU contract and the wrong shape for a batch: a3 is fourteen nodes, ten of
# them belong to other people, and a twenty-four-cell batch submitted one-per-node
# would trickle two at a time for a day. A node is the allocation unit here whether
# or not the job uses it, so leaving seven of eight cards idle is not modesty, it is
# waste. Supersedes ptb_pack6.sbatch, which hardcoded six.
#
# What makes packing safe, in the order the hazards were found:
#
#   - The eval reap is scoped with `nvidia-smi -i $POST_TRAIN_BENCH_VISIBLE_GPUS`
#     (f4fd440). Before that the first cell to finish killed every compute process
#     on the node, i.e. all its neighbours.
#   - CUDA_VISIBLE_DEVICES inside the agent sandbox is the *container* numbering,
#     not the host's, whenever POST_TRAIN_BENCH_ISOLATE_GPUS=1 renumbers the cards
#     from zero. GPU 0 is the one index where the two agree, so single-cell runs
#     never saw it and five of the first pack's six cells died in four minutes with
#     "CUDA is not available". Do not run this against a run_task.sh that predates
#     that fix -- everything on GPU 0 will pass and everything else will not.
#   - TMP_SUBDIR and JOB_DIR carry a uuidgen, so scratch does not collide, and
#     TMP_HF_CACHE is fixed but is the inside-container mount point.
#   - inspect_ai's vllm provider takes a free port when none is given, so N
#     concurrent servers do not fight over 8000. Cells are staggered anyway.
#
# Arms are given per GPU, so interleave them: "which arm" should not be the same
# question as "which node" or "which GPU index".
#
# Usage:  sbatch ptb_pack.sbatch <arm_gpu0> [<arm_gpu1> ...]     # up to 8
ARMS=("$@")
N="${#ARMS[@]}"
{ [ "$N" -ge 1 ] && [ "$N" -le 8 ]; } || { echo "FATAL: need 1..8 arm names, got ${N}" >&2; exit 1; }

TASK="${PTB_TASK:-gsm8k}"
MODEL_TO_TRAIN="${PTB_MODEL:-Qwen/Qwen3-1.7B-Base}"
NUM_HOURS="${PTB_NUM_HOURS:-1}"
AGENT_CONFIG="${PTB_AGENT_CONFIG:-claude-opus-5}"
STAGGER="${PTB_STAGGER:-60}"

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
echo "cells=${N} arms=${ARMS[*]} task=${TASK} hours=${NUM_HOURS} agent_config=${AGENT_CONFIG}"

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

# Echo the two settings that decide whether a packed cell can work at all, so a
# post-mortem does not have to reconstruct them from .env as it was hours later.
echo "isolate_gpus=${POST_TRAIN_BENCH_ISOLATE_GPUS:-<unset>} eval_gpu_reap=${POST_TRAIN_BENCH_EVAL_GPU_REAP:-<unset>}"
echo "run_task_sha=$(git rev-parse --short HEAD 2>/dev/null) run_task_inode=$(ls -i src/run_task.sh | awk '{print $1}')"

# Echo, then enforce, the identity of the payloads being compared.
#
# The b4 effort-tier batch spent eight a3-node-hours measuring nothing, because
# claude_autor_{ctl,pt,fd,fx}/solve.sh were all the SAME 2292 bytes (sha256
# 4c0a7f54...). Four labels, one configuration. Their success rates came out
# 0/12, 2/11, 2/8 and 3/4, and ctl-vs-fd reads Fisher p=0.0071 -- a p of 0.007
# between two byte-identical scripts. Nothing downstream could detect this:
# ptb-results is keyed on the arm NAME, so the duplicate looks like a real arm
# forever. The only place that can see it is here, before the cells start.
#
# agents/ is not tracked in git (tools/ptb_setup.py installs these), so there is
# no commit to diff against either -- the hash has to be taken at run time.
declare -A _ARM_SHA=()
_dupe=""
for _a in "${ARMS[@]}"; do
    _f="agents/${_a}/solve.sh"
    [ -f "$_f" ] || { echo "FATAL: no such arm: ${_f}" >&2; exit 1; }
    _h="$(sha256sum "$_f" | cut -c1-16)"
    echo "arm_payload ${_a} sha=${_h} bytes=$(stat -c%s "$_f") mtime=$(date -r "$_f" --iso-8601=seconds)"
    if [ -n "${_ARM_SHA[$_h]:-}" ] && [ "${_ARM_SHA[$_h]}" != "$_a" ]; then
        _dupe="${_dupe} ${_ARM_SHA[$_h]}==${_a}"
    fi
    _ARM_SHA[$_h]="$_a"
done
if [ -n "$_dupe" ]; then
    echo "FATAL: two arms in this pack have byte-identical solve.sh:${_dupe}" >&2
    echo "       They cannot be compared -- any difference between them is noise" >&2
    echo "       wearing two labels. Fix the payload, or drop one arm. Override" >&2
    echo "       with PTB_ALLOW_DUPLICATE_ARMS=1 only to measure the null." >&2
    [ "${PTB_ALLOW_DUPLICATE_ARMS:-0}" = "1" ] || exit 1
    echo "       PTB_ALLOW_DUPLICATE_ARMS=1 -- continuing as a null calibration." >&2
fi

# A pack is the only unit inside which conditions are held fixed: same node, same
# hour, same CLI build, same HF cache. In b4 the control ran on 08-30 and every
# treatment arm on 08-31, and the same script's success rate moved 2/23 -> 5/12
# (p=0.033) across those two days. So a cross-pack contrast is confounded with
# the day by construction, and the control has to be IN the pack.
CONTROL_ARM="${PTB_CONTROL_ARM:-claude_autor_ctl}"
case " ${ARMS[*]} " in
    *" ${CONTROL_ARM} "*) echo "control_arm=${CONTROL_ARM} in_pack=yes" ;;
    *)  echo "FATAL: pack has no control arm (${CONTROL_ARM}); arms=${ARMS[*]}" >&2
        echo "       Comparing these cells against a control from another job means" >&2
        echo "       comparing them against another day. Add ${CONTROL_ARM} to the" >&2
        echo "       pack, set PTB_CONTROL_ARM, or set PTB_NO_CONTROL=1 if this pack" >&2
        echo "       is deliberately not a comparison." >&2
        [ "${PTB_NO_CONTROL:-0}" = "1" ] || exit 1
        echo "       PTB_NO_CONTROL=1 -- continuing without an in-pack control." >&2 ;;
esac

SCRATCH="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
mkdir -p "$SCRATCH" || { echo "FATAL: cannot create $SCRATCH on $(hostname)" >&2; exit 1; }
[ -w "$SCRATCH" ] || { echo "FATAL: $SCRATCH is not writable on $(hostname)" >&2; exit 1; }
echo "scratch=$SCRATCH free=$(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"

# Run the driver from node-local disk, because bash reads a script incrementally and a
# `git commit` in this checkout replaces the inode under a running job. See
# src/commit_utils/pin_src_locally.sh -- that is how 82165 and 82166 lost ten hours each.
# One pin per job, shared by every cell: they all read it, none of them write it.
RUN_TASK="$(bash src/commit_utils/pin_src_locally.sh "$REPO_ROOT" "${SCRATCH}/ptb-src-${SLURM_JOB_ID}")"
echo "run_task=${RUN_TASK}"

LOGDIR="/rmeng_data/robtang/PostTrainBench/slurm_logs"
declare -a PIDS=()
for (( gpu=0; gpu<N; gpu++ )); do
    arm="${ARMS[$gpu]}"
    cell="${SLURM_JOB_ID}_g${gpu}"
    log="${LOGDIR}/ptb-pack-${SLURM_JOB_ID}-g${gpu}-${arm}.log"
    (
        sleep $(( gpu * STAGGER ))
        export NUM_GPUS=1
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export POST_TRAIN_BENCH_VISIBLE_GPUS="${gpu}"
        echo "cell start gpu=${gpu} arm=${arm} cell=${cell} at $(date --iso-8601=seconds)"
        bash "${RUN_TASK}" "${TASK}" "${arm}" "${MODEL_TO_TRAIN}" "${cell}" "${NUM_HOURS}" "${AGENT_CONFIG}" 1
    ) > "${log}" 2>&1 &
    PIDS+=("$!")
    echo "launched gpu=${gpu} arm=${arm} cell=${cell} pid=${PIDS[$gpu]} log=${log}"
done

rc_all=0
for (( gpu=0; gpu<N; gpu++ )); do
    wait "${PIDS[$gpu]}"
    rc=$?
    echo "cell_exit gpu=${gpu} arm=${ARMS[$gpu]} cell=${SLURM_JOB_ID}_g${gpu} rc=${rc}"
    [ "$rc" -eq 0 ] || rc_all=1
done

echo "pack_exit=${rc_all} end=$(date --iso-8601=seconds)"
exit "${rc_all}"
