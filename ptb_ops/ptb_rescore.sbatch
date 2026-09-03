#!/bin/bash
#SBATCH --job-name=ptb-rescore
#SBATCH --partition=a3
# 11 of a3's 14 nodes are held by reservation robtang-a3; without this line the job is
# unschedulable rather than merely queued. See ptb_pack.sbatch for why it cannot be an
# environment variable.
#SBATCH --reservation=robtang-a3
#SBATCH --exclude=slurm2-a3nodeset0-[0-2]
# a3nodeset0-[0-2] are froilanchoi's. An empty `--reservation=` on the command line
# voids the line above and lands the job there -- see ptb_pack.sbatch, 89727/89809/89810.
# A command-line --exclude REPLACES this line, so it must carry these nodes too.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.out
#SBATCH --error=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.err
set -uo pipefail

# Score a final_model/ an agent already produced, with the fixed scorer.
#
# Runs whose agent trained a model but whose grading step died on the two container-cache
# faults fixed in d22dfc7. The weights are on disk and the grader is frozen, so the score
# is recoverable. What is *not* recoverable is the agent's own reading of it -- it never
# got one -- so nothing downstream of that (did it select between candidates? did it stop
# early because a screen looked bad?) may be inferred from this number. Read the run's
# logs for that.
#
# $1 = eval directory: the run's result dir, the one that contains final_model/
#
# The whole directory is bound in, not just the weights. run_baseline.sh cannot do this
# job -- it binds only RESULT_DIR and REPO_ROOT, and an agent's final_model lives under
# neither, so vLLM starts against a path that does not exist inside the container and
# fails as "Server process exited unexpectedly with code 1", which is the same string the
# cache faults produced. Two rescores were burned finding that out.
EVAL_DIR="${1:?usage: sbatch ptb_rescore.sbatch <eval_dir>}"
TASK="${2:-gsm8k}"
EVAL_DIR="${EVAL_DIR%/}"
MODEL_PATH="${EVAL_DIR}/final_model"

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
echo "eval_dir=${EVAL_DIR}"
echo "task=${TASK}"
[ -f "${MODEL_PATH}/config.json" ] || { echo "FATAL: no config.json under ${MODEL_PATH}" >&2; exit 1; }

export PATH="/rmeng_data/robtang/tools/apt-root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)" >&2; exit 1; }

export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export REPO_ROOT="/rmeng_data/robtang/PostTrainBench"
cd "${REPO_ROOT}"
source src/commit_utils/set_env_vars.sh

TMP_SUBDIR="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}/ptb_rescore_${SLURM_JOB_ID}"
HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
TMP_HF_CACHE="/tmp/hf_cache_rescore"
# Bytecode goes to node-local scratch, not to the container root. Same 64 MiB
# --writable-tmpfs overlay and the same reasoning as run_task.sh's SCORER_PYCACHE:
# nothing in vllm_debug.sif is precompiled (uv installs --no-cache and does not
# byte-compile; 24602 .py against 886 .pyc in dist-packages), one
# `import transformers, torch, vllm` writes 41 MiB of .pyc, and ENOSPC part-way
# through leaves a truncated .pyc that kills every later import. Deliberately NOT
# under ${TMP_HF_CACHE}: that is the fuse-overlayfs merged mount, torn down at the
# bottom of this file. Identity-bound rather than redirected, so the path is the
# same inside and out and check_pycache.sh's message names a path that exists.
PYCACHE="${TMP_SUBDIR}/pycache"
mkdir -p "${HF_MERGED}" "${TMP_SUBDIR}/upper_huggingface" "${TMP_SUBDIR}/fuse_workdir" "${PYCACHE}"
fuse-overlayfs -o "lowerdir=${HF_HOME},upperdir=${TMP_SUBDIR}/upper_huggingface,workdir=${TMP_SUBDIR}/fuse_workdir" "${HF_MERGED}" \
  || { echo "FATAL: overlay failed" >&2; exit 1; }

# The cluster exports HUGGINGFACE_HUB_CACHE globally and this exec is not --cleanenv, so
# all three are named below rather than trusting HF_HOME, which is the lowest-priority of
# the three. XDG_CACHE_HOME is bound rather than redirected: vLLM's VLLM_CACHE_ROOT and
# torch-inductor's triton cache both live under it, and on the 64 MiB --writable-tmpfs
# container root the second one raises OSError: [Errno 28] and kills EngineCore.
CACHE_BIND=()
[ -n "${XDG_CACHE_HOME:-}" ] && [ -d "${XDG_CACHE_HOME}" ] && CACHE_BIND=(--bind "${XDG_CACHE_HOME}:${XDG_CACHE_HOME}")

# Unset by default; the full argument for why it is not pinned on is in
# src/run_task.sh next to POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND. Short version:
# 89810_g5 lost 7 of 43 grader invocations to a CUDA illegal-memory-access until
# it set TRITON_ATTN by hand, but nobody has measured what TRITON_ATTN costs per
# read at these shapes, and a uniform throughput tax is the worse of the two
# mistakes. A rescore is the cheapest place to measure that delta.
ATTN_BACKEND_ENV=()
[ -n "${POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND:-}" ] && \
    ATTN_BACKEND_ENV=(--env "VLLM_ATTENTION_BACKEND=${POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND}")

# check_pycache.sh runs inside this exec, ahead of python, so it sees the bind
# list that is actually in force; a preflight exec of its own would carry a
# second copy of the bind and could not notice this one losing it. $0 carries the
# repo root and the rest are positional, so nothing in the single-quoted script
# is expanded here and `exec` leaves python as the process whose rc is read.
apptainer exec \
    --nv \
    --env HF_HOME="${TMP_HF_CACHE}" \
    --env HF_HUB_CACHE="${TMP_HF_CACHE}/hub" \
    --env HUGGINGFACE_HUB_CACHE="${TMP_HF_CACHE}/hub" \
    --env VLLM_API_KEY="inspectai" \
    "${ATTN_BACKEND_ENV[@]}" \
    "${CACHE_BIND[@]}" \
    --writable-tmpfs \
    --bind "${EVAL_DIR}:${EVAL_DIR}" \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
    --bind "${PYCACHE}:${PYCACHE}" \
    --env PYTHONPYCACHEPREFIX="${PYCACHE}" \
    --pwd "${REPO_ROOT}/src/eval/tasks/${TASK}" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif" \
    bash -c 'bash "$0/src/utils/check_pycache.sh" || exit 1; exec python "$@"' \
        "${REPO_ROOT}" evaluate.py \
        --model-path "${MODEL_PATH}" \
        --templates-dir ../../../../src/eval/templates \
        --limit -1 \
        --json-output-file "${EVAL_DIR}/metrics.json" > "${EVAL_DIR}/final_eval_rescore_${SLURM_JOB_ID}.txt" 2>&1
rc=$?

fusermount -u "${HF_MERGED}" 2>/dev/null; rm -rf "${TMP_SUBDIR}"
echo "score_exit=$rc"
echo "--- metrics.json ---"; cat "${EVAL_DIR}/metrics.json" 2>&1
echo "end=$(date --iso-8601=seconds)"
exit $rc
