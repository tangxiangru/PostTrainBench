#!/bin/bash
set -uo pipefail

# ptb_rescore.sbatch, re-cut to run as an `srun --overlap` step on a node this user
# already holds, instead of queueing behind it.
#
# Why this exists: a3 is EXCLUSIVE, so an agent job holds all 8 H100s of its node for
# its whole wall clock whether or not it is using them. Sampled twice 75 s apart, the
# eleven nodes of reservation robtang-a3 were running 4 of 8 cards on average and one
# node was running none. The rescores were queued behind ~35 h of that. A rescore is a
# single-GPU vLLM eval of weights that already exist, so it fits in the gap.
#
# Safe to overlap *because of what it does not call*. `scripts/rerun_eval_n_times.sh`
# and `src/run_task.sh` both end with an `nvidia-smi --query-compute-apps=pid | xargs
# kill -9`; run_task.sh scopes it to its own device, rerun_eval_n_times.sh does not scope
# it at all. Either one, run as an overlap step, would kill the host job's own training.
# This path reaches neither: it goes straight to `apptainer exec python evaluate.py`.
# Do NOT overlap ptb_pack.sbatch, which does reach run_task.sh.
#
# $1 = eval directory (the run's result dir, containing final_model/)
# $2 = task
# $3 = physical GPU index on the host node -- pick an idle one, do not assume 0
# $4 = unique tag; SLURM_JOB_ID is the *host* job here, so it is the same for every
#      concurrent step and cannot separate their scratch dirs or their logs

EVAL_DIR="${1:?usage: ptb_rescore_overlap.sh <eval_dir> <task> <gpu> <tag>}"
TASK="${2:?}"
GPU="${3:?}"
TAG="${4:?}"
EVAL_DIR="${EVAL_DIR%/}"
MODEL_PATH="${EVAL_DIR}/final_model"

echo "node=$(hostname) host_job=${SLURM_JOB_ID:-none} gpu=${GPU} tag=${TAG} start=$(date --iso-8601=seconds)"
echo "eval_dir=${EVAL_DIR}"
[ -f "${MODEL_PATH}/config.json" ] || { echo "FATAL: no config.json under ${MODEL_PATH}" >&2; exit 1; }
# This script overwrites metrics.json in place, so on a voided cell it would undo the
# quarantine that void_cells.py put there. Rescoring cannot clear answer-key
# contamination anyway: the weights were trained on the leak.
if [ -f "${EVAL_DIR}/VOIDED_ANSWER_KEY.json" ]; then
    echo "FATAL: ${EVAL_DIR} is voided for answer-key contamination -- refusing" >&2
    exit 1
fi

export PATH="/rmeng_data/robtang/tools/apt-root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)" >&2; exit 1; }

export NUM_GPUS=1
# apptainer --nv binds all eight cards regardless of this, so it is the *process* that is
# confined, not the container. That is enough here and it is why the index must be real.
export CUDA_VISIBLE_DEVICES="${GPU}"
export REPO_ROOT="/rmeng_data/robtang/PostTrainBench"
cd "${REPO_ROOT}"
source src/commit_utils/set_env_vars.sh

TMP_SUBDIR="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}/ptb_rescore_ovl_${TAG}"
HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
TMP_HF_CACHE="/tmp/hf_cache_rescore_${TAG}"
rm -rf "${TMP_SUBDIR}"
mkdir -p "${HF_MERGED}" "${TMP_SUBDIR}/upper_huggingface" "${TMP_SUBDIR}/fuse_workdir"
fuse-overlayfs -o "lowerdir=${HF_HOME},upperdir=${TMP_SUBDIR}/upper_huggingface,workdir=${TMP_SUBDIR}/fuse_workdir" "${HF_MERGED}" \
  || { echo "FATAL: overlay failed" >&2; exit 1; }

CACHE_BIND=()
[ -n "${XDG_CACHE_HOME:-}" ] && [ -d "${XDG_CACHE_HOME}" ] && CACHE_BIND=(--bind "${XDG_CACHE_HOME}:${XDG_CACHE_HOME}")

OUT="${EVAL_DIR}/final_eval_rescore_ovl_${TAG}.txt"
apptainer exec \
    --nv \
    --env HF_HOME="${TMP_HF_CACHE}" \
    --env HF_HUB_CACHE="${TMP_HF_CACHE}/hub" \
    --env HUGGINGFACE_HUB_CACHE="${TMP_HF_CACHE}/hub" \
    --env VLLM_API_KEY="inspectai" \
    --env CUDA_VISIBLE_DEVICES="${GPU}" \
    "${CACHE_BIND[@]}" \
    --writable-tmpfs \
    --bind "${EVAL_DIR}:${EVAL_DIR}" \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
    --pwd "${REPO_ROOT}/src/eval/tasks/${TASK}" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif" \
    python evaluate.py \
        --model-path "${MODEL_PATH}" \
        --templates-dir ../../../../src/eval/templates \
        --limit -1 \
        --json-output-file "${EVAL_DIR}/metrics.json" > "${OUT}" 2>&1
rc=$?

fusermount -u "${HF_MERGED}" 2>/dev/null; rm -rf "${TMP_SUBDIR}"
echo "score_exit=$rc"
echo "--- metrics.json ---"; cat "${EVAL_DIR}/metrics.json" 2>&1
echo "end=$(date --iso-8601=seconds)"
exit $rc
