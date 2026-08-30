#!/bin/bash
#SBATCH --job-name=ptb-greedy-board
#SBATCH --partition=a3
# 11 of a3's 14 nodes are held by reservation robtang-a3, the other 3 by froilanchoi's
# UNLIMITED jobs, so a job submitted without the reservation never starts. Not settable
# from the environment -- see the same note in ptb_pack.sbatch.
#SBATCH --reservation=robtang-a3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.out
#SBATCH --error=/rmeng_data/robtang/PostTrainBench/slurm_logs/%x-%j.err
set -uo pipefail

# Re-score the whole finished board under one fixed decode, 8 cells at a time.
#
# Why this exists: `evaluate.py` passes no temperature, so each cell was graded under
# whatever generation_config.json it happened to ship, and two of twenty-three shipped
# `"temperature": 0.0`. Those two are the only cells above 0.5. So the board currently
# measures "which arm" and "did this cell notice the harness's sampling default" in one
# column, and the arm contrast (+0.1839 pooled) drops to +0.0931 with those two removed
# -- same sign, no longer separable from noise. Neither reading is safe. Equalising the
# decode is the only thing that makes the existing cells comparable, and it costs one
# node rather than twenty-three fresh agent hours: the weights are all still on disk.
#
# Nothing under ptb-results/ is written. Each cell gets a symlink shadow whose only
# changed file is generation_config.json (see make_greedy_shadow.py), and the new score
# lands beside the shadow as metrics_greedy.json. The original metrics.json stays as the
# record of what the cell actually submitted -- both numbers are real, they answer
# different questions, and overwriting one with the other would destroy the comparison
# this job exists to make.
#
# Not included, deliberately:
#   - 82165 / 82166, the two recovered ten-hour cells. Both already ship an explicit
#     greedy generation_config.json, so they need no shadow, and both are queued as
#     plain rescores (83998 / 83999). Re-doing them here would race those jobs for the
#     same output path.
#
# Included beyond the 23-cell board: the two mainline one-hour cells (81520 autor,
# 81521 claude_vertex -- the comparator is itself currently a temperature-1.0 number)
# and the untrained base model, so the floor and the target are measured through the
# same decode as everything they are compared against, in the same batch, on the same
# node.
#
# Resumable: a cell whose metrics_greedy.json already parses is skipped, so a job that
# runs out of wall clock can simply be resubmitted.
#
# Usage:  sbatch ptb_greedy_board.sbatch [<eval_dir> ...]     # default: discover all

TASK="${PTB_TASK:-gsm8k}"
SHADOW_ROOT="${PTB_GREEDY_ROOT:-/rmeng_data/robtang/ptb-greedy}"
RESULTS_ROOT="${PTB_RESULTS_ROOT:-/rmeng_data/robtang/ptb-results}"
BASE_SNAPSHOT="${PTB_BASE_SNAPSHOT:-/rmeng_data/cache/robtang_google_com/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1}"
NPAR="${PTB_GREEDY_NPAR:-8}"
STAGGER="${PTB_GREEDY_STAGGER:-20}"
# Two levels up from the snapshot, not the snapshot: an HF snapshot directory is a farm
# of symlinks into a sibling blobs/, so binding only snapshots/ gives the container a
# model whose every file is a dangling link.
BASE_BIND="$(dirname "$(dirname "${BASE_SNAPSHOT}")")"

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date --iso-8601=seconds)"
echo "task=${TASK} shadow_root=${SHADOW_ROOT} npar=${NPAR} stagger=${STAGGER}"

export PATH="/rmeng_data/robtang/tools/apt-root/usr/bin:$PATH"
export LD_LIBRARY_PATH="/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
command -v apptainer >/dev/null || { echo "FATAL: no apptainer on $(hostname)" >&2; exit 1; }
echo "apptainer=$(apptainer --version)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export REPO_ROOT="/rmeng_data/robtang/PostTrainBench"
cd "${REPO_ROOT}" || exit 1
source src/commit_utils/set_env_vars.sh
echo "repo_sha=$(git rev-parse --short HEAD 2>/dev/null)"

# Build the work list as <src_model_dir>:::<label>. A label is arm/cell, and it is what
# names the shadow directory, so two cells of the same job id on different arms cannot
# collide.
declare -a WORK=()
if [ "$#" -gt 0 ]; then
    for d in "$@"; do
        d="${d%/}"
        src="${d}/final_model"; [ -f "${src}/config.json" ] || src="${d}"
        WORK+=("${src}:::$(basename "$(dirname "$d")")/$(basename "$d")")
    done
else
    for d in "${RESULTS_ROOT}"/*/*/; do
        d="${d%/}"
        [ -f "${d}/final_model/config.json" ] || continue
        # metrics.json is the marker of a cell that was actually graded. Its absence is
        # what distinguishes 82165/82166, which are handled elsewhere.
        [ -f "${d}/metrics.json" ] || continue
        WORK+=("${d}/final_model:::$(basename "$(dirname "$d")")/$(basename "$d")")
    done
    [ -f "${BASE_SNAPSHOT}/config.json" ] \
        && WORK+=("${BASE_SNAPSHOT}:::base/Qwen_Qwen3-1.7B-Base") \
        || echo "WARN: no base snapshot at ${BASE_SNAPSHOT}, batch has no untrained reference" >&2
fi

N="${#WORK[@]}"
[ "$N" -gt 0 ] || { echo "FATAL: no cells to score" >&2; exit 1; }
echo "cells=${N}"
printf '  %s\n' "${WORK[@]}"

# One shadow per cell, built on the login-side filesystem before any GPU work starts, so
# a bad path fails in seconds rather than after an hour of evals.
declare -a SHADOWS=()
for item in "${WORK[@]}"; do
    src="${item%%:::*}"; label="${item##*:::}"
    shadow="${SHADOW_ROOT}/${label}"
    mkdir -p "${shadow}" || exit 1
    python3 ptb_ops/make_greedy_shadow.py "${src}" "${shadow}/model" || exit 1
    SHADOWS+=("${shadow}")
done

score_one() {
    local gpu="$1" shadow="$2" label="$3"
    local model="${shadow}/model"
    local out="${shadow}/metrics_greedy.json"

    if python3 -c "import json,sys; json.load(open(sys.argv[1]))" "${out}" 2>/dev/null; then
        echo "skip label=${label} (already scored: $(cat "${out}" | tr -d '\n '))"
        return 0
    fi

    local tmp="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}/ptb_greedy_${SLURM_JOB_ID}_g${gpu}"
    local merged="${tmp}/merged_huggingface"
    local incache="/tmp/hf_cache_rescore"
    rm -rf "${tmp}"; mkdir -p "${merged}" "${tmp}/upper_huggingface" "${tmp}/fuse_workdir"
    fuse-overlayfs -o "lowerdir=${HF_HOME},upperdir=${tmp}/upper_huggingface,workdir=${tmp}/fuse_workdir" "${merged}" \
        || { echo "FATAL: overlay failed for ${label}" >&2; return 1; }

    # XDG_CACHE_HOME is bound rather than redirected: vLLM's VLLM_CACHE_ROOT and the
    # torch-inductor triton cache both live under it, and on the 64 MiB --writable-tmpfs
    # container root the second one raises OSError: [Errno 28] and kills EngineCore.
    local -a cache_bind=()
    [ -n "${XDG_CACHE_HOME:-}" ] && [ -d "${XDG_CACHE_HOME}" ] && cache_bind=(--bind "${XDG_CACHE_HOME}:${XDG_CACHE_HOME}")

    # The shadow links are absolute, so every link target must resolve at the same path
    # inside the container. /rmeng_data/robtang covers the board, the shadows and the
    # repo; the base snapshot lives in the HF cache outside it. Binding it here rather
    # than relying on XDG_CACHE_HOME happening to be its parent -- if that bind is ever
    # missing, the base cell is the one that dies, an hour in, and it is the reference
    # every other number in the batch is read against.
    local -a extra_bind=()
    if [ -n "${BASE_BIND:-}" ] && [ -d "${BASE_BIND}" ] && [ "${BASE_BIND#/rmeng_data/robtang}" = "${BASE_BIND}" ]; then
        extra_bind=(--bind "${BASE_BIND}:${BASE_BIND}")
    fi

    # --nv exposes all eight cards regardless of the allocation, so the card this cell
    # uses is decided by CUDA_VISIBLE_DEVICES inside the container and nothing else.
    # There is no GPU renumbering here (that is run_task.sh's POST_TRAIN_BENCH_ISOLATE_GPUS),
    # so host index and container index agree.
    CUDA_VISIBLE_DEVICES="${gpu}" NUM_GPUS=1 apptainer exec \
        --nv \
        --env CUDA_VISIBLE_DEVICES="${gpu}" \
        --env HF_HOME="${incache}" \
        --env HF_HUB_CACHE="${incache}/hub" \
        --env HUGGINGFACE_HUB_CACHE="${incache}/hub" \
        --env VLLM_API_KEY="inspectai" \
        "${cache_bind[@]}" \
        "${extra_bind[@]}" \
        --writable-tmpfs \
        --bind "/rmeng_data/robtang:/rmeng_data/robtang" \
        --bind "${merged}:${incache}" \
        --pwd "${REPO_ROOT}/src/eval/tasks/${TASK}" \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif" \
        python evaluate.py \
            --model-path "${model}" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            --json-output-file "${out}" > "${shadow}/eval_greedy.txt" 2>&1
    local rc=$?

    fusermount -u "${merged}" 2>/dev/null; rm -rf "${tmp}"
    echo "scored label=${label} rc=${rc} metrics=$(cat "${out}" 2>/dev/null | tr -d '\n ' || echo none)"
    return $rc
}

# Striped work assignment: worker g takes cells g, g+NPAR, g+2*NPAR ... Not load
# balanced, but it needs no lock and no shared cursor, and the spread of eval times
# within a batch of one-hour cells of the same model is small next to that risk.
declare -a PIDS=()
for (( g=0; g<NPAR && g<N; g++ )); do
    (
        # inspect_ai's vllm provider takes a *free* port when none is given, and eight
        # servers probing at the same instant can agree on the same one. ptb_pack.sbatch
        # staggers by 60s for this; 20s is enough here because these are eval-only
        # starts with no training phase behind them.
        sleep $(( g * STAGGER ))
        for (( i=g; i<N; i+=NPAR )); do
            item="${WORK[$i]}"; label="${item##*:::}"
            echo "--- gpu=${g} cell=${i} label=${label} at $(date --iso-8601=seconds)"
            score_one "${g}" "${SHADOWS[$i]}" "${label}"
        done
    ) > "/rmeng_data/robtang/PostTrainBench/slurm_logs/ptb-greedy-${SLURM_JOB_ID}-g${g}.log" 2>&1 &
    PIDS+=("$!")
    echo "launched gpu=${g} pid=${PIDS[$g]}"
done

rc_all=0
for (( g=0; g<${#PIDS[@]}; g++ )); do
    wait "${PIDS[$g]}"; rc=$?
    echo "worker_exit gpu=${g} rc=${rc}"
    [ "$rc" -eq 0 ] || rc_all=1
done

echo "--- greedy board ---"
for (( i=0; i<N; i++ )); do
    item="${WORK[$i]}"; label="${item##*:::}"
    printf '%-64s %s\n' "${label}" "$(cat "${SHADOWS[$i]}/metrics_greedy.json" 2>/dev/null | tr -d '\n ' || echo MISSING)"
done
echo "greedy_board_exit=${rc_all} end=$(date --iso-8601=seconds)"
exit "${rc_all}"
