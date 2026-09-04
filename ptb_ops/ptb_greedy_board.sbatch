#!/bin/bash
#SBATCH --job-name=ptb-greedy-board
#SBATCH --partition=a3
# 11 of a3's 14 nodes are held by reservation robtang-a3, the other 3 by froilanchoi's
# UNLIMITED jobs, so a job submitted without the reservation never starts. Not settable
# from the environment -- see the same note in ptb_pack.sbatch.
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

# ---- hard fence: never occupy froilanchoi's three a3 nodes ----------------------------
# a3 is 23 nodes; 20 sit in reservations robtang-a3 and robtang-a3-ondem, and the
# complement is exactly slurm2-a3nodeset0-[0-2], which are froilanchoi's. A job reaches
# them only when the submit-side guard was bypassed: `--reservation=` with an EMPTY value
# is a PIN onto those three, not a relaxation, and so is any --exclude naming all twenty
# of mine. airs-ops/healthy_nodes.sh drops them from every exclude list it builds, but
# that binds only the callers who ask it. On 2026-09-04 jobs 91755 and 91723 each ran two
# hours on 0-0 and 0-1 before anyone noticed, and froilanchoi refilled all three within
# twenty minutes of being given them back -- so this is contention, not spare capacity.
# Refuse here instead. It forfeits nothing: the node was never ours to run on. Kept ahead
# of every other statement so nothing is half-created on the way out.
_fence_nodes="${SLURM_JOB_NODELIST:-}"
if [ -n "$_fence_nodes" ] && command -v scontrol >/dev/null 2>&1; then
    _fence_nodes="$(scontrol show hostnames "$_fence_nodes" 2>/dev/null | tr '\n' ' ' || true)"
fi
[ -n "${_fence_nodes// /}" ] || _fence_nodes="${SLURMD_NODENAME:-$(hostname -s)}"
for _fence_n in $_fence_nodes; do
    case "$_fence_n" in
        slurm2-a3nodeset0-0|slurm2-a3nodeset0-1|slurm2-a3nodeset0-2)
            echo "FATAL: allocated ${_fence_n}, which belongs to froilanchoi -- one of the three" >&2
            echo "       a3 nodes outside reservations robtang-a3 / robtang-a3-ondem. Resubmit" >&2
            echo "       WITH a reservation; never pass an empty --reservation= on partition a3." >&2
            exit 1 ;;
    esac
done
unset _fence_nodes _fence_n

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
# Physical GPU indices, in order. The default 0..NPAR-1 is right for the whole-node
# allocation this asks for, and wrong for an `srun --overlap` step on a node whose host
# job is already using some cards -- there the free indices are whatever nvidia-smi says
# they are at that moment, and starting a vLLM server on a busy card costs both jobs.
# `PTB_GREEDY_GPUS="0 1 3 4"` names them and NPAR follows from the list.
declare -a GPUS=()
if [ -n "${PTB_GREEDY_GPUS:-}" ]; then
    read -r -a GPUS <<< "${PTB_GREEDY_GPUS//,/ }"
    NPAR="${#GPUS[@]}"
else
    for (( _g=0; _g<NPAR; _g++ )); do GPUS+=("${_g}"); done
fi
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
        # Named by hand, so say why it is being dropped rather than dropping it quietly.
        if [ -f "${d}/VOIDED_ANSWER_KEY.json" ]; then
            echo "SKIP ${d}: voided for answer-key contamination" >&2
            continue
        fi
        src="${d}/final_model"; [ -f "${src}/config.json" ] || src="${d}"
        WORK+=("${src}:::$(basename "$(dirname "$d")")/$(basename "$d")")
    done
else
    for d in "${RESULTS_ROOT}"/*/*/; do
        d="${d%/}"
        [ -f "${d}/final_model/config.json" ] || continue
        # A voided cell has no metrics.json either (void_cells.py renames it), so the
        # next test would already skip it -- but by accident, and silently.
        if [ -f "${d}/VOIDED_ANSWER_KEY.json" ]; then
            echo "SKIP ${d}: voided for answer-key contamination" >&2
            continue
        fi
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
    # Bytecode goes to node-local scratch, not to the 64 MiB --writable-tmpfs
    # container root: nothing in vllm_debug.sif is precompiled (uv installs
    # --no-cache and does not byte-compile; 24602 .py against 886 .pyc in
    # dist-packages) and one `import transformers, torch, vllm` writes 41 MiB of
    # .pyc, so ENOSPC and a truncated .pyc are the default outcome, not an edge
    # case. Per-worker, under ${tmp}, for two reasons the shared ${incache} name
    # does not satisfy: eight workers writing one prefix concurrently would race
    # on the same .pyc paths, and ${tmp} is the thing rm -rf'd at the end of this
    # function, so the bytecode dies with the worker that made it.
    local pycache="${tmp}/pycache"
    rm -rf "${tmp}"; mkdir -p "${merged}" "${tmp}/upper_huggingface" "${tmp}/fuse_workdir" "${pycache}"
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

    # Unset by default; the full argument is in src/run_task.sh next to
    # POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND. 89810_g5 lost 7 of 43 grader
    # invocations to a CUDA illegal-memory-access until it set TRITON_ATTN by
    # hand, but the cost of TRITON_ATTN at these shapes is unmeasured, so pinning
    # it would trade a rare crash for a tax on all 26 cells in this batch. This
    # job is the cheapest place to measure it: same node, same decode, 8 cells
    # at a time.
    local -a attn_backend_env=()
    [ -n "${POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND:-}" ] && \
        attn_backend_env=(--env "VLLM_ATTENTION_BACKEND=${POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND}")

    # --nv exposes all eight cards regardless of the allocation, so the card this cell
    # uses is decided by CUDA_VISIBLE_DEVICES inside the container and nothing else.
    # There is no GPU renumbering here (that is run_task.sh's POST_TRAIN_BENCH_ISOLATE_GPUS),
    # so host index and container index agree.
    #
    # check_pycache.sh runs inside this exec, ahead of python, so it measures the bind
    # list that is actually in force -- a preflight exec of its own would carry a second
    # copy of the pycache bind and could not notice this one losing it. It is reachable
    # because ${REPO_ROOT} lives under the /rmeng_data/robtang bind below. $0 carries the
    # repo root and everything after it is positional, so nothing inside the single-quoted
    # script is expanded here, and `exec` leaves python as the process whose rc this reads.
    # Sourced (`.`) and not run: its remedy for a cache it cannot use is an export (a
    # relocated PYTHONPYCACHEPREFIX, or PYTHONDONTWRITEBYTECODE=1), and an export from a
    # child process reaches nothing. Sourcing also preserves "$@".
    CUDA_VISIBLE_DEVICES="${gpu}" NUM_GPUS=1 apptainer exec \
        --nv \
        --env CUDA_VISIBLE_DEVICES="${gpu}" \
        --env HF_HOME="${incache}" \
        --env HF_HUB_CACHE="${incache}/hub" \
        --env HUGGINGFACE_HUB_CACHE="${incache}/hub" \
        --env VLLM_API_KEY="inspectai" \
        "${attn_backend_env[@]}" \
        "${cache_bind[@]}" \
        "${extra_bind[@]}" \
        --writable-tmpfs \
        --bind "/rmeng_data/robtang:/rmeng_data/robtang" \
        --bind "${merged}:${incache}" \
        --bind "${pycache}:${pycache}" \
        --env PYTHONPYCACHEPREFIX="${pycache}" \
        --pwd "${REPO_ROOT}/src/eval/tasks/${TASK}" \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif" \
        bash -c '. "$0/src/utils/check_pycache.sh" || exit 1; exec python "$@"' \
            "${REPO_ROOT}" evaluate.py \
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
            echo "--- gpu=${GPUS[$g]} cell=${i} label=${label} at $(date --iso-8601=seconds)"
            score_one "${GPUS[$g]}" "${SHADOWS[$i]}" "${label}"
        done
    ) > "/rmeng_data/robtang/PostTrainBench/slurm_logs/ptb-greedy-${SLURM_JOB_ID}-g${GPUS[$g]}.log" 2>&1 &
    PIDS+=("$!")
    echo "launched worker=${g} gpu=${GPUS[$g]} pid=${PIDS[$g]}"
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
