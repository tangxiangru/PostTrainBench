#!/bin/bash
#SBATCH --job-name=ptb-pack
#SBATCH --partition=a3
# 11 of a3's 14 nodes are held by reservation robtang-a3, the other 3 -- a3nodeset0-[0-2]
# -- are froilanchoi's. A pack submitted without the reservation usually never starts: it
# waits on those three and squeue calls it (Resources). Not settable from the environment:
# SBATCH_RESERVATION applies to every partition, and an airs/eval job that inherits it
# dies at submit with "allocation failure".
#SBATCH --reservation=robtang-a3
# (Resources) is the *lucky* outcome. 89727/89809/89810 were submitted with an empty
# `--reservation=` on the command line -- which silently voids the line above -- at
# 23:13 on 2026-09-01, three hours after froilanchoi's last autolab job released
# a3nodeset0-[0-2]. The scheduler did the only thing left: it put three exclusive
# eight-GPU packs on all three of someone else's nodes for sixteen hours. They ran 3.5h
# before being requeued off. So the reservation is not the only guard -- name the nodes:
#SBATCH --exclude=slurm2-a3nodeset0-[0-2]
# A command-line --exclude REPLACES this line rather than adding to it, so any pack
# submitted with its own --exclude must carry slurm2-a3nodeset0-[0-2] in it.
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
# Arms are given as the pack's composition, NOT as a per-GPU assignment: the
# launcher draws the arm -> card map itself (see "ptb seat assignment" below).
# Typing the same order twice used to mean running the same arm on the same
# cards twice.
#
# Usage:  sbatch ptb_pack.sbatch <arm> [<arm> ...]     # up to 8, order ignored
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

# Echo the identity of every payload being compared, then enforce that the pack is
# a comparison rather than half of one.
#
# An arm is its payload/ -- a whole AutoR checkout on its own branch -- not its
# solve.sh, which is a thin shared launcher. So the line worth having in the log is
# the branch and commit each cell actually ran, taken at run time: agents/ is NOT
# tracked in git (tools/ptb_setup.py installs it), and ptb-results/ is keyed on the
# arm NAME, so afterwards there is nowhere left that records what a name meant.
# In b4, fx and fxq turn out to share payload sha 0719436 and rc/rt share ea95d15;
# that is only visible from a line like this one.
#
# The content hash excludes .git/ and __pycache__/ deliberately: two clones of the
# same commit differ in pack files and in .pyc timestamps while being the same arm.
declare -A _ARM_SHA=()
_dupe=""
for _a in "${ARMS[@]}"; do
    _d="agents/${_a}"
    [ -f "${_d}/solve.sh" ] || { echo "FATAL: no such arm: ${_d}/solve.sh" >&2; exit 1; }
    _br="$(git -C "${_d}/payload" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<no-payload>')"
    _psha="$(git -C "${_d}/payload" rev-parse --short HEAD 2>/dev/null || echo '-')"
    _h="$(find "${_d}" -type f ! -path '*/.git/*' ! -path '*__pycache__*' \
              ! -name '*auth*' ! -name 'api_keys.json' -printf '%P\n' 2>/dev/null \
          | LC_ALL=C sort \
          | while read -r _f; do sha256sum "${_d}/${_f}" | cut -d' ' -f1; printf '%s\n' "$_f"; done \
          | sha256sum | cut -c1-16)"
    echo "arm_payload ${_a} branch=${_br} payload_sha=${_psha} content=${_h} solve_sha=$(sha256sum "${_d}/solve.sh" | cut -c1-16)"
    if [ -n "${_ARM_SHA[$_h]:-}" ] && [ "${_ARM_SHA[$_h]}" != "$_a" ]; then
        _dupe="${_dupe} ${_ARM_SHA[$_h]}==${_a}"
    fi
    _ARM_SHA[$_h]="$_a"
done
if [ -n "$_dupe" ]; then
    echo "FATAL: two arms in this pack have identical payload content:${_dupe}" >&2
    echo "       They cannot be compared -- any difference between them is noise" >&2
    echo "       wearing two labels. Fix the payload, or drop one arm. Override" >&2
    echo "       with PTB_ALLOW_DUPLICATE_ARMS=1 only to measure the null." >&2
    [ "${PTB_ALLOW_DUPLICATE_ARMS:-0}" = "1" ] || exit 1
    echo "       PTB_ALLOW_DUPLICATE_ARMS=1 -- continuing as a null calibration." >&2
fi

# A pack is the only unit inside which conditions are held fixed: same node, same
# hour, same CLI build, same HF cache. In b4 the arms were nested inside packs --
# five packs, two arms each, and the control in neither of the days the treatment
# arms ran. That makes the PACK the unit of randomisation, and with 2 treatment
# packs against 3 others the smallest attainable p-value is 0.10: the batch could
# not have reached significance however large the effect. (The pack really is a
# unit and not a formality -- fx ran in two packs on one day for 0/4 and 2/4.)
#
# Interleaving arms across GPUs inside a pack, which this launcher already does,
# does not help; the arm has to be interleaved across PACKS. Eight arms x 1 GPU x
# five packs costs exactly the same GPU-hours as two arms x 4 GPUs x five packs,
# gives 5 replicates per arm instead of 4, and turns the pack from a confound into
# a block you can subtract. Requiring the control in every pack is the cheap half
# of that.
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

# >>> ptb seat assignment >>>
# Which card an arm lands on is a treatment, not a formality.
#
# Until now the map was `arm="${ARMS[$gpu]}"`: seat 0 ran the first name the
# operator typed, seat 1 the second. Every submit line of the ten-hour mainline
# pair typed them in the same order, so across the entire published contrast
# `sn` ran on g0/g2/g4/g6 and `r2` on g1/g3/g5/g7 -- every pack, every day.
# NVLink neighbours, position in the chassis and so inlet temperature, the
# `sleep $((gpu*STAGGER))` start offset, NUMA locality, and rank in the race for
# host page cache and scratch bandwidth are all functions of the seat, and all
# of them were 100% aliased with the arm. "+1.15 pt, p=0.51" is therefore a
# statement about (arm AND even cards) against (arm AND odd cards). A null
# survives that reading; a positive result would not have.
#
# So the seat is drawn, not typed. The draw is a pure function of (job id, pack
# composition): re-running a job id reproduces its own map, and the map is
# recomputable from the log line below years later. The draw is also
# *restricted* -- balanced randomisation, the standard device: a draw that would
# give any multi-seat arm only even cards, only odd cards, or only one half of
# the chassis is rejected and the next draw taken. A one-seat arm is exempt (it
# has no spread to have) and so is a pack below four seats.
#
# This block is extracted verbatim and sourced by tests/test_pack_seat_map.sh.
# Keep it self-contained and keep the sentinel lines.
PTB_SEAT_MAX_DRAWS="${PTB_SEAT_MAX_DRAWS:-1024}"

ptb_seat_shuffle() {
    # $1 seed string, $2 draw counter, $3 n (1..8). Prints a permutation of
    # 0..n-1. Pure: same arguments, same answer, on any host, forever -- which
    # is the whole reason the log line is enough to recover the map.
    local seed="$1" draw="$2" n="$3" i j k tmp off=0 hex
    hex="$(printf '%s|draw=%s' "$seed" "$draw" | sha256sum | cut -d' ' -f1)"
    local -a a=()
    for (( i=0; i<n; i++ )); do a+=("$i"); done
    for (( i=n-1; i>0; i-- )); do   # Fisher-Yates, four hex digits per step
        k=$(( 16#${hex:off:4} )); off=$(( off + 4 ))
        j=$(( k % (i + 1) ))
        tmp="${a[i]}"; a[i]="${a[j]}"; a[j]="$tmp"
    done
    printf '%s' "${a[*]}"
}

ptb_seat_confound() {
    # $1 n, then the realised seat->arm names, seat 0 first. Prints one
    # <arm>:<violation> line per offence and returns 1 when the map reproduces
    # the confound this launcher exists to prevent.
    local n="$1"; shift
    local -a seats=("$@")
    [ "$n" -ge 4 ] || return 0
    local half=$(( n / 2 )) bad=0 a g even odd low high
    local -a uniq=()
    for a in "${seats[@]}"; do
        case " ${uniq[*]:-} " in *" $a "*) ;; *) uniq+=("$a") ;; esac
    done
    for a in "${uniq[@]}"; do
        even=0; odd=0; low=0; high=0
        for (( g=0; g<n; g++ )); do
            [ "${seats[$g]}" = "$a" ] || continue
            if (( g % 2 == 0 )); then even=$(( even + 1 )); else odd=$(( odd + 1 )); fi
            if (( g < half )); then low=$(( low + 1 )); else high=$(( high + 1 )); fi
        done
        (( even + odd >= 2 )) || continue    # one seat cannot be spread over two
        (( even > 0 )) || { echo "${a}:every-seat-odd"; bad=1; }
        (( odd  > 0 )) || { echo "${a}:every-seat-even"; bad=1; }
        (( low  > 0 )) || { echo "${a}:every-seat-in-high-half"; bad=1; }
        (( high > 0 )) || { echo "${a}:every-seat-in-low-half"; bad=1; }
    done
    return "$bad"
}

ptb_seat_plan() {
    # $1 seed, then the arms as typed. On success sets PTB_SEATS (seat -> arm),
    # PTB_SEAT_POS (seat -> index in the typed list), PTB_SEAT_SOURCE and
    # PTB_SEAT_DRAWS. On refusal sets PTB_SEAT_REFUSAL / PTB_SEAT_REFUSAL_KIND
    # and returns 1; the caller must exit before the allocation is spent.
    #
    # PTB_SEAT_MAP pins the map by hand -- "sn r2 r2 sn ..." or the same with
    # commas, one arm per seat starting at g0. It may reorder the pack; it may
    # not change what is in it, and it is held to the same balance rule as a
    # drawn map, because the point of the rule is the map, not its provenance.
    local seed="$1"; shift
    local -a arms=("$@")
    local n="${#arms[@]}" i j draw viol want got
    PTB_SEATS=(); PTB_SEAT_POS=(); PTB_SEAT_DRAWS=0
    PTB_SEAT_REFUSAL=""; PTB_SEAT_REFUSAL_KIND=""; PTB_SEAT_SOURCE="drawn"
    if [ -n "${PTB_SEAT_MAP:-}" ]; then
        PTB_SEAT_SOURCE="pinned"
        read -r -a PTB_SEATS <<< "${PTB_SEAT_MAP//,/ }"
        if [ "${#PTB_SEATS[@]}" -ne "$n" ]; then
            PTB_SEAT_REFUSAL_KIND="malformed"
            PTB_SEAT_REFUSAL="PTB_SEAT_MAP names ${#PTB_SEATS[@]} seats, the pack has ${n}"
            return 1
        fi
        want="$(printf '%s\n' "${arms[@]}" | LC_ALL=C sort | tr '\n' ' ')"
        got="$(printf '%s\n' "${PTB_SEATS[@]}" | LC_ALL=C sort | tr '\n' ' ')"
        if [ "$want" != "$got" ]; then
            PTB_SEAT_REFUSAL_KIND="malformed"
            PTB_SEAT_REFUSAL="PTB_SEAT_MAP is not a permutation of the pack: [${got% }] not [${want% }]"
            return 1
        fi
        local -a used=()
        for (( i=0; i<n; i++ )); do used+=(0); done
        for (( i=0; i<n; i++ )); do
            for (( j=0; j<n; j++ )); do
                if [ "${used[$j]}" = "0" ] && [ "${arms[$j]}" = "${PTB_SEATS[$i]}" ]; then
                    PTB_SEAT_POS+=("$j"); used[$j]=1; break
                fi
            done
        done
    else
        local -a perm=()
        for (( draw=1; draw<=PTB_SEAT_MAX_DRAWS; draw++ )); do
            read -r -a perm <<< "$(ptb_seat_shuffle "$seed" "$draw" "$n")"
            PTB_SEATS=()
            for i in "${perm[@]}"; do PTB_SEATS+=("${arms[$i]}"); done
            PTB_SEAT_DRAWS="$draw"
            if ptb_seat_confound "$n" "${PTB_SEATS[@]}" >/dev/null; then
                PTB_SEAT_POS=("${perm[@]}")
                return 0
            fi
        done
        # Unreachable in practice: every one of the 66 pack shapes with 1 <= n <= 8
        # admits a balanced map (enumerated exhaustively), and the tightest shape,
        # four two-seat arms on eight cards, accepts 96 of 2520 maps. 1024 draws
        # miss it with probability (1-0.038)^1024 ~ 6e-18. So if this line is ever
        # printed it is a fact about the code, not about the luck of the draw --
        # do not paper over it with more draws.
        PTB_SEAT_REFUSAL_KIND="unsatisfiable"
        PTB_SEAT_REFUSAL="no balanced seat map after ${PTB_SEAT_MAX_DRAWS} draws for arms=[${arms[*]}]"
        return 1
    fi
    viol="$(ptb_seat_confound "$n" "${PTB_SEATS[@]}")" || {
        PTB_SEAT_REFUSAL_KIND="confound"
        PTB_SEAT_REFUSAL="PTB_SEAT_MAP reproduces the arm/seat confound: ${viol//$'\n'/ }"
        return 1
    }
    return 0
}
# <<< ptb seat assignment <<<

SEAT_SEED="${SLURM_JOB_ID:-0}|${ARMS[*]}"
if ! ptb_seat_plan "$SEAT_SEED" "${ARMS[@]}"; then
    echo "FATAL: ${PTB_SEAT_REFUSAL}" >&2
    echo "       An arm that only ever draws one parity or one half of the" >&2
    echo "       chassis is not being compared with the other arm, it is being" >&2
    echo "       compared with a seat: NVLink neighbours, inlet temperature," >&2
    echo "       the staggered start and the race for host page cache all" >&2
    echo "       follow the card index. That is the confound this launcher" >&2
    echo "       exists to prevent and it will not reproduce it." >&2
    if [ "${PTB_SEAT_REFUSAL_KIND}" = "confound" ]; then
        echo "       Drop PTB_SEAT_MAP, or set PTB_ALLOW_SEAT_CONFOUND=1 to pin a" >&2
        echo "       historical map on purpose -- which voids the between-arm" >&2
        echo "       contrast for this pack." >&2
        [ "${PTB_ALLOW_SEAT_CONFOUND:-0}" = "1" ] || exit 1
        echo "       PTB_ALLOW_SEAT_CONFOUND=1 -- continuing with a confounded map." >&2
    else
        exit 1
    fi
fi
_seat_flat=""
for (( _g=0; _g<N; _g++ )); do
    _seat_flat="${_seat_flat} g${_g}=${PTB_SEATS[$_g]}"
    echo "seat gpu=${_g} arm=${PTB_SEATS[$_g]} typed_pos=${PTB_SEAT_POS[$_g]}"
done
echo "seat_map${_seat_flat}"
echo "seat_source=${PTB_SEAT_SOURCE} seat_draws=${PTB_SEAT_DRAWS} seat_confound=${PTB_SEAT_REFUSAL_KIND:-none} seat_seed=${SEAT_SEED}"

# `**/test_data.json` is gitignored on purpose -- it is the task's test set -- so a
# `git clone` of this repo produces a checkout that looks complete and has no test data.
# run_task.sh only notices once it is already inside the cell, which is how 84279 and
# 84280 each took an exclusive eight-GPU node and lost all seven cells to one line at
# the 4.5-minute mark. Check it here, before the allocation is spent.
_TD="src/eval/tasks/${TASK}/test_data.json"
if [ ! -s "${_TD}" ]; then
    echo "FATAL: ${_TD} is missing or empty in ${REPO_ROOT}" >&2
    echo "       It is gitignored, so a fresh clone never has it. Copy it from a" >&2
    echo "       working checkout or re-run tools/ptb_setup.py before submitting." >&2
    exit 1
fi
echo "test_data=${_TD} bytes=$(stat -c%s "${_TD}")"

SCRATCH="${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
mkdir -p "$SCRATCH" || { echo "FATAL: cannot create $SCRATCH on $(hostname)" >&2; exit 1; }
[ -w "$SCRATCH" ] || { echo "FATAL: $SCRATCH is not writable on $(hostname)" >&2; exit 1; }
echo "scratch=$SCRATCH free=$(df -h "$SCRATCH" | tail -1 | awk '{print $4}')"

# A cell holds 12-55G of weights and overlay upperdirs here, and on a3 the /tmp
# fallback is the 200G boot PD shared with the OS -- eight concurrent cells took it
# to 42G free on 2026-09-02. set_env_vars.sh points this at /mnt/localssd when the
# node has one; if we still landed on the boot disk, say so loudly rather than
# discovering it from an ENOSPC nine hours into training.
case "$SCRATCH" in
    /tmp*) echo "WARNING: scratch is on the boot disk, not local SSD."
           echo "         localssd_mounted=$(mountpoint -q /mnt/localssd 2>/dev/null && echo yes || echo no)" ;;
esac

# Run the driver from node-local disk, because bash reads a script incrementally and a
# `git commit` in this checkout replaces the inode under a running job. See
# src/commit_utils/pin_src_locally.sh -- that is how 82165 and 82166 lost ten hours each.
# One pin per job, shared by every cell: they all read it, none of them write it.
RUN_TASK="$(bash src/commit_utils/pin_src_locally.sh "$REPO_ROOT" "${SCRATCH}/ptb-src-${SLURM_JOB_ID}")"
echo "run_task=${RUN_TASK}"

LOGDIR="/rmeng_data/robtang/PostTrainBench/slurm_logs"
declare -a PIDS=()
for (( gpu=0; gpu<N; gpu++ )); do
    # PTB_SEATS, not ARMS: the seat is drawn above. The stagger below is keyed on
    # the seat too, deliberately -- start order is one of the things the seat was
    # aliased with, so it has to be randomised with it, not held to typed order.
    arm="${PTB_SEATS[$gpu]}"
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
    echo "cell_exit gpu=${gpu} arm=${PTB_SEATS[$gpu]} cell=${SLURM_JOB_ID}_g${gpu} rc=${rc}"
    # What decode did this cell ship? `evaluate.py` passes no temperature, so vLLM uses
    # whatever the delivered generation_config.json says, and that single flag explains
    # r^2 = 0.914 of the between-arm variance on the one-hour board (nine arms, n=64):
    # an arm that never writes temperature=0.0 averages 0.21, one that always does, 0.61.
    # It is the largest effect on the board and until now it was invisible until someone
    # went digging in final_model/ after the fact. Print it next to the exit code.
    _gc="$(ls -1 "${POST_TRAIN_BENCH_RESULTS_DIR:-/rmeng_data/robtang/ptb-results}"/*/"${TASK}"_*_"${SLURM_JOB_ID}_g${gpu}"/final_model/generation_config.json 2>/dev/null | head -1)"
    if [ -n "${_gc}" ]; then
        echo "cell_decode gpu=${gpu} arm=${PTB_SEATS[$gpu]} $(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(" ".join(f"{k}={d.get(k)}" for k in ("temperature","top_p","top_k","do_sample")))
' "${_gc}" 2>/dev/null || echo 'unparseable')"
    else
        echo "cell_decode gpu=${gpu} arm=${PTB_SEATS[$gpu]} no generation_config.json shipped"
    fi
    [ "$rc" -eq 0 ] || rc_all=1
done

echo "pack_exit=${rc_all} end=$(date --iso-8601=seconds)"
exit "${rc_all}"
