#!/bin/bash
# Submit a PostTrainBench unit of work to Slurm.
#
# This is the Slurm counterpart of src/commit_utils/single_task.sub, which is
# an HTCondor submit file. The resource contract it translates:
#
#     single_task.sub                     this script
#     -------------------------------     ---------------------------------
#     num_gpus = 1                        --gres=gpu:1 + CUDA_VISIBLE_DEVICES=0
#     request_cpus = 16                   --cpus-per-task=16
#     request_memory = 131072             --mem=128G
#     request_disk = 400G                 POST_TRAIN_BENCH_TMP_ROOT on local SSD
#     TARGET.CUDADeviceName ==            --partition=a3 (the only H100 80GB
#       "NVIDIA H100 80GB HBM3"             HBM3 partition on this cluster)
#     (no wall-clock cap)                 --time (Slurm requires one)
#     $(Cluster)                          $SLURM_JOB_ID
#
# Two of those translations are not literal and are worth stating plainly.
#
# CUDA_VISIBLE_DEVICES. Partition a3 is OverSubscribe=EXCLUSIVE, so the
# smallest allocation is a whole node: 8 H100s, whatever --gres asks for. The
# benchmark's hardware contract is one H100, and check_cuda.py enforces it by
# asserting torch.cuda.device_count() == NUM_GPUS. Pinning
# CUDA_VISIBLE_DEVICES=0 keeps the contract the benchmark specifies. Raising
# NUM_GPUS to 8 would satisfy the assert but silently change the result: it
# appends a _8gpu suffix to EVAL_DIR, which renames the method scripts/
# collect.py reads, and it would be a different benchmark.
#
# --time. HTCondor caps nothing; Slurm demands a number. The agent phase is
# bounded by run_task.sh's own `timeout ${NUM_HOURS}h+5m`, so the Slurm cap only
# has to cover that plus the evaluation. Evaluation retries up to 4+3+2 times
# with an 8h timeout each, so a first eval that hangs can outlive any sane cap;
# if that happens the model is already in EVAL_DIR and
# scripts/rerun_eval_n_times.sh can score it in a separate job.
#
# Two things about that recovery path, because it is upstream's script and the
# reap/container fixes in run_task.sh do not reach it.
#
#   * It opens each attempt with an unfiltered
#     `nvidia-smi --query-compute-apps=pid | xargs -r kill -9` -- every GPU
#     process on the host, any owner, ignoring POST_TRAIN_BENCH_EVAL_GPU_REAP.
#     That is exactly the behaviour run_task.sh was patched away from. Only
#     submit `rerun` as a job on an exclusive node; never run the script on a
#     login node or on a node holding someone else's work.
#   * It scores in ${POST_TRAIN_BENCH_CONTAINER_NAME}.sif, while run_task.sh
#     hardcodes vllm_debug.sif. Left alone it would rescore in whatever agent
#     image is configured -- a different scorer than the one that produced
#     metrics.json. This script pins vllm_debug for the rerun payload so the two
#     numbers come from the same image.
#
# And what it produces: <EVAL_DIR>/reruns/run_N.json plus metrics_averaged.json.
# It never writes metrics.json, so it cannot repair a missing one -- it is a
# second, differently named number, not a retry of the first.
#
# Usage:
#   slurm_submit.sh baseline <task> <model>
#   slurm_submit.sh task     <task> <agent> <model> <num_hours> <agent_config>
#   slurm_submit.sh rerun    <eval_dir> [n]
#
# Environment overrides: PTB_PARTITION, PTB_RESERVATION, PTB_TIME, PTB_CPUS,
# PTB_MEM, PTB_LOG_DIR, PTB_DRY_RUN=1.
set -euo pipefail

# No braces in this message: the first '}' would close the expansion early and
# the rest would be appended to the value.
MODE="${1:?usage: slurm_submit.sh baseline|task|rerun ...}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${PTB_PARTITION:-a3}"

# a3 is fourteen nodes: eleven are held by reservation robtang-a3 and the other
# three by froilanchoi's UNLIMITED jobs. An a3 job that does not carry the
# reservation is therefore unschedulable rather than merely queued -- it waits on
# three nodes that never free, and squeue reports the ordinary (Resources).
#
# It must NOT be set for any other partition. The reservation names only a3 nodes,
# so an airs or eval submission carrying it dies at submit with "allocation
# failure: Requested node configuration is not available". That is why this is a
# per-partition line and not SBATCH_RESERVATION in the environment: the variable
# applies to every sbatch this shell ever runs and cannot tell the two apart.
RESERVATION_LINE="# no reservation: partition is ${PARTITION}, not a3"
[ "$PARTITION" = a3 ] && RESERVATION_LINE="#SBATCH --reservation=${PTB_RESERVATION:-robtang-a3}"

CPUS="${PTB_CPUS:-16}"
MEM="${PTB_MEM:-128G}"
LOG_DIR="${PTB_LOG_DIR:-$REPO_ROOT/slurm_logs}"
HEALTHY_NODES="${PTB_HEALTHY_NODES:-$HOME/airs-ops/healthy_nodes.sh}"
APPTAINER_BIN="${APPTAINER_BIN:-/rmeng_data/robtang/tools/apt-root/usr/bin}"
APPTAINER_LIB="${APPTAINER_LIB:-/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu}"

mkdir -p "$LOG_DIR"

# Node eligibility is sinfo's call and nothing else's; healthy_nodes.sh line 2
# is the exclude list. Typing an exclude list here instead would protect only
# this launcher.
EXCLUDE=""
if [ -x "$HEALTHY_NODES" ]; then
    EXCLUDE="$(bash "$HEALTHY_NODES" "$PARTITION" | sed -n '2p')"
fi
EXCLUDE_ARG=()
[ -n "$EXCLUDE" ] && EXCLUDE_ARG=(--exclude="$EXCLUDE")

case "$MODE" in
    baseline)
        TASK="${1:?task}"; MODEL="${2:?model}"
        JOB_NAME="ptb-base-${TASK}"
        TIME="${PTB_TIME:-08:00:00}"
        PAYLOAD="bash \"\$PINNED_SRC/baselines/run_baseline.sh\" '${TASK}' '${MODEL}' \"\$SLURM_JOB_ID\""
        ;;
    task)
        TASK="${1:?task}"; AGENT="${2:?agent}"; MODEL="${3:?model}"
        HOURS="${4:?num_hours}"; CONFIG="${5:?agent_config}"
        JOB_NAME="ptb-${AGENT}-${TASK}"
        # agent budget + 5 min of run_task.sh slack + evaluation headroom
        TIME="${PTB_TIME:-$(printf '%d:00:00' $((HOURS + 8)))}"
        PAYLOAD="bash \"\$RUN_TASK\" '${TASK}' '${AGENT}' '${MODEL}' \"\$SLURM_JOB_ID\" '${HOURS}' '${CONFIG}' 1"
        ;;
    rerun)
        EVAL_DIR_ARG="${1:?eval_dir}"; N="${2:-1}"
        JOB_NAME="ptb-rerun"
        TIME="${PTB_TIME:-08:00:00}"
        # vllm_debug, not the configured agent image: rerun_eval_n_times.sh reads
        # POST_TRAIN_BENCH_CONTAINER_NAME for its evaluation, run_task.sh does not.
        PAYLOAD="POST_TRAIN_BENCH_CONTAINER_NAME=vllm_debug bash \"\$PINNED_ROOT/scripts/rerun_eval_n_times.sh\" '${EVAL_DIR_ARG}' '${N}'"
        ;;
    *)
        echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

JOB_SCRIPT="$(mktemp "${LOG_DIR}/${JOB_NAME}.XXXXXX.sbatch")"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
${RESERVATION_LINE}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.err
set -uo pipefail

echo "node=\$(hostname) job=\$SLURM_JOB_ID start=\$(date --iso-8601=seconds)"

# apptainer is installed node-locally here and is missing on some nodes; the
# unprivileged unpack on the shared filesystem is present on all of them.
export PATH="${APPTAINER_BIN}:\$PATH"
export LD_LIBRARY_PATH="${APPTAINER_LIB}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
command -v apptainer >/dev/null || { echo "FATAL: no apptainer on \$(hostname)" >&2; exit 1; }
echo "apptainer=\$(apptainer --version)"

# Hold the benchmark's one-H100 contract on an exclusive whole-node allocation.
# run_baseline.sh's apptainer calls inherit the host environment; run_task.sh's
# agent container runs --cleanenv and reads POST_TRAIN_BENCH_VISIBLE_GPUS.
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export POST_TRAIN_BENCH_VISIBLE_GPUS=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8

# Baked in at generation time, not inherited. sbatch's --export defaults to ALL
# and this would arrive anyway, but --export is a cluster policy and a pin that
# silently stops applying is worse than no pin: it would let two arms run two
# different CLI builds while the submission record says they did not. Empty here
# does not mean unpinned: set_env_vars.sh below treats set-but-empty as unset, so
# a submitting shell that says nothing leaves the decision to .env, and .env is
# where the pin belongs -- it is read at job start rather than frozen at submit.
export POST_TRAIN_BENCH_SKIP_CLI_UPDATE="${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-}"
echo "skip_cli_update=${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-<unset: each arm updates to latest>}"

cd "${REPO_ROOT}"
source src/commit_utils/set_env_vars.sh

# This cluster exports HUGGINGFACE_HUB_CACHE globally, pointing at the real shared
# cache. Every scoring container is launched without --cleanenv, and huggingface_hub
# ranks HF_HUB_CACHE above HUGGINGFACE_HUB_CACHE above "\$HF_HOME/hub" -- so the
# --env HF_HOME those execs pass loses, the fuse-overlayfs mount goes unread, and the
# unbound host path gets created on the 64 MiB --writable-tmpfs container root. The
# download then dies as ENOSPC with 5.9T free on every host filesystem. The exec
# lines in run_baseline.sh name all three explicitly; unsetting here covers the call
# sites that do not, and costs nothing where they do -- HF_HOME is untouched, so
# host-side resolution lands on the same directory these two named.
unset HUGGINGFACE_HUB_CACHE HF_HUB_CACHE
echo "hf_home=\${HF_HOME:-<unset>} hub_cache_overrides=cleared"
# Fail here rather than deep inside run_task.sh. The scratch root is node-local
# by design -- request_disk = 400G in single_task.sub -- so it is the one path in
# this script whose existence is a property of the node the scheduler picked and
# not of the shared filesystem. A merged model does not fit in a Slurm node's
# small root /tmp, and the failure that produces arrives an hour in, after the
# agent has already spent an hour of its budget.
SCRATCH="\${POST_TRAIN_BENCH_TMP_ROOT:-/tmp}"
mkdir -p "\$SCRATCH" || { echo "FATAL: cannot create \$SCRATCH on \$(hostname)" >&2; exit 1; }
[ -w "\$SCRATCH" ] || { echo "FATAL: \$SCRATCH is not writable on \$(hostname)" >&2; exit 1; }
echo "scratch=\$SCRATCH free=\$(df -h "\$SCRATCH" | tail -1 | awk '{print \$4}')"

# Run the driver from node-local disk. Bash reads a script incrementally and holds a
# handle on its own inode for the whole run, so a \`git commit\` in this checkout kills
# every job already running out of it. That is how 82165 and 82166 died -- both after a
# complete agent phase, both with a finished final_model/ they never got scored. See
# src/commit_utils/pin_src_locally.sh. This script itself is safe without the pin, since
# sbatch takes its own copy at submit; the payload it launches is not.
RUN_TASK="\$(bash src/commit_utils/pin_src_locally.sh "${REPO_ROOT}" "\${SCRATCH}/ptb-src-\${SLURM_JOB_ID}")"
PINNED_SRC="\$(dirname "\$RUN_TASK")"
PINNED_ROOT="\$(dirname "\$PINNED_SRC")"
echo "run_task=\$RUN_TASK"

${PAYLOAD}
rc=\$?
echo "payload_exit=\$rc end=\$(date --iso-8601=seconds)"
exit \$rc
EOF

echo "job script: $JOB_SCRIPT"
[ -n "$EXCLUDE" ] && echo "excluding  : $EXCLUDE"
if [ -n "${PTB_DRY_RUN:-}" ]; then
    echo "--- DRY RUN, not submitting ---"
    cat "$JOB_SCRIPT"
    exit 0
fi
sbatch "${EXCLUDE_ARG[@]}" "$JOB_SCRIPT"
