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
# Usage:
#   slurm_submit.sh baseline <task> <model>
#   slurm_submit.sh task     <task> <agent> <model> <num_hours> <agent_config>
#   slurm_submit.sh rerun    <eval_dir> [n]
#
# Environment overrides: PTB_PARTITION, PTB_TIME, PTB_CPUS, PTB_MEM,
# PTB_LOG_DIR, PTB_DRY_RUN=1.
set -euo pipefail

# No braces in this message: the first '}' would close the expansion early and
# the rest would be appended to the value.
MODE="${1:?usage: slurm_submit.sh baseline|task|rerun ...}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PARTITION="${PTB_PARTITION:-a3}"
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
        PAYLOAD="bash src/baselines/run_baseline.sh '${TASK}' '${MODEL}' \"\$SLURM_JOB_ID\""
        ;;
    task)
        TASK="${1:?task}"; AGENT="${2:?agent}"; MODEL="${3:?model}"
        HOURS="${4:?num_hours}"; CONFIG="${5:?agent_config}"
        JOB_NAME="ptb-${AGENT}-${TASK}"
        # agent budget + 5 min of run_task.sh slack + evaluation headroom
        TIME="${PTB_TIME:-$(printf '%d:00:00' $((HOURS + 8)))}"
        PAYLOAD="bash src/run_task.sh '${TASK}' '${AGENT}' '${MODEL}' \"\$SLURM_JOB_ID\" '${HOURS}' '${CONFIG}' 1"
        ;;
    rerun)
        EVAL_DIR_ARG="${1:?eval_dir}"; N="${2:-1}"
        JOB_NAME="ptb-rerun"
        TIME="${PTB_TIME:-08:00:00}"
        PAYLOAD="bash scripts/rerun_eval_n_times.sh '${EVAL_DIR_ARG}' '${N}'"
        ;;
    *)
        echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

JOB_SCRIPT="$(mktemp "${LOG_DIR}/${JOB_NAME}.XXXXXX.sbatch")"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
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

cd "${REPO_ROOT}"
source src/commit_utils/set_env_vars.sh
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
