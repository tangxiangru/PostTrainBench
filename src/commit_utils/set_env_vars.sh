_SET_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${_SET_ENV_DIR}/../../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE" >&2
    echo "Copy example.env to .env and fill in your values." >&2
    exit 1
fi

# Export all variables from .env, without overriding already-set env vars
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    var_name="${line%%=*}"
    current_value="$(eval echo "\${$var_name:-}")"

    if [ -z "$current_value" ]; then
        eval "export $line"
    fi
done < "$ENV_FILE"

export HF_HOME_NEW="/home/ben/hf_cache"

# Scratch defaults to the node-local SSD array when the node has one.
#
# A cell holds 12-55G of merged HuggingFace weights and fuse-overlayfs upperdirs
# under POST_TRAIN_BENCH_TMP_ROOT. On the a3 nodes that root used to fall through
# to /tmp, which sits on the 200G boot PD shared with the OS: eight concurrent
# cells on 2026-09-02 took / to 42G free, and the next pack would have hit ENOSPC
# mid-training. /mnt/localssd is 5.9T of RAID0 NVMe on the same node -- more room
# on faster disk, and it dies with the node the same way /tmp does.
#
# This lives here rather than in a submit script on purpose: sbatch snapshots the
# batch script at submit time, so a fix in ptb_pack.sbatch cannot reach a job that
# is already queued, but every entry point sources this file at RUN time.
#
# Guarded rather than assumed, and never an override: an unmounted /mnt/localssd
# would silently write to the root fs this is meant to spare, so the branch is
# taken only when it is a real mountpoint with room, and an explicit
# POST_TRAIN_BENCH_TMP_ROOT always wins. Everywhere else this is a no-op.
if [ -z "${POST_TRAIN_BENCH_TMP_ROOT:-}" ] && mountpoint -q /mnt/localssd 2>/dev/null; then
    _LSSD_AVAIL_G="$(df -BG --output=avail /mnt/localssd 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [ "${_LSSD_AVAIL_G:-0}" -ge 500 ] 2>/dev/null; then
        export POST_TRAIN_BENCH_TMP_ROOT="/mnt/localssd/tmp/ptb"
        mkdir -p "$POST_TRAIN_BENCH_TMP_ROOT" 2>/dev/null
    fi
    unset _LSSD_AVAIL_G
fi

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    source /etc/profile.d/modules.sh
fi

export PYTHONNOUSERSITE=1

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi