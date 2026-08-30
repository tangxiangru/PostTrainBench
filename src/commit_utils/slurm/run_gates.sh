#!/bin/bash

set -euo pipefail

GATE="${1:?usage: $0 g1|g2|g3 [node]}"
NODE="${2:-slurm2-a3nodesetondem-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source src/commit_utils/set_env_vars.sh

PARTITION="${POST_TRAIN_BENCH_SLURM_PARTITION:?partition required}"
GATE_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/_slurm_gates/${GATE}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$GATE_DIR"

submit_worker() {
    local mode="$1" hold="$2"
    local -a reservation_args=()
    [ -n "${POST_TRAIN_BENCH_SLURM_RESERVATION:-}" ] && reservation_args+=(--reservation="$POST_TRAIN_BENCH_SLURM_RESERVATION")
    sbatch --parsable --partition="$PARTITION" --nodelist="$NODE" \
        "${reservation_args[@]}" \
        --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G --gres=gpu:1 \
        --time=00:10:00 --chdir="$REPO_ROOT" \
        --output="$GATE_DIR/%x-%j.out" --error="$GATE_DIR/%x-%j.err" \
        --job-name="ptb-${GATE}-${mode}" \
        "$SCRIPT_DIR/gate_worker.sbatch" "$mode" "$GATE_DIR" "$hold" \
        | cut -d';' -f1
}

wait_jobs() {
    local job_ids="$1" states
    while :; do
        states="$(sacct -nX -j "$job_ids" --format=State --parsable2 | awk 'NF {print $1}')"
        if [ "$(printf '%s\n' "$states" | wc -l)" -ge "$(tr ',' '\n' <<< "$job_ids" | wc -l)" ] \
            && ! grep -Eq '^(PENDING|RUNNING|CONFIGURING|COMPLETING)$' <<< "$states"; then
            break
        fi
        sleep 2
    done
    if grep -Evq '^COMPLETED$' <<< "$states"; then
        echo "ERROR: gate jobs did not all complete: $states" >&2
        sacct -j "$job_ids" --format=JobID,JobName,State,ExitCode,AllocCPUS,AllocTRES
        return 1
    fi
}

case "$GATE" in
    g1)
        first="$(submit_worker canary 60)"
        second="$(submit_worker canary 60)"
        jobs="${first},${second}"
        wait_jobs "$jobs"
        [ "$(cat "$GATE_DIR"/*.uuid | sort -u | wc -l)" -eq 2 ]
        sacct -nPX -j "$jobs" --format=JobIDRaw,State,AllocCPUS,AllocTRES > "$GATE_DIR/sacct.txt"
        if ! awk -F'|' 'NF {seen++; if ($2 !~ /^COMPLETED/ || $3 != 16 || $4 !~ /gres\/gpu=1/) bad=1} END {exit !(seen == 2 && !bad)}' "$GATE_DIR/sacct.txt"; then
            echo "ERROR: G1 allocation shape is not cpu=16,gres/gpu=1" >&2
            cat "$GATE_DIR/sacct.txt" >&2
            exit 1
        fi
        ;;
    g2)
        jobs=""
        for _ in $(seq 1 8); do
            job="$(submit_worker canary 90)"
            jobs="${jobs:+${jobs},}${job}"
        done
        ninth="$(submit_worker canary 10)"
        for _ in $(seq 1 60); do
            running="$(squeue -h -j "$jobs" -t R -o '%i' | wc -l)"
            ninth_state="$(squeue -h -j "$ninth" -o '%T' || true)"
            [ "$running" -eq 8 ] && [ "$ninth_state" = "PENDING" ] && break
            sleep 2
        done
        [ "$running" -eq 8 ] || { echo "ERROR: eight canaries never ran concurrently" >&2; exit 1; }
        [ "$ninth_state" = "PENDING" ] || { echo "ERROR: ninth canary was not held pending" >&2; exit 1; }
        wait_jobs "${jobs},${ninth}"
        [ "$(cat "$GATE_DIR"/*.uuid | sort -u | wc -l)" -eq 8 ]
        ;;
    g3)
        survivor="$(submit_worker survivor 90)"
        reaper="$(submit_worker reaper 60)"
        wait_jobs "${survivor},${reaper}"
        [ -s "$GATE_DIR/reaper-survivor-proof.txt" ]
        ;;
    *) echo "ERROR: gate must be g1, g2, or g3" >&2; exit 2 ;;
esac

echo "${GATE^^} PASSED: $GATE_DIR"
