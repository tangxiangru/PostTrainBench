#!/bin/bash
# Synthetic OS/MCP/model acceptance in the deployed image; no PTB score is produced.
set -euo pipefail

[ "$#" -eq 2 ] || { echo "Usage: $0 <agent> <isolated-home>" >&2; exit 2; }
AGENT="$1"
PROBE_HOME="$2"
: "${POST_TRAIN_BENCH_WMA_SIDECAR_CHECKOUT:?private checkout required}"
: "${POST_TRAIN_BENCH_WMA_VALIDATION_DIR:?validation output root required}"
: "${POST_TRAIN_BENCH_WMA_CHECKOUT_SHA:?private source identity required}"
mkdir -p "$PROBE_HOME" "$POST_TRAIN_BENCH_WMA_VALIDATION_DIR"

ENV_ARGS=()
while IFS= read -r name || [ -n "$name" ]; do
    [[ -z "$name" || "$name" =~ ^[[:space:]]*# ]] && continue
    [[ "$name" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "Invalid provider variable name" >&2; exit 1; }
    [ -n "${!name+x}" ] && ENV_ARGS+=(--env "${name}=${!name}")
done < "agents/${AGENT}/env_passthrough.txt"

apptainer exec --containall --cleanenv --writable-tmpfs \
    --env 'PATH=/root/.local/bin:/home/ben/.local/bin:/usr/local/bin:/usr/bin:/bin' \
    --env PYTHONPATH=/opt/awm \
    --env CLAUDE_CONFIG_DIR=/home/ben/.claude-wma-smoke \
    --env "POST_TRAIN_BENCH_WMA_CHECKOUT_SHA=$POST_TRAIN_BENCH_WMA_CHECKOUT_SHA" \
    --env "SLURM_JOB_ID=$SLURM_JOB_ID" --env "SLURMD_NODENAME=$SLURMD_NODENAME" \
    "${ENV_ARGS[@]}" \
    --bind "$POST_TRAIN_BENCH_WMA_SIDECAR_CHECKOUT:/opt/awm:ro" \
    --bind "$POST_TRAIN_BENCH_WMA_VALIDATION_DIR:/acceptance" \
    --home "$PROBE_HOME:/home/ben" --pwd /home/ben \
    "$POST_TRAIN_BENCH_CONTAINERS_DIR/$POST_TRAIN_BENCH_CONTAINER_NAME.sif" \
    python3 -m awm.wma.runtime_smoke --out "/acceptance/$SLURM_JOB_ID" \
        --model "$POST_TRAIN_BENCH_WMA_MODEL" --effort "$POST_TRAIN_BENCH_WMA_EFFORT"

echo "WMA acceptance: $POST_TRAIN_BENCH_WMA_VALIDATION_DIR/$SLURM_JOB_ID/acceptance.json"
