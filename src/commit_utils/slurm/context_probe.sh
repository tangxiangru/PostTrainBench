#!/bin/bash

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <agent> <model> <record.json> <isolated-home>" >&2
    exit 2
fi

AGENT="$1"
REQUESTED_MODEL="$2"
RECORD="$3"
PROBE_HOME="$4"
PROFILE="agents/${AGENT}/profile.env"
PASSTHROUGH="agents/${AGENT}/env_passthrough.txt"

[ -r "$PROFILE" ] || { echo "ERROR: agent profile not found: $PROFILE" >&2; exit 1; }
source "$PROFILE"
[ "$PTB_AGENT_PROVIDER" = "vertex" ] || { echo "ERROR: context probe requires a Vertex agent profile" >&2; exit 1; }

mkdir -p "$(dirname "$RECORD")" "$PROBE_HOME/.claude-context-probe"
RAW="${RECORD%.json}.stream.json"
IMAGE="${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif"

ENV_ARGS=()
while IFS= read -r env_name || [ -n "$env_name" ]; do
    [[ -z "$env_name" || "$env_name" =~ ^[[:space:]]*# ]] && continue
    [[ "$env_name" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "ERROR: invalid passthrough variable: $env_name" >&2; exit 1; }
    [ -n "${!env_name+x}" ] && ENV_ARGS+=(--env "${env_name}=${!env_name}")
done < "$PASSTHROUGH"

set +e
apptainer exec --containall --cleanenv \
    --env 'PATH=/root/.local/bin:/home/ben/.local/bin:/usr/local/bin:/usr/bin:/bin' \
    --env CLAUDE_CONFIG_DIR=/home/ben/.claude-context-probe \
    --env CLAUDE_CODE_EFFORT_LEVEL="$PTB_AGENT_EFFORT" \
    "${ENV_ARGS[@]}" \
    --home "$PROBE_HOME:/home/ben" \
    --pwd /home/ben \
    --writable-tmpfs "$IMAGE" \
    claude --print --verbose --output-format stream-json \
        --model "$REQUESTED_MODEL" --effort "$PTB_AGENT_EFFORT" \
        --setting-sources '' --safe-mode --no-session-persistence \
        --dangerously-skip-permissions 'Reply with exactly OK.' \
    2>&1 | tee "$RAW"
probe_status=${PIPESTATUS[0]}
set -e

python3 - "$RAW" "$RECORD" "$REQUESTED_MODEL" \
    "$PTB_AGENT_REQUESTED_CONTEXT_TOKENS" "$PTB_AGENT_EFFORT" "$IMAGE" "$probe_status" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

raw_path, record_path, requested_model, requested_context, effort, image, probe_status = sys.argv[1:]
events = []
for line in Path(raw_path).read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass
init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
result = next((e for e in reversed(events) if e.get("type") == "result"), {})
usage = result.get("modelUsage") or {}
model_key, model_usage = next(iter(usage.items()), (None, {}))
resolved_context = int(model_usage.get("contextWindow", 0))
requested_context = int(requested_context)
verified = (
    int(probe_status) == 0
    and not result.get("is_error")
    and result.get("api_error_status") in (None, 0)
    and resolved_context >= requested_context
)
raw_bytes = Path(raw_path).read_bytes()
image_path = Path(image)
image_digest = hashlib.sha256()
with image_path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
        image_digest.update(chunk)
actual_image_sha256 = image_digest.hexdigest()
expected_image_sha256 = os.environ.get("POST_TRAIN_BENCH_CONTAINER_SHA256", "")
if expected_image_sha256 and actual_image_sha256 != expected_image_sha256:
    raise SystemExit(
        f"context probe container digest mismatch: actual={actual_image_sha256} "
        f"expected={expected_image_sha256}"
    )
record = {
    "schema_version": 1,
    "verified": verified,
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "provider": "vertex",
    "project": os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"),
    "region": os.environ.get("ANTHROPIC_VERTEX_REGION"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "gpu_ids": os.environ.get("SLURM_JOB_GPUS"),
    "cli_version": init.get("claude_code_version", "unknown"),
    "container": image_path.name,
    "container_sha256": actual_image_sha256,
    "requested_model": requested_model,
    "resolved_model": model_key or init.get("model", "unknown"),
    "canonical_model": model_usage.get("canonicalModel"),
    "requested_context_tokens": requested_context,
    "resolved_context_tokens": resolved_context,
    "effort": effort,
    "terminal_reason": result.get("terminal_reason"),
    "api_error_status": result.get("api_error_status"),
    "raw_trace": str(Path(raw_path).resolve()),
    "raw_trace_sha256": hashlib.sha256(raw_bytes).hexdigest(),
}
Path(record_path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
if not verified:
    raise SystemExit(
        f"context probe failed: model={requested_model} resolved_context={resolved_context} "
        f"api_error_status={record['api_error_status']}"
    )
print(f"Verified {requested_model}: contextWindow={resolved_context}")
PY
