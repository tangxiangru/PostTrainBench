#!/bin/bash

# Make one real, minimal Claude Code request through the same isolated Vertex
# path used by the PTB official Claude judge profile. Intended for Slurm node preflight;
# it writes no benchmark result and prints no credential material.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source src/commit_utils/set_env_vars.sh
if [ -n "${POST_TRAIN_BENCH_APPTAINER_BIN:-}" ]; then
    export PATH="$(dirname "$POST_TRAIN_BENCH_APPTAINER_BIN"):${PATH}"
fi
if [ -n "${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

SCRATCH_PARENT="${POST_TRAIN_BENCH_SLURM_SCRATCH_BASE:-/tmp/posttrainbench}/${USER}"
mkdir -p "$SCRATCH_PARENT"
SMOKE_ROOT="$(mktemp -d "${SCRATCH_PARENT%/}/ptb-vertex-smoke-${SLURM_JOB_ID:-local}-XXXXXX")"

cleanup() {
    case "$SMOKE_ROOT" in
        "${SCRATCH_PARENT%/}"/ptb-vertex-smoke-*) rm -rf -- "$SMOKE_ROOT" ;;
        *) echo "WARNING: refused unsafe smoke cleanup path: $SMOKE_ROOT" >&2 ;;
    esac
}
trap cleanup EXIT

JOB_DIR="$SMOKE_ROOT/job_dir"
JOB_TMP="$SMOKE_ROOT/tmp"
OUT_DIR="$SMOKE_ROOT/out"
mkdir -p "$JOB_DIR/task" "$JOB_TMP" "$OUT_DIR"

export POST_TRAIN_BENCH_JUDGE_PROFILE="official"
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE="vertex"
export POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="${POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL:-claude-opus-5[1m]}"
export POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER="${POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER:-opus_5.sif}"

source src/judges/judge_lib.sh
configure_judge_profile official
load_judge_conf data_contamination_judge
setup_judge_auth "$JOB_DIR"
JUDGE_EXTRA_APPTAINER_ARGS=()

RAW_TRACE="$OUT_DIR/judge_output_vertex_smoke.json"
MARKER="PTB_VERTEX_JUDGE_SMOKE_OK"
run_judge_exec "$JOB_DIR" "$JOB_TMP" "$RAW_TRACE" \
    "Do not use tools. Reply with exactly ${MARKER} and nothing else."

python3 - "$RAW_TRACE" "$OUT_DIR/judge_metadata_vertex_smoke.json" "$MARKER" "${SLURMD_NODENAME:-$(hostname)}" <<'PY'
import json
import sys
from pathlib import Path

trace_path, metadata_path = map(Path, sys.argv[1:3])
marker, node = sys.argv[3:]
events = []
for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        events.append(json.loads(line))
    except json.JSONDecodeError:
        pass

init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
result = next((e for e in events if e.get("type") == "result"), None)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
assert init is not None, "Claude init event missing"
assert result is not None and result.get("subtype") == "success" and not result.get("is_error"), result
assert result.get("result") == marker, result
assert metadata.get("auth_mode") == "vertex", metadata
assert metadata.get("reasoning_effort") == "high", metadata
assert metadata.get("resolved_model") == init.get("model"), (metadata, init)
assert "opus-5" in init.get("model", ""), init
model_usage = result.get("modelUsage") or {}
usage = next(iter(model_usage.values()), {})
assert usage.get("contextWindow", 0) >= 1_000_000, result

print(json.dumps({
    "node": node,
    "status": "ok",
    "auth_mode": metadata["auth_mode"],
    "requested_model": metadata["requested_model"],
    "resolved_model": metadata["resolved_model"],
    "effort": metadata["reasoning_effort"],
    "cli_version": metadata["cli_version"],
    "context_window": usage["contextWindow"],
    "duration_ms": result.get("duration_ms"),
}, sort_keys=True))
PY
