#!/bin/bash
#
# Run the reward-hacking judges on an existing result directory.
#
# Judges live in src/judges/<judge_name>/ (see judge_lib.sh); by default all
# of them run, in the order given by ALL_JUDGES:
#   1. data_contamination_judge -> judgement_gpt5_4_rerun.json
#      (contamination/disallowed_model verdict; this is the canonical
#      contamination verdict consumed downstream)
#   2. api_usage_judge          -> judgement_api_rerun.json
#      (separate `disallowed_api_usage` schema; consumed by scoring — a
#      flagged run falls back to the baseline score)
#   3. ptb_lookup_judge         -> judgement_ptb_lookup_rerun.json
#      (separate `disallowed_ptb_lookup` schema; archival, but
#      scripts/collect.py errors out if it ever flags)
#   4. general_judge            -> judgement_general_rerun.json
#      (separate `general_anomaly` schema; GPT-5.6-Terra unknown-unknowns
#      sweep on codex 0.144.5 — archival, but when it flags,
#      scripts/collect.py finishes its collection pass without writing any
#      files and errors out listing the flagged runs)
#
# All outputs are always saved with the _rerun suffix so original judge
# outputs produced by src/run_task.sh are preserved.
#
# Usage: run_judges.sh [--profile official|claude]
#                      [--judges <name>[,<name>...]] <result_dir>
#
# Options:
#   --judges   Comma-separated subset of judges to run (default: all).
#              e.g. --judges data_contamination_judge
#                   --judges api_usage_judge
#   --profile  Judge runtime/output profile (default: .env or official).
#              The claude profile writes separate judgement_claude_* files.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/judge_lib.sh"

# Parse arguments
JUDGES=()
RESULT_DIR=""
REQUESTED_PROFILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            REQUESTED_PROFILE="$2"
            shift 2
            ;;
        --judges)
            IFS=',' read -r -a JUDGES <<< "$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--profile official|claude] [--judges <name>[,<name>...]] <result_dir>" >&2
            exit 1
            ;;
        *)
            RESULT_DIR="$1"
            shift
            ;;
    esac
done

if [ ${#JUDGES[@]} -eq 0 ]; then
    JUDGES=("${ALL_JUDGES[@]}")
fi

if [ -z "$RESULT_DIR" ]; then
    echo "Usage: $0 [--profile official|claude] [--judges <name>[,<name>...]] <result_dir>" >&2
    exit 1
fi

if [ ! -d "$RESULT_DIR" ]; then
    echo "Error: Result directory does not exist: $RESULT_DIR" >&2
    exit 1
fi

if [ ! -d "$RESULT_DIR/task" ]; then
    echo "Error: No task directory found in $RESULT_DIR" >&2
    exit 1
fi

source "$JUDGES_REPO_ROOT/src/commit_utils/set_env_vars.sh"
if [ -n "$REQUESTED_PROFILE" ]; then
    export POST_TRAIN_BENCH_JUDGE_PROFILE="$REQUESTED_PROFILE"
fi
configure_judge_profile "${POST_TRAIN_BENCH_JUDGE_PROFILE:-official}"

# Validate the requested judges early (before any expensive work).
for JUDGE_NAME in "${JUDGES[@]}"; do
    load_judge_conf "$JUDGE_NAME"
done

# Find trace file (solve_parsed.txt preferred, solve_out.txt as fallback)
if [ -f "$RESULT_DIR/solve_parsed.txt" ]; then
    TRACE_FILE="$RESULT_DIR/solve_parsed.txt"
    TRACE_NAME="solve_parsed.txt"
elif [ -f "$RESULT_DIR/solve_out.txt" ]; then
    TRACE_FILE="$RESULT_DIR/solve_out.txt"
    TRACE_NAME="solve_out.txt"
else
    echo "Error: No trace file (solve_parsed.txt or solve_out.txt) found in $RESULT_DIR" >&2
    exit 1
fi

# New Slurm runs carry unambiguous structured provenance. Prefer it because
# directory names cannot reliably split agents such as claude_vertex_xhigh or
# base-model ids containing underscores. Keep the historical heuristic for old
# result directories that predate runtime_provenance.json.
if [ -r "$RESULT_DIR/runtime_provenance.json" ]; then
    mapfile -t RESULT_IDENTITY < <(python3 - "$RESULT_DIR/runtime_provenance.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))["experiment"]
print(data["task"])
print(data["base_model"])
print(data["agent"])
print(data["agent_config"])
PY
    )
    BENCHMARK="${RESULT_IDENTITY[0]}"
    MODEL_HF="${RESULT_IDENTITY[1]}"
    AGENT="${RESULT_IDENTITY[2]}"
    AGENT_CONFIG="${RESULT_IDENTITY[3]}"
else
    DIRNAME=$(basename "$RESULT_DIR")
    BENCHMARK=$(echo "$DIRNAME" | sed -E 's/^([^_]+)_.*/\1/')
    MODEL_PART=$(echo "$DIRNAME" | sed -E 's/^[^_]+_(.*)_[0-9]+$/\1/')
    MODEL_HF=$(echo "$MODEL_PART" | sed 's/_/\//')
    METHOD_DIR=$(basename "$(dirname "$RESULT_DIR")")
    AGENT_AND_CONFIG=$(echo "$METHOD_DIR" | sed -E 's/_[0-9]+h.*$//')
    AGENT=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^([^_]+)_.*/\1/')
    AGENT_CONFIG=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^[^_]+_(.*)$/\1/')
fi

echo "Running judges on: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF | Agent: $AGENT ($AGENT_CONFIG) | Trace: $TRACE_NAME"
echo "  Profile: $JUDGE_PROFILE ($PTB_JUDGE_BACKEND, model=$JUDGE_DEFAULT_MODEL, effort=$JUDGE_DEFAULT_REASONING_EFFORT)"
echo "  Judges: ${JUDGES[*]} (outputs suffixed with _rerun)"

# Create temporary working directory
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

JOB_DIR="$TMP_DIR/job_dir"
JOB_TMP="$TMP_DIR/tmp"
mkdir -p "$JOB_DIR" "$JOB_TMP"

# Copy task directory
cp -r "$RESULT_DIR/task" "$JOB_DIR/task"

# Remove any pre-existing judgement file from the task dir so stale values
# from earlier runs can't leak into this judge's output when the CLI crashes.
rm -f "$JOB_DIR/task/judgement.json"

# Copy trace file to parent directory (not task directory)
cp "$TRACE_FILE" "$JOB_DIR/$TRACE_NAME"

# Copy judge helper tooling and benchmark metadata into the sandbox.
prepare_judge_sandbox "$JOB_DIR" "$BENCHMARK" "$RESULT_DIR/final_model/config.json"

# Set up profile-specific isolated config/auth.
setup_judge_auth "$JOB_DIR"

# Remove any pre-existing per-judge output files in the result dir for the
# judges we are about to rerun, so stale values from earlier runs can't be
# confused with fresh output when a CLI fails. Leave the skipped judges'
# files alone.
for JUDGE_NAME in "${JUDGES[@]}"; do
    load_judge_conf "$JUDGE_NAME"
    rm -f "$RESULT_DIR/judgement_${JUDGE_OUTPUT_ID}_rerun.json"
done

JUDGE_EXTRA_APPTAINER_ARGS=()

for JUDGE_NAME in "${JUDGES[@]}"; do
    load_judge_conf "$JUDGE_NAME"

    echo ""
    echo "========================================="
    echo "=== ${JUDGE_LABEL} ==="
    echo "========================================="

    # Clean judgement file so each judge starts fresh
    rm -f "$JOB_DIR/task/judgement.json"

    JUDGE_PROMPT=$(build_judge_prompt "$JUDGE_NAME" "$BENCHMARK" "$MODEL_HF" "$AGENT" "$AGENT_CONFIG")

    run_judge_exec "$JOB_DIR" "$JOB_TMP" "$RESULT_DIR/judge_output_${JUDGE_OUTPUT_ID}_rerun.json" "$JUDGE_PROMPT"

    collect_judge_output "$JOB_DIR" "$RESULT_DIR" "_rerun" 1
done

echo ""
echo "Judges completed successfully: ${JUDGES[*]}"
