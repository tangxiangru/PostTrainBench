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
#      sweep on codex 0.144.5 — archival and unconsumed: scripts/collect.py
#      never reads this file, so a flag here has no effect on anything and
#      has to be found by reading the judgements)
#
# All outputs are always saved with the _rerun suffix so original judge
# outputs produced by src/run_task.sh are preserved.
#
# Usage: run_judges.sh [--judges <name>[,<name>...]] <result_dir>
#
# Options:
#   --judges   Comma-separated subset of judges to run (default: all).
#              e.g. --judges data_contamination_judge
#                   --judges api_usage_judge

# pipefail: every judge invocation is `apptainer ... | tee`, and tee always
# succeeds, so without it a codex that dies on a revoked session exits 0 here.
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/judge_lib.sh"

# Parse arguments
JUDGES=()
RESULT_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --judges)
            IFS=',' read -r -a JUDGES <<< "$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--judges <name>[,<name>...]] <result_dir>" >&2
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
    echo "Usage: $0 [--judges <name>[,<name>...]] <result_dir>" >&2
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

source "$JUDGES_REPO_ROOT/src/commit_utils/set_env_vars.sh"

# Parse result directory to get benchmark and model
# Format: {benchmark}_{provider}_{model}_{cluster_id}[_g{gpu_seat}]
#
# The `_g<N>` tail is the GPU seat a pack job ran the cell on, e.g.
# gsm8k_Qwen_Qwen3-1.7B-Base_89727_g3 is seat 3 of job 89727. It has to come off
# before the `_[0-9]+$` anchor below, or that substitution simply does not match
# and sed passes the input through unchanged — MODEL_PART becomes the whole
# directory name and the judge is told the model under test is
# "gsm8k/Qwen_Qwen3-1.7B-Base_89727_g3". No error, just a judge reasoning about a
# model that does not exist.
DIRNAME=$(basename "$RESULT_DIR")
DIRNAME_NOSEAT=$(echo "$DIRNAME" | sed -E 's/_g[0-9]+$//')
BENCHMARK=$(echo "$DIRNAME_NOSEAT" | sed -E 's/^([^_]+)_.*/\1/')
MODEL_PART=$(echo "$DIRNAME_NOSEAT" | sed -E 's/^[^_]+_(.*)_[0-9]+$/\1/')
if [ "$MODEL_PART" = "$DIRNAME_NOSEAT" ]; then
    echo "Error: cannot parse a model out of '$DIRNAME'" >&2
    echo "       expected {benchmark}_{provider}_{model}_{cluster_id}[_g{seat}]" >&2
    exit 1
fi
MODEL_HF=$(echo "$MODEL_PART" | sed 's/_/\//')

# Parse the parent (method) directory to get the agent + its harness model.
# Format: {agent}_{agent_config}_{num_hours}h[_{num_gpus}gpu][{experiment_name}]
METHOD_DIR=$(basename "$(dirname "$RESULT_DIR")")
AGENT_AND_CONFIG=$(echo "$METHOD_DIR" | sed -E 's/_[0-9]+h.*$//')
AGENT=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^([^_]+)_.*/\1/')
AGENT_CONFIG=$(echo "$AGENT_AND_CONFIG" | sed -E 's/^[^_]+_(.*)$/\1/')

echo "Running judges on: $RESULT_DIR"
echo "  Benchmark: $BENCHMARK | Model: $MODEL_HF | Agent: $AGENT ($AGENT_CONFIG) | Trace: $TRACE_NAME"
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

# Set up codex config + ChatGPT Pro subscription auth (JUDGE_CODEX_AUTH_SRC).
setup_judge_codex_auth "$JOB_DIR"

JUDGE_EXTRA_APPTAINER_ARGS=()

for JUDGE_NAME in "${JUDGES[@]}"; do
    load_judge_conf "$JUDGE_NAME"

    echo ""
    echo "========================================="
    echo "=== ${JUDGE_LABEL} ==="
    echo "========================================="

    # Drop this judge's previous verdict immediately before running it, not up
    # front for the whole set. Deleting all of them first means that when judge 2
    # of 4 dies, judges 3 and 4 have lost their old verdicts without producing new
    # ones -- one failure silently erases work the failure had nothing to do with.
    # Now a judge only ever destroys its own prior output, and only once it is
    # about to replace it.
    rm -f "$RESULT_DIR/judgement_${JUDGE_OUTPUT_ID}_rerun.json" \
          "$RESULT_DIR/judgement_${JUDGE_OUTPUT_ID}_rerun.json.REJECTED"

    # Clean judgement file so each judge starts fresh
    rm -f "$JOB_DIR/task/judgement.json"

    JUDGE_PROMPT=$(build_judge_prompt "$JUDGE_NAME" "$BENCHMARK" "$MODEL_HF" "$AGENT" "$AGENT_CONFIG")

    run_judge_exec "$JOB_DIR" "$JOB_TMP" "$RESULT_DIR/judge_output_${JUDGE_OUTPUT_ID}_rerun.json" "$JUDGE_PROMPT"

    collect_judge_output "$JOB_DIR" "$RESULT_DIR" "_rerun" 1
done

echo ""
echo "Judges completed successfully: ${JUDGES[*]}"
