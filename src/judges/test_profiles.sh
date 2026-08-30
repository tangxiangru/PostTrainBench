#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

mkdir -p "$TMP_ROOT/bin" "$TMP_ROOT/containers" "$TMP_ROOT/official/job/task" \
    "$TMP_ROOT/official/tmp" "$TMP_ROOT/official/out" "$TMP_ROOT/claude/job/task" \
    "$TMP_ROOT/claude/tmp" "$TMP_ROOT/claude/out"
ln -s "$SCRIPT_DIR/tests/mock_apptainer.sh" "$TMP_ROOT/bin/apptainer"
export PATH="$TMP_ROOT/bin:$PATH"
export MOCK_APPTAINER_LOG="$TMP_ROOT/apptainer.log"
export POST_TRAIN_BENCH_CONTAINERS_DIR="$TMP_ROOT/containers"
touch "$TMP_ROOT/containers/gpt_5_5.sif" "$TMP_ROOT/containers/opus_5.sif"
printf '%s\n' \
    "POST_TRAIN_BENCH_CONTAINERS_DIR=\"$TMP_ROOT/containers\"" \
    "POST_TRAIN_BENCH_RESULTS_DIR=\"$TMP_ROOT/results\"" \
    'POST_TRAIN_BENCH_PROMPT="prompt"' > "$TMP_ROOT/test.env"
export POST_TRAIN_BENCH_ENV_FILE="$TMP_ROOT/test.env"

source "$SCRIPT_DIR/judge_lib.sh"
JUDGE_EXTRA_APPTAINER_ARGS=()

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_eq() {
    [ "$1" = "$2" ] || fail "expected '$2', got '$1'"
}

# No selection must retain the official/canonical behavior.
unset POST_TRAIN_BENCH_JUDGE_PROFILE
configure_judge_profile
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE="chatgpt"
resolve_judge_auth_mode
load_judge_conf data_contamination_judge
assert_eq "$JUDGE_PROFILE" "official"
assert_eq "$PTB_JUDGE_BACKEND" "codex"
assert_eq "$JUDGE_MODEL" "gpt-5.4"
assert_eq "$JUDGE_OUTPUT_ID" "gpt5_4"
OFFICIAL_PROMPT="$(build_judge_prompt data_contamination_judge gsm8k google/gemma-3-4b-pt claude opus)"
load_judge_conf general_judge
assert_eq "$JUDGE_MODEL" "gpt-5.6-terra"
assert_eq "$JUDGE_CODEX_VERSION" "0.144.5"
assert_eq "$JUDGE_OUTPUT_ID" "general"
load_judge_conf data_contamination_judge

mkdir -p "$TMP_ROOT/official/job/.codex"
touch "$TMP_ROOT/official/codex-auth.json" "$TMP_ROOT/official/job/.codex/auth.json"
JUDGE_CODEX_AUTH_SRC="$TMP_ROOT/official/codex-auth.json"
run_judge_exec "$TMP_ROOT/official/job" "$TMP_ROOT/official/tmp" \
    "$TMP_ROOT/official/out/judge_output_gpt5_4_rerun.json" "OFFICIAL_PROMPT_SENTINEL"
collect_judge_output "$TMP_ROOT/official/job" "$TMP_ROOT/official/out" "_rerun" 1
[ -f "$TMP_ROOT/official/out/judgement_gpt5_4_rerun.json" ] || fail "official judgement missing"
grep -q 'MOCK_CODEX_COMMAND' "$MOCK_APPTAINER_LOG" || fail "official profile did not invoke codex"

# Claude selection changes backend and names, not the judge task/prompt.
export POST_TRAIN_BENCH_JUDGE_PROFILE="claude"
export POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="opus"
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE="claude_oauth"
printf '%s\n' 'judge-only-mock-token' > "$TMP_ROOT/claude-oauth-token"
chmod 600 "$TMP_ROOT/claude-oauth-token"
export POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE="$TMP_ROOT/claude-oauth-token"
configure_judge_profile
load_judge_conf data_contamination_judge
assert_eq "$JUDGE_PROFILE" "claude"
assert_eq "$PTB_JUDGE_BACKEND" "claude"
assert_eq "$JUDGE_MODEL" "opus"
assert_eq "$JUDGE_REASONING_EFFORT" "xhigh"
assert_eq "$JUDGE_OUTPUT_ID" "claude_contamination"
CLAUDE_PROMPT="$(build_judge_prompt data_contamination_judge gsm8k google/gemma-3-4b-pt claude opus)"
assert_eq "$CLAUDE_PROMPT" "$OFFICIAL_PROMPT"

declare -A EXPECTED_CLAUDE_IDS=(
    [data_contamination_judge]="claude_contamination"
    [api_usage_judge]="claude_api"
    [ptb_lookup_judge]="claude_ptb_lookup"
    [general_judge]="claude_general"
)
for judge_name in "${ALL_JUDGES[@]}"; do
    load_judge_conf "$judge_name"
    assert_eq "$JUDGE_OUTPUT_ID" "${EXPECTED_CLAUDE_IDS[$judge_name]}"
    assert_eq "$JUDGE_MODEL" "opus"
    assert_eq "$JUDGE_REASONING_EFFORT" "xhigh"
done
load_judge_conf data_contamination_judge

setup_judge_auth "$TMP_ROOT/claude/job"
run_judge_exec "$TMP_ROOT/claude/job" "$TMP_ROOT/claude/tmp" \
    "$TMP_ROOT/claude/out/judge_output_claude_contamination_rerun.json" "CLAUDE_PROMPT_SENTINEL"
collect_judge_output "$TMP_ROOT/claude/job" "$TMP_ROOT/claude/out" "_rerun" 1

[ -f "$TMP_ROOT/claude/out/judgement_claude_contamination_rerun.json" ] || fail "Claude judgement missing"
[ ! -e "$TMP_ROOT/claude/out/judgement_gpt5_4_rerun.json" ] || fail "Claude profile wrote a canonical judgement"
grep -q 'MOCK_CLAUDE_ARGS --model opus --effort xhigh --prompt CLAUDE_PROMPT_SENTINEL' "$MOCK_APPTAINER_LOG" || \
    fail "Claude command did not receive model/effort/prompt"
grep -q -- '--safe-mode' "$MOCK_APPTAINER_LOG" || fail "Claude judge did not enable safe mode"
grep -q -- '--no-session-persistence' "$MOCK_APPTAINER_LOG" || \
    fail "Claude judge did not disable session persistence"
if grep 'MOCK_CLAUDE_ARGS' "$MOCK_APPTAINER_LOG" | grep -q 'max'; then
    fail "Claude judge command used max instead of xhigh"
fi
grep -q 'Session start — mock-claude' "$TMP_ROOT/claude/out/judge_output_claude_contamination_rerun.txt" || \
    fail "Claude stream-json was not parsed by claude_parser"

python3 - "$TMP_ROOT/claude/out/judge_metadata_claude_contamination_rerun.json" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "profile": "claude",
    "backend": "claude",
    "auth_mode": "claude_oauth",
    "requested_model": "opus",
    "resolved_model": "opus",
    "reasoning_effort": "xhigh",
    "container": "opus_5.sif",
    "cli_version": "2.1.219 (Claude Code)",
}
assert metadata == expected, (metadata, expected)
PY

# With no explicit auth mode, a site-level Vertex declaration must select
# Vertex instead of requiring an Anthropic OAuth token file.
unset POST_TRAIN_BENCH_JUDGE_AUTH_MODE
export CLAUDE_CODE_USE_VERTEX=1
resolve_judge_auth_mode
assert_eq "$JUDGE_AUTH_MODE" "vertex"
export POST_TRAIN_BENCH_JUDGE_AUTH_MODE="claude_oauth"

# Exercise the public standalone selector too; --profile must override .env
# and retain the isolated output id all the way through the runner.
INTEGRATION_RESULT="$TMP_ROOT/results/claude_opus_10h/gsm8k_google_gemma-3-4b-pt_123"
mkdir -p "$INTEGRATION_RESULT/task"
printf '%s\n' 'mock parsed agent trace' > "$INTEGRATION_RESULT/solve_parsed.txt"
bash "$SCRIPT_DIR/run_judges.sh" --profile claude \
    --judges data_contamination_judge "$INTEGRATION_RESULT"
[ -f "$INTEGRATION_RESULT/judgement_claude_contamination_rerun.json" ] || \
    fail "run_judges.sh --profile claude lost the Claude output id"
[ ! -e "$INTEGRATION_RESULT/judgement_gpt5_4_rerun.json" ] || \
    fail "run_judges.sh --profile claude wrote a canonical output"

echo "Judge profile tests passed"
