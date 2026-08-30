#!/bin/bash

set -euo pipefail

: "${MOCK_APPTAINER_LOG:?MOCK_APPTAINER_LOG is required}"

{
    echo "--- invocation ---"
    printf '<%s>\n' "$@"
} >> "$MOCK_APPTAINER_LOG"

home_src=""
previous=""
is_version=0
is_help=0
is_claude=0
is_codex=0
claude_marker_index=0
index=0
for arg in "$@"; do
    index=$((index + 1))
    if [ "$previous" = "--home" ]; then
        home_src="${arg%%:*}"
    fi
    [ "$arg" = "--version" ] && is_version=1
    [ "$arg" = "--help" ] && is_help=1
    [ "$arg" = "claude" ] && is_claude=1
    [[ "$arg" = *codex ]] && is_codex=1
    if [ "$arg" = "ptb-claude-judge" ]; then
        claude_marker_index="$index"
    fi
    previous="$arg"
done

if [ "$is_claude" = "1" ] && [ "$is_version" = "1" ]; then
    echo "2.1.219 (Claude Code)"
    exit 0
fi
if [ "$is_claude" = "1" ] && [ "$is_help" = "1" ]; then
    echo "Usage: claude [--effort level] [--setting-sources sources] [--safe-mode]"
    exit 0
fi

if [ "$claude_marker_index" -gt 0 ]; then
    # Arguments after the bash argv[0] marker are model, effort, prompt.
    eval "model=\${$((claude_marker_index + 1))}"
    eval "effort=\${$((claude_marker_index + 2))}"
    eval "prompt=\${$((claude_marker_index + 3))}"
    printf 'MOCK_CLAUDE_ARGS --model %s --effort %s --prompt %s\n' \
        "$model" "$effort" "$prompt" >> "$MOCK_APPTAINER_LOG"
    mkdir -p "$home_src/task"
    printf '%s\n' '{"contamination":false,"disallowed_model":false,"justification_contamination":"mock clean","justification_disallowed_model":"mock allowed"}' > "$home_src/task/judgement.json"
    printf '%s\n' \
        '{"type":"system","subtype":"init","session_id":"mock-claude","model":"opus","cwd":"/home/ben/task"}' \
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"mock verdict written"}]}}' \
        '{"type":"result","subtype":"success","result":"done"}'
    exit 0
fi

if [ "$is_codex" = "1" ]; then
    echo "MOCK_CODEX_COMMAND" >> "$MOCK_APPTAINER_LOG"
    mkdir -p "$home_src/task"
    printf '%s\n' '{"contamination":false,"disallowed_model":false,"justification_contamination":"mock clean","justification_disallowed_model":"mock allowed"}' > "$home_src/task/judgement.json"
    printf '%s\n' '{"type":"thread.started","thread_id":"mock-codex"}' '{"type":"turn.completed","usage":{}}'
    exit 0
fi

echo "mock apptainer received an unsupported command" >&2
exit 2
