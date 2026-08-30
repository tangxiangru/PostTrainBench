#!/bin/bash

set -euo pipefail

: "${CLAUDE_CODE_USE_VERTEX:?CLAUDE_CODE_USE_VERTEX is required}"
: "${ANTHROPIC_VERTEX_PROJECT_ID:?ANTHROPIC_VERTEX_PROJECT_ID is required}"
: "${ANTHROPIC_VERTEX_REGION:?ANTHROPIC_VERTEX_REGION is required}"
: "${AGENT_CONFIG:?AGENT_CONFIG is required}"
: "${PROMPT:?PROMPT is required}"

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="max"

bash /home/ben/update_agent_cli.sh claude

printf '%s' "$PROMPT" | claude --print --verbose \
    --model "$AGENT_CONFIG" \
    --effort max \
    --output-format stream-json \
    --thinking-display summarized \
    --setting-sources "" \
    --no-session-persistence \
    --dangerously-skip-permissions
