#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

REVISION="0123456789abcdef0123456789abcdef01234567"
SNAPSHOT="/home/ben/hf_cache/hub/models--Qwen--Qwen3-4B-Base/snapshots/${REVISION}"
PROMPT="$(python3 src/eval/general/get_prompt.py \
    --agent claude_vertex_max \
    --model-to-train Qwen/Qwen3-4B-Base \
    --model-revision "$REVISION" \
    --model-snapshot "$SNAPSHOT" \
    --benchmark-id gsm8k \
    --num-hours 10 \
    --num-gpus 1)"

printf -v EXPECTED_REVISION 'immutable starting revision is \\`%s\\`' "$REVISION"
printf -v EXPECTED_SNAPSHOT '\\`%s\\`' "$SNAPSHOT"
EXPECTED_MODEL='Only fine-tune from \`Qwen/Qwen3-4B-Base\`'
grep -Fq "$EXPECTED_REVISION" <<< "$PROMPT"
grep -Fq "$EXPECTED_SNAPSHOT" <<< "$PROMPT"
grep -Fq "$EXPECTED_MODEL" <<< "$PROMPT"

echo "Pinned base-model prompt test passed"
