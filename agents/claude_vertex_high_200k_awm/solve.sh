#!/bin/bash
# claude_vertex_high_200k_awm: claude_vertex_high plus one setup step before the prompt.
#
# The host mounts a read-only checkout of the agentic-world-model repository at
# /home/ben/awm (POST_TRAIN_BENCH_EXTRA_BINDS in run_task.sh) and forwards two
# variables through env_passthrough.txt: AWM_SANDBOX_SETUP, the arguments for
# `awm sandbox setup`, and AWM_CHECKOUT_SHA, the commit of that checkout. This
# script puts the checkout on PATH and PYTHONPATH, runs the setup step in
# /home/ben/task, and then starts Claude Code exactly as claude_vertex_max does.
# What the setup step installs (a skill, a hook, a runtime) is decided by the
# mounted commit, not here; this scaffold does not change between studies.
#
# The one difference from claude_vertex_high: --setting-sources project instead
# of "", because the setup step writes .claude/skills (and possibly hooks) into
# /home/ben/task, and "" would keep Claude Code from loading them.

set -euo pipefail

: "${CLAUDE_CODE_USE_VERTEX:?CLAUDE_CODE_USE_VERTEX is required}"
: "${ANTHROPIC_VERTEX_PROJECT_ID:?ANTHROPIC_VERTEX_PROJECT_ID is required}"
: "${ANTHROPIC_VERTEX_REGION:?ANTHROPIC_VERTEX_REGION is required}"
: "${AGENT_CONFIG:?AGENT_CONFIG is required}"
: "${PROMPT:?PROMPT is required}"
: "${AWM_SANDBOX_SETUP:?AWM_SANDBOX_SETUP is required: the arguments for awm sandbox setup}"

# /home/ben is the sandbox home by PostTrainBench's contract; PTB_SANDBOX_HOME
# exists so test_solve.sh can run this script outside a sandbox.
SANDBOX_HOME="${PTB_SANDBOX_HOME:-/home/ben}"
AWM_CHECKOUT="${SANDBOX_HOME}/awm"
TASK_DIR="${SANDBOX_HOME}/task"

if [ ! -f "${AWM_CHECKOUT}/awm/cli.py" ]; then
    echo "ERROR: no agentic-world-model checkout mounted at ${AWM_CHECKOUT} (POST_TRAIN_BENCH_EXTRA_BINDS)" >&2
    exit 1
fi
export PYTHONPATH="${AWM_CHECKOUT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${SANDBOX_HOME}/.local/bin"
printf '#!/bin/bash\nexec python3 -m awm.cli "$@"\n' > "${SANDBOX_HOME}/.local/bin/awm"
chmod +x "${SANDBOX_HOME}/.local/bin/awm"
export PATH="${SANDBOX_HOME}/.local/bin:${PATH}"
echo "awm checkout: ${AWM_CHECKOUT} sha=${AWM_CHECKOUT_SHA:-unknown}"
python3 -c "import awm.cli" || { echo "ERROR: the mounted checkout does not import as awm" >&2; exit 1; }

# AWM_SANDBOX_SETUP is a whitespace-separated argument list by contract.
# shellcheck disable=SC2086
(cd "${TASK_DIR}" && awm sandbox setup --target "${TASK_DIR}" --sha "${AWM_CHECKOUT_SHA:-unknown}" ${AWM_SANDBOX_SETUP}) \
    || { echo "ERROR: awm sandbox setup failed" >&2; exit 1; }

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"

bash "${SANDBOX_HOME}/update_agent_cli.sh" claude

# The exp_protocol study is opt-in: a null-control cell runs this same scaffold
# without installing the skill and must receive PTB's prompt unchanged.  When
# the skill is present, make protocol discovery the scientist's first action;
# relying on passive skill discovery is too weak (an agent can read the skill
# later, after it has already launched training).
SCIENTIST_PROMPT="$PROMPT"
if [ -r "${TASK_DIR}/.claude/skills/exp_protocol/SKILL.md" ]; then
    read -r -d '' PROTOCOL_BOOTSTRAP <<'EOF' || true
MANDATORY SCIENTIST BOOTSTRAP — do this before the PTB task:
Your first tool/skill action MUST be to invoke the `exp_protocol` skill and read
`skills/exp_protocol/SKILL.md` completely, including the resources it directs you to read.
Do not inspect the repository, edit files, or run any other command before that first action.
Before any model-training or evaluation command, create and fill the experiment card,
run `awm exp_protocol check`, and obtain a successful `awm exp_protocol lock`.
Only after the lock succeeds may you continue with the PTB task below.

PTB TASK:
EOF
    SCIENTIST_PROMPT="${PROTOCOL_BOOTSTRAP}"$'\n'"${PROMPT}"
fi

printf '%s' "$SCIENTIST_PROMPT" | claude --print --verbose \
    --model "$AGENT_CONFIG" \
    --effort high \
    --output-format stream-json \
    --thinking-display summarized \
    --setting-sources project \
    --no-session-persistence \
    --dangerously-skip-permissions
