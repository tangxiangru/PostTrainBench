#!/bin/bash
# AutoR as a PostTrainBench agent.
#
# Installed by tools/ptb_setup.py as agents/claude_autor/solve.sh in a PostTrainBench
# checkout. The harness copies it to /home/ben/agent_solve.sh and runs it inside
# the sandbox with cwd /home/ben/task. AutoR itself is at /home/ben/agent, put
# there by run_task.sh's payload copy.
#
# The prompt is not touched here. The comparison against the control arm is only
# a comparison if both are handed the same bytes, so ptb_agent.py reads $PROMPT
# out of the environment, fences it unedited, and appends exactly one section.
# `ptb_agent.py --print-goal` shows the result and `--print-goal --no-stage-note`
# shows the control's, so the difference between the arms is a diff rather than a
# claim in a docstring.
#
# The three environment variables read below -- BENCHMARK_ID, MODEL_TO_TRAIN,
# NUM_HOURS -- are forwarded by run_task.sh only for an agent that ships a
# payload/ directory. Without that block they are absent: the sandbox runs
# --cleanenv, and the only copies of those facts inside it are prose in $PROMPT
# and a countdown in timer.sh.

set -uo pipefail

# Same two exports as agents/claude/solve.sh, so the CLI underneath both arms is
# configured the same way. BASH_MAX_TIMEOUT_MS is ten hours in milliseconds: a
# training run is one Bash call and the CLI's default cap would kill it.
export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"

# Same CLI, same version, same update path as the control arm.
bash /home/ben/update_agent_cli.sh claude

# Keep the CLI's state out of $HOME. --home puts /home/ben on the job scratch and
# run_task.sh copies task/ out of it into the result directory; a state directory
# full of session transcripts is neither small nor ours to publish.
export CLAUDE_CONFIG_DIR="/tmp/claude-cli-state"
mkdir -p "$CLAUDE_CONFIG_DIR"

# The one line `tools/ptb_setup.py --extra-flag` rewrites. Empty here, so an arm
# built without a flag gets this file verbatim: two arms that differ by one
# mechanism then differ by this single line and share a payload sha, which is
# what makes the pair a comparison rather than two runs of different code.
AUTOR_EXTRA_FLAGS=(--withhold-skills=price-every-full-read-in-the-steps-it-displaces)

echo "autor payload: $(git -C /home/ben/agent rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "task=$BENCHMARK_ID model_to_train=$MODEL_TO_TRAIN num_hours=$NUM_HOURS agent_config=$AGENT_CONFIG"
echo "autor extra flags: ${AUTOR_EXTRA_FLAGS[*]-}"

exec python /home/ben/agent/ptb_agent.py \
    --repo /home/ben/agent \
    --task "$BENCHMARK_ID" \
    --model-to-train "$MODEL_TO_TRAIN" \
    --num-hours "$NUM_HOURS" \
    --workspace /home/ben/task \
    --model "$AGENT_CONFIG" \
    ${AUTOR_EXTRA_FLAGS[@]+"${AUTOR_EXTRA_FLAGS[@]}"}
