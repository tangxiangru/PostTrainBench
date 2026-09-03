#!/bin/bash
# Every path the prompt tells the agent to USE must be absolute.
#
# Why this is a test and not a code read. The arms do not share a working
# directory. The control runs at the task root, so `bash timer.sh` resolves for
# it and a code read done from the control's point of view sees nothing wrong.
# The AutoR operator runs its stages in ${task}/.autor/<stamp>/, two levels
# down, where the same string is an ENOENT. Measured on the live 2026-09-03
# campaign: 4 of 20 AutoR cells hit `bash: timer.sh: No such file or directory`
# and 0 of 7 control cells did. A relative path in a shared prompt is a tax that
# only one arm pays, which is the one thing a benchmark prompt must never be.
#
# get_prompt.py already had --sandbox-home-dir/--sandbox-task-dir, already
# validated them as absolute, and its own error message already explained this
# exact hazard -- and rules 2 and 6 still said "the current directory", because
# nothing rendered those two rules through the plumbing. Ten passing tests on a
# setting no caller applies is not a measurement. So this test asserts the
# RENDERED prompt, not the arguments and not the template.
#
# Usage: bash tests/test_prompt_paths.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

HOME_DIR="/home/ben"
TASK_DIR="/home/ben/task"
FAILED=0
fail() { echo "FAIL: $*" >&2; FAILED=1; }

render() {
    python3 src/eval/general/get_prompt.py \
        --agent "$1" --benchmark-id gsm8k \
        --model-to-train Qwen/Qwen3-1.7B-Base \
        --num-hours 10 --num-gpus 8 \
        --sandbox-home-dir "$HOME_DIR" --sandbox-task-dir "$TASK_DIR" 2>&1
}

# The relative wordings that were in the template before, one per line. Kept in
# one place because [4] greps the real prompt for them and [5] proves that grep
# fires by handing it a string that contains them.
RELATIVE_FORMS='bash timer.sh
in the current directory
folder \`final_model\`
via the evaluate.py script'

# has_relative_form <text> -> prints each relative form found in <text>
has_relative_form() {
    local text="$1" bad
    while IFS= read -r bad; do
        [ -n "$bad" ] || continue
        if printf '%s\n' "$text" | grep -qF -- "$bad"; then printf '%s\n' "$bad"; fi
    done <<< "$RELATIVE_FORMS"
}

render_task() {
    python3 -B src/eval/general/get_prompt.py \
        --agent claude_vertex --benchmark-id "$1" \
        --model-to-train Qwen/Qwen3-1.7B-Base \
        --num-hours 10 --num-gpus 8 \
        --sandbox-home-dir "$HOME_DIR" --sandbox-task-dir "$TASK_DIR" 2>/dev/null
}

CONTROL="$(render claude_vertex)" || { echo "FAIL: control render failed: $CONTROL" >&2; exit 1; }
AUTOR="$(render claude_autor_b3)"  || { echo "FAIL: autor render failed: $AUTOR" >&2; exit 1; }

# [1] Both an arm whose cwd IS the task root and one whose cwd is not. They must
# get byte-identical text: the prompt is shared, so if these two ever diverge the
# benchmark is measuring the prompt and not the agent.
if [ "$CONTROL" != "$AUTOR" ]; then
    fail "[1] the rendered prompt differs between arms; the prompt must not be an arm variable"
    diff <(printf '%s\n' "$CONTROL") <(printf '%s\n' "$AUTOR") | head -20 >&2
fi

# [2] Nothing left unsubstituted. Catches a placeholder added to prompt.txt with
# no matching replace(), which reaches the agent as a literal "{sandbox_task_dir}".
LEFTOVER="$(printf '%s\n' "$CONTROL" | grep -o '{[a-z_]*}' | sort -u | tr '\n' ' ')"
[ -z "${LEFTOVER// /}" ] || fail "[2] unsubstituted placeholders in the rendered prompt: ${LEFTOVER}"

# [3] The three files the agent is told to run or write, each named absolutely.
# Anchored on the containing path, so a rename fails here rather than passing on
# a substring match.
for f in timer.sh evaluate.py final_model; do
    printf '%s\n' "$CONTROL" | grep -qF -- "${TASK_DIR}/${f}" \
        || fail "[3] the prompt never names ${TASK_DIR}/${f} absolutely"
done

# [4] ...and never relatively. This is the assertion that fails on a revert.
FOUND="$(has_relative_form "$CONTROL")"
[ -z "$FOUND" ] || fail "[4] the prompt still contains relative forms: $(echo "$FOUND" | tr '\n' '|')"

# [5] Prove [4] can fire. A predicate that returns empty because it is looking
# for the wrong string passes [4] on a fully reverted prompt, and that is the
# failure this whole file exists to prevent -- a green check that measures
# nothing. So hand it the old wording and require every form to come back.
WANT="$(printf '%s\n' "$RELATIVE_FORMS" | sort)"
GOT="$(has_relative_form "You can query the benchmark via the evaluate.py script.
Store your best trained model in the folder \\\`final_model\\\`.
2. ... by calling \\\`bash timer.sh\\\` in the current directory." | sort)"
if [ "$WANT" != "$GOT" ]; then
    fail "[5] has_relative_form missed a form it is supposed to catch"
    diff <(printf '%s\n' "$WANT") <(printf '%s\n' "$GOT") >&2
fi

# [6] The inspect-ai bullet tracks the benchmark's actual evaluate.py, for every task
# in the repo. This replaced a hand-kept INSPECT_EVALS list that had drifted both ways
# -- it named a task that does not exist and omitted aime2026, which does. The
# expectation here is DERIVED from the same files rather than written out, because a
# second hand-kept list in the test would drift in step with the first and agree with
# it forever.
BULLET='normal behavior for inspect-ai'
for task_dir in src/eval/tasks/*/; do
    task="$(basename "$task_dir")"
    [ -f "${task_dir}evaluate.py" ] || continue
    want="$(python3 -B -c "
import sys; sys.path.insert(0, 'src/utils')
from graded_read import evaluate_uses_inspect
print(1 if evaluate_uses_inspect('${task_dir}evaluate.py') else 0)")"
    got=0
    if render_task "$task" | grep -qF "$BULLET"; then got=1; fi
    [ "$want" = "$got" ] || fail "[6] ${task}: evaluate.py inspect=${want} but prompt bullet=${got}"
done

if [ "$FAILED" -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED" >&2; fi
exit "$FAILED"
