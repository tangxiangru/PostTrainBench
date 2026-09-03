#!/bin/bash
# Drive the "no final_model shipped" rescue block in src/run_task.sh.
#
# WHY THIS EXISTS. `delete_hf_models.py` walks `task/` and removes every
# HuggingFace model folder, and it runs BEFORE `cp -r "${JOB_DIR}/task"`, so
# anything it deletes never reaches the results filesystem. On 2026-09-03 two
# ptb-g3 cells (`91037_g1`, `91039_g4`) finished `exit_code: 0` with
# `final_model_files: 0` -- their agents ended a turn mid-training and the CLI
# killed the background training job -- and that unconditional delete took
# `runs/sft1` and `runs/grpo1/checkpoint-100` with it. The snapshot daemon could
# not cover them either: it guards on `has_model_weights "$src"` against
# `final_model`, which those cells never wrote. Result: nine failed evals, no
# `metrics.json`, a FAILED pack exit, and -- because both cells were in the same
# arm -- an arm-asymmetric hole that no re-score could fill.
#
# The failure was invisible to `bash -n` and to a code read: the delete is one
# unconditional line, and the cell that needs the guard is exactly the cell that
# writes no `final_model`, so every healthy run exercises the other branch. Only
# building a job tree with checkpoints and NO final_model shows it.
#
# Usage: bash tests/test_rescue_when_nothing_shipped.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-rescuetest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Extract the shipped code rather than restating it, so this test fails when the
# block drifts instead of passing against a copy of what it used to say.
sed -n '/^has_model_weights() {/,/^}$/p' "$REPO_ROOT/src/run_task.sh" > "$WORK/lib.sh"
sed -n '/^if ! has_model_weights "\$EVAL_DIR\/final_model" \\$/,/^fi$/p' \
    "$REPO_ROOT/src/run_task.sh" >> "$WORK/lib.sh"
if ! grep -q 'final_model_rescued' "$WORK/lib.sh"; then
    echo "FAIL: could not extract the rescue block from src/run_task.sh" >&2
    exit 1
fi
if ! grep -q '^has_model_weights() {' "$WORK/lib.sh"; then
    echo "FAIL: could not extract has_model_weights from src/run_task.sh" >&2
    exit 1
fi

export POST_TRAIN_BENCH_SNAPSHOT_MIN_FREE_GIB=1

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }

# A job tree with two checkpoints, the second written later, and no final_model.
# `touch -d` sets the mtime the rescue sorts on.
new_case() {
    JOB_DIR="$WORK/$1/job"; EVAL_DIR="$WORK/$1/eval"
    mkdir -p "$JOB_DIR/task/runs/sft1" "$JOB_DIR/task/runs/grpo1/checkpoint-100" "$EVAL_DIR"
    echo w > "$JOB_DIR/task/runs/sft1/model.safetensors"
    echo w > "$JOB_DIR/task/runs/grpo1/checkpoint-100/model.safetensors"
    touch -d '2026-09-03T03:00:00' "$JOB_DIR/task/runs/sft1/model.safetensors"
    touch -d '2026-09-03T04:00:00' "$JOB_DIR/task/runs/grpo1/checkpoint-100/model.safetensors"
}

echo "case 1: nothing shipped -> rescue the newest checkpoint"
new_case c1
# shellcheck disable=SC1090
source "$WORK/lib.sh"
chk "final_model_rescued exists"          '[ -d "$EVAL_DIR/final_model_rescued" ]'
chk "it holds weights"                    'has_model_weights "$EVAL_DIR/final_model_rescued"'
chk "it is the NEWEST checkpoint (grpo1)" 'grep -q "checkpoint-100" "$EVAL_DIR/final_model_rescued/RESCUE_MANIFEST.json"'
chk "manifest denies it shipped"          'grep -q "\"is_shipped_model\": false" "$EVAL_DIR/final_model_rescued/RESCUE_MANIFEST.json"'
chk "manifest refuses board scoring"      'grep -q "\"scoreable_as_a_board_cell\": false" "$EVAL_DIR/final_model_rescued/RESCUE_MANIFEST.json"'

echo "case 2: final_model shipped -> rescue must NOT fire (no double storage)"
new_case c2
mkdir -p "$EVAL_DIR/final_model"; echo w > "$EVAL_DIR/final_model/model.safetensors"
# shellcheck disable=SC1090
source "$WORK/lib.sh"
chk "no final_model_rescued" '[ ! -e "$EVAL_DIR/final_model_rescued" ]'

echo "case 3: a mid-run snapshot survived -> rescue must NOT fire"
new_case c3
mkdir -p "$EVAL_DIR/final_model_snapshot"; echo w > "$EVAL_DIR/final_model_snapshot/model.safetensors"
# shellcheck disable=SC1090
source "$WORK/lib.sh"
chk "no final_model_rescued" '[ ! -e "$EVAL_DIR/final_model_rescued" ]'

echo "case 4: an EMPTY final_model dir is not a shipped model -> rescue fires"
# The original bug's neighbour: `[ -d final_model ]` would call this shipped.
# `has_model_weights` is what makes the distinction, so pin it here.
new_case c4
mkdir -p "$EVAL_DIR/final_model"   # directory, no weights
# shellcheck disable=SC1090
source "$WORK/lib.sh"
chk "rescue still fires" '[ -d "$EVAL_DIR/final_model_rescued" ]'

echo "case 5: nothing shipped and no checkpoint anywhere -> warn, do not crash"
JOB_DIR="$WORK/c5/job"; EVAL_DIR="$WORK/c5/eval"
mkdir -p "$JOB_DIR/task" "$EVAL_DIR"
# shellcheck disable=SC1090
source "$WORK/lib.sh" 2>"$WORK/c5.err"
chk "no final_model_rescued" '[ ! -e "$EVAL_DIR/final_model_rescued" ]'
chk "said so on stderr"      'grep -q "ships no weights" "$WORK/c5.err"'

echo "case 6: free space under the floor -> refuse and say why"
new_case c6
POST_TRAIN_BENCH_SNAPSHOT_MIN_FREE_GIB=999999999
# shellcheck disable=SC1090
source "$WORK/lib.sh" 2>"$WORK/c6.err"
POST_TRAIN_BENCH_SNAPSHOT_MIN_FREE_GIB=1
chk "no final_model_rescued" '[ ! -e "$EVAL_DIR/final_model_rescued" ]'
chk "named the floor"        'grep -q "NOT rescuing" "$WORK/c6.err"'

if [ "$fail" -ne 0 ]; then echo "FAILED"; exit 1; fi
echo "OK"
