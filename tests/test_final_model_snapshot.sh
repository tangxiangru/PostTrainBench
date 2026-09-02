#!/bin/bash
# Drive the final_model snapshot daemon in src/run_task.sh against fake weights.
#
# Worth having as a file rather than a one-off check: the first version of the
# daemon guarded on `ls "$d"/*.safetensors "$d"/*.bin`, which exits nonzero when
# either operand is missing. Real checkpoints ship .safetensors and no .bin, so
# the guard was false on every run the daemon would ever see -- it copied
# nothing, logged nothing, and passed `bash -n` and a code read. Only running it
# against a directory that looks like a checkpoint showed it.
#
# Usage: bash tests/test_final_model_snapshot.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-snaptest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Extract just the snapshot functions from run_task.sh and source them, so the
# test exercises the shipped code rather than a copy that can drift from it.
sed -n '/^has_model_weights() {/,/^trap stop_final_model_snapshots EXIT$/p' \
    "$REPO_ROOT/src/run_task.sh" > "$WORK/daemon.sh"
if ! grep -q '^snapshot_final_model_daemon() {' "$WORK/daemon.sh"; then
    echo "FAIL: could not extract the snapshot daemon from src/run_task.sh" >&2
    exit 1
fi

JOB_DIR="$WORK/job"; EVAL_DIR="$WORK/eval"
RANDOM_UUID="test-uuid"; AGENT="test_agent"; SLURM_JOB_ID="99999"
MODEL="$JOB_DIR/task/final_model"
SNAP="$EVAL_DIR/final_model_snapshot"
LOG="$EVAL_DIR/final_model_snapshot.log"
mkdir -p "$MODEL" "$EVAL_DIR"

export POST_TRAIN_BENCH_SNAPSHOT_INTERVAL=2
export POST_TRAIN_BENCH_SNAPSHOT_MIN_FREE_GIB=1
# shellcheck disable=SC1090
source "$WORK/daemon.sh"

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }
oks() { grep -c ' ok: snapshot updated' "$LOG" 2>/dev/null || echo 0; }

echo "[1] has_model_weights"
chk "false on a missing dir"        '! has_model_weights "$WORK/nope"'
chk "false on an empty dir"         '! has_model_weights "$MODEL"'
touch "$MODEL/config.json"
chk "false on config-only"          '! has_model_weights "$MODEL"'
touch "$MODEL/model.safetensors"
chk "true on .safetensors, no .bin" 'has_model_weights "$MODEL"'
rm -f "$MODEL/model.safetensors"; touch "$MODEL/pytorch_model.bin"
chk "true on .bin, no .safetensors" 'has_model_weights "$MODEL"'
rm -f "$MODEL/pytorch_model.bin" "$MODEL/config.json"

echo "[2] a weightless dir produces no snapshot"
start_final_model_snapshots >/dev/null
sleep 5
chk "no snapshot yet" '[ ! -d "$SNAP" ]'

echo "[3] weights appear -> snapshot created"
head -c 3000000 /dev/urandom > "$MODEL/model.safetensors"
echo '{"model_type":"qwen3"}' > "$MODEL/config.json"
sleep 6
chk "snapshot exists"        '[ -d "$SNAP" ]'
chk "weights copied"         '[ -f "$SNAP/model.safetensors" ]'
chk "config copied"          '[ -f "$SNAP/config.json" ]'
chk "manifest is valid json" 'python3 -c "import json;json.load(open(\"$SNAP/SNAPSHOT_MANIFEST.json\"))"'
chk "manifest has job id"    'grep -q 99999 "$SNAP/SNAPSHOT_MANIFEST.json"'
chk "byte-identical"         'cmp -s "$MODEL/model.safetensors" "$SNAP/model.safetensors"'

echo "[4] unchanged model -> no redundant copy"
n1=$(oks); sleep 7; n2=$(oks)
chk "copy count stayed at $n1" '[ "$n1" = "$n2" ]'

echo "[5] model changes -> snapshot refreshes"
head -c 4000000 /dev/urandom > "$MODEL/model.safetensors"
sleep 6
n3=$(oks)
chk "copy count rose ($n2 -> $n3)" '[ "$n3" -gt "$n2" ]'
chk "byte-identical again"         'cmp -s "$MODEL/model.safetensors" "$SNAP/model.safetensors"'

echo "[6] stop leaves no partial tree"
pid=$SNAPSHOT_PID
stop_final_model_snapshots
sleep 1
chk "daemon reaped"     '! kill -0 "$pid" 2>/dev/null'
chk "no .incoming left" '[ ! -e "$EVAL_DIR/.final_model_snapshot.incoming" ]'
chk "snapshot survives" '[ -d "$SNAP" ]'

# The floor is read into a `local` at daemon entry, so it can only be exercised
# by a daemon started after it is set -- raising it under a running loop
# changes nothing, which is why this restarts rather than just re-exporting.
echo "[7] disk floor is honoured"
export POST_TRAIN_BENCH_SNAPSHOT_MIN_FREE_GIB=999999999
n4=$(oks)
start_final_model_snapshots >/dev/null
head -c 5000000 /dev/urandom > "$MODEL/model.safetensors"
sleep 6
chk "logged a skip"              'grep -q "skip: only" "$LOG"'
chk "no new copy ($n4)"          '[ "$(oks)" = "$n4" ]'
chk "kept the previous snapshot" '[ "$(stat -c%s "$SNAP/model.safetensors")" = "4000000" ]'
stop_final_model_snapshots
export POST_TRAIN_BENCH_SNAPSHOT_MIN_FREE_GIB=1

echo "[8] interval=0 disables the feature"
export POST_TRAIN_BENCH_SNAPSHOT_INTERVAL=0
SNAPSHOT_PID=""
out=$(start_final_model_snapshots)
chk "says disabled"     '[[ "$out" == *disabled* ]]'
chk "no daemon started" '[ -z "$SNAPSHOT_PID" ]'

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; sed 's/^/    /' "$LOG"; fi
exit $fail
