#!/bin/bash
# Pin the answer-key exclusion chain: audit -> manifest -> void -> load_metrics.
#
# Two cells on the 89727/89809 board read RTT1/posttrainbench-gsm8k-recipes, the
# shipped-recipe corpus for the exact task they are graded on. Detecting that is
# not the fix; nothing in the aggregators consulted the detector, so the scores
# would have been counted again on the next pass. What this test guards is the
# whole chain from grep to refusal, and specifically the two ways it silently
# comes apart:
#
#   * the detector flags an honest cell. Every cell that DECLINED the key still
#     writes its name into a ledger, and a ledger is a .py full of open() calls,
#     so file-level co-occurrence flagged 2 false positives out of 4 -- one of
#     them on the control arm, which would have moved the headline the wrong way.
#     The proximity rule is asserted here on a decline that is deliberately built
#     to look guilty at file level.
#   * the exclusion drifts from the detector. The void list is generated from the
#     manifest, never typed, and a marker on disk that the current manifest does
#     not call READ is reported rather than quietly released.
#
# Everything runs on a synthetic results tree. No cell, no GPU, no network.
#
# Usage: bash tests/test_answer_key_void.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="$REPO_ROOT/ptb_ops/answer_key_audit.py"
VOID="$REPO_ROOT/ptb_ops/void_cells.py"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-voidtest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }
note() { echo "       $*"; }

for f in "$AUDIT" "$VOID"; do
    [ -f "$f" ] || { echo "FAIL: no such file: $f" >&2; exit 1; }
done

ROOT="$WORK/results"
KEY="posttrainbench-gsm8k-recipes"
SNAP="/cache/hub/datasets--RTT1--${KEY}/snapshots/c2e13dc"

mk_cell() {  # mk_cell <method> <cell> <accuracy>
    local d="$ROOT/$1/$2"
    mkdir -p "$d/final_model"
    printf '{"accuracy": %s, "stderr": 0.01}\n' "$3" > "$d/metrics.json"
    printf '{"model_type":"qwen3"}\n' > "$d/final_model/config.json"
    : > "$d/solve_out.txt"
    printf '%s' "$d"
}

echo "[1] build a synthetic results tree: one guilty cell, two innocent mentions, one clean"

# GUILTY: opens the corpus where it defines the path, and the file shows a run line.
G="$(mk_cell arm_r2 gsm8k_Qwen_Qwen3-1.7B-Base_89727_g3 0.8415)"
mkdir -p "$G/code"
cat > "$G/code/mine_recipes.py" <<PY
"""Mine RTT1/${KEY} for what actually raised GSM8K accuracy."""
import json
CORPUS = "${SNAP}/gsm8k-gemma-holdout-v1.recipes.jsonl"
recs = [json.loads(l) for l in open(CORPUS)]
print(len(recs))
PY
echo "+ python code/mine_recipes.py" > "$G/solve_out.txt"

# INNOCENT, and built to look guilty at file level: names the key, is a .py, and
# has an open() three lines from the mention. This is the shape that produced the
# false positive on the control arm.
D="$(mk_cell arm_sn gsm8k_Qwen_Qwen3-1.7B-Base_89727_g6 0.8523)"
mkdir -p "$D/code"
cat > "$D/code/ledger.py" <<PY
import json
LEDGER = [
    {"resource": "RTT1/${KEY}",
     "action": "IDENTIFIED AND NOT OPENED -- this is the withheld answer key"},
]
with open("ledger.json", "w") as f:
    json.dump(LEDGER, f)
PY
echo "+ python code/ledger.py" > "$D/solve_out.txt"

# INNOCENT: cites the snapshot path in prose. This is what 87 real cells are -- an
# `ls` of the HF cache scrolling past in a notes file.
P="$(mk_cell arm_sn gsm8k_Qwen_Qwen3-1.7B-Base_89809_g1 0.8102)"
printf 'cache listing:\n%s/README.md\n' "$SNAP" > "$P/notes.md"

# CLEAN: never mentions it.
C="$(mk_cell arm_ctl gsm8k_Qwen_Qwen3-1.7B-Base_89809_g0 0.7788)"

chk "four cells built" '[ "$(find "$ROOT" -name metrics.json | wc -l)" -eq 4 ]'

echo "[2] the audit grades them, and proximity keeps the decliner out of READ"
MAN="$ROOT/_audit/answer_key_audit.json"
out="$(python3 "$AUDIT" "$ROOT" --verbose 2>&1)"; rc=$?
chk "exits 1 because a cell is READ"  '[ "$rc" -eq 1 ]'
chk "scanned all four"                'grep -q "^scanned 4 cells" <<< "$out"'
chk "READ: 1"                         'grep -q "^READ: 1" <<< "$out"'
chk "and it is the miner"             'grep -q "arm_r2/gsm8k_Qwen_Qwen3-1.7B-Base_89727_g3" <<< "$out"'
chk "the decliner is NOT read"        '! grep -q "89727_g6.*READ" <<< "$out"'
chk "manifest written"                '[ -f "$MAN" ]'

q() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))$1)" "$MAN"; }
chk "manifest counts scanned=4"       '[ "$(q "[\"counts\"][\"scanned\"]")" = "4" ]'
chk "manifest counts READ=1"          '[ "$(q "[\"counts\"][\"READ\"]")" = "1" ]'
chk "manifest counts NAME=1"          '[ "$(q "[\"counts\"][\"NAME\"]")" = "1" ]'
chk "manifest counts PATH=1"          '[ "$(q "[\"counts\"][\"PATH\"]")" = "1" ]'
chk "manifest records the window"     '[ "$(q "[\"detector\"][\"window\"]")" = "3" ]'
readers="$(q "[\"cells\"][0][\"executed_readers\"]" 2>/dev/null)"
chk "manifest names the reader file"  '[[ "$readers" == *"code/mine_recipes.py"* ]]'
note "executed_readers: $readers"
# The decliner must be in the manifest -- as NAME, so the record shows it was
# looked at and cleared, not that it was never scanned.
chk "decliner recorded as NAME"       '[ "$(q "[\"cells\"][1][\"verdict\"]")" = "NAME" ]'

echo "[3] --dry-run changes nothing"
out="$(python3 "$VOID" "$ROOT" --dry-run 2>&1)"; rc=$?
chk "dry run exits 0"                 '[ "$rc" -eq 0 ]'
chk "says it would void one"          'grep -q "would change 1 cell" <<< "$out"'
chk "quotes the score at risk"        'grep -q "accuracy=0.8415" <<< "$out"'
chk "metrics.json still there"        '[ -f "$G/metrics.json" ]'
chk "no marker written"               '[ ! -f "$G/VOIDED_ANSWER_KEY.json" ]'

echo "[4] the real run voids exactly the guilty cell"
out="$(python3 "$VOID" "$ROOT" 2>&1)"; rc=$?
chk "exits 0"                         '[ "$rc" -eq 0 ]'
chk "marker written"                  '[ -f "$G/VOIDED_ANSWER_KEY.json" ]'
chk "metrics.json is gone"            '[ ! -f "$G/metrics.json" ]'
chk "score quarantined, not deleted"  '[ -f "$G/metrics.json.VOID_ANSWER_KEY" ]'
chk "and it still holds the number" \
    'grep -q "0.8415" "$G/metrics.json.VOID_ANSWER_KEY"'
m() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))$1)" "$G/VOIDED_ANSWER_KEY.json"; }
chk "marker records the withdrawn acc" '[ "$(m "[\"withdrawn_metrics\"][\"accuracy\"]")" = "0.8415" ]'
chk "marker records the evidence"      '[ "$(m "[\"executed_readers\"][0]")" = "code/mine_recipes.py" ]'
chk "marker points at the audit"       '[ -n "$(m "[\"audit\"][\"generated_at\"]")" ]'
# The three innocent cells must be untouched: an over-broad void is the same
# failure as a missed one, it just costs the other arm instead.
for c in "$D" "$P" "$C"; do
    chk "untouched: $(basename "$c")" '[ -f "$c/metrics.json" ] && [ ! -f "$c/VOIDED_ANSWER_KEY.json" ]'
done

echo "[5] running it again is a no-op, not a second void"
out="$(python3 "$VOID" "$ROOT" 2>&1)"; rc=$?
chk "still exits 0"                   '[ "$rc" -eq 0 ]'
chk "reports already voided"          'grep -q "already voided" <<< "$out"'
chk "changed 0 cells"                 'grep -q "changed 0 cell" <<< "$out"'
chk "did not clobber the quarantine" \
    'grep -q "0.8415" "$G/metrics.json.VOID_ANSWER_KEY"'

echo "[6] load_metrics refuses a voided cell, and says why"
py_load() {  # py_load <cell>
    python3 - "$REPO_ROOT" "$1" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
import utils
try:
    print("OK", utils.load_metrics(os.path.join(sys.argv[2], "metrics.json")))
except utils.VoidedCellError as e:
    print("VOIDED", e)
except Exception as e:
    print(type(e).__name__, e)
PY
}
r="$(py_load "$G")"
chk "voided cell -> VoidedCellError"  '[[ "$r" == VOIDED* ]]'
chk "and the message names the marker" '[[ "$r" == *VOIDED_ANSWER_KEY.json* ]]'
note "$r"
chk "clean cell still loads"          '[[ "$(py_load "$C")" == "OK 0.7788" ]]'
chk "decliner still loads"            '[[ "$(py_load "$D")" == "OK 0.8523" ]]'
# The marker is the statement and the rename is only the enforcement: restoring
# metrics.json by hand must not restore the score.
cp "$G/metrics.json.VOID_ANSWER_KEY" "$G/metrics.json"
chk "a hand-restored metrics.json is still refused" '[[ "$(py_load "$G")" == VOIDED* ]]'
rm -f "$G/metrics.json"

echo "[7] a marker the current manifest does not justify is reported, not released"
touch "$C/VOIDED_ANSWER_KEY.json"
out="$(python3 "$VOID" "$ROOT" 2>&1)"; rc=$?
chk "exits non-zero"                  '[ "$rc" -ne 0 ]'
chk "names it STALE"                  'grep -q "STALE arm_ctl/gsm8k_Qwen_Qwen3-1.7B-Base_89809_g0" <<< "$out"'
chk "and does not delete it"          '[ -f "$C/VOIDED_ANSWER_KEY.json" ]'
rm -f "$C/VOIDED_ANSWER_KEY.json"

echo "[8] --undo restores, for the case where the detector is wrong"
out="$(python3 "$VOID" "$ROOT" --undo 2>&1)"; rc=$?
chk "exits 0"                         '[ "$rc" -eq 0 ]'
chk "metrics.json is back"            '[ -f "$G/metrics.json" ]'
chk "with the original number"        'grep -q "0.8415" "$G/metrics.json"'
chk "marker gone"                     '[ ! -f "$G/VOIDED_ANSWER_KEY.json" ]'
chk "quarantine copy gone"            '[ ! -f "$G/metrics.json.VOID_ANSWER_KEY" ]'
chk "load_metrics works again"        '[[ "$(py_load "$G")" == "OK 0.8415" ]]'
# and the chain is re-appliable
python3 "$VOID" "$ROOT" >/dev/null 2>&1
chk "re-void works"                   '[ -f "$G/VOIDED_ANSWER_KEY.json" ]'

echo "[9] the rescore entry points refuse a voided cell"
RESC_OVL="$REPO_ROOT/ptb_ops/ptb_rescore_overlap.sh"
RESC="$REPO_ROOT/ptb_ops/ptb_rescore.sbatch"
BOARD="$REPO_ROOT/ptb_ops/ptb_greedy_board.sbatch"
for f in "$RESC_OVL" "$RESC" "$BOARD"; do chk "bash -n $(basename "$f")" 'bash -n "$f"'; done
# ptb_rescore_overlap.sh overwrites metrics.json in place, so on a voided cell it
# would undo the quarantine. Run it for real: the guard sits before apptainer, so
# it exits without touching a node.
out="$(bash "$RESC_OVL" "$G" gsm8k 0 t 2>&1)"; rc=$?
chk "overlap rescore exits non-zero"  '[ "$rc" -ne 0 ]'
chk "and says why"                    'grep -q "voided for answer-key contamination" <<< "$out"'
chk "and wrote no metrics.json"       '[ ! -f "$G/metrics.json" ]'
chk "ptb_rescore.sbatch has the guard"  'grep -q "VOIDED_ANSWER_KEY.json" "$RESC"'
# The board scores whatever it globs, in two branches, and only one of them ever
# looked at metrics.json.
chk "board guards the explicit-arg branch" \
    '[ "$(grep -c "VOIDED_ANSWER_KEY.json" "$BOARD")" -ge 2 ]'

echo "[10] the void list is generated, never typed"
# If a cell id is hard-coded anywhere in the chain, the detector and the exclusion
# can disagree, and the one that is wrong is whichever nobody re-ran.
chk "void_cells.py names no cell id"  '! grep -qE "89727|89809" "$VOID"'
# Comments may quote a cell id as an example -- utils.py's run-directory regex
# does. Live code may not: that is where an exclusion list would hide.
chk "utils.py code names no cell id" \
    '! grep -vE "^[[:space:]]*#" "$REPO_ROOT/scripts/utils.py" | grep -qE "89727|89809"'
chk "and holds no exclusion list"     '! grep -qE "^[A-Z_]*(EXCLUDE|SKIP_CELLS|VOIDED_RUNS)" "$REPO_ROOT/scripts/utils.py"'

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
