#!/bin/bash
# What scripts/collect.py does with a pack, and with a tree nobody judged.
#
# collect.py is the benchmark's official score and it has never run to completion
# on the local results tree. Two separate reasons, and each one used to be
# invisible in a different way:
#
#   1. `benchmark, _, model, run_id = entry.split("_")` needs exactly four
#      underscore-separated fields, so every `_g<N>` pack cell -- 286 of the 290
#      run dirs -- raised `ValueError: <entry>, <path>` with no hint of what was
#      wrong. The tempting fix is to parse the suffix and move on, and that is
#      worse: the key is (benchmark, model), every cell here is
#      gsm8k x Qwen3-1.7B-Base, so eight seats of one job collapse into one slot
#      and collect.py cheerfully reports 1/8 of a pack as the method's result.
#      Seat-aware aggregation is a change to what a row means; until someone makes
#      it, the right behaviour is a refusal that says so.
#   2. the tree carries 241 scored cells and zero judgement files, because .env
#      sets POST_TRAIN_BENCH_SKIP_JUDGES="1". collect.py's own coverage rule keeps
#      that out of its CSVs, but the numbers people quote were never read from a
#      CSV -- so it has to say it out loud.
#
# Runs against synthetic result trees. Nothing here reads the real one.
#
# Usage: bash tests/test_collect_seats_and_judges.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-collecttest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }
note() { echo "       $*"; }

walk() {  # walk <method_dir> [min] [max] -> prints "key run_id seat" lines, or the error
    python3 - "$REPO_ROOT" "$@" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
import utils
mn = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
mx = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
try:
    got = utils.walk_latest_runs(sys.argv[2], mn, mx)
except ValueError as e:
    print("ValueError:", e)
    sys.exit(3)
for (b, m), v in sorted(got.items()):
    print(f"{b}|{m} run={v['run_id']} seat={v['seat']}")
PY
}

echo "[1] a classic four-field run dir parses exactly as it always did"
M1="$WORK/t1/claude_x_10h"; mkdir -p "$M1"
mkdir -p "$M1/gsm8k_Qwen_Qwen3-1.7B-Base_82165" "$M1/gsm8k_Qwen_Qwen3-1.7B-Base_82166"
out="$(walk "$M1")"; rc=$?
chk "exits 0"                     '[ "$rc" -eq 0 ]'
chk "one slot"                    '[ "$(wc -l <<< "$out")" -eq 1 ]'
chk "provider dropped from model" 'grep -q "^gsm8k|Qwen3-1.7B-Base " <<< "$out"'
chk "latest run wins"             'grep -q "run=82166" <<< "$out"'
chk "seat is None"                'grep -q "seat=None" <<< "$out"'
note "$out"

echo "[2] two benchmarks and two models stay separate"
M2="$WORK/t2/claude_x_10h"; mkdir -p "$M2"
for d in gsm8k_Qwen_Qwen3-1.7B-Base_100 mmlu_Qwen_Qwen3-1.7B-Base_101 \
         gsm8k_meta_Llama-3-8B_102; do mkdir -p "$M2/$d"; done
out="$(walk "$M2")"
chk "three slots" '[ "$(wc -l <<< "$out")" -eq 3 ]'
note "$(tr '\n' ' ' <<< "$out")"

echo "[3] a seated cell parses instead of raising an opaque ValueError"
M3="$WORK/t3/claude_autor_r2_10h"; mkdir -p "$M3"
mkdir -p "$M3/gsm8k_Qwen_Qwen3-1.7B-Base_89727_g3"
out="$(walk "$M3")"; rc=$?
chk "exits 0"          '[ "$rc" -eq 0 ]'
chk "model is clean"   'grep -q "^gsm8k|Qwen3-1.7B-Base " <<< "$out"'
chk "run id is the job" 'grep -q "run=89727" <<< "$out"'
chk "seat is recorded" 'grep -q "seat=3" <<< "$out"'
note "$out"

echo "[4] a pack is REFUSED, not silently reduced to one seat"
M4="$WORK/t4/claude_autor_r2_10h"; mkdir -p "$M4"
for g in 0 1 2 3 4 5 6 7; do mkdir -p "$M4/gsm8k_Qwen_Qwen3-1.7B-Base_89727_g${g}"; done
out="$(walk "$M4")"; rc=$?
chk "exits non-zero"                '[ "$rc" -eq 3 ]'
chk "it is a ValueError"            'grep -q "^ValueError:" <<< "$out"'
chk "names the job id"              'grep -q "run 89727" <<< "$out"'
chk "names two conflicting seats"   'grep -qE "_g[0-9] and _g[0-9]" <<< "$out"'
chk "names the (benchmark, model)"  'grep -q "(gsm8k, Qwen3-1.7B-Base)" <<< "$out"'
chk "says what to do instead"       'grep -q "seat-per-row" <<< "$out"'
note "$(sed 's/^ValueError: //' <<< "$out" | cut -c1-110)..."
# The refusal must be about seats, not about run ids: an older single-seat job
# alongside a newer single-seat job is the ordinary supersede case.
M5="$WORK/t5/claude_x_10h"; mkdir -p "$M5"
mkdir -p "$M5/gsm8k_Qwen_Qwen3-1.7B-Base_100_g0" "$M5/gsm8k_Qwen_Qwen3-1.7B-Base_200_g0"
out="$(walk "$M5")"; rc=$?
chk "same seat, two jobs -> newest wins" '[ "$rc" -eq 0 ] && grep -q "run=200" <<< "$out"'
# And a run-id filter that leaves one seat standing must collect, not refuse:
# refusing on a pack the caller has already excluded would be a false alarm.
out="$(walk "$M4" "" 89727)"; rc=$?
chk "max-run-id filtering out the pack collects nothing" '[ "$rc" -eq 0 ] && [ -z "$out" ]'

echo "[5] an unparseable name says what shape it expected"
M6="$WORK/t6/claude_x_10h"; mkdir -p "$M6/not-a-run-dir"
out="$(walk "$M6")"; rc=$?
chk "exits non-zero"        '[ "$rc" -eq 3 ]'
chk "quotes the offender"   "grep -q \"'not-a-run-dir'\" <<< \"\$out\""
chk "quotes the expected shape" 'grep -q "{benchmark}_{provider}_{model}_{run_id}" <<< "$out"'
# Bookkeeping dirs the audit writes are not runs and must not look like a defect.
mkdir -p "$WORK/t7/claude_x_10h/_audit" "$WORK/t7/claude_x_10h/gsm8k_Qwen_Qwen3-1.7B-Base_100"
out="$(walk "$WORK/t7/claude_x_10h")"; rc=$?
chk "_audit/ is skipped, not parsed" '[ "$rc" -eq 0 ] && [ "$(wc -l <<< "$out")" -eq 1 ]'

echo "[6] collect.py says out loud that a tree has scores but no verdicts"
warn() {  # warn <root>...
    python3 - "$REPO_ROOT" "$@" <<'PY' 2>&1
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
import collect
collect.warn_if_tree_is_unjudged(list(sys.argv[2:]))
PY
}
R="$WORK/tree"; mkdir -p "$R/method_a"
for i in 1 2 3; do
    d="$R/method_a/gsm8k_Qwen_Qwen3-1.7B-Base_10${i}"
    mkdir -p "$d"; printf '{"accuracy": 0.5}' > "$d/metrics.json"
done
out="$(warn "$R")"
chk "warns"                    'grep -q "^WARNING: 3 scored cells and 0 judgement files" <<< "$out"'
chk "says it is not a score"   'grep -q "is a PostTrainBench score" <<< "$out"'
chk "names the env switch"     'grep -q "POST_TRAIN_BENCH_SKIP_JUDGES" <<< "$out"'
note "$(grep -m1 WARNING <<< "$out")"
printf '{"contamination": false}' > "$R/method_a/gsm8k_Qwen_Qwen3-1.7B-Base_101/judgement_gpt5_4.json"
out="$(warn "$R")"
chk "partial coverage warns differently" 'grep -q "1 of 3 scored cells" <<< "$out"'
for i in 2 3; do
    printf '{"contamination": false}' \
        > "$R/method_a/gsm8k_Qwen_Qwen3-1.7B-Base_10${i}/judgement_gpt5_4.json"
done
out="$(warn "$R")"
chk "a fully judged tree is silent" '[ -z "$(tr -d "[:space:]" <<< "$out")" ]'
# An empty or nonexistent root is not an unjudged tree.
chk "empty root is silent"      '[ -z "$(tr -d "[:space:]" <<< "$(warn "$WORK/nope")")" ]'

echo "[7] the banner and AGENTS.md tell the same story as the code"
chk "collect.py banner exists" \
    'grep -q "READ THIS BEFORE QUOTING A NUMBER" "$REPO_ROOT/scripts/collect.py"'
chk "banner names the skip switch" \
    'grep -q "POST_TRAIN_BENCH_SKIP_JUDGES" "$REPO_ROOT/scripts/collect.py"'
chk "AGENTS.md says no number is judged" \
    'grep -q "No number on the local results tree is a judged number" "$REPO_ROOT/AGENTS.md"'
# The claim in the banner is checkable: .env really does turn the judges off.
if [ -f "$REPO_ROOT/.env" ]; then
    chk ".env really sets it" 'grep -qE "^POST_TRAIN_BENCH_SKIP_JUDGES=\"?1" "$REPO_ROOT/.env"'
else
    note "no .env in this checkout -- skipping the live cross-check"
fi

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
