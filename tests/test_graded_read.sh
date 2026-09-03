#!/bin/bash
# Drive src/utils/graded_read.py through every way a read can fail to be a measurement.
#
# Worth having as a file rather than a code read, because the whole helper is a claim
# about what happens on the *unhappy* paths, and those paths are the ones a code read is
# worst at. The two defects it exists to prevent both passed a code read at the time:
# 89727_g5's chain script called `eval_readout.py --latest` after a read that had already
# failed and wrote an n=500 accuracy of a different checkpoint into a file named
# `*_1319_*`; 89810_g7's run_arm.sh took `SHIP=$?` and rendered ship_rule.py's ENOENT
# exit-2 as a HOLD. Neither crashed, and neither would fail `bash -n`. So the checks below
# are all of the form "the helper exited with the code that names this defect, wrote a
# record saying so, and printed no number".
#
# The evaluation itself is a fake: the real evaluate.py needs a GPU, vLLM and inspect_ai,
# none of which exist on a login node, and a helper whose failure paths can only be tested
# on an H100 is a helper whose failure paths are never tested. Group [0] is what keeps the
# fake honest -- it pins the fake's log schema against a real archived inspect log when one
# is reachable, and says out loud when one is not.
#
# Usage: bash tests/test_graded_read.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="$REPO_ROOT/src/utils/graded_read.py"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-gradedread.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
# Keep __pycache__ out of the tree under test; the same reason
# tests/test_gsm8k_reference_grpo.sh does it.
export PYTHONPYCACHEPREFIX="$WORK/pycache"

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }

# A stand-in for evaluate.py: same three arguments graded_read passes, and it writes the
# same two artefacts a real read leaves behind -- a flattened metrics file and an inspect
# log in $INSPECT_LOG_DIR (falling back to logs/, which is what inspect does). Every
# failure mode is an env knob, so each case below differs from the happy path in exactly
# one thing.
cat > "$WORK/fake_evaluate.py" <<'PYEOF'
import argparse, json, os, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model-path")
p.add_argument("--limit", type=int)
p.add_argument("--json-output-file")
p.add_argument("--max-connections", type=int, default=None)
a, _unknown = p.parse_known_args()

rows = int(os.environ.get("FAKE_ROWS", "1319"))
acc = float(os.environ.get("FAKE_ACC", "0.7808946171341926"))
metrics_acc = float(os.environ.get("FAKE_METRICS_ACC", str(acc)))
model = os.environ.get("FAKE_MODEL", "vllm/" + (a.model_path or ""))
status = os.environ.get("FAKE_STATUS", "success")
completed = int(os.environ.get("FAKE_COMPLETED", str(rows)))

print(f"gsm8k ({rows:,} samples): {model}")

if os.environ.get("FAKE_WRITE_LOG", "1") == "1":
    log_dir = Path(os.environ.get("INSPECT_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "version": 2,
        "status": status,
        "eval": {
            "model": model,
            # 1319 regardless of the limit, exactly as the real logs do -- the field a
            # naive row check would read, and the reason graded_read does not.
            "dataset": {"name": "openai/gsm8k", "samples": 1319},
            "config": {"limit": None if a.limit in (None, -1) else a.limit, "epochs": 1},
        },
        "results": {
            "total_samples": rows,
            "completed_samples": completed,
            "scores": [{
                "name": "match", "scorer": "match",
                "scored_samples": rows, "unscored_samples": 0,
                "params": {"numeric": True},
                "metrics": {
                    "accuracy": {"name": "accuracy", "value": acc, "params": {}},
                    "stderr": {"name": "stderr", "value": 0.011393706634978006,
                               "params": {}},
                },
            }],
        },
    }
    name = os.environ.get("FAKE_LOG_NAME", "2026-09-02T15-45-19+00-00_gsm8k_FAKE.json")
    (log_dir / name).write_text(json.dumps(log))
    print(f"Log: {log_dir / name}")
    if os.environ.get("FAKE_EXTRA_LOG", "0") == "1":
        (log_dir / "2026-09-02T15-59-59+00-00_gsm8k_OTHER.json").write_text(
            json.dumps(log))

if os.environ.get("FAKE_WRITE_METRICS", "1") == "1" and a.json_output_file:
    Path(a.json_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json_output_file).write_text(json.dumps(
        {"accuracy": metrics_acc, "stderr": 0.011393706634978006}, indent=2))

sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
PYEOF

CKPT="$WORK/runs/grpo_main/final"
OTHER="$WORK/runs/grpo_h8_k10/final"
mkdir -p "$CKPT" "$OTHER" "$WORK/out"
head -c 4096 /dev/urandom > "$CKPT/model.safetensors"
head -c 4096 /dev/urandom > "$OTHER/model.safetensors"
echo '{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}' > "$CKPT/config.json"
echo '{"do_sample":true,"temperature":0.0}' > "$CKPT/generation_config.json"

cd "$WORK" || exit 1

# One invocation per case. $1 is the label, the rest are extra graded_read flags; the
# per-case env knobs are exported by the caller and cleared here so a knob cannot leak
# from one case into the next and make a check pass for the wrong reason.
RC=0
run_case() {
    local lbl="$1"; shift
    python3 "$HELPER" --model-path "$CKPT" --label "$lbl" --out-dir out \
        --evaluate "$WORK/fake_evaluate.py" --quiet --print line "$@" \
        > "$WORK/$lbl.stdout" 2> "$WORK/$lbl.stderr"
    RC=$?
    unset FAKE_EXIT FAKE_ROWS FAKE_WRITE_METRICS FAKE_WRITE_LOG FAKE_MODEL \
          FAKE_STATUS FAKE_ACC FAKE_METRICS_ACC FAKE_EXTRA_LOG FAKE_COMPLETED \
          FAKE_LOG_NAME
    return 0
}
# Read one field out of a record with the stdlib, so the assertions do not depend on jq.
rec() { python3 -c "
import json,sys
d=json.load(open('$WORK/out/$1.graded_read.json'))
for k in '$2'.split('.'):
    d = d.get(k) if isinstance(d, dict) else None
print(d)
"; }
has_key() { python3 -c "
import json
d=json.load(open('$WORK/out/$1.graded_read.json'))
raise SystemExit(0 if '$2' in d else 1)
"; }

echo "[0] the fake's log schema, pinned against a real archived inspect log"
REAL_LOG_DIR="/rmeng_data/robtang/ptb-results/claude_vertex_claude-opus-5_10h/gsm8k_Qwen_Qwen3-1.7B-Base_84729_g7/task/logs"
if [ -d "$REAL_LOG_DIR" ]; then
    python3 - "$REAL_LOG_DIR" <<'PYEOF'
import glob, json, sys
seen_limited = seen_full = False
for p in sorted(glob.glob(sys.argv[1] + "/*gsm8k*.json")):
    d = json.load(open(p))
    r = d["results"]
    ok = all(k in r for k in ("total_samples", "completed_samples", "scores"))
    ok = ok and "scored_samples" in r["scores"][0] and "status" in d
    ok = ok and "model" in d["eval"] and "samples" in d["eval"]["dataset"]
    print(("  PASS " if ok else "  FAIL ") +
          f"real log has every field graded_read reads ({p.split('/')[-1][:34]})")
    if not ok:
        raise SystemExit(1)
    if r["total_samples"] != d["eval"]["dataset"]["samples"]:
        seen_limited = True
    else:
        seen_full = True
print(("  PASS " if seen_limited else "  FAIL ") +
      "a real limited read has dataset.samples != total_samples (the trap)")
print(("  PASS " if seen_full else "  FAIL ") + "a real full read has them equal")
raise SystemExit(0 if (seen_limited and seen_full) else 1)
PYEOF
    [ $? -eq 0 ] || fail=1
else
    echo "  SKIP no archived run at $REAL_LOG_DIR -- schema unpinned this run"
fi

echo "[1] happy path"
run_case happy --rows 1319 --full
chk "exit 0"                       '[ "$RC" = 0 ]'
chk "record written"               '[ -f out/happy.graded_read.json ]'
chk "verdict verified"             '[ "$(rec happy verdict)" = verified ]'
chk "metrics file kept"            '[ -f out/happy.metrics.json ]'
chk "no UNVERIFIED file"           '[ ! -e out/happy.metrics.json.UNVERIFIED ]'
chk "rows_found matches"           '[ "$(rec happy rows_found.total_samples)" = 1319 ]'
chk "accuracy on the record"       '[ "$(rec happy accuracy)" = 0.7808946171341926 ]'
chk "resolved checkpoint recorded" '[ "$(rec happy checkpoint.resolved)" = "$(python3 -c "import os;print(os.path.realpath(\"$CKPT\"))")" ]'
chk "wall time recorded"           'python3 -c "import json;assert json.load(open(\"out/happy.graded_read.json\"))[\"wall_seconds\"]>=0"'
chk "command recorded"             'grep -q fake_evaluate.py out/happy.graded_read.json'
chk "--full sent --limit -1"       'python3 -c "import json;c=json.load(open(\"out/happy.graded_read.json\"))[\"command\"];assert c[c.index(\"--limit\")+1]==\"-1\",c"'
chk "stdout says VERIFIED"         'grep -q "graded_read: VERIFIED happy rows=1319" happy.stdout'

# Everything after a literal `--` reaches evaluate.py untouched. Without this the helper
# is unusable on the machine where it matters: --max-connections is what the prompt tells
# an agent to lower when the GPU runs out of memory, and a wrapper that cannot pass it is
# a wrapper the agent works around rather than through.
run_case passthru --rows 1319 --full -- --max-connections 1
chk "passthrough exit 0"           '[ "$RC" = 0 ]'
chk "extra arg reached the child"  'python3 -c "import json;c=json.load(open(\"out/passthru.graded_read.json\"))[\"command\"];assert c[-2:]==[\"--max-connections\",\"1\"],c"'

echo "[2] the command exits nonzero"
export FAKE_EXIT=1
run_case rc1 --rows 1319 --full
chk "exit 3 (command_failed)" '[ "$RC" = 3 ]'
chk "record written"          '[ -f out/rc1.graded_read.json ]'
chk "kind is command_failed"  '[ "$(rec rc1 failure.kind)" = command_failed ]'
chk "child exit code kept"    '[ "$(rec rc1 exit_code)" = 1 ]'
chk "no accuracy key at all"  '! has_key rc1 accuracy'
chk "verdict refused"         '[ "$(rec rc1 verdict)" = refused ]'
chk "nothing on stdout"       '[ ! -s rc1.stdout ]'
chk "stderr names the defect" 'grep -q "REFUSED \[command_failed\]" rc1.stderr'
# The 89727_g5 half: a read that failed still wrote a metrics file, and it must not be
# left anywhere a `*.metrics.json` glob can pick it up as a result.
chk "metrics quarantined"     '[ -f out/rc1.metrics.json.UNVERIFIED ]'
chk "no *.metrics.json"       '[ ! -e out/rc1.metrics.json ]'

echo "[3] exit 0, writes nothing"
export FAKE_WRITE_METRICS=0 FAKE_WRITE_LOG=0
run_case silent --rows 1319 --full
chk "exit 4 (output_missing)" '[ "$RC" = 4 ]'
chk "record written"          '[ -f out/silent.graded_read.json ]'
chk "kind is output_missing"  '[ "$(rec silent failure.kind)" = output_missing ]'
chk "child exit code was 0"   '[ "$(rec silent exit_code)" = 0 ]'
chk "no accuracy key"         '! has_key silent accuracy'
chk "nothing on stdout"       '[ ! -s silent.stdout ]'

echo "[3b] exit 0, writes a log but no metrics file"
export FAKE_WRITE_METRICS=0
run_case nometrics --rows 1319 --full
chk "exit 4 (output_missing)" '[ "$RC" = 4 ]'
chk "kind is output_missing"  '[ "$(rec nometrics failure.kind)" = output_missing ]'
chk "the log was still found" 'grep -q inspect_log out/nometrics.graded_read.json'

echo "[4] 500 rows when 1319 were requested -- the 89727_g5 defect"
export FAKE_ROWS=500
run_case n500 --rows 1319 --full
chk "exit 6 (row_count_mismatch)" '[ "$RC" = 6 ]'
chk "record written"              '[ -f out/n500.graded_read.json ]'
chk "kind is row_count_mismatch"  '[ "$(rec n500 failure.kind)" = row_count_mismatch ]'
chk "rows_requested kept"         '[ "$(rec n500 rows_requested)" = 1319 ]'
chk "detail names both counts"    'grep -q "asked for 1319 rows, the log says total_samples=500" out/n500.graded_read.json'
chk "no accuracy key"             '! has_key n500 accuracy'
chk "nothing on stdout"           '[ ! -s n500.stdout ]'
chk "metrics quarantined"         '[ -f out/n500.metrics.json.UNVERIFIED ]'
# The check must not be satisfiable by dataset.samples, which the fake writes as 1319
# exactly as the real logs do. If it ever is, this is the check that goes red.
chk "not fooled by dataset.samples" 'grep -q row_count_mismatch out/n500.graded_read.json'

echo "[4b] a read cut short: total 1319, completed 900"
export FAKE_COMPLETED=900
run_case cutshort --rows 1319 --full
chk "exit 6"                      '[ "$RC" = 6 ]'
chk "names completed_samples"     'grep -q "completed_samples=900" out/cutshort.graded_read.json'

echo "[4c] the requested n is met exactly"
export FAKE_ROWS=500
run_case n500ok --rows 500
chk "exit 0"                      '[ "$RC" = 0 ]'
chk "sent --limit 500"            'python3 -c "import json;c=json.load(open(\"out/n500ok.graded_read.json\"))[\"command\"];assert c[c.index(\"--limit\")+1]==\"500\",c"'

echo "[5] the log is a read of a different checkpoint"
export FAKE_MODEL="vllm/$OTHER"
run_case wrongckpt --rows 1319 --full
chk "exit 8 (checkpoint_mismatch)" '[ "$RC" = 8 ]'
chk "kind is checkpoint_mismatch"  '[ "$(rec wrongckpt failure.kind)" = checkpoint_mismatch ]'
chk "names the other checkpoint"   'grep -q grpo_h8_k10 out/wrongckpt.graded_read.json'
chk "no accuracy key"              '! has_key wrongckpt accuracy'

echo "[6] the metrics file and the log disagree"
export FAKE_METRICS_ACC=0.5
run_case disagree --rows 1319 --full
chk "exit 5 (output_unreadable)"  '[ "$RC" = 5 ]'
chk "kind is output_unreadable"   '[ "$(rec disagree failure.kind)" = output_unreadable ]'
chk "detail quotes both numbers"  'grep -q "0.5 vs 0.7808946171341926" out/disagree.graded_read.json'

echo "[7] two candidate logs for one read"
export FAKE_EXTRA_LOG=1
run_case ambiguous --rows 1319 --full
chk "exit 5 (output_unreadable)"  '[ "$RC" = 5 ]'
chk "says it is not decidable"    'grep -q "not decidable" out/ambiguous.graded_read.json'

echo "[8] inspect did not finish"
export FAKE_STATUS=error
run_case notdone --rows 1319 --full
chk "exit 5 (output_unreadable)"  '[ "$RC" = 5 ]'
chk "names the status"            "grep -q \"status='error'\" out/notdone.graded_read.json"

echo "[9] a stale metrics file from an earlier read cannot be adopted"
# Same label, run twice: the first read succeeds, the second fails before evaluate.py
# writes anything. Without the pre-run unlink the first read's metrics file would still be
# on disk, agree with nothing, and be one missing check away from being reported again.
run_case reused --rows 1319 --full
chk "first read verified"       '[ "$RC" = 0 ]'
chk "metrics file present"      '[ -f out/reused.metrics.json ]'
export FAKE_EXIT=1 FAKE_WRITE_METRICS=0 FAKE_WRITE_LOG=0
run_case reused --rows 1319 --full
chk "second read exit 3"        '[ "$RC" = 3 ]'
chk "old metrics file gone"     '[ ! -e out/reused.metrics.json ]'
chk "removal is on the record"  'grep -q removed_stale out/reused.graded_read.json'

echo "[10] --print accuracy gives a shell a bare number and nothing else"
run_case printacc --rows 1319 --full --print accuracy
chk "exit 0"                    '[ "$RC" = 0 ]'
chk "stdout is just the number" '[ "$(cat printacc.stdout)" = 0.7808946171341926 ]'
export FAKE_EXIT=1
run_case printacc2 --rows 1319 --full --print accuracy
chk "refusal prints nothing"    '[ ! -s printacc2.stdout ]'
chk "and exits 3"               '[ "$RC" = 3 ]'

echo "[11] setup refusals happen before anything is spent"
python3 "$HELPER" --model-path "$WORK/no-such-checkpoint" --rows 1319 --label nockpt \
    --out-dir out --evaluate "$WORK/fake_evaluate.py" --quiet >nockpt.stdout 2>nockpt.stderr
RC=$?
chk "exit 7 (setup_refused)"    '[ "$RC" = 7 ]'
chk "record still written"      '[ -f out/nockpt.graded_read.json ]'
chk "code 7 on the record"      '[ "$(rec nockpt failure.code)" = 7 ]'
chk "no command was run"        '! has_key nockpt exit_code'
python3 "$HELPER" --model-path "$CKPT" --rows 1319 --label noeval --out-dir out \
    --evaluate "$WORK/no-such-evaluate.py" --quiet >noeval.stdout 2>noeval.stderr
RC=$?
chk "missing evaluate.py -> 7"  '[ "$RC" = 7 ]'
chk "and on its record"         '[ "$(rec noeval failure.code)" = 7 ]'
python3 "$HELPER" --model-path "$CKPT" --rows 0 --label zerorows --out-dir out \
    --evaluate "$WORK/fake_evaluate.py" --quiet >zerorows.stdout 2>zerorows.stderr
RC=$?
chk "--rows 0 -> 7"             '[ "$RC" = 7 ]'

echo "[12] every record is valid JSON carrying the schema tag"
chk "all records parse" 'python3 -c "
import glob,json
ps=sorted(glob.glob(\"out/*.graded_read.json\"))
assert len(ps)>=12, ps
for p in ps:
    d=json.load(open(p))
    assert d[\"schema\"]==\"ptb.graded_read/1\", (p,d.get(\"schema\"))
    assert d[\"verdict\"] in (\"verified\",\"refused\"), (p,d[\"verdict\"])
    assert (\"accuracy\" in d) == (d[\"verdict\"]==\"verified\"), p
    assert d[\"finished_utc\"] and d[\"started_utc\"], p
"'
chk "no leftover .tmp records"   '[ -z "$(ls out/*.tmp 2>/dev/null)" ]'
# The filename invariant an audit relies on: every *.metrics.json in the directory belongs
# to a record that says "verified".
chk "*.metrics.json => verified" 'python3 -c "
import glob,json,os
for m in glob.glob(\"out/*.metrics.json\"):
    lbl=os.path.basename(m)[:-len(\".metrics.json\")]
    d=json.load(open(f\"out/{lbl}.graded_read.json\"))
    assert d[\"verdict\"]==\"verified\", (m, d[\"verdict\"])
"'

echo "[13] run_task.sh actually puts the helper where the prompt says it is"
# A helper the agent cannot see is worth nothing, and nothing else in the repo would
# notice: the sandbox has no repo bind, the copy list is a hardcoded whitelist that globs
# nothing, and a file left out of it is simply absent with no error anywhere. So run the
# real block out of the shipped script rather than asserting on a grep of it.
#
# Three guards on the extraction, not the one tests/test_final_model_snapshot.sh uses. Its
# sed has a start-anchor guard only, so an end anchor that drifts makes the range run to
# EOF and the test then executes the rest of run_task.sh -- apptainer calls included. The
# extra two guards here are a line-count bound and an explicit "no apptainer in what we
# are about to run".
cd "$REPO_ROOT" || exit 1
SNIP="$WORK/copy_block.sh"
sed -n '/^mkdir "${JOB_DIR}\/task"$/,/^cp -r "containers\/other_home_data\/\.codex" "${JOB_DIR}\/"$/p' \
    src/run_task.sh > "$SNIP"
NLINES=$(wc -l < "$SNIP")
chk "extraction found the block"    'grep -q "graded_read.py" "$SNIP"'
chk "extraction closed ($NLINES l)" '[ "$NLINES" -gt 10 ] && [ "$NLINES" -lt 80 ]'
chk "last line is the end anchor"   '[ "$(tail -1 "$SNIP")" = "cp -r \"containers/other_home_data/.codex\" \"\${JOB_DIR}/\"" ]'
chk "no apptainer in the snippet"   '! grep -q apptainer "$SNIP"'

for TASK in gsm8k aime2025; do
    JD="$WORK/jobdir_$TASK"; mkdir -p "$JD"
    ( JOB_DIR="$JD" EVALUATION_TASK="$TASK" EVAL_SCRIPT="evaluate.py" \
      bash "$SNIP" ) >/dev/null 2>&1
    chk "$TASK: evaluate.py copied"    '[ -f "$JD/task/evaluate.py" ]'
    chk "$TASK: templates copied"      '[ -d "$JD/task/templates" ]'
    chk "$TASK: graded_read.py copied" '[ -f "$JD/task/graded_read.py" ]'
    chk "$TASK: helper is intact"      'cmp -s "$JD/task/graded_read.py" "$HELPER"'
done
# The conditional half: gsm8k has a reference/ directory and nothing else does, and the
# prompt bullet names paths under reference/, so the directory has to arrive as a
# directory rather than flattened into the task root.
chk "gsm8k: reference/ arrives"        '[ -d "$WORK/jobdir_gsm8k/task/reference" ]'
chk "gsm8k: reference/train_grpo.py"   '[ -f "$WORK/jobdir_gsm8k/task/reference/train_grpo.py" ]'
chk "gsm8k: not flattened"             '[ ! -e "$WORK/jobdir_gsm8k/task/train_grpo.py" ]'
chk "aime2025: no reference/"          '[ ! -e "$WORK/jobdir_aime2025/task/reference" ]'
# ...and the prompt only promises it where it lands.
chk "prompt promises it for gsm8k"     'python3 src/eval/general/get_prompt.py --model-to-train M --benchmark-id gsm8k --num-hours 10 --num-gpus 1 --agent claude | grep -q "reference/train_grpo.py"'
chk "prompt silent for aime2025"       '! python3 src/eval/general/get_prompt.py --model-to-train M --benchmark-id aime2025 --num-hours 10 --num-gpus 1 --agent claude | grep -q "reference/train_grpo.py"'
# Subshells, not bare loops: chk runs its argument through `eval` in THIS shell, so an
# `exit` inside one ends the test run early -- silently, and with whatever status the exit
# carried. Every multi-command check below is parenthesised for that reason.
chk "graded_read announced for both"   '( for b in gsm8k aime2025; do python3 src/eval/general/get_prompt.py --model-to-train M --benchmark-id $b --num-hours 10 --num-gpus 1 --agent claude | grep -q "graded_read.py" || exit 1; done )'
# An unrendered {placeholder} reaching a real prompt is the failure mode the whole
# conditional shape exists to avoid, and it is invisible unless something looks.
chk "no unrendered placeholders"       '( for b in gsm8k aime2025 humaneval; do python3 src/eval/general/get_prompt.py --model-to-train M --benchmark-id $b --num-hours 10 --num-gpus 1 --agent claude | grep -q "{" && exit 1; done; exit 0 )'

echo "[14] a benchmark that writes no inspect log is degraded, not refused"
# healthbench and arenahardwriting are not inspect_ai based: evaluate.py generates, judges
# and writes a metrics dict, and that is the whole artefact. graded_read.py is copied into
# their sandboxes too, and the strict path found no log, refused every single time, and --
# because a refusal quarantines -- renamed their one good metrics file to *.UNVERIFIED.
# That is not conservatism, it is data loss on a task the helper was never able to check.
cd "$WORK" || exit 1
# Deliberately mentions neither inspect_ai nor a log, exactly like the two real ones.
cat > "$WORK/fake_healthbench.py" <<'PYEOF'
import argparse, json, os, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--model-path")
p.add_argument("--limit", type=int)
p.add_argument("--json-output-file")
a, _unknown = p.parse_known_args()

metrics = {"accuracy": 0.3412, "stderr": 0.0189}
if os.environ.get("HB_ROWS"):
    metrics["n_examples"] = int(os.environ["HB_ROWS"])
if os.environ.get("HB_BAD_ACC") == "1":
    metrics["accuracy"] = "0.34"
if os.environ.get("HB_NO_ACC") == "1":
    metrics.pop("accuracy")
print("scored 5000 completions with the grader model")
if os.environ.get("HB_WRITE_METRICS", "1") == "1" and a.json_output_file:
    Path(a.json_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json_output_file).write_text(json.dumps(metrics, indent=2))
sys.exit(int(os.environ.get("HB_EXIT", "0")))
PYEOF
chk "the no-log fake really names no inspect_ai" '! grep -q inspect_ai "$WORK/fake_healthbench.py"'

hb_case() {
    local lbl="$1"; shift
    python3 "$HELPER" --model-path "$CKPT" --label "$lbl" --out-dir out \
        --evaluate "$WORK/fake_healthbench.py" --quiet --print line "$@" \
        > "$WORK/$lbl.stdout" 2> "$WORK/$lbl.stderr"
    RC=$?
    unset HB_ROWS HB_EXIT HB_WRITE_METRICS HB_BAD_ACC HB_NO_ACC
    return 0
}

export HB_ROWS=5000
hb_case hb --rows 5000 --full
chk "exit 0 instead of a refusal"   '[ "$RC" = 0 ]'
chk "verdict verified"              '[ "$(rec hb verdict)" = verified ]'
# The data-loss regression, stated as its own check because it is the part that destroyed
# results rather than merely annoying someone.
chk "the metrics file is STILL THERE" '[ -f out/hb.metrics.json ]'
chk "and was not quarantined"       '[ ! -e out/hb.metrics.json.UNVERIFIED ]'
chk "accuracy reported"             '[ "$(rec hb accuracy)" = 0.3412 ]'
chk "decided_on says so"            '[ "$(rec hb decided_on)" = "exit_code+metrics_file" ]'
chk "and gives the reason"          'grep -q "does not mention inspect_ai" out/hb.graded_read.json'
chk "stdout carries decided_on"     'grep -q "decided_on=exit_code+metrics_file" hb.stdout'
# It must not claim the checks it did not make...
chk "no inspect_log key"            '! has_key hb inspect_log'
chk "no rows_found key"             '! has_key hb rows_found'
chk "checkpoint identity: not checked, and says so" \
    '[ "$(rec hb checkpoint_identity_checked)" = False ]'
# ...but n_examples is a row count it CAN check, so it does.
chk "rows verified off n_examples"  '[ "$(rec hb rows_verified)" = True ]'
chk "and names the key it used"     'grep -q "n_examples" out/hb.graded_read.json'
chk "stdout does not say UNVERIFIED" '! grep -q UNVERIFIED hb.stdout'

# arenahardwriting's metrics dict has no row count at all. "The check did not run" and
# "the check passed" have to be different words, on the record and on stdout.
hb_case hbnorows --rows 500 --full
chk "still verified without a row count" '[ "$RC" = 0 ]'
chk "rows_verified is false"        '[ "$(rec hbnorows rows_verified)" = False ]'
chk "the reason is on the record"   'grep -q rows_check out/hbnorows.graded_read.json'
chk "stdout marks the count UNVERIFIED" 'grep -q "rows=500(UNVERIFIED)" hbnorows.stdout'
chk "metrics file kept here too"    '[ -f out/hbnorows.metrics.json ]'

# The weaker basis is still a basis: it refuses everything it can still see.
export HB_ROWS=500
hb_case hbrows --rows 5000 --full
chk "a row count that disagrees is refused" '[ "$RC" = 6 ]'
chk "and quarantines, because this one IS a bad read" '[ -f out/hbrows.metrics.json.UNVERIFIED ]'
export HB_EXIT=2
hb_case hbfail --rows 5000 --full
chk "a nonzero exit is still refused" '[ "$RC" = 3 ]'
export HB_WRITE_METRICS=0
hb_case hbnofile --rows 5000 --full
chk "no metrics file is still refused" '[ "$RC" = 4 ]'
chk "and the message says there is no log either" \
    'grep -q "no inspect log either" out/hbnofile.graded_read.json'
export HB_NO_ACC=1
hb_case hbnoacc --rows 5000 --full
chk "a metrics file with no accuracy is refused" '[ "$RC" = 5 ]'
export HB_BAD_ACC=1
hb_case hbstracc --rows 5000 --full
chk "a non-numeric accuracy is refused"          '[ "$RC" = 5 ]'

echo "[14b] the mode is a decision, and it can be forced either way"
export HB_ROWS=5000
hb_case hbforce --rows 5000 --full --expect-inspect-log yes
chk "--expect-inspect-log yes refuses the no-log task" '[ "$RC" = 4 ]'
chk "and the record says the flag decided it" \
    'grep -q "expect-inspect-log yes" out/hbforce.graded_read.json'
run_case gsforce --rows 1319 --full --expect-inspect-log no
chk "--expect-inspect-log no is obeyed on an inspect task" '[ "$RC" = 0 ]'
chk "and downgrades the basis"      '[ "$(rec gsforce decided_on)" = "exit_code+metrics_file" ]'
# auto is not allowed to be wrong in the dangerous direction: a script the predicate reads
# as no-log that then writes a log is checked against the log it wrote. This is what stops
# a stale substring from silently disarming the row-count and checkpoint checks.
cp "$WORK/fake_evaluate.py" "$WORK/fake_unlabelled.py"
python3 - "$WORK/fake_unlabelled.py" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read().replace("inspect", "insp" + "ect".upper()[:0] + "ect_x")
open(p, "w").write(s)
PYEOF
chk "the relabelled fake no longer names inspect_ai" '! grep -q inspect_ai "$WORK/fake_unlabelled.py"'
python3 "$HELPER" --model-path "$CKPT" --label autoup --out-dir out \
    --evaluate "$WORK/fake_unlabelled.py" --quiet --print line --rows 1319 --full \
    > autoup.stdout 2> autoup.stderr
RC=$?
chk "auto upgraded to the log it found"  '[ "$(rec autoup decided_on)" = inspect_log ]'
chk "and it verified"                    '[ "$RC" = 0 ]'
chk "the record explains the upgrade"    'grep -q "wrote an inspect log anyway" out/autoup.graded_read.json'
# ...and the strong checks really are back on, not just relabelled.
FAKE_ROWS=500 python3 "$HELPER" --model-path "$CKPT" --label autoup500 --out-dir out \
    --evaluate "$WORK/fake_unlabelled.py" --quiet --print line --rows 1319 --full \
    > autoup500.stdout 2> autoup500.stderr
RC=$?
chk "the row check bites after the upgrade" '[ "$RC" = 6 ]'

echo "[14c] the invariants hold across every record this run wrote"
chk "every record names a decision basis" 'python3 -c "
import glob,json
ps=sorted(glob.glob(\"out/*.graded_read.json\"))
assert len(ps)>=20, len(ps)
for p in ps:
    d=json.load(open(p))
    if \"exit_code\" not in d:      # refused during setup, before the decision is made
        continue
    assert d.get(\"decided_on\") in (\"inspect_log\",\"exit_code+metrics_file\"), (p,d.get(\"decided_on\"))
    assert d.get(\"decided_on_because\"), p
    if d[\"verdict\"]==\"verified\":
        assert isinstance(d.get(\"rows_verified\"), bool), p
"'
chk "*.metrics.json => verified, still" 'python3 -c "
import glob,json,os
n=0
for m in glob.glob(\"out/*.metrics.json\"):
    lbl=os.path.basename(m)[:-len(\".metrics.json\")]
    d=json.load(open(f\"out/{lbl}.graded_read.json\"))
    assert d[\"verdict\"]==\"verified\", (m, d[\"verdict\"])
    n+=1
assert n>=3, n
"'

echo "[15] the prompt hands the agent paths that resolve on EVERY arm"
# The AutoR operator runs each stage with cwd <task>/.autor/<stamp>/, two levels below the
# task root. A bare `reference/train_grpo.py` in the prompt resolves for the control arm
# and for nothing else -- and produces no error anywhere, just a run that looks like it
# ignored the advice. Nothing in the repo would have reported it.
cd "$REPO_ROOT" || exit 1
GP() { python3 src/eval/general/get_prompt.py --model-to-train M --benchmark-id "$1" \
       --num-hours 10 --num-gpus 1 --agent claude; }
# The two path constants come out of run_task.sh, not out of this file.
eval "$(grep -E '^SANDBOX_(HOME|TASK_DIR)=' src/run_task.sh)"
chk "run_task.sh still defines both"  '[ -n "${SANDBOX_HOME:-}" ] && [ -n "${SANDBOX_TASK_DIR:-}" ]'
chk "get_prompt.py's defaults match the bind" "python3 -c '
import sys; sys.path.insert(0, \"src/eval/general\")
import get_prompt as g
assert g.DEFAULT_SANDBOX_HOME == \"$SANDBOX_HOME\", g.DEFAULT_SANDBOX_HOME
assert g.DEFAULT_SANDBOX_TASK_DIR == \"$SANDBOX_TASK_DIR\", g.DEFAULT_SANDBOX_TASK_DIR
'"
for B in gsm8k healthbench aime2025; do
    GP "$B" > "$WORK/prompt_$B.txt" 2>"$WORK/prompt_$B.err"
    chk "$B: prompt renders"  '[ -s "$WORK/prompt_'"$B"'.txt" ]'
    # Every backticked path the harness itself put in the prompt must be absolute. Anchored
    # on the file names this change owns rather than on all backticks, because the upstream
    # template legitimately says `evaluate.py` and `final_model` as task-root-relative
    # contract terms and those are not ours to move.
    chk "$B: no harness-owned relative path" 'python3 - "$WORK/prompt_'"$B"'.txt" <<'"'"'PY'"'"'
import re, sys
text = open(sys.argv[1]).read()
OWNED = ("graded_read.py", "train_grpo.py", "smoke.sh", "contamination_check.py",
         "test_data.json", "reference/README.md")
bad = []
for m in re.finditer(r"\\\\`([^`]+)\\\\`", text):
    tok = m.group(1).split()[0] if m.group(1).split() else ""
    if any(tok.endswith(o) or o in tok for o in OWNED):
        if not tok.startswith("/"):
            bad.append(tok)
if bad:
    print("relative:", bad); sys.exit(1)
PY'
done
chk "gsm8k: reference path is the bound one" \
    'grep -qF "$SANDBOX_TASK_DIR/reference/train_grpo.py" "$WORK/prompt_gsm8k.txt"'
chk "gsm8k: smoke.sh too"  'grep -qF "$SANDBOX_TASK_DIR/reference/smoke.sh" "$WORK/prompt_gsm8k.txt"'
chk "gsm8k: graded_read.py too" 'grep -qF "$SANDBOX_TASK_DIR/graded_read.py" "$WORK/prompt_gsm8k.txt"'
# The decontamination section only renders where the benchmark ships test_data.json, and
# that file is downloaded, not committed (.gitignore: **/test_data.json). Skipped LOUDLY on
# a checkout that has not downloaded it -- a silently absent case is a check that never ran.
DECON_TASK=""
for B in gsm8k healthbench aime2025; do
    [ -f "src/eval/tasks/$B/test_data.json" ] && { DECON_TASK="$B"; break; }
done
if [ -n "$DECON_TASK" ]; then
    chk "$DECON_TASK: the decontamination tool is under the home bind" \
        'grep -qF "$SANDBOX_HOME/contamination_check.py" "$WORK/prompt_'"$DECON_TASK"'.txt"'
    chk "$DECON_TASK: and the test-set copy with it" \
        'grep -qF "$SANDBOX_HOME/test_data.json" "$WORK/prompt_'"$DECON_TASK"'.txt"'
else
    echo "  SKIP no task here has downloaded its test_data.json, so the decontamination"
    echo "  SKIP section did not render and its two absolute paths are unchecked this run"
fi
chk "gsm8k: the reference bullet names MODEL_TO_TRAIN" \
    'grep -q "MODEL_TO_TRAIN" "$WORK/prompt_gsm8k.txt"'
# The prompt's description of graded_read.py must match what it will actually do there.
chk "gsm8k: the strong wording, since gsm8k writes a log" \
    'grep -q "the evaluation log confirms the row count" "$WORK/prompt_gsm8k.txt"'
chk "healthbench: the honest wording instead" \
    'grep -q "writes no inspect log" "$WORK/prompt_healthbench.txt"'
chk "healthbench: does not promise a row-count check" \
    '! grep -q "the evaluation log confirms the row count" "$WORK/prompt_healthbench.txt"'
chk "healthbench: names decided_on so the agent can read the record" \
    'grep -q "decided_on" "$WORK/prompt_healthbench.txt"'
# The wording is chosen by graded_read.py's own predicate, so the two cannot drift.
chk "the prompt and the tool share one predicate" 'python3 -c "
import sys; sys.path.insert(0, \"src/eval/general\")
import get_prompt as g, graded_read as r
assert g.evaluate_uses_inspect is r.evaluate_uses_inspect
assert g.evaluate_uses_inspect(g.task_evaluate_script(\"gsm8k\")) is True
assert g.evaluate_uses_inspect(g.task_evaluate_script(\"healthbench\")) is False
assert g.evaluate_uses_inspect(g.task_evaluate_script(\"arenahardwriting\")) is False
"'
# A relative path must be refused rather than rendered: an absolute default is only a fix
# while nothing can pass something else.
chk "a relative --sandbox-task-dir is refused" \
    '! python3 src/eval/general/get_prompt.py --model-to-train M --benchmark-id gsm8k \
        --num-hours 10 --num-gpus 1 --agent claude --sandbox-task-dir task >/dev/null 2>&1'

echo "[16] nothing ships bytecode into the 64 MiB overlay"
# A stray reference/__pycache__/*.cpython-313.pyc was on disk in this repo: py3.13 bytecode
# that `cp -r` would have shipped into a py3.10 container, in the very directory whose
# sibling intervention is "keep bytecode off the overlay". .gitignore hides it from
# `git status`, so nothing would have said a word.
# src/eval is the tree the copy block reaches with `cp -r` (templates/, evaluation_code/,
# task_context/, reference/); src/utils/graded_read.py is copied as a single named file, so
# its bytecode cannot ride along -- which is itself asserted below rather than assumed.
chk "no __pycache__ anywhere run_task.sh copies recursively from" \
    '[ -z "$(find src/eval -name "__pycache__" -print -quit)" ]'
chk "no .pyc/.pyo either" \
    '[ -z "$(find src/eval \( -name "*.pyc" -o -name "*.pyo" \) -print -quit)" ]'
chk "graded_read.py is copied as a file, not as its directory" \
    'grep -q "^cp src/utils/graded_read.py " "$SNIP"'
# And the copy block strips it, so a stray one on the run node cannot reach the sandbox.
FIX="$WORK/jobdir_pyc"; mkdir -p "$FIX"
mkdir -p src/eval/tasks/gsm8k/reference/__pycache__
printf '\x00fake pyc' > src/eval/tasks/gsm8k/reference/__pycache__/train_grpo.cpython-313.pyc
mkdir -p src/utils/__pycache__
printf '\x00fake pyc' > src/utils/__pycache__/graded_read.cpython-313.pyc
( JOB_DIR="$FIX" EVALUATION_TASK=gsm8k EVAL_SCRIPT=evaluate.py bash "$SNIP" ) >/dev/null 2>&1
rm -rf src/eval/tasks/gsm8k/reference/__pycache__ src/utils/__pycache__
chk "the fixture pyc did exist to be stripped" '[ -d "$FIX/task" ]'
chk "no __pycache__ reached the sandbox copy" \
    '[ -z "$(find "$FIX/task" -name "__pycache__" -print -quit)" ]'
chk "no .pyc reached the sandbox copy" \
    '[ -z "$(find "$FIX/task" -name "*.pyc" -print -quit)" ]'
chk "and the real files still arrived" \
    '[ -f "$FIX/task/reference/train_grpo.py" ] && [ -f "$FIX/task/graded_read.py" ]'

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
