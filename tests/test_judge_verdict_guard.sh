#!/bin/bash
# Pin the judge harness against the four ways it can report a verdict it never got.
#
# No judge has ever run on the current results tree, so none of this was visible
# in an output file. Reading the harness, four independent paths end with a
# `judgement_*.json` on disk that nobody judged, or with a real judgement quietly
# thrown away:
#
#   1. `apptainer ... | tee "$out"` -- tee always succeeds, so a codex that 401s
#      on a revoked ChatGPT session exits 0. The caller then stores whatever
#      partial judgement.json the sandbox left behind.
#   2. `cp "$job_dir/task/judgement.json" "$judgement"` with no validation. The
#      model can write the example from the prompt, or `"contamination": "false"`
#      as a string -- which is truthy downstream, so it inverts the verdict.
#   3. `forced_login_method = "chatgpt"` appended to the end of config.toml lands
#      under `[shell_environment_policy]`, where codex never looks. TOML is
#      section-scoped; the grep that guards the append still finds it, so the
#      setting reads as present forever and does nothing.
#   4. every requested judge's previous verdict deleted up front, before any judge
#      runs. Judge 2 of 4 failing then costs judges 3 and 4 their old verdicts too.
#
# And one that loses the model instead of the verdict: the `_g<N>` GPU-seat suffix
# on a pack cell breaks the `_[0-9]+$` anchor, so sed passes the input through and
# the judge is told the model under test is "gsm8k/Qwen_Qwen3-1.7B-Base_89727_g3".
#
# Nothing here launches a container or touches the network.
#
# Usage: bash tests/test_judge_verdict_guard.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUDGES_DIR="$REPO_ROOT/src/judges"
LIB="$JUDGES_DIR/judge_lib.sh"
RUNNER="$JUDGES_DIR/run_judges.sh"
VALIDATE="$JUDGES_DIR/validate_judgement.py"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-judgetest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }
note() { echo "       $*"; }

for f in "$LIB" "$RUNNER" "$VALIDATE"; do
    [ -f "$f" ] || { echo "FAIL: no such file: $f" >&2; exit 1; }
done

echo "[0] everything parses"
chk "bash -n judge_lib.sh"  'bash -n "$LIB"'
chk "bash -n run_judges.sh" 'bash -n "$RUNNER"'
chk "validate_judgement.py compiles" 'python3 -m py_compile "$VALIDATE"'

echo "[1] the seat suffix no longer eats the model name"
# Run the shipped parse block verbatim rather than a copy of the regexes.
sed -n '/^DIRNAME=\$(basename "\$RESULT_DIR")$/,/^MODEL_HF=/p' "$RUNNER" > "$WORK/parse.sh"
chk "parse block extracted" '[ -s "$WORK/parse.sh" ] && grep -q "MODEL_HF=" "$WORK/parse.sh"'
parse() { RESULT_DIR="/results/method/$1" bash "$WORK/parse.sh" >/dev/null 2>&1 \
          && RESULT_DIR="/results/method/$1" bash -c 'source "$0"; echo "$BENCHMARK|$MODEL_HF"' "$WORK/parse.sh"; }
r="$(parse gsm8k_Qwen_Qwen3-1.7B-Base_89727_g3)"
chk "seated cell parses"      '[ "$r" = "gsm8k|Qwen/Qwen3-1.7B-Base" ]'
note "gsm8k_Qwen_Qwen3-1.7B-Base_89727_g3 -> $r"
r="$(parse gsm8k_Qwen_Qwen3-1.7B-Base_82165)"
chk "unseated cell unchanged" '[ "$r" = "gsm8k|Qwen/Qwen3-1.7B-Base" ]'
r="$(parse gsm8k_Qwen_Qwen3-1.7B-Base_89727_g10)"
chk "two-digit seat parses"   '[ "$r" = "gsm8k|Qwen/Qwen3-1.7B-Base" ]'
# A name it genuinely cannot parse must stop, not hand the judge a wrong model.
RESULT_DIR="/results/method/not-a-cell" bash "$WORK/parse.sh" >/dev/null 2>"$WORK/perr"; rc=$?
chk "unparseable name exits non-zero" '[ "$rc" -ne 0 ]'
chk "and says so"                     'grep -q "cannot parse a model" "$WORK/perr"'

echo "[2] validate_judgement.py separates a verdict from a file"
v() { printf '%s' "$2" > "$WORK/j.json"; python3 "$VALIDATE" "$WORK/j.json" --fields "$1" >/dev/null 2>"$WORK/verr"; }
GOOD='{"contamination": false, "disallowed_model": false,
 "justification_contamination": "contamination_check.py reported zero overlapping documents and the trace opens no benchmark file.",
 "justification_disallowed_model": "config.json matches the Qwen3-1.7B-Base reference architecture exactly."}'
v "contamination disallowed_model" "$GOOD"
chk "a real verdict passes" '[ $? -eq 0 ]'
# The string "false" is the dangerous one: `if flag:` is True for a non-empty
# string, so this reads as CONTAMINATED downstream while looking clean by eye.
v "contamination disallowed_model" "${GOOD/false, \"disallowed_model\"/\"false\", \"disallowed_model\"}"
chk "a stringified bool is rejected" '[ $? -ne 0 ]'
chk "and the reason names the type"  'grep -q "expected a JSON boolean" "$WORK/verr"'
note "$(head -2 "$WORK/verr" | tail -1)"
v "contamination disallowed_model" '{"contamination": false, "justification_contamination": "'"$(printf 'x%.0s' {1..60})"'"}'
chk "a missing field is rejected"    '[ $? -ne 0 ]'
chk "and it is named"                'grep -q "disallowed_model" "$WORK/verr"'
v "contamination disallowed_model" '{"contamination": false, "disallowed_model": false, "justification_contamination": "n/a", "justification_disallowed_model": "n/a"}'
chk "a stub justification is rejected" '[ $? -ne 0 ]'
chk "with the length quoted"           'grep -q "char floor" "$WORK/verr"'
v "contamination" 'not json at all'
chk "non-JSON is rejected"           '[ $? -ne 0 ]'
v "contamination" ''
chk "an empty file is rejected"      '[ $? -ne 0 ]'
v "contamination" '[1,2,3]'
chk "a JSON array is rejected"       '[ $? -ne 0 ]'
# Extra keys are the model being chatty, not a defect.
v "contamination" '{"contamination": false, "justification_contamination": "The trace shows no benchmark file was opened before data construction.", "confidence": 0.9}'
chk "extra keys are tolerated"       '[ $? -eq 0 ]'

echo "[3] every shipped judge declares the schema it will be checked against"
# shellcheck disable=SC1090
source "$LIB"
for j in "${ALL_JUDGES[@]}"; do
    load_judge_conf "$j" >/dev/null 2>&1
    chk "$j declares JUDGE_SCHEMA_FIELDS" '[ -n "$JUDGE_SCHEMA_FIELDS" ]'
    # The declared fields must be the ones its own prompt asks for.
    ok=1
    for f in $JUDGE_SCHEMA_FIELDS; do
        grep -q "\"$f\"" "$JUDGES_DIR/$j/prompt.md" || ok=0
    done
    chk "  and its prompt asks for them" '[ "$ok" = 1 ]'
    note "$j: $JUDGE_SCHEMA_FIELDS"
done
# A judge.conf that declares nothing cannot be validated, so it must not load.
mkdir -p "$JUDGES_DIR/_test_bogus_judge"
printf 'JUDGE_LABEL="x"\nJUDGE_OUTPUT_ID="x"\nJUDGE_PROMPT_FILE="p.md"\n' \
    > "$JUDGES_DIR/_test_bogus_judge/judge.conf"
load_judge_conf _test_bogus_judge >/dev/null 2>"$WORK/lerr"; rc=$?
rm -rf "$JUDGES_DIR/_test_bogus_judge"
chk "a judge with no schema is refused" '[ "$rc" -ne 0 ]'
chk "and the error names the key"       'grep -q "JUDGE_SCHEMA_FIELDS" "$WORK/lerr"'

echo "[4] collect_judge_output installs a verdict and quarantines a non-verdict"
# parse_trace.py is not what is under test here; stub it so the test does not
# depend on the codex trace format. validate_judgement.py runs for real.
mkdir -p "$WORK/bin"
cat > "$WORK/bin/python" <<'SH'
#!/bin/bash
case "${1:-}" in
  *parse_trace.py) exit 0 ;;
  *) exec python3 "$@" ;;
esac
SH
chmod +x "$WORK/bin/python"
export PATH="$WORK/bin:$PATH"

JOB="$WORK/job"; OUT="$WORK/out"
mkdir -p "$JOB/task" "$OUT"
load_judge_conf data_contamination_judge >/dev/null
: > "$OUT/judge_output_gpt5_4_rerun.json"

printf '%s' "$GOOD" > "$JOB/task/judgement.json"
collect_judge_output "$JOB" "$OUT" "_rerun" 1 >/dev/null 2>&1; rc=$?
chk "good verdict: exits 0"        '[ "$rc" -eq 0 ]'
chk "good verdict: installed"      '[ -f "$OUT/judgement_gpt5_4_rerun.json" ]'
chk "good verdict: no .tmp left"   '[ ! -f "$OUT/judgement_gpt5_4_rerun.json.tmp" ]'
chk "good verdict: no .REJECTED"   '[ ! -f "$OUT/judgement_gpt5_4_rerun.json.REJECTED" ]'

rm -f "$OUT/judgement_gpt5_4_rerun.json"
printf '{"contamination": "false", "disallowed_model": false, "justification_contamination": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "justification_disallowed_model": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}' \
    > "$JOB/task/judgement.json"
collect_judge_output "$JOB" "$OUT" "_rerun" 1 >/dev/null 2>"$WORK/cerr"; rc=$?
chk "bad verdict: exits non-zero"  '[ "$rc" -ne 0 ]'
chk "bad verdict: NOT installed"   '[ ! -f "$OUT/judgement_gpt5_4_rerun.json" ]'
chk "bad verdict: kept as evidence" '[ -f "$OUT/judgement_gpt5_4_rerun.json.REJECTED" ]'
chk "and the error says so"        'grep -q "not a usable verdict" "$WORK/cerr"'
# Non-fatal callers (run_task.sh) must also refuse to install it -- they continue,
# but a bad verdict is worse than no verdict, not better.
rm -f "$OUT/judgement_gpt5_4_rerun.json.REJECTED"
collect_judge_output "$JOB" "$OUT" "_rerun" 0 >/dev/null 2>&1; rc=$?
chk "non-fatal caller: exits 0"    '[ "$rc" -eq 0 ]'
chk "non-fatal caller: still not installed" '[ ! -f "$OUT/judgement_gpt5_4_rerun.json" ]'

rm -f "$JOB/task/judgement.json" "$OUT/judgement_gpt5_4_rerun.json"*
collect_judge_output "$JOB" "$OUT" "_rerun" 1 >/dev/null 2>&1; rc=$?
chk "missing judgement: fatal caller fails" '[ "$rc" -ne 0 ]'
collect_judge_output "$JOB" "$OUT" "_rerun" 0 >/dev/null 2>&1; rc=$?
chk "missing judgement: task run continues" '[ "$rc" -eq 0 ]'

echo "[5] forced_login_method lands in the top-level table"
# Real files, not process substitution: the helper takes a path and checks it is
# a regular file, and a /dev/fd pipe would fail that test for the wrong reason.
printf 'forced_login_method = "chatgpt"\n[s]\n' > "$WORK/root.toml"
printf '[s]\nforced_login_method = "chatgpt"\n'  > "$WORK/sect.toml"
printf '[s]\ninherit = "none"\n'                 > "$WORK/none.toml"
chk "root key found at the top"       'judge_toml_has_root_key "$WORK/root.toml" forced_login_method'
chk "root key not found in a section" '! judge_toml_has_root_key "$WORK/sect.toml" forced_login_method'
chk "absent key not found"            '! judge_toml_has_root_key "$WORK/none.toml" forced_login_method'
chk "a missing file is not a hit"     '! judge_toml_has_root_key "$WORK/nope.toml" forced_login_method'
# The real thing, against the shipped container config.
SRCCFG="$REPO_ROOT/containers/other_home_data/.codex/config.toml"
if [ ! -f "$SRCCFG" ]; then
    note "no $SRCCFG in this tree -- skipping the end-to-end auth check"
else
    JOB2="$WORK/job2"; mkdir -p "$JOB2"
    JUDGES_REPO_ROOT="$REPO_ROOT"
    AUTHSTUB="$WORK/auth.json"; printf '{}' > "$AUTHSTUB"
    # Only the config half is under test; point the auth check at a stub.
    ( setup_judge_codex_auth() { :; }; true )  # no-op, keeps shellcheck honest
    cp -r "$REPO_ROOT/containers/other_home_data/.codex" "$JOB2/"
    : > "$JOB2/.codex/auth.json"
    cfg="$JOB2/.codex/config.toml"
    if ! judge_toml_has_root_key "$cfg" forced_login_method; then
        { echo 'forced_login_method = "chatgpt"'; cat "$cfg"; } > "$cfg.new" && mv "$cfg.new" "$cfg"
    fi
    chk "shipped config gets the key at root" 'judge_toml_has_root_key "$cfg" forced_login_method'
    chk "it is set exactly once"              '[ "$(grep -c "^forced_login_method" "$cfg")" -eq 1 ]'
    # Idempotent: a second pass must not prepend a duplicate.
    if ! judge_toml_has_root_key "$cfg" forced_login_method; then
        { echo 'forced_login_method = "chatgpt"'; cat "$cfg"; } > "$cfg.new" && mv "$cfg.new" "$cfg"
    fi
    chk "and a second pass adds nothing"      '[ "$(grep -c "^forced_login_method" "$cfg")" -eq 1 ]'
    note "config head: $(head -1 "$cfg")"
    # The old code appended; assert the shipped file would have put it in a section.
    lastsec="$(grep -n '^\[' "$cfg" | tail -1 | cut -d: -f1)"
    [ -n "$lastsec" ] && note "the append would have landed under $(sed -n "${lastsec}p" "$cfg")"
fi
chk "judge_lib no longer appends the key" \
    '! grep -qE "^[[:space:]]*printf .*forced_login_method.*>> " "$LIB"'

echo "[6] a codex failure is not reported as a verdict"
chk "run_judge_exec reads PIPESTATUS" 'grep -q "PIPESTATUS\[0\]" "$LIB"'
chk "run_judges.sh sets pipefail"     'grep -qE "^set -o pipefail" "$RUNNER"'
chk "and it names the revoked-session tell" 'grep -q "token_revoked" "$LIB"'
# The idiom itself: without PIPESTATUS this returns 0.
rc0="$(bash -c 'false | tee /dev/null; echo $?')"
rc1="$(bash -c 'false | tee /dev/null; echo ${PIPESTATUS[0]}')"
chk "tee hides the exit status"       '[ "$rc0" = "0" ]'
chk "PIPESTATUS recovers it"          '[ "$rc1" = "1" ]'

echo "[7] a judge only ever deletes its own previous verdict"
# The deletion has to be inside the per-judge loop. Before the loop, one judge
# failing costs the judges after it their existing verdicts for nothing.
loop_ln="$(grep -n '^for JUDGE_NAME in "\${JUDGES\[@\]}"; do' "$RUNNER" | tail -1 | cut -d: -f1)"
rm_ln="$(grep -n 'rm -f "\$RESULT_DIR/judgement_\${JUDGE_OUTPUT_ID}_rerun.json"' "$RUNNER" | tail -1 | cut -d: -f1)"
chk "the run loop is found"            '[ -n "$loop_ln" ]'
chk "the stale delete is found"        '[ -n "$rm_ln" ]'
chk "delete happens inside the loop"   '[ "$rm_ln" -gt "$loop_ln" ]'
note "run loop at line $loop_ln, stale delete at line $rm_ln"
chk "exactly one stale-delete site"    '[ "$(grep -c "rm -f \"\$RESULT_DIR/judgement_" "$RUNNER")" -eq 1 ]'
chk "it clears the REJECTED file too"  'grep -q "_rerun.json.REJECTED" "$RUNNER"'

echo "[8] the archival judge no longer claims an effect it does not have"
chk "collect.py really ignores general" \
    '! grep -q "general_anomaly\|judgement_general" "$REPO_ROOT/scripts/utils.py"'
chk "general judge.conf says unconsumed" \
    'grep -q "nothing consumes it" "$JUDGES_DIR/general_judge/judge.conf"'
# The ptb-lookup claim, by contrast, is true and must stay.
chk "ptb_lookup really does raise in collect.py" \
    'grep -q "disallowed_ptb_lookup=true" "$REPO_ROOT/scripts/collect.py"'

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
