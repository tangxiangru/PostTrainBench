#!/bin/bash
# Check the CPU-testable half of src/eval/tasks/gsm8k/reference/train_grpo.py.
#
# Worth having as a file rather than a code read, twice over. The generation_config fix
# is a fix for a ValueError that only fires on a *write*, an hour into a GPU run --
# `GenerationConfig(do_sample=False, temperature=0.0)` constructs fine, validates fine,
# and blows up in save_pretrained, which is why five cells found it the expensive way. So
# [4] does the write. And the reward is a transcription of a scorer that lives in a pinned
# third-party wheel; a transcription that drifts is a model trained for a test nobody is
# giving, with nothing in any log to show for it, so [2] and [3] pin the transcription
# against strings taken from the real gsm8k test set.
#
# Usage: bash tests/test_gsm8k_reference_grpo.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="$REPO_ROOT/src/eval/tasks/gsm8k/reference"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-grpotest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
# py_compile and the import below both drop __pycache__ next to the source otherwise, and
# a test that leaves artefacts in the tree it is testing is a test people stop running.
export PYTHONPYCACHEPREFIX="$WORK/pycache"

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }

echo "[1] the files parse"
chk "train_grpo.py compiles"  'python3 -m py_compile "$REF/train_grpo.py"'
chk "smoke.sh parses"         'bash -n "$REF/smoke.sh"'
chk "README exists"           '[ -s "$REF/README.md" ]'
chk "README says BEATEN"      'grep -qi "beat it, not adopt it" "$REF/README.md"'
# Parenthesised: chk runs its argument through `eval` in THIS shell, so a bare `exit 1`
# inside a loop ends the whole run -- silently, and looking like the checks after it
# passed. Mutation-tested: without the subshell, reverting the README killed this file at
# [1] and printed four PASSes and nothing else.
chk "README names all four swept base models" \
    '( for m in gemma-3-4b-pt Qwen3-4B-Base Qwen3-1.7B-Base SmolLM3-3B-Base; do
           grep -q "$m" "$REF/README.md" || exit 1; done )'
chk "no __pycache__ shipped beside the reference" '[ ! -e "$REF/__pycache__" ]'
chk "no .pyc shipped beside the reference" \
    '[ -z "$(find "$REF" -name "*.pyc" -o -name "*.pyo")" ]'

echo "[1b] smoke.sh forwards its arguments to train_grpo.py"
# smoke.sh took an output dir and nothing else, and passed no --model. With the model now
# mandatory, that made it a script that cannot be run outside the sandbox at all -- and,
# worse, one whose obvious repair is to hardcode a model in a second place. Driven, not
# grepped: train_grpo.py is stubbed out so the real argv smoke.sh builds is what is read.
mkdir -p "$WORK/smoke/reference"
cp "$REF/smoke.sh" "$WORK/smoke/reference/smoke.sh"
cat > "$WORK/smoke/reference/train_grpo.py" <<'STUB'
import sys
open(sys.argv[0] + ".argv", "w").write("\n".join(sys.argv[1:]) + "\n")
STUB
SARGV="$WORK/smoke/reference/train_grpo.py.argv"
# The checkpoint half of smoke.sh must not run: it needs a GPU. Stopping at the missing
# checkpoint is fine -- the argv file is already written by then.
bash "$WORK/smoke/reference/smoke.sh" "$WORK/smoke/out" --model google/gemma-3-4b-pt \
     --learning-rate 3e-6 >/dev/null 2>&1
chk "smoke.sh ran train_grpo.py"          '[ -s "$SARGV" ]'
chk "output dir still positional"         'grep -A1 -x -- --output-dir "$SARGV" | grep -qx "$WORK/smoke/out"'
chk "--model reaches train_grpo.py"       'grep -A1 -x -- --model "$SARGV" | grep -qx google/gemma-3-4b-pt'
chk "so does a second forwarded flag"     'grep -A1 -x -- --learning-rate "$SARGV" | grep -qx 3e-6'
chk "smoke.sh hardcodes no model"         '! grep -Eq "Qwen/|google/|HuggingFaceTB/" "$REF/smoke.sh"'
rm -f "$SARGV"
# A leading flag must be a flag, not an output directory.
bash "$WORK/smoke/reference/smoke.sh" --model Qwen/Qwen3-4B-Base >/dev/null 2>&1
chk "a leading flag is not eaten as the out dir" \
    'grep -A1 -x -- --model "$SARGV" | grep -qx Qwen/Qwen3-4B-Base'
chk "and the out dir then falls back to a temp path" \
    'grep -A1 -x -- --output-dir "$SARGV" | grep -q grpo_smoke'

# The python half runs as one process: importing transformers costs seconds, and the
# point of each check is the shipped function, not the interpreter startup. It emits the
# same "PASS "/"FAIL " vocabulary chk does, so the two halves read as one list.
cat > "$WORK/probe.py" <<'PYEOF'
import os, sys, tempfile

# $MODEL_TO_TRAIN is set inside every sandbox this file runs in, and [7]/[10] are about
# what happens when it is and is not there. Whatever this machine has is not the fixture.
os.environ.pop("MODEL_TO_TRAIN", None)

sys.path.insert(0, sys.argv[1])
# Importing the shipped module, not a copy of it: everything below therefore fails if
# train_grpo.py drifts. This is also the assertion that the module's heavy imports really
# are deferred -- trl is not installed on a login node, and this line would raise if
# train_grpo.py imported it at module scope.
import train_grpo as t

fail = 0
def chk(desc, ok):
    global fail
    print(("  PASS " if ok else "  FAIL ") + desc)
    if not ok:
        fail = 1

# Real rows from the gsm8k test split (openai/gsm8k, config "main"), answers verbatim
# including the #### delimiter, so the target extraction is exercised on real text.
ROWS = [
    ("Janet's ducks lay 16 eggs per day...", "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
     "She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18", "18"),
    ("A robe takes 2 bolts of blue fiber...", "It takes 2/2=<<2/2=1>>1 bolt of white fiber\n"
     "So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3", "3"),
    ("Mark's basketball team scores...", "...\n#### 201", "201"),
]

print("[2] the grader's target extraction")
chk("#### split, whitespace stripped", [t.gsm8k_target(a) for _, a, _ in ROWS] == [x for _, _, x in ROWS])
chk("comma-grouped target kept as written", t.gsm8k_target("blah\n#### 1,450,000") == "1,450,000")
chk("negative target kept as written", t.gsm8k_target("blah\n#### -10") == "-10")

print("[3] grade_completion reproduces match(numeric=True, location='end')")
g = lambda c, tgt, **kw: t.grade_completion(c, tgt, **kw)
chk("plain last-line answer", g("Reasoning: ...\n\nANSWER: 18", "18")[1])
chk("wrong answer rejected", not g("Reasoning: ...\n\nANSWER: 19", "18")[1])
# The defect this whole reference is a reaction to: answer correct, then keeps talking.
chk("text after the answer is graded", not g("ANSWER: 18\nWait, actually it is 42", "18")[1])
chk("extracted answer is the LAST number", g("ANSWER: 18\nWait, actually it is 42", "18")[0] == "42")
chk("$ and , stripped from the completion", g("ANSWER: $1,450,000", "1450000")[1])
chk("trailing period stripped", g("The answer is 3.", "3")[1])
chk("normalisation makes 3.0 match 3", g("ANSWER: 3.0", "3")[1])
# 16/1319 test targets are not isnumeric(); the grader takes its string branch on those
# and demands the literal target as a suffix, commas and all.
chk("comma target: literal suffix accepted", g("ANSWER: 1,450,000", "1,450,000")[1])
chk("comma target: de-comma'd answer REJECTED", not g("ANSWER: 1450000", "1,450,000")[1])
chk("negative target matched as a string", g("ANSWER: -10", "-10")[1])
# The grader's own endswith quirk, reproduced by default and removable with strict=True.
chk("grader mode credits 25 against target 5", g("ANSWER: 25", "5")[1])
chk("strict mode does not", not g("ANSWER: 25", "5", strict=True)[1])
chk("strict still accepts the real match", g("ANSWER: 5", "5", strict=True)[1])
chk("no number at all is not a crash", g("I have no idea", "18")[1] is False)
chk("empty completion is not a crash", g("", "18")[1] is False)

print("[4] the reward function TRL will actually call")
r = t.make_reward_fn()
conv = [[{"role": "assistant", "content": "ANSWER: 18"}], [{"role": "assistant", "content": "ANSWER: 19"}]]
chk("conversational completions", r(completions=conv, target=["18", "18"]) == [1.0, 0.0])
chk("plain-string completions", r(completions=["ANSWER: 3"], target=["3"]) == [1.0])
chk("tolerates TRL's extra kwargs", r(completions=["ANSWER: 3"], target=["3"], prompts=["x"],
                                      completion_ids=[[1]], trainer_state=None) == [1.0])
chk("strict variant is a subset", t.make_reward_fn(strict=True)(completions=["ANSWER: 25"],
                                                                target=["5"]) == [0.0])

print("[5] sanitise_generation_config against the real transformers validator")
try:
    from transformers import GenerationConfig
except ImportError:
    print("  FAIL transformers is not importable, so [5] proves nothing")
    fail = 1
else:
    def save(cfg):
        with tempfile.TemporaryDirectory() as d:
            cfg.save_pretrained(d)

    # First: reproduce the five-cell failure, so the fix below is measured against a
    # bug that is demonstrably present in this transformers, not one we assume.
    broken = GenerationConfig()
    broken.do_sample = False
    broken.temperature = 0.0
    reproduced = False
    try:
        save(broken)
    except ValueError as exc:
        reproduced = "do_sample" in str(exc) and "temperature" in str(exc)
    chk("the bug reproduces here", reproduced)

    fixed = t.sanitise_generation_config(broken)
    ok = True
    try:
        save(fixed)
    except Exception:
        ok = False
    chk("sanitised config saves", ok)
    chk("temperature survives the fix", fixed.temperature == 0.0)
    chk("do_sample flipped, not the temperature", fixed.do_sample is True)
    chk("top_k is transformers-legal (0, not -1)", fixed.top_k == 0)

    # top_k=0 with do_sample False is the *second* ValueError, on a different flag.
    tk = GenerationConfig()
    tk.do_sample = False
    tk.top_k = 0
    ok = True
    try:
        save(t.sanitise_generation_config(tk))
    except Exception:
        ok = False
    chk("also clears the top_k form of the same refusal", ok)

    custom = t.sanitise_generation_config(GenerationConfig(), temperature=1e-6)
    chk("--shipped-temperature is honoured", custom.temperature == 1e-6)
    chk("1e-6 is below vLLM's greedy epsilon 1e-5", custom.temperature < 1e-5)

    print("[6] ensure_stop_tokens")
    base = GenerationConfig()
    base.eos_token_id = 151643              # what Qwen3-1.7B-Base actually ships
    t.ensure_stop_tokens(base, 151645)      # <|im_end|>, what qwen3.jinja emits
    chk("im_end added", 151645 in base.eos_token_id)
    chk("endoftext kept", 151643 in base.eos_token_id)
    t.ensure_stop_tokens(base, 151645)
    chk("idempotent", base.eos_token_id.count(151645) == 1)
    empty = GenerationConfig(); empty.eos_token_id = None
    t.ensure_stop_tokens(empty, 151645)
    chk("handles a config with no eos at all", empty.eos_token_id == [151645])
    ok = True
    try:
        save(t.sanitise_generation_config(base))
    except Exception:
        ok = False
    chk("both fixes together still save", ok)

print("[7] the CLI exposes the knobs, with defaults")
a = t.parse_args([])
chk("beta defaults to 0.0", a.beta == 0.0)
chk("loss_type defaults to dr_grpo", a.loss_type == "dr_grpo")
chk("scale_rewards defaults to none", a.scale_rewards == "none")
chk("vllm colocate by default", a.use_vllm is True and a.vllm_mode == "colocate")
chk("save_steps is low enough to leave candidates", 0 < a.save_steps <= 100)
chk("fewshot matches the grader's 10 / seed 42", a.fewshot == 10 and a.fewshot_seed == 42)
chk("max_prompt_length clears the measured 2370", a.max_prompt_length >= 2370)
chk("shipped temperature is greedy under vLLM", a.shipped_temperature < 1e-5)
# No model, no template, no stop token in the defaults -- all three are resolved from the
# model this cell is actually about. See [10]; asserting a literal here is the defect.
chk("no default model", a.model is None)
chk("no default chat template", a.chat_template is None)
chk("no default stop token", a.stop_token is None)
# Nothing a user would want to change may be missing from the CLI.
for flag in ("learning_rate", "num_generations", "per_device_train_batch_size",
             "gradient_accumulation_steps", "max_completion_length", "temperature",
             "num_train_epochs", "max_steps", "seed", "output_dir", "model",
             "vllm_gpu_memory_utilization", "save_total_limit", "gradient_checkpointing"):
    chk(f"--{flag.replace('_', '-')} exists", hasattr(a, flag))
# TRL raises unless generation_batch_size % num_generations == 0, and the default has to
# satisfy it or the reference does not start.
gen_batch = a.per_device_train_batch_size * a.gradient_accumulation_steps
chk(f"default batch {gen_batch} is divisible by G={a.num_generations}",
    gen_batch % a.num_generations == 0)
chk("default is the arm's median 128 rollouts/step", gen_batch == 128)

print("[8] the vLLM rollout stop reaches BOTH modes")
# Regression. The first version of this guarded on `vllm_mode == "colocate"`, so
# `--vllm-mode server` trained on rollouts that never stopped -- silently, because a
# rollout that runs to the cap looks exactly like a verbose one in the log. TRL merges
# args.generation_kwargs into the SamplingParams dict last in both modes (colocate
# grpo_trainer.py:1420-1433, server grpo_trainer.py:1325-1335 -> vllm_serve.py:591-604),
# so the correct answer does not depend on the mode at all.
IM_END = 151645
for mode in ("colocate", "server"):
    a = t.parse_args(["--vllm-mode", mode])
    chk(f"stop_token_ids set in {mode} mode",
        t.build_generation_kwargs(a.use_vllm, IM_END) == {"stop_token_ids": [IM_END]})
chk("no generation_kwargs without vllm", t.build_generation_kwargs(False, IM_END) is None)
chk("no generation_kwargs without a stop token", t.build_generation_kwargs(True, None) is None)
# --stop-token '' means "keep the tokenizer's own", and must not fabricate an empty stop.
chk("empty --stop-token yields no stop", t.parse_args(["--stop-token", ""]).stop_token == "")

print("[9] contamination guard")
bad = t.parse_args(["--train-split", "test"])
raised = False
try:
    t.build_dataset(bad)
except SystemExit:
    raised = True
chk("refuses --train-split test", raised)

print("[10] the base model is taken from the run, not from a literal")
# This same reference/ directory is copied into every gsm8k cell, and gsm8k is swept over
# four base models. The file used to default --model to Qwen/Qwen3-1.7B-Base and
# --chat-template to templates/qwen3.jinja, which on three cells out of four is a script
# that trains the wrong model under the wrong template and says nothing about it -- while
# rule 8 of the same prompt forbids exactly that.
#
# The model list is READ OUT of src/commit_utils/commit.sh, the file that actually
# launches the sweep, and not restated here. A fifth model added there fails this test
# until train_grpo.py can map it, which is the only version of this check that stays true.
import re
sweep = open(sys.argv[2]).read()
block = re.search(r"^models=\(\n(.*?)^\)", sweep, re.S | re.M)
MODELS = re.findall(r'"([^"]+)"', block.group(1)) if block else []
chk("read the swept model list out of commit.sh", len(MODELS) >= 4)

chk("MODEL_ENV_VAR is the harness variable MODEL_TO_TRAIN", t.MODEL_ENV_VAR == "MODEL_TO_TRAIN")
raised = ""
try:
    t.resolve_model(None, env={})
except SystemExit as exc:
    raised = str(exc)
chk("no --model and no env var stops the run", bool(raised))
chk("the message names the env var", "MODEL_TO_TRAIN" in raised)
chk("the message names the flag", "--model" in raised)
empty_stopped = False
try:
    t.resolve_model(None, env={"MODEL_TO_TRAIN": "   "})
except SystemExit:
    empty_stopped = True
chk("a blank env var is not a model either", empty_stopped)
chk("the env var is used when set",
    t.resolve_model(None, env={"MODEL_TO_TRAIN": "Qwen/Qwen3-4B-Base"}) == "Qwen/Qwen3-4B-Base")
chk("--model beats the env var",
    t.resolve_model("google/gemma-3-4b-pt", env={"MODEL_TO_TRAIN": "Qwen/Qwen3-4B-Base"})
    == "google/gemma-3-4b-pt")

tdir = t.default_templates_dir()
chk("the default templates dir is absolute", tdir.startswith("/"))
chk("the default templates dir exists from a checkout", os.path.isdir(tdir))

# The grader's own mapping, from src/eval/tasks/gsm8k/evaluate.py's template_kwargs().
EXPECT = {
    "google/gemma-3-4b-pt":          ("gemma3.jinja", "<end_of_turn>"),
    "Qwen/Qwen3-4B-Base":            ("qwen3.jinja",  "<|im_end|>"),
    "Qwen/Qwen3-1.7B-Base":          ("qwen3.jinja",  "<|im_end|>"),
    "HuggingFaceTB/SmolLM3-3B-Base": ("smollm.jinja", "<|im_end|>"),
}
for model in MODELS:
    chk(f"{model}: is in this test's expectations", model in EXPECT)
    if model not in EXPECT:
        continue
    want_tpl, want_stop = EXPECT[model]
    ct = None
    try:
        ct = t.select_chat_template(model, tdir)
    except SystemExit as exc:
        print(f"    (select_chat_template refused: {str(exc).splitlines()[0]})")
    chk(f"{model}: template is {want_tpl}", ct is not None and os.path.basename(ct) == want_tpl)
    chk(f"{model}: template path is absolute", ct is not None and ct.startswith("/"))
    # Derived by rendering the template file the grader will be handed, not by a lookup
    # table -- so a template that changes its turn marker changes this answer with it.
    got_stop = t.derive_stop_token(ct) if ct else None
    chk(f"{model}: stop token derived as {want_stop}", got_stop == want_stop)
    # And end to end through the CLI, with the env var doing the work the harness does.
    ns = t.parse_args([])
    os.environ["MODEL_TO_TRAIN"] = model
    try:
        ns = t.resolve_run_identity(ns)
    finally:
        os.environ.pop("MODEL_TO_TRAIN", None)
    chk(f"{model}: end to end from $MODEL_TO_TRAIN",
        ns.model == model and os.path.basename(ns.chat_template) == want_tpl
        and ns.stop_token == want_stop)

# Not one of the four: refuse rather than pick a template at random.
odd = False
try:
    t.select_chat_template("mistralai/Mistral-7B-v0.3", tdir)
except SystemExit:
    odd = True
chk("an unmapped model family stops rather than guessing", odd)

# The two deliberate overrides still win over everything derived.
os.environ["MODEL_TO_TRAIN"] = "Qwen/Qwen3-4B-Base"
try:
    ns = t.resolve_run_identity(t.parse_args(["--stop-token", ""]))
    chk("--stop-token '' survives resolution (tokenizer's own eos)", ns.stop_token == "")
    ns = t.resolve_run_identity(t.parse_args(["--chat-template", os.path.join(tdir, "gemma3.jinja")]))
    chk("an explicit --chat-template is not overridden",
        os.path.basename(ns.chat_template) == "gemma3.jinja")
    chk("and the stop token follows the template that was passed",
        ns.stop_token == "<end_of_turn>")
finally:
    os.environ.pop("MODEL_TO_TRAIN", None)

sys.exit(fail)
PYEOF

python3 "$WORK/probe.py" "$REF" "$REPO_ROOT/src/commit_utils/commit.sh" > "$WORK/probe.out" 2>"$WORK/probe.err"
rc=$?
cat "$WORK/probe.out"
if [ "$rc" != 0 ] && ! grep -q '  FAIL ' "$WORK/probe.out"; then
    # A nonzero exit with no FAIL line means the probe died before it could report --
    # an import error or a syntax error, which is a worse result than any failed check.
    echo "  FAIL the python probe crashed (exit $rc)"
    sed 's/^/    /' "$WORK/probe.err"
fi
[ "$rc" = 0 ] || fail=1

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
