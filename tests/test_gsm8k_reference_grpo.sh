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
echo "[9] boolean flags accept the HuggingFace spelling as well as the argparse one"
# argparse.BooleanOptionalAction takes `--bf16` and `--no-bf16` and nothing else.
# HuggingFace TrainingArguments takes `--bf16 True`, which is the form an agent
# writing a TRL command has actually read, and on the live 2026-09-03 campaign
# cell 295f00fb spent two probe launches on
#     train_grpo.py: error: unrecognized arguments: false
# before logging "Fix flag and relaunch probes". Both spellings now work. The
# last two cases matter as much as the rest: a boolean action that accepts
# anything is not more permissive, it is silently wrong, and one that quietly
# drops the value would pass every positive case here.
cat > "$WORK/boolflag.py" <<'PYBOOL'
import importlib.util, sys, io, contextlib
spec = importlib.util.spec_from_file_location("tg", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bad = 0
def chk(argv, want, label):
    global bad
    got = m.parse_args(argv).bf16
    ok = got == want
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL'} {label}: {' '.join(argv) or '(default)'} -> {got}")
chk([], True, "default is unchanged")
chk(["--bf16"], True, "bare flag")
chk(["--no-bf16"], False, "bare negated")
chk(["--bf16", "false"], False, "the live failure")
chk(["--bf16", "True"], True, "HF capitalised")
chk(["--bf16", "0"], False, "numeric")
chk(["--no-bf16", "false"], True, "double negative")
for flag, want in [("--use-vllm", False), ("--gradient-checkpointing", False),
                   ("--log-completions", False), ("--save-only-model", False),
                   ("--sanitise-generation-config", False),
                   ("--mask-truncated-completions", True)]:
    dest = flag[2:].replace("-", "_")
    got = getattr(m.parse_args([flag, str(want).lower()]), dest)
    bad += got != want
    print(f"  {'PASS' if got == want else 'FAIL'} {flag} takes a value too -> {got}")
err = io.StringIO()
try:
    with contextlib.redirect_stderr(err):
        m.parse_args(["--bf16", "banana"])
    print("  FAIL a non-boolean value was accepted"); bad += 1
except SystemExit:
    print("  PASS a non-boolean value is still rejected")
sys.exit(1 if bad else 0)
PYBOOL
# -B, because this probe imports train_grpo.py from the reference directory and
# checks [2]/[3] above assert that no __pycache__ is shipped beside it -- without
# it the suite fails on its own side effect.
python3 -B "$WORK/boolflag.py" "$REF/train_grpo.py" || fail=1

echo "[10] the greedy probe arms, halts, and never takes the run down with it"
# The failure this guards is invisible in every number the trainer prints. 91039_g7
# had completions/clipped_ratio max 0.086, reward 0.231 -> 0.562 and "no kill
# criterion fired", and the graded read of that same checkpoint decoded 13,425
# characters a row into the grader's 4000-token cap and scored 19.41%. The reward is
# computed on sampled rollouts; the grade is a greedy decode. So the probe generates
# rather than reading a metric, and the cases worth pinning are the un-armed one
# (nothing happens), the halt, and every way the probe itself can go wrong -- a probe
# that can crash a ten-hour run is worse than no probe at all.
cat > "$WORK/monitor.py" <<'PYMON'
import importlib.util, sys, types, json, os, tempfile
import torch
spec = importlib.util.spec_from_file_location("train_grpo", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["train_grpo"] = m
spec.loader.exec_module(m)

bad = 0
def chk(label, cond):
    global bad
    print(("  PASS " if cond else "  FAIL ") + label)
    if not cond:
        bad = 1

Monitor = m._make_termination_monitor_class()
STOP = 151645
PROMPT_LEN = 3

class FakeTok:
    pad_token_id = 0
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "P"
    def __call__(self, texts, return_tensors=None, padding=None, padding_side=None):
        return {"input_ids": torch.ones(len(texts), PROMPT_LEN, dtype=torch.long)}

class FakeModel:
    """Emits `stop_n` sequences that reach STOP and the rest that never do."""
    def __init__(self, stop_n, total, raises=False):
        self.stop_n, self.total, self.raises = stop_n, total, raises
        self.training = True
        self.calls = 0
        self.eval_calls = 0
    def eval(self): self.training = False; self.eval_calls += 1
    def train(self): self.training = True
    def parameters(self):
        return iter([torch.zeros(1)])
    def generate(self, **kw):
        self.calls += 1
        if self.raises:
            raise RuntimeError("CUDA out of memory (simulated)")
        new = kw["max_new_tokens"]
        rows = []
        for i in range(self.total):
            body = [7] * new
            if i < self.stop_n:
                body[4] = STOP
            rows.append([1] * PROMPT_LEN + body)
        return torch.tensor(rows, dtype=torch.long)

def make(stop_n=16, total=16, floor=0.9, steps=5, halt=True, raises=False, prompts=16):
    out = tempfile.mkdtemp()
    mon = Monitor(floor, steps, prompts, 32, halt=halt, tokenizer=FakeTok(),
                  prompts=[[{"role": "user", "content": "q"}]] * prompts,
                  stop_token_id=STOP)
    model = FakeModel(stop_n, total, raises=raises)
    args = types.SimpleNamespace(output_dir=out, max_completion_length=512)
    return mon, model, args, out

def step(mon, model, args, n):
    st = types.SimpleNamespace(global_step=n)
    ctl = types.SimpleNamespace(should_save=False, should_training_stop=False)
    mon.on_step_end(args, st, ctl, model=model)
    return ctl

# Cadence: the probe is not free, so it must only run on its multiples.
mon, model, args, out = make(steps=5)
for n in (1, 2, 3, 4):
    step(mon, model, args, n)
chk("no probe off-cadence", model.calls == 0)
ctl = step(mon, model, args, 5)
chk("probes on the multiple", model.calls == 1)
chk("a passing probe arms", mon.armed is True)
chk("a passing probe checkpoints", ctl.should_save is True)
chk("...and records where good was", mon.last_good_step == 5)
chk("the model is put back in training mode", model.training is True)
chk("...having actually been switched out of it", model.eval_calls == 1)

# The 91039_g7 shape: healthy, then greedy decode stops terminating.
mon.armed = True
model.stop_n = 0
ctl = step(mon, model, args, 10)
chk("collapse is detected", mon.collapsed is True and mon.collapsed_step == 10)
chk("training halts", ctl.should_training_stop is True)
chk("and saves on the way out", ctl.should_save is True)
chk("the last good step is still readable", mon.last_good_step == 5)
chk("a collapse marker is written",
    os.path.exists(os.path.join(out, "grpo_termination_collapse.json")))
ctl = step(mon, model, args, 15)
chk("collapse does not re-fire", mon.collapsed_step == 10)

# Never healthy: a base model that has not learned to stop is not a model being
# destroyed, and halting there would kill every run that was going to work.
mon, model, args, out = make(stop_n=0, steps=5)
ctl = step(mon, model, args, 5)
chk("an un-armed failure arms nothing", mon.armed is False)
chk("an un-armed failure does not halt", ctl.should_training_stop is False)
chk("...and does not checkpoint", ctl.should_save is False)
chk("...and is not a collapse", mon.collapsed is False)

# Partial credit: 12 of 16 is 0.75, under the 0.9 floor.
mon, model, args, out = make(stop_n=12, total=16, steps=5)
step(mon, model, args, 5)
chk("0.75 does not clear a 0.9 floor", mon.armed is False)
chk("the fraction is recorded as measured", abs(mon.probes[0][1] - 0.75) < 1e-6)

# --no-termination-halt: report, do not act.
mon, model, args, out = make(steps=5, halt=False)
step(mon, model, args, 5)
model.stop_n = 0
ctl = step(mon, model, args, 10)
chk("halt=False still detects", mon.collapsed is True)
chk("halt=False does not stop training", ctl.should_training_stop is False)

# A probe that throws must never reach the training loop, and must give up rather
# than throw once a step for the rest of the run.
mon, model, args, out = make(steps=5, raises=True)
ctl = step(mon, model, args, 5)
chk("a raising probe does not propagate", mon.failures == 1 and mon.disabled is False)
ctl = step(mon, model, args, 10)
chk("two failures disable the probe", mon.disabled is True)
before = model.calls
step(mon, model, args, 15)
chk("a disabled probe stops generating", model.calls == before)
chk("a disabled probe does not halt training", ctl.should_training_stop is False)

# probe_steps=0 is the off switch.
mon, model, args, out = make(steps=0)
step(mon, model, args, 5)
chk("--termination-probe-steps 0 disables", model.calls == 0)

# The sampled statistic: watched, reported once, never acted on. This is the
# 91038_g7 / 91036_g6 defect, which both cells diagnosed by hand at ~0.40.
mon, model, args, out = make(steps=5)
st = types.SimpleNamespace(global_step=1)
ctl = types.SimpleNamespace(should_save=False, should_training_stop=False)
mon.on_log(args, st, ctl, logs={"completions/clipped_ratio": 0.40})
mon.on_log(args, st, ctl, logs={"completions/clipped_ratio": 0.42})
chk("truncation is recorded every step", len(mon.clipped) == 2)
chk("but warned about once", mon.truncation_warned is True)
chk("truncation never halts", ctl.should_training_stop is False)
mon2, _, args2, _ = make(steps=5)
mon2.on_log(args2, st, ctl, logs={"completions/clipped_ratio": 0.05})
chk("a healthy truncation rate is not warned about", mon2.truncation_warned is False)
mon2.on_log(args2, st, ctl, logs={"loss": 0.3})
chk("a log without the key is inert", len(mon2.clipped) == 1 and mon2.missing_warned)
chk("the key TRL 0.27.2 emits is the one watched",
    "completions/clipped_ratio" in Monitor.CLIPPED_KEYS)

# The trace is written on every run, collapse or not.
mon, model, args, out = make(steps=5)
step(mon, model, args, 5)
mon.on_train_end(args, types.SimpleNamespace(global_step=5),
                 types.SimpleNamespace())
p = os.path.join(out, "grpo_termination_trace.json")
chk("a trace is written on a clean run", os.path.exists(p))
d = json.load(open(p))
chk("the trace says it was clean", d["collapsed"] is False and d["armed"] is True)
chk("the trace carries the greedy series", len(d["greedy_probes"]) == 1)
chk("the trace names the last good step", d["last_good_step"] == 5)

sys.exit(1 if bad else 0)
PYMON
python3 -B "$WORK/monitor.py" "$REF/train_grpo.py" || fail=1

echo "[11] the greedy probe renders at the grader's few-shot depth, not at --fewshot"
# [10] pins that the probe fires. This pins WHAT it decodes, which is the half that
# decides whether a pass means anything. 91036_g6 trained at --fewshot 2, and under that
# rendering its completions stopped at 82 tokens -- healthy by every reading the trainer
# had. The graded read of the same weights renders 10-shot, ran every row into the
# 4000-token cap, and came back 69.52. A probe that inherits --fewshot reports that model
# green, so the probe must render the way the grader renders even when training does not.
# No GPU and no network: the dataset is stubbed, but build_fewshot_prefix is the real one,
# so a change to the prefix layout still moves these numbers.
cat > "$WORK/rendering.py" <<'PYREN'
import importlib.util, sys, types, random
spec = importlib.util.spec_from_file_location("train_grpo", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules["train_grpo"] = m
spec.loader.exec_module(m)

bad = 0
def chk(label, cond):
    global bad
    print(("  PASS " if cond else "  FAIL ") + label)
    if not cond:
        bad = 1

class FakeDS:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]
    def shuffle(self, seed=0):
        r = list(self.rows); random.Random(seed).shuffle(r); return FakeDS(r)
    def select(self, rng): return FakeDS([self.rows[i] for i in rng])

ROWS = [{"question": f"Q{i}?", "answer": f"reasoning {i}\n#### {i}"} for i in range(64)]
fake = types.ModuleType("datasets")
fake.load_dataset = lambda *a, **k: FakeDS(ROWS)
sys.modules["datasets"] = fake

def args(fewshot, probe_fewshot):
    return types.SimpleNamespace(
        fewshot=fewshot, termination_probe_fewshot=probe_fewshot,
        dataset="openai/gsm8k", dataset_config="main", train_split="train",
        max_train_samples=0, fewshot_seed=m.DEFAULT_FEWSHOT_SEED,
    )

def shots_in(prompts):
    """How many worked examples the system message carries."""
    if not prompts or prompts[0][0]["role"] != "system":
        return 0
    return prompts[0][0]["content"].count("ANSWER:")

# The default: training shallow, probe at the grader's depth.
p_default, n_default = m.build_probe_prompts(args(fewshot=2, probe_fewshot=m.DEFAULT_FEWSHOT), 8)
chk("the default probe depth IS the grader's", n_default == m.DEFAULT_FEWSHOT)
chk("the probe renders 10 shots while training renders 2", shots_in(p_default) == m.DEFAULT_FEWSHOT)
chk("the probe returns the prompts it was asked for", len(p_default) == 8)

# The 91036_g6 shape: the two renderings must actually differ, or this fix is a no-op.
p_mirror, n_mirror = m.build_probe_prompts(args(fewshot=2, probe_fewshot=-1), 8)
chk("-1 mirrors --fewshot (the old behaviour is still reachable)", n_mirror == 2)
chk("mirroring renders 2 shots", shots_in(p_mirror) == 2)
chk("the grader rendering and the training rendering are NOT the same string",
    p_default[0][0]["content"] != p_mirror[0][0]["content"])
chk("the graded rendering is the longer one",
    len(p_default[0][0]["content"]) > len(p_mirror[0][0]["content"]))

# The 91039_g7 shape: a ladder rung at --fewshot 0 has no system message at all, and the
# probe must still ask the 10-shot question.
p_zero, n_zero = m.build_probe_prompts(args(fewshot=0, probe_fewshot=m.DEFAULT_FEWSHOT), 4)
chk("--fewshot 0 still probes at 10", n_zero == m.DEFAULT_FEWSHOT and shots_in(p_zero) == 10)
p_zm, _ = m.build_probe_prompts(args(fewshot=0, probe_fewshot=-1), 4)
chk("mirroring --fewshot 0 carries no system message", shots_in(p_zm) == 0)

# The user turn is the graded one, not a bare question.
user = [t for t in p_default[0] if t["role"] == "user"][0]["content"]
chk("the user turn goes through MATH_PROMPT_TEMPLATE",
    "Q0?" in user and user != "Q0?")
chk("the rows are training rows", all(
    any(f"Q{i}?" in t["content"] for t in p[1:]) for i, p in enumerate(p_default)))

# Asking for more prompts than the split holds must clamp, not raise.
p_big, _ = m.build_probe_prompts(args(fewshot=10, probe_fewshot=m.DEFAULT_FEWSHOT), 10_000)
chk("a probe count past the end of the split clamps", len(p_big) == len(ROWS))

# The DEFAULT is the whole fix. Every check above passes an explicit depth, so every one
# of them still passes with the default reverted to -1 -- mutation-tested, and it caught
# exactly that. What an agent actually runs is `train_grpo.py` with no probe flag at all,
# so the value argparse hands it is the thing to pin.
d = m.parse_args(["--output-dir", "/tmp/x", "--model", "Qwen/Qwen3-1.7B-Base"])
chk("with NO probe flag, argparse hands back the grader's depth",
    d.termination_probe_fewshot == m.DEFAULT_FEWSHOT)
chk("and that is not the mirroring sentinel", d.termination_probe_fewshot != -1)
chk("a shallow --fewshot does not drag the probe down with it",
    m.parse_args(["--output-dir", "/tmp/x", "--model", "Q/M", "--fewshot", "2"]
                 ).termination_probe_fewshot == m.DEFAULT_FEWSHOT)
p_def, n_def = m.build_probe_prompts(
    m.parse_args(["--output-dir", "/tmp/x", "--model", "Q/M", "--fewshot", "2"]), 4)
chk("and end to end, the unflagged default renders 10 shots",
    n_def == m.DEFAULT_FEWSHOT and shots_in(p_def) == m.DEFAULT_FEWSHOT)

# The trace has to say which question was asked, or a green trace is unreadable.
Monitor = m._make_termination_monitor_class()
mon = Monitor(0.9, 25, 16, 1024, probe_fewshot=m.DEFAULT_FEWSHOT, train_fewshot=2)
chk("the monitor records the depth it rendered at", mon.probe_fewshot == m.DEFAULT_FEWSHOT)
chk("the monitor records the depth training used", mon.train_fewshot == 2)

sys.exit(1 if bad else 0)
PYREN
python3 -B "$WORK/rendering.py" "$REF/train_grpo.py" || fail=1

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
