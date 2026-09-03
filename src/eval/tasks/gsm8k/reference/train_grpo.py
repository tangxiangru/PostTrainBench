#!/usr/bin/env python3
"""A correct, deliberately unremarkable GRPO run on gsm8k -- the baseline to beat.

This is not the entrypoint. Nothing in ``src/run_task.sh`` calls it and no score depends
on it. It exists because five independent cells of one twelve-cell arm spent their first
hours of RL rediscovering the same three defects, and because the arm's own numbers say
that RL is the only variable that moved the score at all:

    Six of the twelve cells shipped a lineage that had received a real verifier-reward
    RL run (>=500 GRPO optimizer steps and >=50k rollouts). Those six averaged **81.70**
    against **73.93** for the six that did not -- **+7.77 pt**, Welch p=0.0042, about
    half of the arm's whole 15.99 pt spread. The within-cell measurement agrees and is
    not subject to the post-hoc grouping critique: GRPO's own pre/post delta was measured
    six times on the same base and the same grader (+4.93, +5.2, +5.7, +6.9, ~+7.8,
    +10.0), all six positive, sign test p=0.0156. Four of the six reward curves were
    still climbing when the clock stopped.

So the expensive failure in this arm is not "trained badly", it is "never got an RL run
started". Everything below is aimed at that and nothing else.

**The three defects this file pre-empts.**

1. ``ValueError: GenerationConfig is invalid: temperature 0.0 with do_sample False`` at
   the FIRST checkpoint save. 89809_g3 lost ~9 min and 50 steps, 89727_g1 lost 14.85
   GPU-min and 100 steps and produced no weights at all, 89810_g3 lost 25 min and 20
   steps, 89810_g7's ``soup.py`` died on it and had to be rewritten, and 89727_g7's
   ``grpo_v1`` died at step 40 of 1500. It costs ~75-90 min of direct wall clock across
   the arm, but the real damage is that it lands at the moment the first checkpoint would
   have existed, and a cell that hits it there often abandons RL entirely.
   :func:`sanitise_generation_config` is the fix; :class:`GenerationConfigGuard` is why it
   is still applied at step 1650 and not just at step 0.

2. A reward that grades differently from the grader. The frozen scorer is
   ``inspect_evals/gsm8k``'s ``match(numeric=True)``, which reads the LAST number in the
   completion. :func:`grade_completion` below is a transcription of that scorer, not a
   reimplementation of it -- see the comment on that function for what was copied and the
   two places this copy is deliberately *stricter*.

3. A model that cannot stop. Qwen3-1.7B-Base's tokenizer ends an assistant turn with
   ``<|im_end|>`` (151645) but its ``eos_token`` is ``<|endoftext|>`` (151643), so
   rollouts and graded generations both run to the token cap. ``ptb_ops/make_greedy_shadow.py``
   records the consequence: the model "states a correct ``ANSWER: n``, misses the stop
   token and runs on until the 4000-token cap", and the last-number scorer then grades
   the tail. ``--stop-token`` fixes both ends of that, and it is *derived* rather than
   given: :func:`derive_stop_token` renders one assistant turn through the very chat
   template the grader will serve this model with and reads the closing token off the
   result, so it is ``<|im_end|>`` for the Qwen bases and SmolLM3 and ``<end_of_turn>``
   for gemma-3-4b-pt without anyone having to remember which.

**Why it is deliberately unremarkable, and why that is the point.** The top of this arm
came from method diversity, not from this recipe: 89809_g5 shipped a uniform fp32 soup of
two GRPO checkpoints and won, 89727_g3 shipped Dr. GRPO on a 201k external corpus and came
second, 89809_g3 shipped plain GRPO at G=16 and came third. A canned script that everyone
adopts pulls the ceiling down toward its own number. So every default here is the boring
one -- TRL's own, or the arm's median -- and ``reference/README.md`` says in its first
paragraph that the job is to beat it. Run it, confirm the path works, then do something
else.

Usage (single H100, inside the task container)::

    python reference/train_grpo.py --output-dir runs/grpo_ref
    python reference/train_grpo.py --help     # every knob, with its default

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
from pathlib import Path

logger = logging.getLogger("train_grpo")


# --------------------------------------------------------------------------------------
# Constants. Each one is a measurement, not a preference.
# --------------------------------------------------------------------------------------

#: The prompt ``inspect_evals/gsm8k`` wraps every question in, copied verbatim from
#: ``src/inspect_evals/gsm8k/gsm8k.py`` at the commit the container pins
#: (``containers/opus_5.def``: ``git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8``).
#: Training on a differently-worded prompt trains the model to answer a question the
#: grader will never ask; the "ANSWER: $ANSWER" convention in particular is the only
#: reason the last-number scorer lands on the right number.
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

#: The grader's own defaults. ``inspect_evals``' ``gsm8k`` task signature is
#: ``gsm8k(fewshot=10, fewshot_seed=42, shuffle_fewshot=True)`` and
#: ``src/eval/tasks/gsm8k/evaluate.py`` passes no ``-T`` overrides, so the graded prompt is
#: always 10-shot with seed 42. Reproduced here exactly rather than approximately:
#: inspect builds the prefix with ``hf_dataset(..., shuffle=True, seed=42, limit=10)``,
#: which is ``datasets`` ``.shuffle(seed=42).select(range(10))`` -- the same two calls
#: :func:`build_fewshot_prefix` makes. Measured: the resulting system message is 5872
#: characters / 2048 Qwen3 tokens.
DEFAULT_FEWSHOT = 10
DEFAULT_FEWSHOT_SEED = 42

#: Measured on all 7473 train and 1319 test rows, rendering the 10-shot system message
#: plus :data:`MATH_PROMPT_TEMPLATE` through ``src/eval/templates/qwen3.jinja``: 2170-2370
#: tokens on train, 2181-2346 on test. 2560 clears the observed maximum by 190 tokens.
#: This number is load-bearing in a quiet way -- TRL left-truncates any prompt longer than
#: ``max_prompt_length``, which would silently eat the front of the few-shot prefix and
#: leave a run training on a prompt the grader never shows.
DEFAULT_MAX_PROMPT_LENGTH = 2560

#: Measured over the 7473 gold gsm8k solutions rendered as ``{reasoning}\n\nANSWER: {t}``:
#: p50 113 tokens, p99 290, max 455. 512 covers every gold solution in the training set.
#: Note this is also the Dr. GRPO loss normaliser (``loss_type="dr_grpo"`` divides by
#: ``max_completion_length``), so changing it changes the gradient scale, not just the cap.
DEFAULT_MAX_COMPLETION_LENGTH = 512

#: 128 completions per optimizer step. This is the arm's own median, not a guess: five of
#: the six cells that cleared the RL threshold ran at exactly 128 rollouts/step
#: (89727_g3 600 steps/76.8k rollouts, 89810_g1 815/104.3k, 89809_g3 1050/134.4k,
#: 89727_g1 900/115.2k, 89727_g5 2782/356k) and the winner 89809_g5 ran at 64
#: (1650/105.6k). With ``num_generations=8`` that is 16 distinct prompts per step.
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 16
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_NUM_GENERATIONS = 8

#: Two epochs, because one is 33 optimizer steps short of the bar this file argues for.
#: With the defaults above TRL consumes 16 distinct prompts per optimizer step
#: (``generation_batch_size = 16 * 1 * 8 = 128`` completions, ``/ num_generations = 8``),
#: so gsm8k's 7473 train rows are 467 steps and 59.8k rollouts per epoch -- over the 50k
#: rollout threshold, under the 500 step one. Two epochs is 934 steps and 119.6k rollouts,
#: inside the range every cell in the arm's high group actually ran (600-2782 steps,
#: 76.8k-356k rollouts). At the observed 8.6-54.88 s/step that is 2.2 to 14.2 hours, so on
#: a slow node bound it with ``--max-steps`` rather than letting the harness kill a
#: half-written save.
DEFAULT_NUM_TRAIN_EPOCHS = 2.0

#: Observed GRPO step rate across the arm: 8.6 s/step (89809_g3) to 54.88 s/step
#: (89810_g3). Saving every 50 steps therefore puts checkpoints 7 to 46 minutes apart, so
#: even the slowest cell has a handful of ship candidates within its first hour of RL and
#: the fastest has dozens. That mattered: the arm's winner shipped a *soup of two
#: intermediate checkpoints*, which is only possible if intermediate checkpoints exist.
DEFAULT_SAVE_STEPS = 50

#: Ten kept checkpoints at ~3.4 GB each (1.7B params, bf16) is ~34 GB. Keeping the
#: optimizer state as well would be roughly four times that, and these are ship candidates
#: rather than resume points -- hence ``save_only_model`` below, which is also what makes
#: ``--resume-from-checkpoint`` unavailable. Say so out loud rather than let someone
#: discover it after a preemption.
DEFAULT_SAVE_TOTAL_LIMIT = 10

#: Rollout truncation rate above which TerminationMonitor says something. Not a knob and
#: not acted on: it is the point at which a rate stops being background and starts being
#: the thing to fix. 91038_g7 and 91036_g6, two of the three cells in the arm's ~69 tail,
#: both ran at ~0.40 here and both worked it out by hand -- 91038_g7 filed it as "a defect
#: in the training configuration ... 40% of rollouts are truncated and scored zero",
#: having reasoned that a truncated completion rarely ends on its final answer, so those
#: rollouts contribute a zero that says nothing about the policy.
TRUNCATION_WARN = 0.25

#: What goes into the shipped ``generation_config.json``. vLLM 0.11.0 (the version
#: ``containers/opus_5.def`` pins, and the one the grader runs behind
#: ``inspect_ai``) switches to ``SamplingType.GREEDY`` when ``temperature <
#: _SAMPLING_EPS``, and ``_SAMPLING_EPS = 1e-5`` (``vllm/sampling_params.py:20,524``).
#: ``src/eval/tasks/gsm8k/evaluate.py`` passes no temperature at all, so whatever is in
#: this file *is* the graded decode. ``ptb_ops/make_greedy_shadow.py`` writes the same 0.0
#: for the same reason, and records that on the 23-cell one-hour board the only two cells
#: above 0.5 were the two that happened to write it.
#:
#: 0.0 and not 1e-6: 0.0 is what the repo already writes, and both are greedy under vLLM.
#: The cost of 0.0 is that ``transformers``' own sampler rejects it --
#: ``TemperatureLogitsWarper(0.0)`` raises "has to be a strictly positive float" -- so a
#: checkpoint written by this script cannot be sampled by ``model.generate(do_sample=True)``
#: without overriding the temperature. Pass ``--shipped-temperature 1e-6`` if you need
#: both; it is below vLLM's epsilon and above the warper's floor.
DEFAULT_SHIPPED_TEMPERATURE = 0.0


# --------------------------------------------------------------------------------------
# Which model, which template, which stop token -- none of them a literal.
# --------------------------------------------------------------------------------------
#
# gsm8k is swept over FOUR base models. `src/commit_utils/commit.sh` runs it against
# google/gemma-3-4b-pt, Qwen/Qwen3-4B-Base, Qwen/Qwen3-1.7B-Base and
# HuggingFaceTB/SmolLM3-3B-Base, and `reference/` is copied into the sandbox of every one
# of those cells because run_task.sh's copy is per-task, not per-model. An earlier version
# of this file defaulted `--model` to Qwen/Qwen3-1.7B-Base and `--chat-template` to
# templates/qwen3.jinja, and `smoke.sh` passed neither. On three of the four models that is
# a script that downloads the wrong base model and trains it -- which rule 8 of the agent's
# own prompt forbids ("You are not allowed to fine-tune any model other than {model}") --
# and it fails in the *quiet* direction: it runs, it converges, and the run is void.
#
# So the model is mandatory and comes from the harness. `src/run_task.sh` passes
# `--env MODEL_TO_TRAIN=${MODEL_TO_TRAIN}` into the agent sandbox (the same string
# get_prompt.py renders into the prompt's `{model}`), which is the one copy of this fact
# that is inside the container and not prose. Absent it, this file stops rather than
# guesses: a SystemExit costs a minute, and the wrong base model costs the cell.

#: The environment variable ``src/run_task.sh`` sets to the base model this cell must train.
MODEL_ENV_VAR = "MODEL_TO_TRAIN"

#: Chat template per model family, transcribed from ``src/eval/tasks/gsm8k/evaluate.py``'s
#: ``model_type()`` / ``template_kwargs()`` -- the grader's own mapping, in the grader's own
#: order (``qwen`` before ``llama`` before ``gemma`` before ``smollm``; the order matters
#: only if a model id ever contains two of them, and it is copied rather than tidied for
#: exactly that reason). Whatever the grader serves the final checkpoint with is what the
#: rollouts have to be generated under, or training and scoring see different prompts.
MODEL_TYPE_MARKERS = ("qwen", "llama", "gemma", "smollm")
CHAT_TEMPLATE_BY_MODEL_TYPE = {
    "qwen": "qwen3.jinja",
    "llama": "llama3.jinja",
    "gemma": "gemma3.jinja",
    "smollm": "smollm.jinja",
}

#: Rendered into a one-turn conversation to find the token that closes an assistant turn.
#: Any string that cannot occur in a template's own text will do.
_STOP_PROBE_MARKER = "PTB_REFERENCE_ASSISTANT_TURN_MARKER"

#: What counts as a special token in the rendered tail: either the ``<|...|>`` form (qwen,
#: smollm, llama) or a bare ``<word>`` (gemma's ``<end_of_turn>``). Applied only to the text
#: a template emits *after* the assistant content, which is template text and never model
#: output, so there is nothing here for a stray angle bracket in a completion to confuse.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|<>\s]+\|>|<[A-Za-z0-9_]+>")


def default_templates_dir() -> str:
    """``templates/`` beside this script's own parent, resolved absolutely.

    Not the relative ``templates/`` this file used to carry. run_task.sh copies
    ``src/eval/templates`` to ``<task>/templates`` and ``reference/`` to
    ``<task>/reference``, so the relation between the two is fixed no matter where the
    process was started from -- and *where it was started from* is not fixed. The control
    arm's agent has cwd ``<task>``; the claude_autor arm's operator runs its stages with cwd
    ``<task>/.autor/<stamp>/``, and a relative ``templates/qwen3.jinja`` opened from there
    is a FileNotFoundError two minutes into a ten-hour cell.

    Two candidates, in order: ``../templates`` (the sandbox layout, which is the only one
    that matters at run time) and ``../../../../templates`` (this repo's ``src/eval/templates``,
    where the same four files live before run_task.sh copies them). The second exists so
    that the derivation can be exercised for every swept base model from a checkout, with
    no container and no GPU -- a mapping that can only be tested inside a ten-hour cell is
    a mapping that gets tested by the cell.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "templates",           # <task>/templates, in the sandbox
                      here.parents[2] / "templates"):      # src/eval/templates, in the repo
        if candidate.is_dir():
            return str(candidate)
    return str(here.parent / "templates")


def resolve_model(explicit: str | None, env=None) -> str:
    """The base model to train: the flag, else :data:`MODEL_ENV_VAR`, else stop.

    No default. See the block comment above :data:`MODEL_ENV_VAR` for why a default here is
    a script that trains the wrong model on three cells out of four and never says so.
    """
    if explicit:
        return explicit
    env = os.environ if env is None else env
    from_env = (env.get(MODEL_ENV_VAR) or "").strip()
    if from_env:
        return from_env
    raise SystemExit(
        f"no base model: --model was not given and ${MODEL_ENV_VAR} is unset or empty.\n"
        "  This benchmark is swept over four different base models and this script will not\n"
        "  guess which one your cell is about -- training the wrong one is forbidden by rule\n"
        "  8 of your prompt and produces a run that looks fine and does not count.\n"
        f"  Inside the task sandbox ${MODEL_ENV_VAR} is set by the harness. If you are\n"
        "  running somewhere else, pass the model named in your prompt explicitly:\n"
        "      python train_grpo.py --model <org>/<model> ..."
    )


def model_type(model_id: str) -> str:
    """Which template family a model id belongs to, by the grader's own rule.

    Transcribed from ``src/eval/tasks/gsm8k/evaluate.py``'s ``model_type()``, first-match
    over :data:`MODEL_TYPE_MARKERS` on the lowercased id. The grader's version has a second
    branch that reads ``config.json['architectures']`` when the id matches nothing; that
    branch exists because the grader is handed a local ``final_model/`` directory whose name
    says nothing. Here the argument is a hub id, so the fall-through is an error rather than
    a second guess.
    """
    lowered = model_id.lower()
    for marker in MODEL_TYPE_MARKERS:
        if marker in lowered:
            return marker
    raise SystemExit(
        f"--model {model_id!r} matches none of {list(MODEL_TYPE_MARKERS)}, so this script\n"
        "  cannot tell which chat template the grader will serve it with. Pass\n"
        "  --chat-template <path> and --stop-token <token> explicitly, or add the family to\n"
        "  MODEL_TYPE_MARKERS here AND to model_type() in src/eval/tasks/gsm8k/evaluate.py --\n"
        "  the two have to agree or you train under a different prompt than you are scored on."
    )


def select_chat_template(model_id: str, templates_dir: str) -> str:
    """The template file the grader will serve this model with."""
    name = CHAT_TEMPLATE_BY_MODEL_TYPE[model_type(model_id)]
    path = Path(templates_dir) / name
    if not path.is_file():
        raise SystemExit(
            f"--model {model_id!r} needs chat template {path}, which does not exist.\n"
            f"  {templates_dir} holds: "
            f"{sorted(p.name for p in Path(templates_dir).glob('*.jinja')) if Path(templates_dir).is_dir() else '(no such directory)'}"
        )
    return str(path)


def render_assistant_turn(template_text: str, marker: str = _STOP_PROBE_MARKER) -> str:
    """One user turn plus one assistant turn, rendered through a chat template.

    Plain jinja2 with the three extras ``transformers`` injects into its own chat-template
    environment (``raise_exception``, ``tojson``, ``strftime_now``) and with the
    ``{% generation %}`` markers stripped, since those are a transformers extension that
    only marks spans for assistant-token masking and emits nothing. Rendering here rather
    than through ``tokenizer.apply_chat_template`` so that the derivation depends on the
    template *file the grader is handed* and on nothing that has to be downloaded -- which
    is also what makes it checkable on a login node against all four shipped templates.
    """
    import json as _json

    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def _raise(message):
        raise RuntimeError(message)

    source = template_text.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda value, **kw: _json.dumps(value)
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: "01 January 2026"
    return env.from_string(source).render(
        messages=[{"role": "user", "content": "q"},
                  {"role": "assistant", "content": marker}],
        bos_token="<bos>",
        eos_token="</s>",
        add_generation_prompt=False,
    )


def stop_token_from_rendered_turn(rendered: str, marker: str = _STOP_PROBE_MARKER) -> str | None:
    """The first special token a template emits after the assistant's own content.

    That token, and not the tokenizer's ``eos_token``, is what a chat-templated generation
    has to stop on. The two disagree on every base model in this sweep -- Qwen3-1.7B-Base
    ships ``eos_token = <|endoftext|>`` (151643) while ``qwen3.jinja`` closes the turn with
    ``<|im_end|>`` (151645) -- so out of the box nothing stops the rollouts during training
    and nothing stops the grader afterwards; the last-number scorer then grades whatever
    the model said after the answer, all the way to the 4000-token cap.
    """
    if marker not in rendered:
        return None
    match = _SPECIAL_TOKEN_RE.search(rendered.split(marker)[-1])
    return match.group(0) if match else None


def derive_stop_token(chat_template_path: str) -> str | None:
    """:func:`stop_token_from_rendered_turn` on a template file, or ``None`` if it will not
    render. ``None`` is a refusal, not a default -- :func:`main` turns it into a SystemExit
    naming ``--stop-token``, because silently training without a stop is defect 3 again."""
    try:
        text = Path(chat_template_path).read_text(encoding="utf-8")
        return stop_token_from_rendered_turn(render_assistant_turn(text))
    except Exception as exc:  # noqa: BLE001 - any render failure is the same answer
        logger.warning("could not derive a stop token from %s: %s: %s",
                       chat_template_path, type(exc).__name__, exc)
        return None


# --------------------------------------------------------------------------------------
# The grader, transcribed.
# --------------------------------------------------------------------------------------
#
# `src/eval/tasks/gsm8k/evaluate.py` does not extract anything itself -- it runs
# `inspect_evals/gsm8k`, whose scorer is `match(numeric=True)`. So "copy the grader"
# means copying `inspect_ai.scorer._common.match_str` at the revision the container
# installs, which `containers/opus_5.def` pins by cloning
# `rank-and-file/inspect_ai_vllm_stdout` (HEAD 64db0afd, 2026-01-12). The four helpers
# below are that function and the two `inspect_ai._util.text` helpers it calls, unrolled
# for `location="end"`, `ignore_case=True`, `numeric=True` -- the only configuration the
# gsm8k task ever uses.
#
# Transcribing rather than importing, even though `inspect_ai` is installed in the
# container: the reward has to be readable next to the thing it rewards, and a reward
# function that silently changes when someone bumps a dependency is exactly the kind of
# drift that produces a model tuned for a scorer nobody looked at.
#
# TWO DELIBERATE DEVIATIONS, both in the strict direction. The invariant is that anything
# this file rewards, the grader also marks correct -- never the reverse, because a reward
# that is more generous than the grader trains the model to pass a test it will not be
# given:
#
#   * `--reward-match strict` compares the extracted number with `==` where the grader
#     uses `.endswith`. The grader's version credits an extracted "25" against a target of
#     "5". Left off by default, because faithful is the safer default and the brief for
#     this file is to match the grader.
#   * `_normalize_number` returns the token unchanged where inspect falls back to a
#     unicode-aware parser (vulgar fractions, superscripts). That can lose a reward on a
#     completion whose final number is "½"; it cannot invent one.


def _strip_numeric_punctuation(s: str) -> str:
    """``inspect_ai._util.text.strip_numeric_punctuation``, verbatim."""
    # strip $, €, £, and , -- and *,_ the formatting characters LLMs sometimes add
    stripped = re.sub(r"[$,£,€,*,_]", "", s)
    # strip . if it's followed by a space, the end of the string, or a non-digit
    stripped = re.sub(r"\.(?=\s|$|\D)", "", stripped)
    return stripped


def _strip_punctuation(s: str) -> str:
    """``inspect_ai._util.text.strip_punctuation``, verbatim."""
    return s.strip(string.whitespace + string.punctuation)


def _normalize_number(number: str, precision: int = 5) -> str:
    """``inspect_ai.scorer._common.normalize_number``, minus the unicode fallback."""
    if number.replace(".", "").isnumeric():
        try:
            num = float(number)
        except ValueError:
            # inspect reaches for `unicode_number_to_float` here. Returning the token
            # unchanged instead can only cost a reward, never grant one.
            return number
        return format(num, f".{precision}g")
    return number


def _first_number_normalized(words: list[str]) -> str:
    """``inspect_ai.scorer._common.first_number_normalized``, verbatim.

    Note the fallback is ``words[0]`` and not the empty string: when the completion
    contains no number at all, the grader compares against its last whitespace-delimited
    word. Kept, because a "cleaner" fallback would score differently.
    """
    number = next(
        (word for word in words if word.replace(".", "").isnumeric()), words[0]
    )
    return _normalize_number(number)


def grade_completion(completion: str, target: str, strict: bool = False) -> tuple[str, bool]:
    """Return ``(extracted_answer, is_correct)`` the way the frozen grader would.

    ``inspect_ai.scorer._common.match_str`` unrolled for the gsm8k configuration. The
    branch on ``t.isnumeric()`` is the part everyone gets wrong by omission: 16 of the
    1319 gsm8k test targets and 82 of the 7473 train targets are *not* ``isnumeric()``
    -- comma-grouped ("2,125", "1,450,000") or negative ("-10", "-3"). On those rows the
    grader never extracts a number at all; it requires the whole completion to *end with*
    the literal target string, commas and all. A verifier that helpfully normalised
    "2,125" to 2125 would reward completions the grader marks wrong on 1.2% of rows, and
    would teach the model to drop exactly the thousands separators it needs to keep.
    """
    # strip ws
    v = completion.strip()
    t = target.strip()

    # baseline answer (will only change for numeric)
    answer = v

    # ignore_case=True
    v = v.casefold()
    t = t.casefold()

    if t.isnumeric():
        v = _strip_numeric_punctuation(v)
        t = _strip_numeric_punctuation(t)
        t = _normalize_number(t)
        # location="end": the LAST number in the completion, which is why a model that
        # answers correctly and then keeps generating is graded on its tail.
        words = re.split(r"\s+", v)
        words.reverse()
        v = _first_number_normalized(words)
        answer = v
        return answer, (v == t) if strict else v.endswith(t)

    # ignore_punctuation=True
    v = _strip_punctuation(v)
    t = _strip_punctuation(t)
    return answer, v.endswith(t)


def gsm8k_target(answer: str) -> str:
    """The gold answer, extracted the way ``inspect_evals``' ``record_to_sample`` does.

    ``record["answer"]`` is the worked solution, a ``####`` delimiter and the final
    number. inspect splits on ``####`` and pops the last field, so a solution that itself
    contained ``####`` would keep the trailing field -- copied rather than simplified to
    ``rsplit(..., 1)`` for that reason, even though no gsm8k row exercises it.
    """
    parts = answer.split("####")
    return parts.pop().strip()


def _completion_text(completion) -> str:
    """The assistant text out of whatever shape TRL hands the reward function.

    Conversational datasets (which this script builds, because the grader serves the model
    a chat-templated request) give ``[{"role": "assistant", "content": ...}]``; standard
    text datasets give a bare string. Handling both means the reward can be reused if
    someone swaps the dataset builder out, which is the most likely thing to be swapped.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content") or ""
        return str(last)
    return ""


def make_reward_fn(strict: bool = False):
    """Build the single verifier reward: 1.0 if the frozen grader would mark it correct.

    Binary and unshaped on purpose. A format bonus, a length penalty and a partial-credit
    term are each a reasonable idea and each one is a place where this reference would
    stop being a baseline and start being a recipe -- see ``reference/README.md``.
    """

    def correct_final_answer(completions, target, **kwargs):
        return [
            1.0 if grade_completion(_completion_text(c), t, strict=strict)[1] else 0.0
            for c, t in zip(completions, target)
        ]

    return correct_final_answer


# --------------------------------------------------------------------------------------
# The generation_config fix.
# --------------------------------------------------------------------------------------


def sanitise_generation_config(gen_cfg, temperature: float = DEFAULT_SHIPPED_TEMPERATURE):
    """Make ``gen_cfg`` state greedy decoding in the one form that survives both libraries.

    This is the fix for the failure that hit 89809_g3, 89727_g1, 89810_g3, 89810_g7 and
    89727_g7 -- five of twelve cells, all at their first RL checkpoint save, all with the
    same message::

        ValueError: GenerationConfig is invalid:
        - `temperature`: `do_sample` is not set to `True`. However, `temperature` is set
          to `0.0` -- this flag is only used in sample-based generation modes.

    The trap is that both halves of the pair are things a careful person does on purpose.
    ``temperature = 0.0`` is correct and necessary: ``src/eval/tasks/gsm8k/evaluate.py``
    passes no temperature, so vLLM falls back to this file and anything at or above
    ``_SAMPLING_EPS`` is *sampled*. ``do_sample = False`` is what Qwen3-1.7B-Base ships.
    Put them together, as anyone forcing greedy decode on this base model will, and
    ``save_pretrained`` refuses -- not at load, not at ``from_pretrained``, but at the
    first ``Trainer._save_checkpoint``, an hour into the run.

    Measured against transformers 5.15.0 (the container pins 4.57.3; the validator rule is
    the same), saving a ``GenerationConfig`` with:

        do_sample=False, temperature=0.0                 -> ValueError
        do_sample=False, temperature=1.0                 -> ok, but vLLM then SAMPLES
        do_sample=False, temperature unset               -> ok, but vLLM samples at 1.0
        do_sample=False, temperature=1.0, top_k=0        -> ValueError (now on top_k)
        do_sample=True,  temperature=0.0, top_p=1.0, top_k=0 -> ok, and vLLM is greedy

    So the resolution is the last row: keep the temperature the grader will read, and flip
    ``do_sample`` to the value that makes transformers agree to write it down.
    ``do_sample`` is a transformers field vLLM ignores entirely, so flipping it changes
    nothing about the graded decode.

    Duck-typed rather than annotated ``GenerationConfig`` so the unit test can drive it
    with a bare object and no transformers import.
    """
    if gen_cfg is None:
        return None
    gen_cfg.do_sample = True
    gen_cfg.temperature = float(temperature)
    # Written for the record rather than out of necessity: vLLM ignores top_p/top_k once
    # the sampler is greedy. top_k is 0 (transformers' "disabled") and not vLLM's -1,
    # because this object has to pass transformers' validator on the way to disk.
    gen_cfg.top_p = 1.0
    gen_cfg.top_k = 0
    return gen_cfg


def ensure_stop_tokens(gen_cfg, stop_token_id: int):
    """Add ``stop_token_id`` to ``gen_cfg.eos_token_id``, keeping whatever was there.

    Separate from :func:`sanitise_generation_config` because it fixes a different failure:
    that one is about the file being writable, this one is about the model being able to
    stop. Qwen3-1.7B-Base's ``generation_config.json`` says ``eos_token_id: 151643``
    (``<|endoftext|>``) while ``src/eval/templates/qwen3.jinja`` closes an assistant turn
    with ``<|im_end|>`` (151645), so a chat-templated generation has no reachable stop and
    runs to ``--max-tokens 4000``. The grader reads the last number in that, which is how
    a checkpoint that answers correctly scores like one that does not.

    Appended rather than replaced, and ``<|endoftext|>`` deliberately left in place: cells
    in this arm shipped both ``[151645, 151643]`` and ``151645`` alone, and dropping a stop
    the model already knows can only make generations longer.
    """
    if gen_cfg is None:
        return None
    existing = getattr(gen_cfg, "eos_token_id", None)
    if existing is None:
        ids = []
    elif isinstance(existing, int):
        ids = [existing]
    else:
        ids = list(existing)
    if stop_token_id not in ids:
        ids.insert(0, stop_token_id)
    gen_cfg.eos_token_id = ids
    return gen_cfg


def build_generation_kwargs(use_vllm: bool, stop_token_id) -> dict | None:
    """The ``GRPOConfig.generation_kwargs`` that make vLLM rollouts stop at the chat turn.

    Neither vLLM path puts a stop into the ``SamplingParams`` it builds, and the vLLM
    engine takes its default EOS from the *checkpoint on disk*: TRL constructs the colocate
    engine as ``LLM(model=model.name_or_path, ...)`` (``grpo_trainer.py:694-695``), so it
    reads the base model's ``<|endoftext|>`` and never sees the tokenizer object
    :func:`main` patches. Without this, every rollout runs the full
    ``--max-completion-length`` and the run pays for tokens nobody grades.

    Returned for BOTH vLLM modes rather than colocate alone. Each builds its own dict and
    merges ``args.generation_kwargs`` into it *last*, so one setting covers them: colocate
    at ``grpo_trainer.py:1420-1433``, server at ``grpo_trainer.py:1325-1335``, which
    forwards the dict to ``vllm_serve.py:591-604`` where it lands in the same
    ``SamplingParams(**generation_kwargs)``. An earlier draft of this file guarded on
    ``vllm_mode == "colocate"``; that made ``--vllm-mode server`` train on rollouts that
    never stopped, which is the defect ``--stop-token`` exists to fix, reintroduced by the
    flag that fixes it. Hence the argument this function does *not* take.

    Returns ``None`` on the transformers-generate path (``--no-use-vllm``), which already
    stops on ``tokenizer.eos_token_id`` (``grpo_trainer.py:728-745``) and would take
    ``stop_token_ids`` as an unknown ``GenerationConfig`` field.
    """
    if not use_vllm or stop_token_id is None:
        return None
    return {"stop_token_ids": [stop_token_id]}


def _make_guard_class():
    """Build :class:`GenerationConfigGuard`, importing transformers only when called.

    The class has to subclass ``transformers.TrainerCallback``, but the reward function and
    the sanitiser above are the parts under unit test and neither needs transformers, trl
    or torch. Deferring the import keeps ``import train_grpo`` cheap and, more usefully,
    keeps it possible on a login node where ``trl`` is not installed -- which is where
    ``tests/test_gsm8k_reference_grpo.sh`` runs.
    """
    from transformers import TrainerCallback

    class GenerationConfigGuard(TrainerCallback):
        """Re-apply the generation_config fixes on every step, and audit every save.

        Applying the fix once before ``train()`` is not enough, and the reason is
        structural rather than hypothetical: ``generation_config`` is a mutable object
        that the trainer, a user callback, an evaluation helper or a resumed checkpoint can
        all reach. The failure this guards against lands at ``_save_checkpoint``, so the
        only hook that is guaranteed to have run immediately before it is
        ``on_step_end`` -- ``transformers.Trainer`` calls
        ``callback_handler.on_step_end(...)`` and then ``_maybe_log_save_evaluate(...)``,
        which is where the save happens. ``on_save`` fires *after* the write and is
        therefore useless as a preventative; it is used here only to audit.

        Cost of running every step: four attribute writes, against a measured 8.6 to 54.88
        seconds per GRPO step in this arm.
        """

        def __init__(self, model, temperature=DEFAULT_SHIPPED_TEMPERATURE, stop_token_id=None):
            self.model = model
            self.temperature = temperature
            self.stop_token_id = stop_token_id
            self.saves_audited = 0
            self.saves_bad = 0

        def _apply(self):
            gen_cfg = getattr(self.model, "generation_config", None)
            if self.temperature is not None:
                sanitise_generation_config(gen_cfg, temperature=self.temperature)
            if self.stop_token_id is not None:
                ensure_stop_tokens(gen_cfg, self.stop_token_id)

        def on_train_begin(self, args, state, control, **kwargs):
            self._apply()

        def on_step_end(self, args, state, control, **kwargs):
            self._apply()

        def on_save(self, args, state, control, **kwargs):
            """Confirm the checkpoint the trainer just wrote carries a readable decode.

            Logged and not raised. A missing or unreadable ``generation_config.json`` means
            the grader will sample the checkpoint at 1.0 -- a wrong answer with no error
            attached, which is the same class of defect as 89727_g5's chain script writing
            ``*_1319_*`` filenames over an n=500 accuracy. But if ``save_pretrained``
            returned at all it wrote the file, so the honest reading of a failure here is
            "something outside this script is wrong", and killing a six-hour run on that
            suspicion costs more than the line of log it replaces.
            """
            ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            path = os.path.join(ckpt, "generation_config.json")
            self.saves_audited += 1
            try:
                with open(path) as f:
                    cfg = json.load(f)
            except (OSError, ValueError) as exc:
                self.saves_bad += 1
                logger.error("save audit: cannot read %s (%s)", path, exc)
                return
            logger.info(
                "save audit: ok: %s do_sample=%s temperature=%s eos_token_id=%s",
                ckpt,
                cfg.get("do_sample"),
                cfg.get("temperature"),
                cfg.get("eos_token_id"),
            )

    return GenerationConfigGuard


def _make_termination_monitor_class():
    """Build :class:`TerminationMonitor`, importing transformers only when called.

    Deferred for the same reason as :func:`_make_guard_class`.
    """
    from transformers import TrainerCallback

    class TerminationMonitor(TrainerCallback):
        """Decode the way the grader decodes, during training, and halt when it breaks.

        This is README defect 3 wearing its third hat, and the hat is a train/serve
        skew rather than a bug in either half. GRPO optimises a reward computed on
        *sampled* rollouts capped at ``--max-completion-length`` (512). The grade is
        computed on a *greedy* decode capped at 4000. Nothing in the training loop
        ever performs the second one, so the two can come apart silently and every
        number the trainer prints stays green while the thing being graded falls
        apart.

        Cell 91039_g7 of the 2026-09-03 campaign is the worked example, and it is
        worth stating precisely because the obvious reading of it is wrong. Its
        ladder picked rung 1e-5 on a pre-registered rule, and its own telemetry
        endorsed the pick: reward 0.231 -> 0.562, ``completions/clipped_ratio``
        **max 0.086**, mean completion length 185 of a 512 cap, entropy 0.632 ->
        0.495, and -- in its own words -- "no kill criterion fired through step 30".
        Sampled rollouts terminated. Then the graded read of that checkpoint decoded
        greedily and produced completions averaging 13,425 characters, ran into the
        grader's 4000-token cap, and scored 19.41% on the 170 rows it got through
        before being killed: "the completions are running to the grader's 4000-token
        cap and being scored on whatever number happens to be last in a page of
        unrelated text". The cell concluded "there is no rate in the ladder that does
        both", abandoned RL and shipped its SFT plateau at 73.01. The fourteen cells
        that shipped GRPO at 1e-5 without this happening averaged 83.2.

        No sampled statistic would have caught that -- 0.086 clipped is a healthy
        run -- which is the whole reason this probe generates instead of reading a
        metric. Every ``--termination-probe-steps`` steps it decodes a handful of
        training prompts greedily, at length, and reports the fraction that reach the
        stop token. A rollout sampled at temperature 1.0 wanders into the stop token
        eventually; a greedy decode of the same policy can loop forever, and only the
        second one is what the grader will do.

        Two consequences, both of which the cells asked for in their own logs:

        * A probe that passes marks a checkpoint worth keeping (``should_save``), so
          the last decode-healthy model is on disk rather than reconstructable.
        * A probe that fails after one has passed halts training (``should_save``
          then ``should_training_stop``). Continuing is not neutral: the run that
          produced the 19.41% read had spent its remaining budget making the model
          worse while its loss curve improved.

        Separately, ``on_log`` watches TRL's ``completions/clipped_ratio`` -- not to
        act on, but because 91038_g7 and 91036_g6, the other two cells in the ~69
        tail, both hit ~40% of rollouts truncating at the cap and both diagnosed it
        by hand. 91038_g7 filed it as "a defect in the training configuration ... 40%
        of rollouts are truncated and scored zero", having worked out that a
        truncated completion cannot contain a parseable final answer, so two rollouts
        in five were contributing a zero that says nothing about the policy. That is
        worth one loud line at the point it becomes true rather than an afternoon of
        somebody re-deriving it.

        Nothing here shapes the reward, picks a learning rate, or touches the
        dataset. It measures the quantity the score is computed from, which the
        training loop otherwise never looks at, and refuses to keep spending GPU
        hours after that quantity has collapsed.
        """

        CLIPPED_KEYS = ("completions/clipped_ratio", "clipped_ratio")

        def __init__(self, floor, probe_steps, probe_prompts, probe_tokens,
                     halt=True, tokenizer=None, prompts=None, stop_token_id=None):
            self.floor = floor
            self.probe_steps = probe_steps
            self.probe_tokens = probe_tokens
            self.halt = halt
            self.tokenizer = tokenizer
            self.prompts = list(prompts or [])[:probe_prompts]
            self.stop_token_id = stop_token_id
            self.trainer = None
            self.armed = False        # a greedy probe has passed at least once
            self.collapsed = False
            self.collapsed_step = None
            self.last_good_step = None
            self.best = 0.0
            self.probes = []          # [(step, stop_fraction, mean_new_tokens)]
            self.clipped = []         # [(step, clipped_ratio)]
            self.failures = 0
            self.disabled = False
            self.truncation_warned = False
            self.missing_warned = False

        def bind(self, trainer) -> None:
            """Hand the callback the trainer. Kept for symmetry with the guard.

            A callback is only ever passed ``args`` -- the TrainingArguments -- and
            some of what is worth reading (the vLLM handle, the config actually in
            force) lives on the trainer, which does not exist until after the
            callback list has been built.
            """
            self.trainer = trainer

        # -- the sampled statistic, watched but not acted on ------------------------

        def _ratio(self, logs):
            for key in self.CLIPPED_KEYS:
                value = (logs or {}).get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return float(value)
            return None

        def on_log(self, args, state, control, logs=None, **kwargs):
            ratio = self._ratio(logs)
            if ratio is None:
                if not self.missing_warned:
                    self.missing_warned = True
                    logger.warning(
                        "termination monitor: no %s in the trainer's logs (saw %s)",
                        " or ".join(self.CLIPPED_KEYS), sorted(logs or {}),
                    )
                return
            self.clipped.append((int(state.global_step), round(ratio, 4)))
            if ratio >= TRUNCATION_WARN and not self.truncation_warned:
                self.truncation_warned = True
                logger.warning(
                    "%.0f%% of rollouts hit the %d-token completion cap at step %d. A "
                    "truncated completion rarely ends on its final answer, so most of "
                    "those score 0 and contribute a gradient that says nothing about "
                    "the policy. Raise --max-completion-length, or pass "
                    "--mask-truncated-completions to drop them from the loss. This is "
                    "what 91038_g7 and 91036_g6 both found by hand.",
                    100.0 * ratio, args.max_completion_length, state.global_step,
                )

        # -- the greedy probe, which is the one the grade is computed from ----------

        def on_step_end(self, args, state, control, **kwargs):
            if self.disabled or self.probe_steps <= 0 or not self.prompts:
                return
            step = int(state.global_step)
            if step <= 0 or step % self.probe_steps:
                return
            result = self._probe(kwargs.get("model"))
            if result is None:
                return
            stop_fraction, mean_new = result
            self.best = max(self.best, stop_fraction)
            self.probes.append((step, round(stop_fraction, 4), round(mean_new, 1)))
            logger.info(
                "greedy probe at step %d: %.1f%% of %d prompts reached the stop token "
                "in %d tokens (mean %.0f generated)",
                step, 100.0 * stop_fraction, len(self.prompts), self.probe_tokens,
                mean_new,
            )
            if stop_fraction >= self.floor:
                if not self.armed:
                    self.armed = True
                    logger.info("greedy probe: armed at step %d", step)
                self.last_good_step = step
                # Keep the last decode-healthy weights on disk. save_steps defaults to
                # 50 and the collapse above happened by step 30, so without this the
                # only artefact of a run that dies is the model that died.
                control.should_save = True
                return
            if not self.armed or self.collapsed:
                # Never yet healthy: the model has not learned to stop, which is a
                # model still training rather than a model being destroyed.
                return

            self.collapsed = True
            self.collapsed_step = step
            logger.error(
                "GREEDY TERMINATION COLLAPSE at step %d: %.1f%% of probes reach the "
                "stop token, best was %.1f%%, floor %.1f%%. Sampled rollouts can still "
                "look healthy here -- 91039_g7 had clipped_ratio max 0.086 and no kill "
                "criterion at the moment its graded read was decoding 13k characters "
                "per row and scoring 19.41%%. The reward is computed on sampled "
                "rollouts; the grade is not.",
                step, 100.0 * stop_fraction, 100.0 * self.best, 100.0 * self.floor,
            )
            self._write(args, "grpo_termination_collapse.json")
            if self.halt:
                control.should_save = True
                control.should_training_stop = True
                logger.error(
                    "halting. The last probe to pass was step %s; prefer that "
                    "checkpoint over `final`, and score it before shipping. Pass "
                    "--no-termination-halt to train through this instead.",
                    self.last_good_step,
                )

        def _probe(self, model):
            """Greedy-decode the probe prompts. Never raises; disables itself instead.

            Runs on every rank rather than on rank zero alone: under ZeRO-3 or FSDP a
            forward pass is a collective, and a probe that only rank zero enters is a
            hang rather than a measurement. The duplicated work is a few seconds of a
            1.7B model.
            """
            if model is None:
                return None
            try:
                import torch

                was_training = model.training
                model.eval()
                try:
                    texts = [
                        self.tokenizer.apply_chat_template(
                            p, tokenize=False, add_generation_prompt=True
                        )
                        for p in self.prompts
                    ]
                    enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                                         padding_side="left")
                    device = next(model.parameters()).device
                    enc = {k: v.to(device) for k, v in enc.items()}
                    pad_id = self.tokenizer.pad_token_id
                    if pad_id is None:
                        pad_id = self.stop_token_id
                    with torch.no_grad():
                        out = model.generate(
                            **enc,
                            max_new_tokens=self.probe_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                            eos_token_id=self.stop_token_id,
                            pad_token_id=pad_id,
                        )
                finally:
                    if was_training:
                        model.train()
                grown = out[:, enc["input_ids"].shape[1]:]
                stopped, lengths = 0, []
                for row in grown.tolist():
                    if self.stop_token_id is not None and self.stop_token_id in row:
                        stopped += 1
                        lengths.append(row.index(self.stop_token_id) + 1)
                    else:
                        lengths.append(len(row))
                return stopped / len(grown), sum(lengths) / len(lengths)
            except Exception as exc:  # noqa: BLE001 -- see the docstring
                self.failures += 1
                logger.error("greedy probe failed (%s): %s", type(exc).__name__, exc)
                if self.failures >= 2:
                    self.disabled = True
                    logger.error(
                        "greedy probe disabled after %d failures. Training continues "
                        "UNWATCHED: nothing is now checking that the model can still "
                        "stop when decoded the way the grader decodes it.",
                        self.failures,
                    )
                return None

        def on_train_end(self, args, state, control, **kwargs):
            # Written on every run. "Termination held the whole way" is the sentence a
            # cell needs in order to rule this out and go and look somewhere else, and
            # it is worth as much as the alarm is.
            self._write(args, "grpo_termination_trace.json")

        def _write(self, args, name) -> None:
            path = os.path.join(args.output_dir, name)
            payload = {
                "floor": self.floor,
                "probe_steps": self.probe_steps,
                "probe_tokens": self.probe_tokens,
                "probe_prompts": len(self.prompts),
                "halt": self.halt,
                "armed": self.armed,
                "collapsed": self.collapsed,
                "collapsed_at_step": self.collapsed_step,
                "last_good_step": self.last_good_step,
                "best_stop_fraction": round(self.best, 4),
                "probe_disabled": self.disabled,
                "greedy_probes": self.probes,
                "sampled_clipped_ratio": self.clipped,
            }
            try:
                os.makedirs(args.output_dir, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2)
                logger.info("wrote %s", path)
            except OSError as exc:
                logger.error("cannot write %s (%s)", path, exc)

    return TerminationMonitor


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


def build_fewshot_prefix(dataset_path: str, dataset_config: str, n: int, seed: int) -> str:
    """The grader's own few-shot system message, reproduced call for call.

    ``inspect_evals``' gsm8k task builds it with
    ``hf_dataset(path, data_dir="main", split="train", shuffle=True, seed=42, limit=10)``,
    and inspect's ``hf_dataset`` is ``datasets.shuffle(seed=seed)`` followed by
    ``.select(range(limit))`` (``inspect_ai/dataset/_sources/hf.py:113-119``) -- so the two
    calls below select the identical ten rows, in the identical order. The join and the
    per-shot layout are ``sample_to_fewshot``.
    """
    from datasets import load_dataset

    shots = load_dataset(dataset_path, dataset_config, split="train")
    shots = shots.shuffle(seed=seed).select(range(n))

    rendered = []
    for row in shots:
        parts = row["answer"].split("####")
        target = parts.pop().strip()
        reasoning = "####".join(parts).strip()
        rendered.append(f"{row['question']}\n\nReasoning:\n{reasoning}\n\nANSWER: {target}")
    return "\n\n".join(rendered)


def build_dataset(args):
    """The GRPO prompt set: the same messages the grader will send, plus the gold target.

    Conversational rather than plain text because that is what the grader does -- it
    serves the checkpoint through vLLM with the ``templates/`` file its own
    ``template_kwargs()`` picks for this model family, and sends a system message and a
    user message. Training on raw concatenated text and then
    being graded through a chat template is a train/serve skew that costs points and
    leaves no trace in any log.
    """
    # Checked before the import so the refusal costs nothing and needs nothing installed.
    # src/eval/tasks/gsm8k/info.json allows "gsm8k training subset" and MetaMathQA and
    # nothing else, and run_task.sh runs containers/contamination_check.py against the
    # shipped model. Cheaper to refuse here than to find out from the judge.
    if args.train_split.startswith("test"):
        raise SystemExit(
            f"refusing --train-split {args.train_split!r}: the graded split is gsm8k test, "
            "and info.json allows the training subset only"
        )

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split=args.train_split)
    if args.max_train_samples > 0:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    system_prefix = ""
    if args.fewshot > 0:
        system_prefix = build_fewshot_prefix(
            args.dataset, args.dataset_config, args.fewshot, args.fewshot_seed
        )

    def to_prompt(row):
        messages = []
        if system_prefix:
            messages.append({"role": "system", "content": system_prefix})
        messages.append(
            {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=row["question"])}
        )
        return {"prompt": messages, "target": gsm8k_target(row["answer"])}

    # remove_columns so that "question"/"answer" do not reach the reward function: TRL
    # forwards every surviving dataset column to it as a keyword argument, and a reward
    # signature that has to absorb the whole row is one rename away from silently
    # swallowing the column it was meant to grade against.
    return ds.map(to_prompt, remove_columns=ds.column_names)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


class BooleanFlag(argparse.Action):
    """`--bf16`, `--no-bf16` and `--bf16 false` all mean what they look like.

    argparse.BooleanOptionalAction accepts only the first two. HuggingFace's own
    TrainingArguments accepts `--bf16 True`, and that is the spelling an agent
    writing a TRL command has read a hundred times, so it is the one it actually
    types. Measured on the live 2026-09-03 campaign: cell 295f00fb burned two
    probe launches on

        train_grpo.py: error: unrecognized arguments: false

    before logging "Fix flag and relaunch probes". A reference script advertised
    to the agent as pre-validated should not charge it for the ecosystem's
    spelling of a boolean.

    Safe as nargs="?" only because this parser has no positional arguments -- if
    one is ever added, `--bf16 <positional>` would swallow it.
    """

    TRUE = frozenset({"true", "t", "yes", "y", "1", "on"})
    FALSE = frozenset({"false", "f", "no", "n", "0", "off"})

    def __init__(self, option_strings, dest, default=None, help=None, **kwargs):
        opts = []
        for opt in option_strings:
            opts.append(opt)
            if opt.startswith("--") and not opt.startswith("--no-"):
                opts.append("--no-" + opt[2:])
        super().__init__(option_strings=opts, dest=dest, nargs="?", default=default,
                         help=help, metavar="BOOL", **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        negated = (option_string or "").startswith("--no-")
        if value is None:
            # The bare flag: `--bf16` is True, `--no-bf16` is False.
            parsed = not negated
        else:
            low = str(value).strip().lower()
            if low in self.TRUE:
                parsed = True
            elif low in self.FALSE:
                parsed = False
            else:
                raise argparse.ArgumentError(self, (
                    f"expected a boolean, got {value!r}. Accepted: "
                    f"{'/'.join(sorted(self.TRUE))} or {'/'.join(sorted(self.FALSE))}, "
                    "or pass the flag bare."
                ))
            # `--no-bf16 false` reads as "no-bf16 is false", i.e. bf16 is on.
            if negated:
                parsed = not parsed
        setattr(namespace, self.dest, parsed)

    def format_usage(self):
        return " | ".join(self.option_strings)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    b = BooleanFlag

    g = p.add_argument_group("model and data")
    # No defaults on these three. `None` means "resolve it", `''` means "deliberately
    # none", and there is no third value that quietly means Qwen. resolve_model(),
    # select_chat_template() and derive_stop_token() do the resolving in main(), and each
    # of them stops with a message rather than picking something.
    g.add_argument("--model", default=None,
                   help=f"base checkpoint to train (default: ${MODEL_ENV_VAR} from the harness)")
    g.add_argument("--templates-dir", default=None,
                   help="directory of graded jinja templates (default: templates/ beside "
                        "this script's parent, i.e. the task root's copy)")
    g.add_argument("--chat-template", default=None,
                   help="jinja template the grader will serve with; default is chosen from "
                        "the model id by the grader's own rule; '' keeps the tokenizer's own")
    g.add_argument("--dataset", default="openai/gsm8k")
    g.add_argument("--dataset-config", default="main")
    g.add_argument("--train-split", default="train", help="must not be the graded test split")
    g.add_argument("--max-train-samples", type=int, default=0, help="0 = the whole split")
    g.add_argument("--fewshot", type=int, default=DEFAULT_FEWSHOT,
                   help="few-shot exemplars in the system message; the grader uses 10")
    g.add_argument("--fewshot-seed", type=int, default=DEFAULT_FEWSHOT_SEED)

    g = p.add_argument_group("GRPO")
    g.add_argument("--learning-rate", type=float, default=1e-6,
                   help="TRL's GRPO default, and what every cell in this arm but one picked")
    g.add_argument("--lr-scheduler-type", default="constant_with_warmup")
    g.add_argument("--warmup-steps", type=int, default=10)
    g.add_argument("--beta", type=float, default=0.0,
                   help="KL coefficient; 0.0 drops the reference model entirely")
    g.add_argument("--loss-type", default="dr_grpo",
                   choices=["grpo", "dapo", "bnpo", "dr_grpo"],
                   help="dr_grpo normalises by a constant, removing the length bias in 'grpo'")
    g.add_argument("--scale-rewards", default="none", choices=["group", "batch", "none"],
                   help="'none' is what scale_rewards=False coerces to (grpo_config.py:886)")
    g.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS)
    g.add_argument("--per-device-train-batch-size", type=int,
                   default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE)
    g.add_argument("--gradient-accumulation-steps", type=int,
                   default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    g.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    g.add_argument("--max-completion-length", type=int, default=DEFAULT_MAX_COMPLETION_LENGTH)
    g.add_argument("--temperature", type=float, default=1.0, help="rollout temperature")
    g.add_argument("--top-p", type=float, default=1.0, help="rollout top_p")
    g.add_argument("--top-k", type=int, default=0, help="rollout top_k; 0 disables")
    g.add_argument("--num-iterations", type=int, default=1, help="mu in the GRPO paper")
    g.add_argument("--epsilon", type=float, default=0.2, help="PPO-style clip range")
    # TRL's default, and worth leaving alone until the model can stop. Early in training a
    # base model almost never emits the stop token, so nearly every rollout is truncated;
    # masking them then zeroes most of the batch and the run makes no progress while
    # looking perfectly healthy in the logs. Grading the tail is also what the frozen
    # scorer does, just at 4000 tokens instead of 512. TerminationMonitor prints the
    # rollout truncation rate and says so when it gets high enough to be worth this flag.
    g.add_argument("--mask-truncated-completions", action=b, default=False,
                   help="drop rollouts that hit the completion cap instead of scoring their tail")
    g.add_argument("--reward-match", default="grader", choices=["grader", "strict"],
                   help="'strict' compares the extracted number with == instead of endswith")
    # The reward is computed on sampled rollouts capped at --max-completion-length;
    # the grade is computed on a GREEDY decode capped at 4000. Nothing else in this
    # loop ever performs the second one, and 91039_g7 shipped an SFT plateau at 73.01
    # because of the gap. See TerminationMonitor.
    g.add_argument("--termination-probe-steps", type=int, default=25,
                   help="greedy-decode a few prompts every N steps; 0 disables the probe")
    g.add_argument("--termination-probe-prompts", type=int, default=16,
                   help="how many training prompts each greedy probe decodes")
    g.add_argument("--termination-probe-tokens", type=int, default=1024,
                   help="token budget per probe; long enough that a looping policy shows")
    g.add_argument("--termination-floor", type=float, default=0.90,
                   help="fraction of probes that must reach the stop token")
    g.add_argument("--termination-halt", action=b, default=True,
                   help="stop training if a passing greedy probe is later failed")
    g.add_argument("--num-train-epochs", type=float, default=DEFAULT_NUM_TRAIN_EPOCHS)
    g.add_argument("--max-steps", type=int, default=-1, help="-1 = bounded by epochs")
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("vLLM")
    g.add_argument("--use-vllm", action=b, default=True)
    g.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"],
                   help="colocate shares the training GPU; server needs `trl vllm-serve`")
    g.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3,
                   help="the fraction evaluate.py also asks for")
    g.add_argument("--vllm-max-model-length", type=int, default=0,
                   help="0 = infer from the model config; must exceed prompt + completion")

    g = p.add_argument_group("checkpoints and the shipped decode")
    g.add_argument("--output-dir", default="runs/grpo_ref")
    g.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    g.add_argument("--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT)
    g.add_argument("--save-only-model", action=b, default=True,
                   help="drops optimizer state (~4x smaller) and with it --resume support")
    g.add_argument("--shipped-temperature", type=float, default=DEFAULT_SHIPPED_TEMPERATURE,
                   help="temperature written into every saved generation_config.json")
    g.add_argument("--sanitise-generation-config", action=b, default=True,
                   help="disable only if you want the five-cell ValueError back")
    g.add_argument("--stop-token", default=None,
                   help="extra EOS for chat-templated turns; default is read off the chosen "
                        "chat template; '' to leave the tokenizer's own")

    g = p.add_argument_group("runtime")
    g.add_argument("--bf16", action=b, default=True)
    g.add_argument("--gradient-checkpointing", action=b, default=True)
    g.add_argument("--attn-implementation", default="flash_attention_2")
    g.add_argument("--logging-steps", type=int, default=1)
    g.add_argument("--log-completions", action=b, default=True,
                   help="prints sampled completions; the cheapest way to see a format collapse")
    g.add_argument("--report-to", default="none")

    return p.parse_args(argv)


def resolve_run_identity(args) -> argparse.Namespace:
    """Fill in ``model``, ``templates_dir``, ``chat_template`` and ``stop_token`` on ``args``.

    Split out of :func:`main` and taking no I/O beyond reading the template file, so the
    three decisions that used to be Qwen literals can be exercised for all four swept base
    models on a machine with no GPU and no model download -- which is the only way this
    stays right the next time a model is added to ``src/commit_utils/commit.sh``.
    """
    args.model = resolve_model(args.model)
    if args.templates_dir is None:
        args.templates_dir = default_templates_dir()
    if args.chat_template is None:
        args.chat_template = select_chat_template(args.model, args.templates_dir)
    if args.stop_token is None:
        if not args.chat_template:
            # No chat template means the tokenizer's own turn markers, whatever they are;
            # there is nothing here to read a stop token off.
            args.stop_token = ""
        else:
            args.stop_token = derive_stop_token(args.chat_template)
            if args.stop_token is None:
                raise SystemExit(
                    f"could not read the assistant turn's stop token out of "
                    f"{args.chat_template}.\n"
                    "  Training without it is defect 3 in this file's docstring: nothing\n"
                    "  stops a chat-templated generation, every rollout runs to the token\n"
                    "  cap, and the last-number scorer grades the tail. Pass it explicitly:\n"
                    "      --stop-token '<|im_end|>'      (or '' to accept the tokenizer's own)"
                )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    resolve_run_identity(args)
    logger.info("model=%s chat_template=%s stop_token=%r",
                args.model, args.chat_template, args.stop_token)

    # Imported here rather than at module scope so that the reward function and the
    # generation_config fixes stay importable without trl or torch. See _make_guard_class.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template:
        # The grader passes this exact file to vLLM (evaluate.py's template_kwargs), and a
        # base model's own bundled template is close but not byte-identical to it. Use the
        # graded one so the prompt the model trains on is the prompt it is scored on.
        with open(args.chat_template) as f:
            tokenizer.chat_template = f.read()

    stop_token_id = None
    if args.stop_token:
        stop_token_id = tokenizer.convert_tokens_to_ids(args.stop_token)
        if stop_token_id is None or stop_token_id == tokenizer.unk_token_id:
            # The cross-check that makes the template-derived answer safe: the token this
            # model's tokenizer does not have is a template/model mismatch, and it is caught
            # here at second zero instead of by a rollout that never stops.
            raise SystemExit(
                f"--stop-token {args.stop_token!r} (read off {args.chat_template}) is not in "
                f"{args.model}'s tokenizer. The template and the model do not match."
            )
        # TRL masks a rollout at the first `tokenizer.eos_token_id` (grpo_trainer.py:323,
        # 1570) and passes that id to the transformers generate path, so the tokenizer is
        # where the training side of the stop has to be set.
        tokenizer.eos_token = args.stop_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if args.bf16 else None,
        attn_implementation=args.attn_implementation,
    )

    if args.sanitise_generation_config:
        sanitise_generation_config(model.generation_config, temperature=args.shipped_temperature)
    if stop_token_id is not None:
        ensure_stop_tokens(model.generation_config, stop_token_id)

    dataset = build_dataset(args)
    logger.info("train prompts: %d", len(dataset))

    generation_kwargs = build_generation_kwargs(args.use_vllm, stop_token_id)

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        mask_truncated_completions=args.mask_truncated_completions,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length or None,
        generation_kwargs=generation_kwargs,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        save_strategy="steps",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        log_completions=args.log_completions,
        report_to=args.report_to,
    )

    monitor = _make_termination_monitor_class()(
        args.termination_floor,
        args.termination_probe_steps,
        args.termination_probe_prompts,
        args.termination_probe_tokens,
        halt=args.termination_halt,
        tokenizer=tokenizer,
        # The probe decodes real training prompts, so what it measures is the policy
        # on the distribution it is being optimised for rather than on a fixture.
        prompts=[dataset[i]["prompt"]
                 for i in range(min(args.termination_probe_prompts, len(dataset)))],
        stop_token_id=stop_token_id,
    )

    callbacks = [monitor]
    if args.sanitise_generation_config or stop_token_id is not None:
        callbacks.append(
            _make_guard_class()(
                model,
                temperature=args.shipped_temperature if args.sanitise_generation_config else None,
                stop_token_id=stop_token_id,
            )
        )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(strict=args.reward_match == "strict")],
        callbacks=callbacks,
    )
    # After construction: TRL copies mask_truncated_completions out of the config in
    # GRPOTrainer.__init__ and reads it off the trainer thereafter, so the monitor
    # needs the trainer itself and there is no earlier moment it exists.
    monitor.bind(trainer)
    trainer.train()

    final = os.path.join(args.output_dir, "final")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    logger.info("wrote %s", final)
    logger.info(
        "this is a floor, not a deliverable: score it against the base model with "
        "`python evaluate.py --model-path %s --limit 150` before you ship anything",
        final,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
