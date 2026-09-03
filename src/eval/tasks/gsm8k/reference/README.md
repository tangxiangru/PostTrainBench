# A reference GRPO run — and why you should beat it, not adopt it

`train_grpo.py` is a **reference**, not an entrypoint. Nothing in `src/run_task.sh` calls
it, no score depends on it, and you are free to delete it. It is here for one reason: in a
twelve-cell arm of this task, whether the shipped weights had received a real
verifier-reward RL run explained about half the entire 15.99 pt spread — the six cells
that cleared 500 GRPO steps and 50k rollouts averaged **81.70** against **73.93** for the
six that did not (+7.77 pt, Welch p=0.0042), and GRPO's own within-cell pre/post delta was
positive all six times it was measured (+4.93, +5.2, +5.7, +6.9, ~+7.8, +10.0; sign test
p=0.0156, four of the six curves still climbing at stop). The cells that did not get there
mostly did not fail at RL. They failed at *starting* RL, on the same three defects, one
after another. `bash reference/smoke.sh` runs two steps on 32 rows and re-reads the
checkpoint it wrote; if that passes in your first ten minutes you have skipped all three.

**This is a floor, and treating it as a recipe will cost you the top of the range.** The
best results in that arm came from method diversity, not from a shared recipe: 89809_g5
shipped a uniform fp32 **soup of two GRPO checkpoints** and won at 84.53; 89727_g3 shipped
**Dr. GRPO with `scale_rewards=False` on a 201k external corpus** and came second at
84.15; 89809_g3 shipped **plain GRPO at G=16** and came third at 83.85. Three different
methods in the top three. If everyone runs this file unchanged, the ceiling falls toward
whatever this file happens to score and the spread narrows from the *top* rather than from
the bottom — which is the opposite of the point. So every default here is deliberately the
boring one: TRL's own value, or the arm's median. Learning rate is `1e-6` because that is
what every cell but one picked; the one that did not ran a `1e-6/3e-6/1e-5/3e-5` ladder,
killed the `3e-5` rung at step 75 when `clipped_ratio` hit 0.93, shipped from `1e-5`, and
won the arm. Nothing in this file ladders anything. Reward shaping, curriculum,
checkpoint selection, souping, dataset choice and the learning rate itself are all
untouched and all yours. Run it, confirm the path works, then go and do something else —
and score against the base model before you ship, because a run that trained is not the
same thing as a run that helped.

## The defects it pre-empts

1. **`ValueError: GenerationConfig is invalid: temperature 0.0 with do_sample False`, at
   the first checkpoint save.** Five of twelve cells, independently: 89809_g3 (~9 min and
   50 steps), 89727_g1 (14.85 GPU-min, 100 steps, no weights produced), 89810_g3 (25 min,
   20 steps), 89810_g7 (its `soup.py` died on it and had to be rewritten), 89727_g7 (its
   `grpo_v1` died at step 40 of 1500). Both halves of the pair are things a careful person
   sets on purpose — `temperature=0.0` because `evaluate.py` passes no temperature so vLLM
   reads this file, `do_sample=False` because that is what Qwen3-1.7B-Base ships — and
   transformers refuses the combination only when something tries to *write* it, an hour
   in. `sanitise_generation_config()` writes `do_sample=True, temperature=0.0` instead,
   which transformers accepts and which vLLM still decodes greedily (`_SAMPLING_EPS =
   1e-5`). `GenerationConfigGuard` re-applies it on `on_step_end`, the last hook that runs
   before `Trainer._save_checkpoint` — fixing the config once before `train()` does not
   survive anything that touches it later, and `on_save` fires after the write.
2. **A reward that grades differently from the frozen grader.** `evaluate.py` extracts
   nothing itself; it runs `inspect_evals/gsm8k`, whose scorer is `match(numeric=True)` —
   the *last number* in the completion. `grade_completion()` is a transcription of
   `inspect_ai.scorer._common.match_str` at the revision `containers/opus_5.def` installs,
   including the branch nobody reproduces: 16 of the 1319 test targets and 82 of the 7473
   train targets are not `isnumeric()` (comma-grouped like `2,125`, or negative like
   `-10`), and on those rows the grader requires the completion to *end with the literal
   target string*, commas included. A verifier that "helpfully" normalised those would
   reward answers the grader marks wrong.
3. **A model that cannot stop.** Qwen3-1.7B-Base's `eos_token` is `<|endoftext|>` (151643)
   but `templates/qwen3.jinja` closes an assistant turn with `<|im_end|>` (151645), so a
   chat-templated generation has no reachable stop and runs to the 4000-token cap — and
   the last-number scorer then grades the tail. `ptb_ops/make_greedy_shadow.py` documents
   the same failure from the scoring side. `--stop-token` sets it on the tokenizer (which
   is what TRL masks rollouts against), in vLLM's `SamplingParams` in *both* vLLM modes
   (colocate and server merge `generation_kwargs` into that dict the same way), and
   in the saved `generation_config.json` so the grader stops there too. You do not pass it:
   the script renders one assistant turn through the chat template the grader will actually
   serve your model with and reads the closing token out of the result, so `<|im_end|>` for
   the two Qwen bases and SmolLM3, `<end_of_turn>` for `gemma-3-4b-pt`. If that token is not
   in your tokenizer the script stops at second zero, because a template/model mismatch is
   this same defect wearing a different hat.

4. **RL training the stop token back off again.** Defect 3 is about a stop token the
   model cannot *reach*, and it is fixed before step zero on the decode side. This is the
   same defect from the other end: the model can stop, and GRPO unlearns it. Nothing in
   the objective prices termination -- `make_reward_fn` is binary on `grade_completion`,
   which grades the trailing number and never asks whether the rollout ended -- so a
   rollout that runs into the 512-token cap with the right number in its tail earns the
   full reward and the full gradient, and "keep talking until a correct number goes past"
   is a policy this reward pays for. 91039_g7 ran the ladder cleanly, found `1e-6` and
   `3e-6` moved rollout reward *down* (-0.0266, -0.0203) and `1e-5` moved it up +0.3313,
   then found that at `1e-5` greedy `stop_fraction` hit 0.0 and `mean_greedy_tokens` the
   full 1024. It wrote `grpo_greedy_termination_failure.json`, concluded "the learning
   rates that move rollout reward are exactly the learning rates that destroy greedy
   termination -- there is no rate in the ladder that does both", abandoned RL and shipped
   its SFT plateau at 73.01. The fourteen cells that shipped GRPO at `1e-5` *without*
   collapsing averaged 83.2. It was not wrong about anything it measured; it was missing
   the flag that breaks the tie, and the flag is hard to reach for because both of its
   settings are wrong half the time. `--mask-truncated-completions False` leaves the hole
   open; `True` closes it and opens another, because a base model that cannot stop yet has
   nearly every rollout truncated, so masking zeroes most of the batch and the run makes no
   progress while every number in the log looks healthy. The default is therefore neither:
   `auto` keeps masking off until termination has actually been reached and latches it on
   if it is then lost, which answers both objections at once and is a **no-op on a run that
   never learns to stop** -- exactly the runs the second objection is about. `TerminationMonitor`
   reads TRL's own `completions/clipped_ratio` (`grpo_trainer.py:1782`, the fraction of
   rollouts whose last token is neither eos nor pad), so it costs no extra generation, and
   it writes `grpo_termination_trace.json` next to the checkpoints on every run --
   "termination held all the way through" is worth as much as the alarm is. Reward,
   dataset, curriculum and learning rate are all still untouched: the only thing this
   refuses to do is let the loss keep paying for completions that never terminated.

## Which model it trains

**It has no default model, on purpose.** This benchmark is swept over four base models —
`google/gemma-3-4b-pt`, `Qwen/Qwen3-4B-Base`, `Qwen/Qwen3-1.7B-Base`,
`HuggingFaceTB/SmolLM3-3B-Base` — and this same `reference/` directory is copied into every
cell. A file that defaulted to one of them would train the wrong model in three cells out
of four, produce a run that looks entirely normal, and break rule 8 of your prompt while
doing it. So inside the sandbox the script reads `$MODEL_TO_TRAIN`, which the harness sets
to the model named in your prompt, and it exits immediately if that is unset and `--model`
was not given. The chat template follows from the model id by the *grader's own*
`model_type()` rule (transcribed from `evaluate.py`), and the stop token follows from the
template, so there is exactly one thing to say and the other two cannot drift away from it.
`bash reference/smoke.sh` needs no arguments in the sandbox; outside it, say
`bash reference/smoke.sh /tmp/out --model <org>/<model>` — every argument after the output
directory is forwarded to `train_grpo.py`.

## Things it is not

It does not choose a checkpoint for you, soup anything, tune anything, or touch the
1319-row official score. `--save-only-model` is on, so these checkpoints are ship
candidates and not resume points. `--shipped-temperature 0.0` makes the saved config
greedy under vLLM but unusable with `model.generate(do_sample=True)`, because
transformers' `TemperatureLogitsWarper` rejects 0.0 — pass `1e-6` if you need both. And
the honest caveat: **nobody has run this file on a GPU.** It is checked by
`bash tests/test_gsm8k_reference_grpo.sh` (the reward, the extraction, and the
`generation_config` fixes, on CPU) and by `bash -n`, and its first real execution is your
`smoke.sh`. If it is broken, that is a five-minute fix and a fixed reference; the five
cells above each paid an hour for the same information.
