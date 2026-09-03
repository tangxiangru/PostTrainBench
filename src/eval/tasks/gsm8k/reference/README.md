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

4. **A policy that stops when sampled and never stops when decoded greedily.** Defect 3
   is a stop token the model cannot *reach*, and it is fixed before step zero. This is the
   same token going missing again for a completely different reason, and the reason is a
   train/serve skew rather than a bug in either half: GRPO optimises a reward computed on
   **sampled** rollouts capped at `--max-completion-length` (512), and the grade is
   computed on a **greedy** decode capped at 4000. Nothing in the training loop ever
   performs the second one. A rollout sampled at temperature 1.0 wanders into the stop
   token eventually; a greedy decode of that same policy can loop forever.

   91039_g7 is the worked example, and the obvious reading of it is wrong. Its telemetry
   endorsed the checkpoint it picked: reward 0.231 → 0.562, `completions/clipped_ratio`
   **max 0.086**, mean completion length 185 of a 512 cap, and in its own words "no kill
   criterion fired through step 30". Sampled rollouts terminated. The graded read of that
   checkpoint then produced completions averaging **13,425 characters**, ran into the
   grader's 4000-token cap, and scored **19.41%** on the 170 rows it got through before
   being killed — "scored on whatever number happens to be last in a page of unrelated
   text". It concluded "there is no rate in the ladder that does both", abandoned RL and
   shipped its SFT plateau at 73.01. The fourteen cells that shipped GRPO at `1e-5`
   without this happening averaged 83.2. **No sampled statistic would have caught it**,
   which is why `TerminationMonitor` generates instead of reading a metric: every
   `--termination-probe-steps` steps (25) it greedy-decodes `--termination-probe-prompts`
   (16) real training prompts for `--termination-probe-tokens` (1024) and reports the
   fraction reaching the stop token. A probe that passes marks the checkpoint worth
   keeping — `save_steps` defaults to 50 and this collapse landed by step 30, so without
   that the only artefact of a dead run is the model that died. A probe that fails *after
   one has passed* saves and halts, because continuing is not neutral: the run that
   produced the 19.41% read spent its remaining budget making the model worse while its
   loss curve improved. A model that has never yet stopped is left alone —
   `--no-termination-halt` turns the acting off and keeps the reporting. The probe runs on
   every rank (a forward pass is a collective under ZeRO-3 and FSDP, so a rank-zero-only
   probe is a hang, not a measurement) and it can never take the run down: two failures
   disable it, loudly.

   Separately, the monitor prints the rollout truncation rate and says something once it
   passes 25%. It does not act on it. 91038_g7 and 91036_g6 — the other two cells in this
   arm's ~69 tail — both ran at **~40%** and both worked it out by hand, 91038_g7 filing
   it as "a defect in the training configuration … 40% of rollouts are truncated and
   scored zero", having reasoned that a truncated completion rarely ends on its final
   answer, so two rollouts in five contribute a zero that says nothing about the policy.
   That is worth one line at the moment it becomes true.

   `grpo_termination_trace.json` is written next to the checkpoints on every run, not just
   on collapse: "termination held the whole way through" is what lets a cell rule this out
   and go and look somewhere else. Reward, dataset, curriculum and learning rate are all
   still untouched — this measures the quantity the score is computed from, which the
   training loop otherwise never looks at.

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
