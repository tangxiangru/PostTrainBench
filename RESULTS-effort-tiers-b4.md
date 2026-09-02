# PostTrainBench × AutoR — batch 4 (effort tiers + stage-scoped skills)

Cluster: slurm2. Task: gsm8k. Base model: `Qwen/Qwen3-1.7B-Base`. Agent model: Claude Opus 5 on Vertex.
Grader: the frozen `src/eval/tasks/gsm8k/evaluate.py`, `--limit -1` (all 1319 rows).

## Reference points

| what | accuracy | provenance |
| --- | --- | --- |
| published zero-shot | 0.12661 | PostTrainBench v1.1 paper |
| **untrained, measured here** | **0.11979 ± 0.0089** | job 81735, full split, COMPLETED |
| published few-shot | **0.46679** | the number an arm has to beat |
| leaderboard, Opus 5 + Claude Code | 0.8034 ± 0.037 | PostTrainBench leaderboard |

The measured untrained number is what makes the rest of this file readable: the scoring
half reproduces the published zero-shot figure on this cluster, so a low arm score is the
arm and not the scorer.

## What is running

### Paper-faithful pair, 10 h (submitted 07:44 UTC)

| job | arm | payload | node |
| --- | --- | --- | --- |
| 82165 | `claude_autor` | `f36dfd5` (pre-intervention) | ondem-1 |
| 82166 | `claude_vertex` | bare Claude Code | ondem-0 |

This is the published contract — 10 h, one GPU — and answers "does AutoR beat bare Claude
Code at all". `POST_TRAIN_BENCH_SKIP_CLI_UPDATE` is empty in these two, so each updated the
CLI at job start.

### Batch 4: 6 vs 6 at 1 h (submitted 08:0x UTC)

| job | node | GPU 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 82647 | A | ctl | pt | ctl | pt | ctl | pt |
| 82648 | B | pt | ctl | pt | ctl | pt | ctl |

Six cells per node, one GPU each, via `slurm_logs/ptb_pack6.sbatch`. Arms are interleaved
so that "which arm" is not the same question as "which node" or "which GPU index".
`POST_TRAIN_BENCH_SKIP_CLI_UPDATE=1` in all twelve: this batch compares payloads, so the
CLI build is pinned.

| arm | agent dir | payload | worktree |
| --- | --- | --- | --- |
| ctl | `claude_autor_ctl` | `f36dfd5` | `~/autor-s2ctl` |
| pt | `claude_autor_pt` | `1243fdf` | `~/autor-s2tiers` |

Cell result dirs are `${RESULTS}/claude_autor_{ctl,pt}_claude-opus-5_1h/gsm8k_Qwen_Qwen3-1.7B-Base_<job>_g<gpu>`.

## What the treatment changes

Two commits on top of the control, both aimed at one defect: **the run spends itself on
paper apparatus that a scored benchmark never reads.**

- `6f2880f` — set P. Fourteen gsm8k skill pins, two new skills, two paragraphs in the
  Stage 05 prompt. Measured effect on the real brief: skills installed 41 → 55, skills
  actually rendered into a stage prompt 3 → 53.
- `1243fdf` — set T. `ARTIFACT_TIERS` inverts the effort table for a scored-artifact front
  end (01/02/03 routine, 04/05/06/07 deliberative), and `install_run_skills` takes
  `stage_slugs` so a run that stops at 05 stops installing 06/07/08 skills. The stage
  filter drops exactly 7 of 55 on this brief.

**Attribution between P and T is not resolvable from this batch.** Twelve cells is enough
power for one contrast, not three. If ctl vs pt moves, the next batch splits it.

The tier half has a live control behind it: job 81520 spent 06:34→07:02 — **28 of its 60
budgeted minutes** — inside Stage 01 Literature Survey, over three attempts plus a
reviewer audit, on a task whose method the brief already named.

## Recovered scores

Two earlier live runs trained a model and then died at grading, on container cache faults
since fixed (`4fb9184`) and a missing bind (see `slurm_logs/ptb_rescore.sbatch`). The
weights are on disk and the grader is frozen, so the score is recoverable — but the
*agent's own reading of it* is not, so nothing downstream of the agent seeing a number may
be inferred from these.

| job | run | rescore job | accuracy |
| --- | --- | --- | --- |
| 81521 | `claude_vertex` 1 h | 82097 | **0.47157 ± 0.01375** |
| 81520 | `claude_autor` 1 h | 82148 | pending |

0.47157 is the first agent score this cluster has produced. Read against the table at the
top: it is +35 pp on the untrained model, and it lands on top of the published few-shot
number (0.46679) rather than above it — a 0.005 difference against se 0.0138 is not a
difference. It is also less than two thirds of the 0.8034 the leaderboard reports for the
same agent and model, which is the expected shape for one hour against ten and is the
reason batch 4 is a within-batch comparison and not a comparison to the leaderboard.

82102 accidentally duplicated 82097. If both land, their agreement is a free cross-check on
the two scoring paths; that is a consolation, not the reason it was submitted.

## Reading rules for this file

- The harness grades on all 1319 rows; the agent-facing `evaluate.py` defaults to
  `--limit 150`. An agent's own reported number therefore carries se ≈ 4 pp and two
  independent 150-row reads differ by ≈ 5 pp for nothing. Never compare an agent's
  self-report to a harness score.
- A `state.json` on disk means a cell started, not that it finished.
- A cell that produces no `final_model/` is not a zero; it is a missing observation, and
  the count of them is itself an arm difference worth reporting.

---

# Batch 4 results, and what they say about the walk

Added 2026-08-30 ~11:00 UTC. Everything below is a **within-batch** reading: pack size
differs across packs (82647/82648 are six cells, 82822/82823 are eight), so cells are
compared inside a job, never across.

## Every cell scored so far, against how deep its walk got

`deepest` is the highest stage number the ledger records; `01 min` is wall-clock inside
`01_literature_survey` out of a ~54.5 min usable budget.

| score | cell | arm | deepest stage | 01 min |
| ---: | --- | --- | ---: | ---: |
| **0.7309** | 82822_g6 | pt | 2 | 49.9 |
| **0.4284** | 82648_g2 | pt | 2 | 49.9 |
| 0.2570 | 82648_g4 | pt | 4 | 7.7 |
| 0.1243 | 82647_g0 | ctl | 4 | 13.0 |
| 0.1175 | 82648_g1 | ctl | 3 | 32.5 |
| 0.1099 | 82648_g0 | pt | 3 | 30.9 |
| 0.0728 | 82648_g5 | ctl | 4 | 12.6 |
| 0.0485 | 82648_g3 | ctl | 5 | 15.2 |

Untrained, measured here: **0.11979**. Six of these eight cells are at or below it.

Two things fall out, and they point opposite ways.

**The only cell that completed the walk scored last.** 82648_g3 is the single cell to
reach `05_experimentation`, and at 0.0485 it is the worst number in the batch — three
quarters of the way below the model it started from. The two best cells never got past
Stage 02.

**But "deepest stage" is an outcome, not an assignment.** A cell that spends its hour
training gets stuck in Stage 01 *because* training ate the clock; the ledger for both
0.73 and 0.43 cells reads `01_literature_survey, 49.9 min, 5 attempts` and the reviewer
text on one of them says it ran "an actual 3-epoch SFT run that completed all 1404 steps"
and "correctly tried to bank a floor early rather than only writing a reading list". So
this table cannot separate "the walk costs score" from "training costs walk depth". It is
a correlation among outcomes. The experiment that separates them is `fx` vs `fxq` below.

## 82648, complete at 6/6

| arm | payload | n | cells | mean |
| --- | --- | ---: | --- | ---: |
| `claude_autor_pt` | `1243fdf` s2-tiers | 3 | 0.1099, 0.4284, 0.2570 | 0.2651 |
| `claude_autor_ctl` | `f36dfd5` main | 3 | 0.1175, 0.0485, 0.0728 | 0.0796 |

pt is above ctl by 0.186 within the one job, and every ctl cell is below the untrained
model. Three per arm against a within-arm spread of 0.11–0.43 is not a result; it is a
reason to finish 82822/82823, which add eight more cells of the same contrast.

## Why the runs were losing: a census of what refused them

Across every pack cell with a ledger, validator refusals break down as:

| count | stage | refusal |
| ---: | --- | --- |
| ~28 | 03, 04 | `... is covered by artifact '../final_model', which does not exist` |
| 2 | 02 | `'Files Produced' references missing file(s): `Qwen/Qwen3-1.7B-Base`` |
| 2 | 03 | `covered by headline number '4', which this plan does not declare` |
| 2 | 02 | `'Files Produced' references missing file(s): `outputs/format_audit.json`` |
| 1 | 02 | `'Paper Claims (Provisional)' must include at least one typed identifier` |

**Four in five validator refusals in this batch are one bug**, and the second entry is
another. Both are now fixed:

- `01a359c` — the `artifact:` existence check ran as soon as `report.md` existed on disk,
  a proxy for "the report is being written" that an agent flips during Stage 01 by
  drafting a header. A study design was therefore refused for naming the object the study
  is designed to produce. The check moves to Stage 07 coverage, where it is answerable,
  and `../final_model` is refused at Stage 03 for its *shape* instead, with a message that
  says the path can never resolve and names the two ways out. The old message invited the
  agent to rewrite the same path and read the same refusal until the clock ran out — run
  81520 spent two attempts and eleven minutes doing exactly that.
- `ff2c5d3` — `Qwen/Qwen3-1.7B-Base` is `org/name` with a version dot, so its last
  segment ends in `.7B-Base`, which the path heuristic read as a file extension. On a
  benchmark where naming the model you trained is the task, that refused Stage 02 for a
  `Files Produced` entry that was never a file. Which ids were caught was an accident of
  length: `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` passed only because its tail runs past the
  twelve-character bound.

## The measurement failure underneath run 81520

Separate from the refusals, and not fixed by them. The agent measured the **base model**
at `--limit 30, --max-tokens 1024` and got **0/30 → 0.0**. Its entire comparison frame was
against zero, so an SFT probe at 0.0667 read as +6.7 pp when against the true 0.1198 it
was −5.3 pp. It then found a `--max-tokens` confound, **fixed it on the candidates only**,
and never re-ran the control — leaving the control at limit 30 / 1024 tokens while
candidates moved to 150 / 4000.

`audit_measurements` (in `01a359c`) walks the workspace for recorded accuracies and reports
two things it can check without knowing the task: everything recorded came in below the
published untrained rate, and more than one `--limit` was used. Replayed against 81520 it
reproduces the whole story in four rows and both problems fire.

Note this kills an intervention proposed earlier in this file's history: **shipping the
base model as a score floor is a rules violation.** Rule 8 of the harness prompt is "Only
fine-tune from `Qwen/Qwen3-1.7B-Base` and submit one of your fine-tuned versions as your
final_model."

## Batch 5, submitted: does bounding the walk help or hurt?

Job **84425**, 8 cells on one node. Submitted as 83765, which carried the unpinned
launcher (see below); cancelled at 0:00 elapsed and resubmitted 2026-08-31 with identical
argv and the pinned one. Cells are therefore `84425_g0..g7`, not `83765_g*`.

| arm | payload | `--stage-quotas` |
| --- | --- | --- |
| `claude_autor_fx` | `0719436` | off |
| `claude_autor_fxq` | `0719436` | **on** |

Same payload SHA, byte-identical goal text (3875 bytes, diffed), interleaved across GPUs
0–7 four and four. The **only** difference is the flag. `0719436` is `1243fdf` plus the
two fixes above plus two commits that had been written, tested and left unmerged —
`f2c3324` (a baseline carries where it was measured) and `7401c78` (stage quotas). Neither
was an ancestor of `main` or of either running arm, so the quota work was reaching zero
experiments.

Quotas re-cut the hour by `PTB_STAGE_WEIGHTS` — 01: 8%, 02: 7%, 03: 10%, 04: 25%, 05: 50%
— which at 1 h gives Stage 01 about 4.4 min instead of the 30–50 it has been taking.

**This is a real test and the table above predicts it may lose.** The two cells that
scored 0.73 and 0.43 got there inside a 49.9-minute Stage 01, which is the one block long
enough to hold a training run; quotas would cut exactly that block. The counter-argument is
that quotas move the long block to Stage 05 (50% = 27.5 min) and put it after the stage
that builds the training code. Which effect dominates is what 83765 measures.

## The largest effect in this batch is decoding, and it is not the agents' fault

`src/eval/tasks/gsm8k/evaluate.py` passes `max_tokens` and `max_connections` to
`inspect_eval` and **no temperature**. The eval logs the cells left behind record the
consequence directly — every one of them reports

```
model_generate_config = {'max_connections': 16, 'max_tokens': 4000}
```

so nothing sets a temperature at the harness layer. vLLM then falls back to the
checkpoint's `generation_config.json`, and `Qwen/Qwen3-1.7B-Base` ships

```json
{"bos_token_id": 151643, "do_sample": false, "eos_token_id": 151643,
 "max_new_tokens": 2048, "transformers_version": "4.37.0"}
```

`do_sample` is a transformers field; vLLM reads `temperature`, which is absent, so
sampling runs at the library default of **1.0**.

### What that does to a completion

From `82822_g6`'s own 0.0400 run, verbatim, target 20:

```
If Amir eats five cookies, Cody eats 3 * 5 = 15 cookies
Together, Amir and Cody eat 5 + 15 = 20 cookies.

ANSWER: 20ﴼ
```

— a correct answer, then a stray low-probability token where `<|im_end|>` belonged, then
11,308 further characters of invented problems, ending `ANSWER: 2(wrapper)สนใจ`. The task
scores with `match(numeric=True)`, which reads the **last** number. Judged wrong.

In that single 150-item log: 144 misses, **93 of them (64.6%) contain the correct number
somewhere in the completion**; 50 of 150 hit `max_tokens`; median completion 5,671
characters.

### The size of it, over every log in the batch

All 40 evaluation logs the cells wrote, 5,750 graded items, re-scored offline on the
first `ANSWER:` line each completion states instead of on the last number it contains —
same weights, same completions, same items, only the extraction rule changes:

| | accuracy |
| --- | ---: |
| as the harness scored it | 0.2210 |
| reading the first stated answer | **0.4092** |
| | **+0.1882** |

The per-log deltas are diagnostic rather than uniform. They are ~0 for logs that already
scored well (0.7067 → 0.7067, 0.7600 → 0.7600) and ~0 for logs where the model never
stated a well-formed answer at all (0.1333 → 0.1333). They are **+0.39 to +0.47** for the
logs in between. The defect does not depress a weak model; it deletes the gains of a
model that has learned to answer.

### It separates the leaderboard exactly

| score | cell | shipped `temperature` |
| ---: | --- | --- |
| 0.7309 | 82822_g6 | **0.0** |
| 0.7293 | 82823_g3 | **0.0** |
| 0.4284 | 82648_g2 | absent |
| ... 20 more, 0.4284 down to 0.0485 | | absent |

Two of twenty-three cells wrote an explicit greedy `generation_config.json` into
`final_model/`. They are the only two above 0.5. Several losing cells wrote
`"do_sample": false` with no temperature, which does nothing.

### The untrained reference is affected too, by a different channel

Eighteen agent evaluations of `Qwen/Qwen3-1.7B-Base` straight from the HF cache score
0.08–0.20 — consistent with the harness's own 0.11979 on the full split. The one cell that
copied the base model locally and gave that copy a greedy config measured **0.2667**. For
the base model the first-answer re-score gains nothing (a base model rarely states a
clean answer line), so this is not the tail: it is temperature-1.0 noise in the arithmetic
itself. Two channels, one missing setting.

**Consequence for every number above this section.** "Six of eight are at or below the
model they started from" is measured against a floor that is itself depressed. Nothing in
this file that compares a cell to 0.11979 survives unchanged.

### The change under test

`575664a` on `ptb/s2-decoding` adds one section to `AUTOR_STAGE_NOTE`, kept task-agnostic
so the control arm is not handed a recipe (the leak-token guard — `metamath`, `gsm8k`,
`lora`, `sft`, `0.12`, `baseline` — still passes): what an evaluation script does not pass
is chosen by something you did not measure; the generation config inside the deliverable
is part of the submission, so measuring one decode and shipping another measures nothing;
and stop tokens are as much a setting as the sampling fields.

Job **83997**, 8 cells, interleaved 4/4:

| arm | payload | difference |
| --- | --- | --- |
| `claude_autor_fx` | `0719436` | control |
| `claude_autor_fd` | `575664a` | `0719436` + the decoding section, nothing else |

`solve.sh` is byte-identical between them and the rendered goals differ by exactly the
22 added lines (3800 → 5339 bytes, diffed). Not bundled with 83765's quota flag: one
mechanism per arm.

## 82165 and 82166 were killed by a commit, and both are recovered

Both ten-hour cells died with `src/run_task.sh: error reading input file: Stale file
handle`, `payload_exit=2`. Not the agent, not the model, not NFS flakiness:

- Both started at **07:44:28**. Commit `2775447` rewrote `src/run_task.sh` at
  **08:28:29**, forty-four minutes in, and git writes a new file and renames it over the
  old one.
- Bash reads a script incrementally and seeks back to the byte after the last command it
  parsed, so a script whose body is one ten-hour `timeout` holds an open handle on its own
  inode for the whole run. Both jobs held handles on the replaced inode.
- Neither noticed for hours because neither needed to read. Each died the instant its
  agent phase returned — 82165 after 10:01:47, 82166 after 08:27:27. The two elapsed
  times are two agent phases, not two events.

Both had a finished 3.3 GB `final_model/` on node-local disk, recovered from
`slurm2-a3nodesetondem-1` and `-0` and rescored as **83998** (`claude_autor`) and
**83999** (`claude_vertex`). Both shipped `temperature: 0.0` — in ten hours each arm found
the decoding defect on its own.

Fixed in `6bece37`: `src/commit_utils/pin_src_locally.sh` mirrors the shell half of `src/`
plus `scripts/` to node-local scratch (840 KB; `src/eval` and `.env` symlinked), and
`run_task.sh` resolves both of its `source` lines against `${BASH_SOURCE[0]}` so sourcing
follows the copy. The working directory deliberately stays on the checkout — line 542
takes `REPO_ROOT` from `pwd` and the scoring container binds it by that path.

**83765 and 83818 were submitted before the fix and still carry the unpinned launcher.**
But `grep -c pin_src_locally` is the wrong test and it over-freezes the tree. What matters
is whether bash holds a handle on a checkout file *across* the long command, so read what
each queued job actually runs:

| job | long command | exposed? |
| --- | --- | --- |
| ~~83765~~, 83818 | `bash src/run_task.sh …`, multi-hour | **yes** |
| 83998, 83999, 84024 | `source …/set_env_vars.sh` then `apptainer exec` inline | no |
| 83997, 84363, 84364, **84425** | pinned | no |

A `source` is one short read at that instant, and `python3 foo.py` reads the whole file at
import; neither holds a handle. 83998, 83999 and 84024 all grep as "unpinned" and are all
safe, because their long body lives in the **sbatch**, which slurm snapshotted at submit.

### 83765 is now 84425, and the freeze is down to one job

A freeze is a rule a person has to remember, and the thing that would break it is another
session's `Write`, which no hook here can intercept — a `pre-commit` guard fires after the
inode has already been replaced. So where the job can be reissued exactly, reissuing it is
the only remedy with teeth.

83765 could: it was PENDING at **0:00 elapsed**, its argv is in `sacct SubmitLine`, its
`--time` is the sbatch default, and this section records it ran "at 1 h", i.e. no
`PTB_NUM_HOURS` override — the five env knobs are all at their defaults. Cancelled and
resubmitted as **84425**; its frozen script hashes `491ad070`, byte-identical to 83997's,
and slurm still estimates the same start (2026-09-01T11:36:33), because all seven pending
a3 jobs are waiting on the same eleven reservation nodes to drain together. A new job id
costs no queue position here.

**83818 stays.** It was submitted `--time=16:00:00` against a 4 h default, which means an
env override this file never recorded, and `scontrol`/`sacct`/`--json` expose a pending
job's argv but never its environment. Resubmitting it would mean guessing `PTB_NUM_HOURS`,
and silently running a different experiment is worse than the conditional risk of the
handle. It is documented, not fixed.

So the freeze is one file — `src/run_task.sh` — for one job, 83818, over one window,
2026-09-01T11:36 to roughly 2026-09-02T03:36. Everything else in the checkout, and every
other queued job, stays editable throughout.

---

# The board re-read from the checkpoints, 2026-08-30

Everything in this section is computed from what is on disk right now — 27 `final_model/`
directories under `ptb-results/`, 25 of them with a `metrics.json` — and needed no GPU.
The two unscored ones are the recovered ten-hour cells, queued as 83998/83999.

## Every scored cell, with the decode it shipped

| score | arm | cell | `temperature` | `do_sample` | `eos_token_id` |
| ---: | --- | --- | --- | --- | --- |
| 0.7309 | `pt` | 82822_g6 | **0.0** | false | `[151645,151643]` |
| 0.7293 | `pt` | 82823_g3 | **0.0** | false | `[151645,151643]` |
| 0.4716 | `vertex-1h` | 81521 | absent | absent | `[151645,151643]` |
| 0.4284 | `pt` | 82648_g2 | absent | absent | `151645` |
| 0.4177 | `pt` | 82823_g4 | absent | false | `[151645,151643]` |
| 0.3669 | `pt` | 82823_g0 | absent | absent | `[151643,151645]` |
| 0.3632 | `ctl` | 82822_g4 | absent | absent | `[151645,151643]` |
| 0.3033 | `ctl` | 82823_g2 | absent | absent | `[151645,151643]` |
| 0.2570 | `pt` | 82648_g4 | absent | false | `[151645,151643]` |
| 0.2123 | `pt` | 82823_g7 | absent | absent | `[151645,151643]` |
| 0.1259 | `ctl` | 82823_g1 | absent | false | `151643` |
| 0.1243 | `ctl` | 82647_g0 | absent | absent | `[151645,151643]` |
| 0.1236 | `ctl` | 82822_g7 | absent | false | `151643` |
| 0.1175 | `pt` | 82822_g5 | absent | false | `151643` |
| 0.1175 | `ctl` | 82648_g1 | absent | false | `151643` |
| 0.1168 | `ctl` | 82823_g6 | absent | false | `151643` |
| 0.1152 | `ctl` | 82823_g5 | absent | false | `151643` |
| 0.1130 | `pt` | 82822_g1 | absent | false | `151643` |
| 0.1099 | `pt` | 82648_g0 | absent | false | `151643` |
| 0.0849 | `ctl` | 82822_g3 | absent | absent | `[151645,151643]` |
| 0.0728 | `ctl` | 82648_g5 | absent | absent | `151643` |
| 0.0591 | `ctl` | 82822_g0 | absent | absent | `[151645,151643]` |
| 0.0569 | `pt` | 82822_g2 | absent | absent | `[151645,151643]` |
| 0.0553 | `autor-1h` | 81520 | absent | absent | `151645` |
| 0.0485 | `ctl` | 82648_g3 | absent | false | `[151645,151643]` |

All twenty-five shipped a `generation_config.json`. Two set a temperature.

## The arm contrast, and how much of it is decoding

`pt` and `ctl` are interleaved across the same packs (82648, 82822, 82823), so this is a
same-batch comparison on the same nodes. One-sided permutation test on the difference of
means, every partition enumerated exactly:

| comparison | n | mean | diff | p | partitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pt` vs `ctl`, all cells | 11 / 12 | 0.3218 / 0.1379 | **+0.1839** | **0.0104** | 1,352,078 |
| same, `pt`'s two greedy cells removed | 9 / 12 | 0.2311 / 0.1379 | **+0.0931** | **0.0477** | 293,930 |

So `pt` is ahead, and **about half its margin is the two cells that fixed the decode**.
The remainder is real but sits exactly on the conventional line, from one batch, and it is
one of several comparisons drawn off this board — treat it as suggestive, not established.

## Channel one: sampling temperature

| | n | mean | min | max |
| --- | ---: | ---: | ---: | ---: |
| shipped `temperature: 0.0` | 2 | 0.7301 | 0.7293 | 0.7309 |
| everything else | 23 | 0.1853 | 0.0485 | 0.4716 |

Perfectly separated: the lower greedy cell beats the highest non-greedy cell by **0.2578**.
Two specific cells landing in the top two of twenty-five by chance is 1/C(25,2) = **0.0033**.

## Channel two: the stop token, which is a separate defect

`Qwen3-1.7B-Base` is a base model; the agents fine-tune it against a chat template whose
turn ends with `<|im_end|>` (151645). A `generation_config.json` that lists only 151643
(`<|endoftext|>`) therefore names a stop token the fine-tuned model does not emit, so
generation runs to the 4000-token cap and `match(numeric=True)` reads the tail. Same
outcome as temperature 1.0, different cause, and it is not the same cells:

| | n | mean | min | max | above the 0.1198 floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eos_token_id` includes 151645 | 16 | 0.2944 | 0.0485 | 0.7309 | 11/16 |
| `eos_token_id` is 151643 alone | 9 | 0.1125 | 0.0728 | 0.1259 | 2/9 |

`diff = +0.1819`, one-sided permutation `p = 0.00656` (2,042,975 partitions). It is not a
proxy for the temperature finding — dropping the two greedy cells leaves **+0.1196,
p = 0.0164** — and it is not a proxy for the arm, because it holds inside both:

| arm | with 151645 | 151643 alone |
| --- | --- | --- |
| `pt` | n=8, mean 0.3999 | n=3, mean 0.1135 |
| `ctl` | n=6, mean 0.1639 | n=6, mean 0.1120 |

The nine `151643`-only cells span 0.0728–0.1259. That is not a weak spread of results, it
is the untrained model's score with noise on it: **a wrong stop token pins the cell to the
floor no matter what the training did.**

### This changes what job 84024 will settle

84024 equalises temperature across the board but deliberately leaves `eos_token_id` as each
cell shipped it, on the grounds that forcing a stop token a cell did not train toward would
swap one confound for another. That reasoning is right about *replacing* the list and wrong
about *extending* it: adding 151645 to a cell that lists only 151643 cannot truncate a model
that really does emit `<|endoftext|>`, because 151643 stays in the list — generation stops
at whichever arrives first. As written, 84024 will hand back a board still carrying a
+0.12 confound in nine of its cells.

## Half the board is no better than not training at all

| arm | n | above 0.1198 | at or below |
| --- | ---: | ---: | ---: |
| `pt` | 11 | 7 | 4 |
| `ctl` | 12 | 5 | 7 |
| `autor-1h` | 1 | 0 | 1 |
| `vertex-1h` | 1 | 1 | 0 |
| **total** | **25** | **13** | **12 (48%)** |

Three cells of twenty-five clear 0.4668, the reference this benchmark is actually read
against. And 0.1198 is itself a temperature-1.0 number — the one greedy measurement of the
untrained model is 0.2667 — so under a fixed decode the floor rises and the 48% gets worse,
not better. That is the single largest fact on this board, and it is not a tuning problem:
a run that ships something worse than its own starting point had no gate stopping it.

# The arm aimed at that, built 2026-08-30

`claude_autor_rt` against `claude_autor_rc`. Built but **not submitted**: six PTB jobs are
already pending behind eleven `airsgpu` jobs holding the `robtang-a3` reservation.

## What it changes

AutoR `ea95d15` ([#499](https://github.com/tangxiangru/AutoR/pull/499),
`src/ptb_ratchet.py`). One mechanism, aimed at the 48% and at the finding two sections up
that the deeper the walk goes the worse the score:

* **Restore, no numbers needed.** `final_model/` is hardlinked into a snapshot at every
  stage boundary, with a per-file `(path, size, mtime)` manifest. If what ships does not
  load, the newest snapshot that still matches its manifest is put back. A hardlink
  survives the unlink-and-rename `save_pretrained` does, but not a truncating write; the
  manifest tells those apart, and a snapshot that changed is dropped rather than repaired.
* **Promote, numbers needed and comparability required.** A checkpoint declares its own
  score in `AUTOR_CANDIDATE.json` **inside** the directory. Snapshots are partitioned by
  `(limit, measured decode)` and maximised only within a partition, so a candidate scored
  on 200 items is never compared against one scored on 1,319 and a greedy measurement is
  never compared against a temperature-1.0 one — which is exactly the confound the two
  sections above are about. The bar is `max(0.005, sqrt(2·p·(1−p)/n))`, ≈1.8 pp at n=1319.

`measured_decode` (declared) and `shipping_decode` (the checkpoint's own
`generation_config.json`) are tracked separately. A disagreement blocks promotion — the
number does not describe what would ship — but not restore.

## Why the pair is a controlled one

The `fd`/`fx` pair had byte-identical `solve.sh` files and differed by a **payload sha**,
so it also carried whatever else landed between the two builds. This pair does not:

```
$ diff agents/claude_autor_{rt,rc}/solve.sh
43c43
< AUTOR_EXTRA_FLAGS=(--ratchet)
---
> AUTOR_EXTRA_FLAGS=()
$ git -C agents/claude_autor_rt/payload rev-parse HEAD   # ea95d157... , both arms
```

Payload trees hash identically outside `.git`; `api_keys.json` is `{"allowed_api_keys": []}`
on both; `--check` reports no problems on either, and reports one if the flag is misspelled,
which would otherwise be an `unrecognized arguments` exit ten hours into a queue.

## What it is not

Half of the mechanism is enforced in code and needs nothing from the agent: the export-time
ratchet. The other half — bank a real checkpoint in the first 45% of the budget — is a
directive that appears in the prompt once that deadline passes with nothing banked, and
repeats until it is met. It is not a gate. **An agent that trains nothing for ten hours
still ships nothing**, by design: the ratchet only ever re-selects among checkpoints the
agent saved to `final_model/` itself.

## What the arm can and cannot show

It can show whether the *lower half* of the board comes up: the prediction is a rise in the
"above 0.1198" count, not a rise in the best cell. It cannot show that on n=1 per arm —
the decode census above spreads cells of one arm from 0.0485 to 0.4284, so a single-seed
contrast here sits well inside its own noise. Six cells per arm is the minimum worth
reading, and the comparison to make is the count above the floor, not the mean.

# The 1 h tier could not have reached significance, whatever the effect (2026-09-01)

The count-above-the-floor reading above is the right one. Applied to the whole 1 h tier,
it says the tier is unreadable — not because the arms are weak, but because of how the
cells were distributed over packs.

Threshold, derived rather than chosen: the pooled 1 h scores (n=65) have their largest gap
at **0.549 → 0.636**, so "worked" is > 0.5925. The base model is 0.1198, and nothing sits
between 0.472 and 0.535 either. The distribution is two clumps: a cell produces a working
fine-tune, or it produces something at or under the untrained model. Means are the wrong
summary for it; successes-out-of-cells is the right one.

## The control never ran on a day any treatment arm ran

| arm | 08-30 | 08-31 | payload branch | payload sha |
|---|---|---|---|---|
| ctl | 0/12 | — | `main` | f36dfd5 |
| pt | 2/11 | — | `ptb/s2-tiers` | 1243fdf |
| fd | — | 3/4 | `ptb/s2-decoding` | 575664a |
| fx | — | 2/8 | `ptb/s2-tiers` | 0719436 |
| fxq | — | 0/4 | `ptb/s2-tiers` | 0719436 |
| dg | — | 0/4 | `ptb/gate-on-tiers` | dcf487b |
| dgc | — | 1/4 | `ptb/gate-on-tiers` | dcf487b |
| rt | — | 5/8 | `ptb/arm-build` | ea95d15 |
| rc | — | 6/8 | `ptb/arm-build` | ea95d15 |

The table separates perfectly. So the headline anyone would write from the arm means —
`rc` 6/8 vs `ctl` 0/12, Fisher p = 0.00072 — is a comparison between two calendar days.
There is no same-day contrast against the control anywhere in this batch.

The arms themselves are real: nine distinct payload contents over five AutoR branches.
(An arm is its `payload/`, a whole AutoR checkout; `solve.sh` is a thin launcher and
`ctl`, `pt`, `fd` and `fx` all share the same 2292 bytes of it. Hashing `solve.sh` says
nothing about whether two arms differ — `fx` and `fxq` are the pair worth noticing, and
they share a payload *commit*, 0719436.)

## The pack is the unit, and there are five of them

Arms are nested two-per-pack, four GPUs each:

| pack | arms | worked / 8 |
|---|---|---|
| 83997 | fd, fx | 3 |
| 84425 | fx, fxq | 2 |
| 84437 | dg, dgc | 1 |
| 84363 | rt, rc | 5 |
| 84364 | rt, rc | 6 |

The pack is a real source of variance and not a formality: **`fx` ran in two packs on the
same day, one payload, and went 0/4 then 2/4.** So the treatment contrast has to be taken
with the pack as the unit. Exact permutation over which 2 of the 5 packs are `rc`/`rt`:
observed difference 3.50 successes per pack, **p = 1/10 = 0.10**, which is the most extreme
of the ten arrangements.

**0.10 is the floor.** With 2 treatment packs against 3 others there are only ten
assignments, so no effect of any size could have produced p < 0.10. The batch was
unable to reach significance before the first cell started.

Taken at the cell level instead, `rc`+`rt` 11/16 vs every other 08-31 arm 6/24 reads
Fisher p = 0.0095 — that number is what the nesting manufactures, and it should not be
quoted.

## The fix is free

Interleaving arms across GPUs *within* a pack, which this launcher already does, does not
address this. The arm has to be interleaved across **packs**:

- as run: 5 packs × 2 arms × 4 GPUs → 4 replicates/arm, arm confounded with pack
- instead: 5 packs × 8 arms × 1 GPU → **5 replicates/arm**, every arm in every pack

Identical GPU-hours, one more replicate per arm, and the pack becomes a block to subtract
rather than a confound. `rc` remains the best point estimate (6/8, Wilson [0.41, 0.93]) and
is worth re-running that way; detecting 75% vs 25% at 80% power needs about **17 cells per
arm**, so roughly three blocked rounds.

## What is enforced now

`ptb_ops/ptb_pack.sbatch`, before any cell starts:

- logs `arm_payload <arm> branch= payload_sha= content= solve_sha=` for every arm, so the
  meaning of an arm name survives the run — `ptb-results/` is keyed on the name and
  `agents/` is untracked, so nothing else records it;
- refuses two arms with identical payload content (`PTB_ALLOW_DUPLICATE_ARMS=1` overrides,
  for deliberately measuring the null);
- refuses a pack with no in-pack control (`PTB_CONTROL_ARM`, default `claude_autor_ctl`;
  `PTB_NO_CONTROL=1` overrides). The content hash skips `.git/` and `__pycache__/`: two
  clones of one commit differ there while being the same arm.

# The one-hour board measures a decode flag, not a training recipe (2026-09-01)

The section above says the b4 design could not have reached significance. That is
still true and it is now the *second* problem. The first is that the outcome
variable is not measuring the treatment.

`src/eval/tasks/gsm8k/evaluate.py` passes no temperature, so vLLM samples under
whatever `generation_config.json` the cell delivered in `final_model/`. That file is
part of what an arm ships. Across the 95 scored cells, 50 ship `temperature: 0.0`,
44 ship a config with no decode keys at all, and one ships `temperature: 1.0`.

    ships temperature=0.0    n=68   mean 0.5901
    ships anything else      n=27   mean 0.2290

## On b4 the flag is the result

Cross-tabulating the 40 b4 cells against the 0.5925 success threshold derived above:

                    win   lose
      temp=0.0       17      5
      otherwise       0     18      Fisher exact p = 3.35e-07

Not one of the eighteen cells without the flag ever cleared the bar. And the flag is
almost perfectly determined by the arm:

    arm    ships temp=0.0    "wins"
    rc          8/8           6/8
    rt          7/8           5/8
    fd          3/4           3/4
    fx          3/8           2/8
    dgc         1/4           1/4
    dg          0/4           0/4
    fxq         0/4           0/4

Within the 22 cells that all shipped `temperature: 0.0`, 17 land in a band of
**mean 0.6757, sd 0.0133**, and the arms are not separable inside it (fd 0.668,
dgc 0.683, rt's winners 0.68, rc's winners 0.68). The other 5 are 0.111, 0.187,
0.466, 0.535, 0.549.

Across all nine one-hour arms, the arm mean against the fraction of that arm's cells
shipping the flag gives **Pearson r = 0.9558, r² = 0.914** — 91% of the between-arm
variance on the one-hour board is one boolean. The fitted line is: an arm that never
writes it scores 0.211, an arm that always writes it scores 0.612.

## The greedy re-score says the same thing from the other direction

Job 84024 re-scored 27 b3 cells under one fixed greedy decode, same weights, nothing
else changed (`ptb_ops/ptb_greedy_board.sbatch`, shadows under `/rmeng_data/robtang/ptb-greedy`).
Equalising the decode moved the mean **+0.2912** (median +0.2237, max +0.6368). The
four cells that moved by ~0 (−0.0015, +0.0015, +0.0030, +0.0068) are exactly the four
that had already shipped a greedy config — the mechanism is confirmed, not assumed.

After equalising, the board has two states and nothing between them. The untrained
base model scores **0.3328** greedy:

    ctl  n=12   at-or-below base: 7   above: 5    the 5 average 0.6893 (sd 0.0182)
    pt   n=11   at-or-below base: 3   above: 8    the 8 average 0.6952 (sd 0.0235)

Ten of twenty-three cells land within ±0.02 of the untrained model and two land
*below* it. A one-hour gsm8k score carries about one bit: did this run ship trained
weights, or the base weights it started from.

**So the +0.1839 "pins and tiers wins" headline is a reliability effect, not a recipe
effect** — pt produced a model at all 8/11 times against ctl's 5/12 (Fisher p = 0.21),
and conditional on producing one the two arms differ by 0.006 against a within-arm sd
of ~0.02. Under the equalised decode the arm gap is +0.1280, Welch t = 1.66. It was
+0.1839, t = 2.38, under the un-equalised one. The decode nuisance is larger than the
signal it sits on.

## What survives

The ten-hour tier is not contaminated this way. `sn` ships the flag 8/8, `sc` 4/4,
`sk` 1/1, `sq` 1/1; `claude_vertex` 6/8 inside the paired packs, and its two
non-setters are 0.7157 and 0.8469, one either side, so they do not move its mean.
Restricted to decode-matched cells the head-to-head is vertex 0.8127 (n=6) against
`sn` 0.7979 (n=8) — the same tie the full n=8 pairing gives. That comparison stands.

## What is enforced and what is running

- `ptb_pack.sbatch` now prints `cell_decode gpu= arm= temperature= top_p= top_k=
  do_sample=` beside every `cell_exit`, so the largest effect on the board is in the
  pack log instead of buried in `final_model/` for someone to find later.
- Job **86742** re-scores every cell that has no `metrics_greedy.json` yet — the 40
  b4 cells and the ten-hour tier — under the same fixed greedy decode. It is
  resumable and skips what is already done. Until it lands, no b4 arm contrast should
  be quoted, including the ones in the section above.
- `ptb_pack.sbatch` also now refuses to start if `src/eval/tasks/$TASK/test_data.json`
  is missing. `**/test_data.json` is gitignored by design, so a `git clone` of this
  repo yields a checkout that looks complete and has no test data; that is how
  jobs 84279 and 84280 each took an exclusive eight-GPU node and lost all seven cells
  to one error line at the 4.5-minute mark.

# 86742 landed: equalise the decode and the b4 ranking inverts (2026-09-01)

Job 86742 finished COMPLETED in 2:34:10, exit 0. All 95 cells now carry a
`metrics_greedy.json` — the 40 b4 cells and the whole ten-hour tier re-scored off
symlink shadows under one fixed greedy decode, same weights, `ptb-results/` untouched.

## The mechanism, at n=40 and with nothing left to argue about

    cells that already shipped temperature=0.0   shift +0.0002   sd 0.0038   n=22
    cells that did not                           shift +0.3889   sd 0.1610   n=18

The re-score is a no-op for every cell that was already being graded greedily. Its
sd of 0.0038 is the re-scoring noise floor, so the +0.3889 on the other eighteen is
not noise, and it is not a property of those cells' training.

## The ranking inverts

    arm    sampled            greedy             shift
    rc     #1  0.5937         #6  0.5940         +0.0003
    rt     #2  0.5601         #7  0.5612         +0.0011
    fd     #3  0.5224         #2  0.6738         +0.1514
    fx     #4  0.3651         #3  0.6131         +0.2479
    dgc    #5  0.3254         #4  0.5970         +0.2716
    dg     #6  0.2993         #1  0.6816         +0.3823
    fxq    #7  0.1486         #5  0.5961         +0.4475

Spearman rho between the two orderings is **−0.464**. `rc` and `rt` — the two arms
the previous sections called the clean winners at 6/8 and 5/8 — come **last**. They
gained nothing from the re-score because they had nothing to gain: they were already
setting the flag. `dg`, which scored 0/4 and sat sixth, is first.

**Every b4 arm claim in the earlier sections of this document is withdrawn.** Not
weakened — reversed.

## And after equalising there is no arm effect at all

35 of 40 cells produced a model above the untrained base (0.3328); those 35 average
**0.6654, sd 0.0498**. Conditional on producing one:

    dg  0.6816 (4)   dgc 0.6806 (3)   fd  0.6738 (4)   fx  0.6845 (7)
    fxq 0.6798 (3)   rc  0.6519 (7)   rt  0.6329 (7)

Seven arms inside a 5.2-point band; permutation test on the arm factor, 20,000
draws, **p = 0.728**. The five that failed are 0.059, 0.113, 0.189, 0.345, 0.346 —
the last two are the base weights shipped unchanged, the first three are below the
model the run started from.

**At the one-hour budget every arm that produces a model produces the same model.**
The board's entire spread is two binary facts: did the run ship trained weights, and
did it write `temperature: 0.0`.

## The ten-hour tier moved 1 point and still does not resolve

    sn      0.7979 -> 0.7985   (+0.0007)      sc  0.7864 -> 0.7854  (-0.0009)
    vertex  0.8182 -> 0.8287   (+0.0106)      sq  0.7043 -> 0.7028  (-0.0015)

AutoR's ten-hour arms were already greedy, so they do not move; `claude_vertex` gains
0.0106 from its three non-greedy cells. Paired by slot inside 84998/84999:

    vertex - sn = +0.0191   t = 1.33   95% CI [-0.0149, +0.0532]   vertex takes 7/8

Equalising moved the pair from a dead tie (−0.0070, t −0.37, vertex 6/8) to a
1.9-point lead for `claude_vertex` that still does not clear. Report it as unresolved
with the point estimate against AutoR, not as a tie.

# The ten-hour tier gets a clean design and returns a null: `sn` vs `r2` (2026-09-02)

Jobs **89727**, **89809**, **89810**, all COMPLETED, ~10.2 h each, 24 cells, `pack_exit=0`
on all three and `rc=0` on all twenty-four. This is the first PTB contrast on this cluster
where every nuisance the sections above identified is off by construction rather than
argued away afterwards.

## The design

| | |
| --- | --- |
| arms | `claude_autor_sn` (`f99b4db`) vs `claude_autor_r2` (`12c2d46`) |
| layout | 3 packs x 8 GPUs, arms interleaved even/odd, **4 + 4 in every pack** |
| task / model | gsm8k, `Qwen/Qwen3-1.7B-Base`, agent Claude Opus 5, 10 h |
| in-pack control | `claude_autor_r2`, enforced by `ptb_pack.sbatch` |

Both arms appear in all three packs, so the pack is a block to subtract and not a
confound. That is the fix the b4 post-mortem above prescribed and could not apply
retroactively.

## What the treatment is

`r2` is `sn` plus **two prose sections appended to `AUTOR_STAGE_NOTE`** in
`src/posttrainbench.py`, and nothing else. Diffing the two payload trees outside `.git/`
and `__pycache__/` returns exactly two files: that one, and `docs/iclr.md`. The two
sections are:

- *The rules you are under, and the rules you inferred* — a restriction you derived from
  a prohibition costs the same as breaking one; find the file that states what is
  permitted and quote the clause before dropping a resource.
- *Two stages of optimisation, and more than one finishable candidate* — reserve the
  second half of the clock for a second optimisation stage and let that reservation
  constrain the first half; the cheapest extra candidate is a weight average of
  checkpoints you already hold; report how many candidates you finished next to the
  score you shipped.

Both were written against defects the 27-cell board showed. `docs/iclr.md` differs by 82
lines and nothing on the PTB entry path reads it, but it is in the checkout and so is
named here rather than left for someone to rediscover.

## Neither nuisance that ruined the one-hour board is present

- **Decode.** All 24 cells shipped `temperature: 0.0`. The pack log records it per cell
  (`cell_decode`), so this is read off the run and not inferred from `final_model/`.
  The flag that carried 91% of the between-arm variance at 1 h is constant here.
- **Reliability.** All 24 produced a `final_model/` above the greedy untrained base
  (0.3328) — the closest cell is more than twice it. The "did the run ship trained
  weights" bit, which was the other half of the 1 h board's entire spread, is saturated.

So the outcome variable is measuring the treatment. That was not true of any earlier
contrast in this file.

## The result is a null, on all three readings

| reading | n | `sn` | `r2` | `sn` - `r2` | |
| --- | ---: | ---: | ---: | ---: | --- |
| cells, Welch | 12 / 12 | 78.97 | 77.82 | **+1.15** (SE 1.73) | p = 0.52, 95% CI [-2.25, +4.55] |
| pack as the unit | 3 / 3 | | | **+1.15** (SE 1.90) | t_2 CI [-7.05, +9.35] |
| blocked exact permutation | 343,000 arrangements | | | +1.15 | **p = 0.514** |

Per pack, `sn` - `r2` is **-1.54, +0.15, +4.83**. The sign is not stable across three
packs of the same contrast, which is the same message the interval gives.

Direction matters for what to do next: `r2` is the arm *with* the two sections, so the
point estimate on adding them is **-1.15 pt**, and the honest summary is that they buy
nothing and may cost a little.

## What this can and cannot rule out

Pooled sd is 4.25 pt, so 12 v 12 has 80% power against about **4.9 pt**. The contrast
therefore excludes an effect the size of anything else on this board and says nothing
about a one- or two-point one: detecting the observed +1.15 would need roughly **215
cells per arm**, which at 10 h and one GPU per cell is not a measurement anyone is going
to make here.

The two arms are also not equally noisy — `sn` sd 2.84, `r2` sd 5.30, and `r2` owns the
three lowest cells on the board (0.6854, 0.7127, 0.7157). Welch is used above for that
reason; pricing the interval at the control's spread would have understated it.

## Cells not in the contrast

`claude_autor_sn` carries 8 further scored cells from 84998/84999 and `claude_autor_r2`
carries 18 from 86725-87815. Pooling all of them gives +1.49 +/- 1.13 (Welch p = 0.19) —
the same null — but those cells come from different builds on different days, which is
the drift the b4 sections above show inverting a whole ranking. The 12 v 12 in-pack
number is the one to quote.
