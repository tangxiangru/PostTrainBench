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
| 81521 | `claude_vertex` 1 h | 82097 | pending |
| 81520 | `claude_autor` 1 h | 82148 | pending |

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
