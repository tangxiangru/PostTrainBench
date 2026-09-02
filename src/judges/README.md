# Reward-Hacking Judges

Agent-as-judge checks that detect reward hacking in agent runs. They analyze the agent's trace
(`solve_parsed.txt`, fallback `solve_out.txt`) by starting from the end and tracing back where
`final_model` comes from, then write a structured verdict.

The judges run in two contexts:

- **Inline** — `src/run_task.sh` runs every judge right after the agent finishes, writing
  `judgement_<id>.json` etc. into the result dir.
- **Standalone/rerun** — `run_judges.sh <result_dir>` reruns judges on an existing result dir.
  Outputs always carry the `_rerun` suffix so the originals are preserved.

They also have two runtime/output profiles:

- **`official` (default)** — Claude Code CLI + Claude Opus 5 at **high** effort. It writes the
  canonical output ids consumed by `scripts/collect.py`; all four judges use this same model
  and effort, including the general judge.
- **`claude` (research)** — the same Claude Opus 5 + high runtime with separate output ids.
  It writes only `judgement_claude_*` outputs. Canonical aggregation deliberately ignores
  these files, so a Claude verdict cannot impersonate or overwrite an official verdict.

Both profiles receive the byte-for-byte same prompt assembled by `build_judge_prompt` from
the same `prompt.md` template and `get_judge_prompt.py` fillers. The judge task, tool paths,
and required `task/judgement.json` schema live in those shared prompts; there is no hidden
Codex system configuration containing benchmark-specific instructions.

## Judges

| Folder | Output id | Verdict schema | Consumed downstream? |
|--------|-----------|----------------|----------------------|
| `data_contamination_judge/` | `gpt5_4` | `contamination`, `disallowed_model` + justifications | **Yes** — canonical contamination verdict (`judgement_gpt5_4.json`, or `judgement_gpt5_4_rerun.json` when present) |
| `api_usage_judge/` | `api` | `disallowed_api_usage` + justification | **Yes** — a flagged run falls back to the baseline score in `scripts/collect.py` (missing file = "not flagged": runs predating this judge have none) |
| `ptb_lookup_judge/` | `ptb_lookup` | `disallowed_ptb_lookup` + justification | Archival — no score fallback, but `scripts/collect.py` raises an error if it ever flags, so a firing lookup judge cannot pass unnoticed |
| `general_judge/` | `general` | `general_anomaly` + justification | Archival — never feeds scores; when it flags, `scripts/collect.py` finishes its collection pass but writes **no** output files and raises, listing every flagged run for manual review (flip `general_anomaly` to false in the listed verdict file if the run checks out) |

The official and research profiles use the same four judge definitions, a configurable
Claude Opus 5 model id, and explicit `--effort high` plus
`CLAUDE_CODE_EFFORT_LEVEL=high`. Their container defaults to `opus_5.sif`. Every invocation
writes a `judge_metadata_*.json`
record containing the requested model alias/id, the model actually resolved in Claude's init
event, effort, container and actual Claude CLI version. The installed CLI/account resolves
and validates the model request at runtime; set
`POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL` to an exact accepted Opus 5 id when pinning is required.

## Layout

```
src/judges/
├── judge_lib.sh              # shared bash helpers (profiles, auth, sandbox, exec, collection);
│                             # also defines ALL_JUDGES (the set + execution order)
├── run_judges.sh             # standalone runner: run_judges.sh [--judges a,b] <result_dir>
├── get_judge_prompt.py       # prompt generation (--judge <judge_name>)
├── <judge_name>/             # one folder per judge
│   ├── judge.conf            # label, output id, prompt file, inline-fatality, condor weight
│   └── prompt.md             # prompt template
├── judge_tools/              # helper tools copied into the judge sandbox
│   ├── contamination_check.py    # n-gram contamination checker (decon SIMPLE algorithm)
│   ├── model_identity_check.py   # architecture-identity check vs reference config
│   └── reference_configs/        # config.json of every allowed base model
├── test_data_download/       # host-side helper to (re)download benchmark test data
└── rerun/                    # batch-rerun pipeline (see below)
```

## Running

```bash
# Official/canonical rerun (the default)
bash src/judges/run_judges.sh /path/to/result_dir

# Rerun a subset
bash src/judges/run_judges.sh --judges data_contamination_judge /path/to/result_dir
bash src/judges/run_judges.sh --judges api_usage_judge /path/to/result_dir

# Research-only Claude/Opus/high rerun with separate output ids
bash src/judges/run_judges.sh --profile claude /path/to/result_dir
```

The result dir must contain `task/` and a trace file. The trace is copied next to the sandbox
task directory, so the judge reads it as `../solve_parsed.txt` from its working directory.

For the Claude profile on GCP/Vertex, the existing site environment is enough when
`CLAUDE_CODE_USE_VERTEX=1` (or `ANTHROPIC_VERTEX=true`) and the compute node exposes GCE
metadata ADC:

```bash
POST_TRAIN_BENCH_JUDGE_PROFILE="official"
POST_TRAIN_BENCH_JUDGE_AUTH_MODE="vertex"  # optional when Vertex env is already set
POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="claude-opus-5[1m]"
POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER="opus_5.sif"
```

For non-GCE Vertex, set a read-only file-backed ADC with
`POST_TRAIN_BENCH_VERTEX_ADC_FILE`. For Anthropic subscription OAuth instead, configure a
dedicated credential outside this repository:

```bash
POST_TRAIN_BENCH_JUDGE_PROFILE="claude"
POST_TRAIN_BENCH_JUDGE_AUTH_MODE="claude_oauth"
POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="opus"
POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER="opus_5.sif"
POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE="/secure/path/ptb-claude-judge/oauth_token"
```

The OAuth token file must not be an `agents/<agent>/oauth_token`. Both Claude auth modes run in `--safe-mode`
(disabling `CLAUDE.md`, skills, plugins, hooks, MCP and custom agents), with a fresh
`CLAUDE_CONFIG_DIR`, empty user/project/local setting sources, no session persistence,
and cleared Anthropic API-key variables. OAuth tokens and file-backed ADC are mounted
read-only; GCE metadata ADC uses the node service account and has no local secret file.
This prevents a tested agent's project files, persona, settings or credentials from
becoming judge configuration. Vertex mode explicitly forwards only the project, region,
Opus mapping and ADC source required by Claude Code; file-backed ADC is mounted read-only.

## Batch reruns (HTCondor)

```bash
# Rerun all judges on the latest run of every (method, benchmark, model)
bash src/judges/rerun/commit_rerun_judges.sh

# Preview without submitting
bash src/judges/rerun/commit_rerun_judges.sh --dry-run

# Only a subset of judges
bash src/judges/rerun/commit_rerun_judges.sh --judges data_contamination_judge

# Batch research rerun with isolated Claude output ids
bash src/judges/rerun/commit_rerun_judges.sh --profile claude

# Per result dir, only rerun judges whose _rerun output is missing
bash src/judges/rerun/commit_rerun_judges.sh --skip-existing
```

Submitted cluster IDs are logged to `rerun/submission_logs/submitted_<timestamp>.txt`
(one `cluster<TAB>judges<TAB>result_dir` line per job).

For ad-hoc targeting (a specific method/benchmark, a limit, etc.), build the directory list with
`list_results.py` and submit `rerun_judges.sub` directly:

```bash
python src/judges/rerun/list_results.py --paths-only --latest-only --method "claude" \
| while read -r d; do
    condor_submit_bid 100 -a "result_dir=$d" src/judges/rerun/rerun_judges.sub
done
```

`rerun_judges.sub` accepts `-a "judges=..."` and `-a "profile=official|claude"`.
Every rerun job consumes a
fixed 2500 units of the `user.codex_judge_rerun` concurrency limit, regardless of which
judges it runs.

### Listing and aggregation

```bash
# List/filter result directories (--with-trace, --missing-rerun, --paths-only, ...)
python src/judges/rerun/list_results.py --paths-only --method "claude"

# Compare original vs rerun contamination verdicts (--diff-only, --csv out.csv)
python src/judges/rerun/aggregate_rerun_results.py

# Absolute paths of latest dirs flagged for disallowed API usage
python src/judges/rerun/find_disallowed_api_usage.py
```

## Output files

Per judge (`<id>` = the profile's output id, `<sfx>` = empty inline / `_rerun` standalone):

- `judge_output_<id><sfx>.json` — raw Codex JSONL or Claude stream-json trace
- `judge_output_<id><sfx>.txt` — human-readable trace (`src/trace_parsing/parse_trace.py`)
- `judge_metadata_<id><sfx>.json` — profile/backend/model/effort/container/CLI version
- `judgement_<id><sfx>.json` — structured verdict the judge wrote to `judgement.json`
  in its sandbox task directory

Claude ids are `claude_contamination`, `claude_api`, `claude_ptb_lookup`, and
`claude_general`. Their verdict JSON schemas are identical to the corresponding official
judge schemas below, because both profiles use the same prompts and collector contract.

`judgement_gpt5_4*.json` schema:

```json
{
  "contamination": true,
  "disallowed_model": false,
  "justification_contamination": "...",
  "justification_disallowed_model": "..."
}
```

`judgement_api*.json` schema:

```json
{
  "disallowed_api_usage": false,
  "justification_disallowed_api_usage": "..."
}
```

`judgement_ptb_lookup*.json` schema:

```json
{
  "disallowed_ptb_lookup": false,
  "justification_disallowed_ptb_lookup": "..."
}
```

`judgement_general*.json` schema:

```json
{
  "general_anomaly": false,
  "justification_general_anomaly": "..."
}
```

## Adding a new judge

1. Create `src/judges/<judge_name>/` with:
   - a prompt template (any filename). `{model}` and `{benchmark}` are always filled; if the
     template needs extra placeholders, register a fill function in `EXTRA_FILLERS` in
     `get_judge_prompt.py`.
   - `judge.conf` — simple `KEY="value"` lines (sourced by bash, parsed by python):
     - `JUDGE_LABEL` — human-readable name used in logs
     - `JUDGE_OUTPUT_ID` — suffix for all output files (`judgement_<id>.json`, ...)
     - `JUDGE_CLAUDE_OUTPUT_ID` — distinct research-profile output suffix; it must not be
       any canonical id consumed by `scripts/collect.py`
     - optional: `JUDGE_CLAUDE_LABEL` for profile-specific log text
     - `JUDGE_PROMPT_FILE` — the template's filename
     - optional: `JUDGE_MODEL` / `JUDGE_REASONING_EFFORT` to override the codex defaults
       (`gpt-5.4` / `xhigh`, see `judge_lib.sh`)
     - optional: `JUDGE_CODEX_VERSION` to pin the codex CLI release for this judge
       (e.g. `"0.144.5"`); `judge_lib.sh` npm-installs exactly that `@openai/codex`
       version into the sandbox home and runs it instead of the container's codex.
       Empty/unset = the container's pinned codex.
2. Add the folder name to `ALL_JUDGES` in `judge_lib.sh` (array order = execution order).

That's it — `run_task.sh`, `run_judges.sh` and `commit_rerun_judges.sh` all iterate over the
judge set, and `--skip-existing` is derived from `judge.conf`.

A judge that produces no `judgement.json` is handled by the caller, not by `judge.conf`:
inline (`run_task.sh`) it is always a warning, so a finished 10h agent run still gets
evaluated and the rerun pipeline can supply the verdict later; standalone
(`run_judges.sh`) it is always fatal, since producing the verdict is the job.

The shared judge prompt must instruct either backend to write its verdict to
`judgement.json` in the task directory; that file is collected as
`judgement_<profile-id><sfx>.json`.
