# Slurm adapter

This fork can run one PostTrainBench cell as a non-exclusive, single-GPU Slurm job while preserving the benchmark's declared resources. Upstream PostTrainBench still targets HTCondor; this directory is the scheduler adapter and does not change prompts, time budgets, judges, evaluators, or scoring.

## Site configuration

Copy `example.env` to `.env` and set the normal PostTrainBench paths plus the Slurm variables. Site-specific node names, partitions, accounts, paths, and credentials belong only in the gitignored `.env`.

```bash
POST_TRAIN_BENCH_JOB_SCHEDULER="slurm"
POST_TRAIN_BENCH_SLURM_PARTITION="ptb-gpu"
POST_TRAIN_BENCH_SLURM_NODELIST="gpu-node-[0-3]"
# POST_TRAIN_BENCH_SLURM_RESERVATION="site-reservation"
POST_TRAIN_BENCH_SLURM_CPUS_PER_TASK="16"
POST_TRAIN_BENCH_SLURM_MEMORY="128G"
POST_TRAIN_BENCH_SLURM_SCRATCH_BASE="/mnt/localssd/posttrainbench"
POST_TRAIN_BENCH_SLURM_MIN_SCRATCH_GB="400"
POST_TRAIN_BENCH_SLURM_TIME_OVERHEAD_MINUTES="240"
POST_TRAIN_BENCH_SLURM_GPU_MODE="gres"
```

Judge profiles are scheduler-independent. The default remains the official
Codex/GPT profile. To run the research Claude profile on these GCP nodes, use
Vertex/GCE metadata ADC and the installed `opus_5.sif`:

```bash
POST_TRAIN_BENCH_JUDGE_PROFILE="claude"
POST_TRAIN_BENCH_JUDGE_AUTH_MODE="vertex"
POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL="opus"
POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER="opus_5.sif"
# GCE metadata ADC is used automatically. Only non-GCE/file-backed ADC needs:
# POST_TRAIN_BENCH_VERTEX_ADC_FILE="/secure/path/google-application-credentials.json"
```

The Claude judge runs through Claude Code with explicit `xhigh` effort. GCE
metadata credentials are node-local and require no secret file. If a different
site uses file-backed ADC or Anthropic OAuth, that credential must be judge-only,
outside the repository, and visible at the same path on every selected node.

Use `POST_TRAIN_BENCH_SLURM_GPU_MODE="gres"` for shared-node runs. It submits `--gres=gpu:<n>` without `--exclusive` and requires `CfgTRES`/`AllocTRES` to contain `gres/gpu`, Slurm to set `SLURM_JOB_GPUS` and `CUDA_VISIBLE_DEVICES`, and `ConstrainDevices=yes` to enforce the device cgroup.

Use `POST_TRAIN_BENCH_SLURM_GPU_MODE="manual"` only when all of the following are true:

- the selected nodes are dedicated to these runs;
- every job receives a whole node through `--exclusive`;
- GPU GRES is unavailable or broken;
- `nvidia-container-cli` is installed.

Manual mode exposes devices `0..num_gpus-1` and uses Apptainer `--nvccli` so the agent sandbox cannot see the other device files. It is safe for cross-node parallelism, not multiple concurrent PTB jobs on one node.

For a shared/unpacked Apptainer installation, also set:

```bash
POST_TRAIN_BENCH_APPTAINER_BIN="/shared/apptainer/usr/bin/apptainer"
POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH="/shared/apptainer/usr/lib/x86_64-linux-gnu"
```

## Validation

Run a scheduler and asset preflight without starting an agent:

```bash
bash src/commit_utils/slurm/submit.sh \
  --eval gsm8k \
  --agent codex_non_api \
  --model google/gemma-3-4b-pt \
  --hours 10 \
  --agent-config gpt-5.4 \
  --run-branch gangda_trial_0828 \
  --job-name gangda_trial_0828.ptb.example.gsm8k.preflight.r1 \
  --experiment-name _gangda_trial_0828_example_gsm8k_preflight_r1 \
  --preflight-only
```

Run a short GPU check inside the configured agent SIF:

```bash
bash src/commit_utils/slurm/submit.sh \
  --eval gsm8k \
  --agent codex_non_api \
  --model google/gemma-3-4b-pt \
  --hours 10 \
  --agent-config gpt-5.4 \
  --run-branch gangda_trial_0828 \
  --job-name gangda_trial_0828.ptb.example.gsm8k.runtime-smoke.r1 \
  --experiment-name _gangda_trial_0828_example_gsm8k_runtime_smoke_r1 \
  --runtime-smoke
```

`--preflight-only` and `--runtime-smoke` are validation jobs and must not be reported as benchmark results. Use `--dry-run` to inspect the generated `sbatch` command without submitting it.

## Formal run

After preflight and runtime smoke pass, omit the validation flags:

```bash
bash src/commit_utils/slurm/submit.sh \
  --eval gsm8k \
  --agent codex_non_api \
  --model google/gemma-3-4b-pt \
  --hours 10 \
  --agent-config gpt-5.4 \
  --run-branch gangda_trial_0828 \
  --job-name gangda_trial_0828.ptb.example.gsm8k.formal.r1 \
  --judge-profile official \
  --experiment-name _gangda_trial_0828_example_gsm8k_formal_r1
```

Pass `--judge-profile claude` to override `.env` for one submission. Its
`judgement_claude_*` outputs are research artifacts and do not satisfy the
canonical verdict coverage required by `scripts/collect.py`; run the official
profile separately before canonical aggregation.

The default Slurm walltime is the agent budget plus the configured judge/evaluation overhead. Override it with `--walltime`. Scheduler logs are written to `logs/slurm/`; benchmark artifacts remain under `POST_TRAIN_BENCH_RESULTS_DIR`.

Each run uses `$SLURM_JOB_ID` as its run ID. Scratch is isolated under:

```text
${POST_TRAIN_BENCH_SLURM_SCRATCH_BASE}/${USER}/${SLURM_JOB_ID}
```

It is removed on exit unless `POST_TRAIN_BENCH_SLURM_KEEP_SCRATCH=1` is set.

Formal runs should also set:

```bash
POST_TRAIN_BENCH_REQUIRE_COMPLETE=1
POST_TRAIN_BENCH_EVAL_GPU_REAP=own
POST_TRAIN_BENCH_CODEX_JUDGE_AUTH_FILE="$HOME/.codex/auth.json"
POST_TRAIN_BENCH_JUDGE_LOCK_FILE="/shared/ptb-locks/official-judges.lock"
```

The completion contract requires `final_model/config.json`, `metrics.json`, and all four canonical official verdicts. Runtime provenance records the source commits, Slurm allocation/GPU UUID, agent model/context/effort/provider, CLI version, and agent/judge SIF digests. Shared ChatGPT subscription auth is protected by the cross-node lock for the entire official judge phase.

Claude Vertex experiment profiles freeze effort and requested context together. The reusable Opus
profiles are `claude_vertex_max`, `claude_vertex_xhigh`, and `claude_vertex_high` for 1M context,
plus `claude_vertex_max_200k` for native 200K context. `context_probe.sh` requires the provider's
resolved context window to equal the profile value exactly; a larger or smaller route is not an
acceptable substitute for the requested experiment setup.

Before a pilot, execute the ordered shared-node gates on an otherwise idle target node. The
required batch id plus the current top-level Git branch are embedded in every gate job name:

```bash
bash src/commit_utils/slurm/run_gates.sh g1 gpu-node-0 my-batch-id
bash src/commit_utils/slurm/run_gates.sh g2 gpu-node-0 my-batch-id
bash src/commit_utils/slurm/run_gates.sh g3 gpu-node-0 my-batch-id
```

G1 checks two simultaneous allocations, H100 visibility and device cgroups. G2 checks eight unique GPUs and a ninth pending job. G3 proves the evaluation cleanup function cannot kill a survivor on another allocation.

## Current limits

- Shared Unix accounts are not ownership evidence. Every Slurm submission must carry the current
  top-level branch in `JobName` and the result suffix; project launchers must additionally freeze
  spec/batch/cell/commit identity in a receipt. Never cancel jobs by username or a generic prefix.

- The adapter submits one benchmark cell at a time. A project-level experiment launcher may submit several cells; Slurm distributes them across eligible nodes.
- A full official run still requires the agent/eval/judge SIFs, task `test_data.json`, model access, agent credentials, ChatGPT judge auth, and a real-provider context-validation record when that gate is enabled.
- A Claude-profile run requires `opus_5.sif`, a working Claude Code binary, and either Vertex ADC (GCE metadata on this cluster) or a dedicated Claude judge OAuth token. Preflight records the CLI version and requested model alias/id but does not spend a model request to resolve the alias; the real judge invocation validates it.
- `POST_TRAIN_BENCH_JUDGE_AUTH_MODE=skip` is smoke-only. `apikey` remains unsupported; profile defaults are `chatgpt` for official, `vertex` for Claude when Vertex env is present, and otherwise `claude_oauth`.
- Do not run the worker directly on a shared/login node. Evaluation cleanup is restricted to the current Slurm GPU allocation and current Unix user; setting the reaper to a whole-node mode is unsupported.

## Extra sandbox binds

`POST_TRAIN_BENCH_EXTRA_BINDS="src:dst[:ro],src2:dst2"` adds one `--bind` per
comma-separated entry to the agent sandbox only, never to the evaluation or
judge containers. It is read on the host by `run_task.sh`; nothing about it
reaches the sandbox environment, and a missing source is an error before the
agent starts. Unset or empty, the default, changes nothing. A study uses it to
mount its own read-only code or data next to the task; `agents/claude_vertex_max_awm`
expects a checkout of `agentic-world-model` at `/home/ben/awm` this way.
