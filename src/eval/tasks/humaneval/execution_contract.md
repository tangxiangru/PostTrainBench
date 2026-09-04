# Pinned isolated HumanEval: construction checkpoint, 2026-09-04

Not scheduler/site admission. This common evaluator is for all three approved
Opus4.8 HumanEval arms, not an exp_protocol treatment or another repeat.
No real model or benchmark was evaluated during construction.

## Shared task contract

- Public dataset openai/openai_humaneval, revision
  7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544, test split. Local parquet bytes are
  hash-checked before parsing, never fetched implicitly. data_provenance.json
  binds164 rows/IDs and the contamination reference. Provision under
  `$HF_HOME/hub/datasets--openai--openai_humaneval/snapshots/<revision>/openai_humaneval/test-00000-of-00001.parquet`.
  Data/tests are evaluator/checker material, never training/watch-set examples.
- Exact pinned native instruction, mapper, generate, code extraction and verify.
  Explicit one epoch; actual metric is verification accuracy and sample SEM,
  not pass@5. Developer default150; formal164.
- Formal concurrency1, GPU memory fraction0.3; caps4000 for attempts1–4,
  3000 for5–7,2000 for8–9. No implicit additional attempts.
- New common execution contract: public CPython3.10/NumPy closure, private
  namespaces, resource limits and30s admitted timeout described in
  [the backend guide](../../ptb_python_sandbox.md). Not old local equivalence.

The launcher requires site variable POST_TRAIN_BENCH_PYTHON_BWRAP. Both
scientist/formal containers receive that public binary and the helper as explicit
read-only binds; runtime.py verifies the accepted binary SHA. No local fallback.
Only the frozen public closure is copied to separately mounted scratch, because
the actual Apptainer image overlay is unbindable; original identities are kept.

## Evidence and failures

Fresh official_eval/UUID directories retain pre-eval selection/request/model
fingerprints, full runtime/transport receipts and native Inspect JSON. Structured
execution reports enter sample store even on exception/cancellation: Inspect
otherwise drops custom exception.report. The native scorer is unchanged.

Admitted ordinary nonzero exits, including a generated nonexistent import, are
I. Untrusted stderr keywords cannot establish evaluator failure. Admitted wall
timeout is I. Proven bootstrap, policy, resource/supervision or cleanup failures
fail the evaluation and cannot publish a score; raw attempts remain available.

Publication requires all selected typed ID/epoch keys, bound input/target/test
metadata, one supported score/execution per sample, clean teardown, exact
code/runtime/limits, counts and recomputed accuracy/SEM. Model bytes must be
unchanged. An exact-byte private log snapshot is synchronized before a single
no-clobber metrics commit. Host validation checks the independently frozen full
selection, evaluator source, provenance, model files, runtime/receipt hashes and
limit profile, not embedded expected counts or cached hash labels alone.

Retries accept existing metrics only after this validation. Invalid existing
metrics are preserved for investigation, never overwritten. The completed-run
validator accepts an independent expected task; a missing task field cannot
select legacy numeric-only validation. Judges and receipt placement stay separate.

## Tested scope and remaining gates

Backend49 native host tests passed; exact helper
94f41494e049d6a1e2e40c2057f25b205356eff72c343b13ef47e9b550035da7.
Actual vllm_debug.sif CPU: invented C/I/import-I/timeout-I programs, native
sample-store/raw-log, snapshot publication and revalidation passed. Policy
rejection retained partial stdout/report and produced status:error, no metrics.
No benchmark loaded. Pure filesystem/evidence/host-provenance/model checks:
104 passed,22 native skips in the original environment.

Metadata-only data check confirmed164 rows and the full-selection hash; static
prompt/test imports were copy/math/random/string/typing, with no parse failures
or explicit dynamic imports. No canonical solution dependency analysis or
dataset-code execution occurred. These are not scientist sessions/results.

Before launch: actual scientist-image and compute-node execution permissions,
outer timeout and GPU-enabled containment; shared offline data/bwrap provisioning;
common frozen PTB source admission; expected-task validation in receipt-backed
AWM consumers and synthetic end-to-end tests; full manifest/site checks.
The approved40-cell budget, ownership, native two-node isolation and useful
eight-held floor are unchanged. This file grants no GPU or Slurm authority.
