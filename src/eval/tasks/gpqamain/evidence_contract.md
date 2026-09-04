# GPQA Main evidence contract

Construction only: real gated data is unavailable at this checkpoint. No real
`data_provenance.json` or `test_data.json` is supplied, and task admission must
remain disabled until legal source access, data identity and deployment checks
are complete. Synthetic fixtures in tests are not benchmark results.

The task directory/receipt identity is `gpqamain`; native Inspect task identity
is `gpqa_main`. Preserve native `multiple_choice(cot=True)` / `choice`, one epoch,
full formal population, six connections, 0.8 memory fraction and the existing
16k×4 / 12k×3 / 8k×2 formal retry schedule. No code-execution scorer is involved.

`task.py` accepts only locally cached bytes from the declared official dataset
revision, checked against an adjacent source-frozen profile. The profile needs
observed source size/SHA, population, typed Record IDs, ordered row hashes and
the contamination-reference hash. The actual CSV must still be compared with
the original loader, including inferred types/null handling, when authorized
data becomes available. A `train` split name is not training permission.

Production preserves native unseeded choice randomization over the whole
population before slicing a developer subset. A shadow dataset of unique
position markers exercises the original native shuffle and then maps positions
back to real options. Markers never reach the model; duplicate option strings
cannot lose their original correct/incorrect identity. Each presented sample
records the permutation and source-row hash. Integer seeds exist solely as a
CPU differential-test seam; the formal contract rejects them.

Each attempt freezes request, full selection, rendered COT prompts, source,
model, template, serving limits and runtime provenance before generation.
Successful publication requires complete native JSON; exact model-event inputs
and config; successful event output → retained completion/assistant message →
native single-answer parse → choice score; and recomputed accuracy/SEM/counts.
The host's stdlib single-answer projection is differentially checked against
the pinned native parser, not used to replace the runtime solver. In particular,
lowercase answer letters, multi-answer text and first-match semantics remain
unchanged. Empty output is an ordinary incorrect answer; missing output evidence
is not a valid zero. Only local log attachments may resolve content references.

Verified raw bytes are fsynced to an immutable no-clobber snapshot before metrics
are committed. Failed or conflicting attempts are retained. The completed-run
validator receives the expected task independently from AWM; no legacy fallback
may turn a rejected GPQA check into completion. Host revalidation also checks
the full frozen data/reference/model/source identities, not saved hash labels
alone. Development can fingerprint existing HF file symlinks; formal final-model
files remain strict, and an identifier resolves only to its provided frozen
base snapshot, never to an unpinned download.

CPU acceptance: both pinned opus_5 and vllm_debug images exercised invented rows
and MockLLM, 300 native shuffle seeds covering all 24 permutations, 672 parser
differentials, C/I/duplicate-text/lowercase/empty/structured-content outputs,
durable publication and the reviewer's output-tampering/deletion negatives.
No GPU, real model or benchmark question was used. Source hashes are checked by
`native_modules()`. Actual data, compute-node/GPU-enabled acceptance and common
source adoption are separate gates, not established by these CPU checks.
