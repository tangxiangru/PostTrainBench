# HumanEval Python execution boundary

Construction component, **not full-task/site admission**. This module does not
modify HumanEval `find_code`, `verify`, its 30-second timeout, dataset or metric.
The integration owner must import/register this backend and select its name
instead of `local`, freeze the common runtime for every arm, and retain failures.
There is no local/Docker fallback and no network/package installation.

## Callable interface

Run this trusted setup in the evaluator, never in generated Python:

```python
from dataclasses import asdict
from ptb_python_sandbox import build_runtime, execute_python, register_inspect

runtime = build_runtime(
    bwrap="/operator/frozen/bwrap",
    python="/usr/bin/python3.10",
    stdlib="/usr/lib/python3.10",
    library_dirs=["/lib/x86_64-linux-gnu", "/lib64"],
    public_packages=[
        "/usr/local/lib/python3.10/dist-packages/numpy",
        "/usr/local/lib/python3.10/dist-packages/numpy.libs",
        "/usr/local/lib/python3.10/dist-packages/numpy-2.2.6.dist-info",
    ],
)
# Persist asdict(runtime) and runtime.identity BEFORE relying on an evaluation.
backend = register_inspect(runtime)  # registers "ptb_python", never overwrites local
# Existing task construction then uses sandbox="ptb_python".
# Direct synthetic acceptance, without constructing a Task/model/dataset:
result = execute_python("print(sum([3, 5, 7]))", runtime)
assert result.success
print(result.evidence)
```

Registration deliberately **is not idempotent**: reuse the returned class.
Names must be unqualified lowercase identifiers; `local`, `docker`, `default`
are reserved. Every existing name/foreign package alias is rejected, including
an identical runtime/limits registration. This helper serializes its own
concurrent registrations. Register during evaluator setup; unrelated code can
still mutate Inspect's global registry directly, outside this helper's contract.

The module must be importable in the actual evaluator process. The registered
class implements the pinned Inspect SandboxEnvironment, sample lifecycle and
`exec(["python", "-c", code], timeout=30)`. Arbitrary shell commands, injected
environment/user/cwd/input and persistent cross-exec sample files are explicitly
unsupported. Normal multiple functions, stdlib algorithms, file operations in
the private work area, NumPy, threads, fork and Python subprocesses are not AST
restricted. Each execution gets a new filesystem; there is no code rewrite.
The class advertises default sandbox concurrency one. Each invocation owns its
supervisor worker independently of AnyIO's shared thread pool; sample cleanup
cancels active invocations and closed environments refuse further execution.
`environment.execution_evidence` retains per-call reports for the integration
owner to archive; Inspect's ordinary ExecResult does not automatically archive
that extra field. Direct API results/exceptions also expose their reports.

## Runtime trust and compatibility

Only individually enumerated public files are read-only mounted. `readelf`
parses the interpreter/extensions and resolves their ELF dependencies; it does
not execute inspected ELF files. Stdlib excludes site/dist-packages and bytecode
caches. Explicit package directories (including wheel `.libs` and metadata)
are frozen separately. Whole site-packages, Inspect benchmark packages and
startup `.pth`/sitecustomize hooks cannot be supplied as package roots.

The operator must audit the selected assets as public and keep the runtime and
this trusted module immutable to the scientist. Hashes do not sanitize a
credential-bearing arbitrary file or defend concurrent hostile host mutation.
The runtime manifest includes every source/destination/SHA256 and bwrap hash;
execution evidence adds the backend/code hashes and actual limits. The adapter
checks the three pinned Inspect interface-source hashes before registration.

The initial host-test interpreter is CPython 3.10.12, SHA256
`94df6ff4b32aafdba90e38c0aed70880fe44ee4cf88ecdf411641158f120c1af`,
identical to the existing opus_5 extraction. Tested bwrap 0.6.1 SHA256
`d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca`;
host libseccomp SHA256
`0a10b36f83b58889352ef9de563fddeb80ba649d6a7108f32f31cfb08693e2a3`.
The public NumPy test uses actual extracted 2.2.6 files, not an import mock.
That mixed host/public-library runtime alone did not establish actual-image
compatibility; the separate nested-image acceptance below addresses that path.

## Nested Apptainer public-file transport

In the actual `vllm_debug.sif --containall --writable-tmpfs` route, the image root
is `fuse.fuse-overlayfs` marked **unbindable**. Direct nested file binds returned
EINVAL, independent of destination spelling. A byte-identical libbz2 copy on
the separately bound task ext4 filesystem bound successfully with every
isolation flag unchanged. This is a source-mount issue, not a Python ABI failure.

Use explicit materialization once during evaluator setup, before registration:

```python
from pathlib import Path
import tempfile
from ptb_python_sandbox import materialize_runtime

parent = Path(tempfile.mkdtemp(prefix="public-runtime-", dir="/home/ben"))
runtime = materialize_runtime(
    runtime, directory=parent / "transport",
    source_image_reference="/rmeng_data/robtang/ptb-containers/vllm_debug.sif",
    source_image_sha256="72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8",
)
# register_inspect(runtime) once, as above
```

The parent directory must be an independently bindable ordinary filesystem or
tmpfs, not the same unbindable image overlay. Only the frozen public file closure
is copied; all inner mounts remain individual read-only file binds. No host/root,
cache directory or complete image tree is exposed. Copy bytes are checked against
both source and frozen SHA256; original source modes are recorded and transport
modes narrowed to owner read, plus execute permission only for originally executable files. Source/copy
hashes, modes, canonical paths, unexpected symlinks, receipt identity and complete
file coverage are rechecked before every execution. The original source image
files must remain available in the evaluator namespace; this is not a detached
cache that silently substitutes for a missing/different image.

Existing destinations are never overwritten/merged/reused automatically. Failed
partial copies remain with a failed receipt for inspection, never a complete
transport claim. There is no recursive deletion. The transport's Runtime identity
is distinct; `materialization.source_runtime_sha256` preserves the original
Runtime identity. Use the original identity plus frozen image/helper provenance
for cross-session comparisons, not the session-specific copy location. The
image reference/SHA is explicitly **caller provenance, not rehashed here**;
per-file source bytes are independently revalidated. Archive `materialization.json`
and the materialized Runtime manifest before task scratch disappears: per-exec
evidence contains references, not a full copy of that receipt.

## Isolation and limits

Linux x86_64 only. Mandatory user/mount/PID/network/IPC/UTS namespaces; no shared
root bind, repository/results/cache/weights, parent environment, GPU devices or
host process namespace. Only null/zero/random/urandom devices are exposed.
Private POSIX shared memory lives within the same bounded work filesystem.
Network namespace has isolated loopback only, no host listener or external route.

Trusted bootstrap mounts a **64 MiB / 4096 inode** private tmpfs (also covers
`/tmp`, home and `/dev/shm`), then removes all live/bounding capabilities, sets
no-new-privileges and installs seccomp before admitting generated bytes.
The control socket and notification listener are closed in the generated
process. An external trusted supervisor owns the listener. Dangerous namespace,
mount/ptrace/kernel-key/BPF/io_uring/filter-replacement and unbounded SysV-IPC or
memfd paths are denied. This is shared-kernel isolation, not a VM/kernel-exploit
defense or adversarial same-user-host security boundary.

Default hard limits per process: 1 GiB address space, 30 CPU seconds, 16 MiB per
file, 128 FDs, zero core dump; 10 MiB captured bytes per output stream; 120 KiB
code argv. Wall timeout is 30 seconds **after bootstrap admission**; startup has
its own 10-second deadline. There are no execution retries.
RLIMIT_CPU includes accumulated bootstrap CPU; an excessively small configured
budget can therefore cause an infrastructure rejection before admission.

The supervisor permits at most 32 lifetime task-creation attempts across all
descendants/threads; failed attempts still consume permission. This is not 32
reusable concurrency slots and does not rely on a shared-host-UID NPROC limit.
`clone3` receives ENOSYS for libc fallback; classic clone's namespace flags are
scalar-checked. Later seccomp/prctl filter replacement is denied. No dereferenced
user pointer is treated as a safe syscall argument by the notification manager.
Threads share AS; a conservative userspace virtual-memory ceiling is 33 GiB
for 33 separate processes, plus bounded scratch. **This is not a cgroup aggregate
RSS/kernel-memory guarantee.** A delegated cgroup memory/pids/cpu contract is
still appropriate for stronger per-verification accounting at deployment.
Numerical-library native thread defaults are explicitly capped at one via
OPENBLAS/OMP/MKL environment settings; this is a shared new execution contract.

The generated interpreter is namespace PID 1. Its exit kills every descendant,
including setsid/double-fork descendants. Cleanup uses its host pidfd, waits for
the bubblewrap monitor, and reaps adopted children in a dedicated subreaper.
Cancellation waits for cleanup. Unconfirmed cleanup is infrastructure failure,
not success; uninterruptible kernel failure or supervisor SIGKILL cannot promise
a durable final report. Namespace/die-with-parent containment still applies.

## Outcomes and remaining admission work

- Ordinary admitted exit zero/nonzero becomes Inspect's unchanged ExecResult.
- Admitted wall timeout raises TimeoutError: the original scorer maps it to I.
- Missing/changed runtime, denied namespace setup, bootstrap, supervision or
  cleanup failure raises SandboxInfrastructureError, **not an incorrect answer**.
- Independently observed forbidden syscall/task-budget events, output bounds and
  the resource-or-signal termination path remain SandboxDependencyError.
- For ordinary admitted exits, `ModuleNotFoundError`, `ImportError`, `MemoryError`
  and matching OSError text are only `stderr_diagnostics`, explicitly sourced
  from untrusted program output with `affects_outcome: false`. Missing imports,
  forged error text and ordinary ValueError at rc=1 return ExecResult(False,1,...)
  and the original scorer returns I. Text cannot promote a program failure into
  an evaluator failure or alter an otherwise successful exit. Kernel AS/tmpfs/
  file limits remain enforced; a Python-handled limit error followed by ordinary
  rc=1 follows the same rule unless independently observed supervisory evidence
  requires a different outcome. Actual dependency-profile completeness remains
  a separate pre-deployment validation requirement, not an inference from text.
  Exit zero also cannot prove tests genuinely executed (unchanged scorer limit).

Every available execution report now records `outcome`, `error_category`, the
original `error`, and bounded `output.stdout`/`output.stderr`: UTF-8 display text,
byte-exact `bytes_base64`, retained/observed byte counts, EOF, truncation,
capture coverage and decoding errors. The raw prefix never exceeds the chosen
limit (at most Inspect's 10 MiB per stream); base64/display transport adds bounded
encoding overhead. Output remaining in pipes is drained after teardown with a
bounded nonblocking drain. `complete` describes captured bytes through observed
EOF, **not successful program completion**. Missing helper evidence is explicitly
`unavailable`, not an invented empty/complete capture. Invalid UTF-8 still fails
closed while retaining exact bytes; it is not silently repaired into a score.
Cancellation retains the supervisor's interruption under `supervisor_termination`.
Secondary cleanup failures retain the primary error/category and cleanup errors;
unconfirmed cleanup raises infrastructure failure, never a timeout score. Abrupt
helper death can still prevent final report recovery, and is labelled unavailable.

Do not turn these exceptions into zero scores or silently omit failed samples.
Registering the backend does not establish old-environment dependency parity:
add the exact necessary public packages/assets, freeze their transitive closure,
and run harmless native imports/functional tests. Missing arbitrary public
packages, timezone/certificate/system tools and unsupported dynamic dlopen paths
remain a visible compatibility gap—not a claim all HumanEval programs are ready.

Before release, independently test the remaining compute-node/deployment
nested namespace/seccomp/pidfd permissions, public dependency profile, startup
cost, concurrent evaluator load, cancellation and outer timeout behavior; freeze
all source/image/profile identities across arms. No benchmark data or model is
needed for those mechanism tests. Only then integrate task-level admission.

## CPU reproduction

From this PTB checkout, using the already installed/extracted runtime:

```bash
PTB_SANDBOX_NATIVE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3.10 -c '
import sys
sys.path.append("/home/robtang_google_com/gangda_workspace/agentic-world-model-exp-protocol-operator/.venv/lib/python3.13/site-packages")
import pytest
raise SystemExit(pytest.main(["src/eval/test_ptb_python_sandbox.py", "-vv", "-p", "no:cacheprovider"]))
'
```

Native tests require explicit execution permission; the desktop sandbox blocks
its local socket/netlink setup. Report that as infrastructure denial; do not
remove namespace flags to make it pass. Without the opt-in, native tests skip.
Fixtures are invented arithmetic/functions, benign dependency calls and bounded
processes only. The scorer test imports original library code and uses invented
prompt/test strings; it never constructs or loads the benchmark dataset.

Pre-review 2026-09-04 verification: **26 passed in 148.34 s** (7 dependency-free checks,
19 real namespace/native tests). The original environment without native opt-in
passes 7 and explicitly skips 19; Ruff passes. One representative 1759-file,
122,277,089-byte public manifest built in 1.314 s; a trivial admitted call took
3.972 s total, including 3.677 s namespace/bootstrap. These are head-node CPU
measurements, not compute-node throughput. The tested runtime identity is
`dd627a7b6d92420a2b36094b498400384379daa3ef3fb196362cf393d4fd435c`.

Reviewed predecessor identities (not the corrected source):

```text
backend b6bcd61075a782e13083a70fe2c9dc6ef56d4178303b698b2fb537363aef9043
tests   f68bfc565b4d68c6ac764e04f61174302e1cf526b371b850f86e8bc1d9203d18
guide   6c80d1740c878aa0f7fe74efcf94946e3bc34f468239a5691d9418aabe4734b0
```

Independent review demonstrated reserved/duplicate registration overwrites and
lost exception output in that predecessor. The fixes and additional regression
coverage are changes to the construction component, not task/site admission.

First post-review source regression: **38 passed in 166.24 s**: 17
dependency-free checks and 21 runtime-dependent cases, including real Inspect
registry-only verification and one explicitly fault-injected pidfd-cleanup case
around real isolated execution. Missing-dependency, policy-denial, timeout,
cancellation, truncation and invalid-UTF8 cases assert retained bytes/diagnostics;
the native adapter retains the report and the unchanged scorer's real 30-second
timeout still maps to I. Without native opt-in: 17 passed, 21 skipped; Ruff passes.

Second reviewed predecessor (registration/output fixes present, but stderr-only
classification still changed the original score semantics):

```text
backend a9195aad0420e261263694d871888dc8c0889449491c469ca5fce8ba5502941e
tests   14f6f1c90b381ab299cc7a01719619cc79e6c942fc27b0a1073ea24c55542972
guide   6b7b0a916ead19f39d9e1e17d6517014501535291b3d0019889712b4bd6f63e6
```

That predecessor's missing-import exception test encoded the mapping defect;
its passing count did not establish scorer equivalence. The current correction
removes only stderr-based promotion and adds native original-scorer missing
import/forged stderr/ValueError I contrasts plus a correct-program C contrast.

Mapping-corrected regression: **38 passed in 179.43 s** (17 dependency-free,
21 runtime-dependent cases, including the labelled cleanup fault injection).
The existing scorer test now exercises all three new I contrasts and the C
control with actual isolated Python; retained partial output and diagnostics
are asserted. Original-environment checks: 17 passed, 21 skipped; Ruff passes.
The real 30-second timeout and independent isolation/cleanup protections still
pass. These results do not certify full dependency parity or compute-node/image
deployment, and do not retrospectively validate the predecessor's score mapping.

Pre-transport identities (host tests valid, actual-image direct binds failed):

```text
backend 9ad4c924faf77b8bf40f0cfe0060a9b2177b208beaf971156b2c410b5324ae41
tests   231d3ee3ca1b789a5ac9140bceb6ac8bae6ac92cbcb6f0e22925b41bb5f49984
guide   505de38268cd3051b42fd8c4ec5403aa924623900f01579725bc82dbc8aae20a
```

Transport-corrected regression: **49 passed in 185.33 s** (27 dependency-free
filesystem/logic checks and 22 runtime-dependent cases). Without native opt-in:
27 passed, 22 skipped. Filesystem unit fixtures are explicitly non-executable;
they do not stand in for the real native/image execution tests. Ruff passes.

Separately, actual-image acceptance recorded 13 successful checks/expected
rejections in `/tmp/ptb-nested-mount.n7oNq5y5/materialized-observations.json`:
median, NumPy, private visibility, concurrency, missing-import rc1, timeout/reaping,
policy denial, original scorer C/I/I, altered copies/symlinks/source-reference
substitution, and non-overwrite. The image's Python is genuinely different from
the host binary: `7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86`.
Its original public Runtime identity remains
`01d60204dd8b6023e9177c3c27f0cf0ac882ad1306073f67d17695f92c652178`.
The final 1754-file/122,158,273-byte transport took 4.496 s including verification
(copy 3.504 s); individual executions took about 3.8–4.7 s while the host suite
ran concurrently. This establishes this head-node CPU image route, not compute-
node permissions, GPU-enabled outer containment, full dependency parity or model
evaluation. Exact repro commands and minimal mount evidence are retained in
`/tmp/ptb-nested-mount.n7oNq5y5/README.md`; archive them with the source/receipts.
