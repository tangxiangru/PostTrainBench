"""Trusted evaluator-side wiring for the isolated Python backend.

The helper and bubblewrap are explicit read-only outer-container binds. No
fallback to local execution is permitted. Generated code cannot access this
module, its logs, model, source dataset, credentials, or the trusted evaluator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

BWRAP_SHA256 = "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca"
HELPER_PATH = Path("/opt/ptb-python/ptb_python_sandbox.py")


def register_backend(evidence_dir, *, image_reference, image_sha256):
    from inspect_ai.util import store

    helper_bytes = HELPER_PATH.read_bytes()
    spec = importlib.util.spec_from_file_location("ptb_python_execution_runtime", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    bwrap = "/opt/ptb-python/bwrap"
    if hashlib.sha256(Path(bwrap).read_bytes()).hexdigest() != BWRAP_SHA256:
        raise ValueError("HumanEval bubblewrap differs from the accepted binary")
    source = helper.build_runtime(
        bwrap=bwrap,
        python="/usr/bin/python3.10",
        stdlib="/usr/lib/python3.10",
        library_dirs=["/lib/x86_64-linux-gnu", "/lib64"],
        public_packages=[
            "/usr/local/lib/python3.10/dist-packages/numpy",
            "/usr/local/lib/python3.10/dist-packages/numpy.libs",
            "/usr/local/lib/python3.10/dist-packages/numpy-2.2.6.dist-info",
        ],
    )
    parent = Path(tempfile.mkdtemp(prefix="ptb-public-runtime-"))
    runtime = helper.materialize_runtime(
        source,
        directory=parent / "public",
        source_image_reference=image_reference,
        source_image_sha256=image_sha256,
    )
    record = {
        "schema_version": 1,
        "helper_sha256": hashlib.sha256(helper_bytes).hexdigest(),
        "source_runtime_sha256": source.identity,
        "runtime_sha256": runtime.identity,
        "runtime": asdict(runtime),
        "limits": asdict(helper.DEFAULT_LIMITS),
        "materialization": json.loads(Path(runtime.materialization["receipt"]).read_text()),
    }
    with (Path(evidence_dir) / "python-runtime.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    cls = helper.register_inspect(runtime, name="ptb_python")
    original = cls.exec

    async def recording_exec(self, *args, **kwargs):
        before = len(self.execution_evidence)
        try:
            return await original(self, *args, **kwargs)
        finally:
            # Runs after the helper's shielded cleanup, including error/cancel.
            # Inspect does not serialize arbitrary exception.report attributes.
            observations = self.execution_evidence[before:]
            previous = store().get("ptb_python_execution", [])
            store().set("ptb_python_execution", previous + observations)

    # This freshly registered class belongs to this evaluator; no foreign or
    # builtin registry entry is replaced and native verify() is unchanged.
    cls.exec = recording_exec
    return record
