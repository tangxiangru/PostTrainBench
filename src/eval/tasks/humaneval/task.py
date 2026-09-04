"""Pinned HumanEval task; reference contents never enter operator diagnostics.

This module is shared by development and official evaluation. It deliberately
does not call the upstream task factory, which downloads an unpinned dataset.
The upstream mapper, instruction, solver and verification scorer are unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

DATASET = "openai/openai_humaneval"
REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
SOURCE_FILE = "openai_humaneval/test-00000-of-00001.parquet"
SOURCE_SHA256 = "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6"
SOURCE_BYTES = 83920
POPULATION = 164
ID_EPOCH_SHA256 = "43c813f8fd5eb6bf78c107f565a21dd556c8fd314e22f3cb5781f75464b8e25f"
FULL_SELECTION_SHA256 = "7aeeca4daf4f06680efd86ea58b1d2233a100c2912e2618285fde80072f5494d"
FORMAL_PUBLIC_RUNTIME_SHA256 = "2d7606e28dfd14ed33e5940ce62ca0961319239a85ac85cde1147e57b65be2a7"
EXECUTION_LIMITS = {
    "wall_seconds": 30,
    "startup_seconds": 10,
    "address_space_bytes": 1073741824,
    "cpu_seconds": 30,
    "scratch_bytes": 67108864,
    "scratch_inodes": 4096,
    "file_bytes": 16777216,
    "output_bytes": 10485760,
    "open_files": 128,
    "new_tasks": 32,
    "code_bytes": 122880,
}
UPSTREAM_SHA256 = "e7bd00d5002afa39f8c42ac5183bcb65feb7692f069eca41a32bad44024f6aaa"
COLUMNS = ("task_id", "prompt", "canonical_solution", "test", "entry_point")


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def dataset_path(hf_home=None):
    home = Path(hf_home or os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return (
        home / "hub" / "datasets--openai--openai_humaneval" / "snapshots" / REVISION / SOURCE_FILE
    )


def validate_rows(rows):
    """Check metadata and types, without executing or emitting any row contents."""
    if len(rows) != POPULATION:
        raise ValueError("HumanEval pinned population mismatch")
    if any(
        set(row) != set(COLUMNS)
        or any(type(row[key]) is not str or not row[key] for key in COLUMNS)
        for row in rows
    ):
        raise ValueError("HumanEval pinned row schema mismatch")
    ids = [row["task_id"] for row in rows]
    if len(set(ids)) != POPULATION:
        raise ValueError("HumanEval pinned IDs are not unique")
    if canonical_hash(sorted([["str", item, 1] for item in ids])) != ID_EPOCH_SHA256:
        raise ValueError("HumanEval pinned ID/epoch identity mismatch")


def load_rows(hf_home=None):
    """Read only the already-provisioned, hash-pinned public parquet. No network."""
    path = dataset_path(hf_home)
    raw = path.read_bytes()
    if len(raw) != SOURCE_BYTES or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("HumanEval pinned source bytes mismatch")
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Parse the same bytes that were checked, not a second pathname lookup.
    rows = pq.read_table(pa.BufferReader(raw), columns=list(COLUMNS)).to_pylist()
    validate_rows(rows)
    return rows


def upstream_module():
    module = importlib.import_module("inspect_evals.humaneval.humaneval")
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("HumanEval native mapper/scorer source differs from accepted pin")
    return module


def make_task(*, sandbox_name, hf_home=None, metadata=None):
    if sandbox_name != "ptb_python":
        raise ValueError("HumanEval requires the accepted isolated Python backend")
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.solver import generate

    native = upstream_module()
    samples = [native.record_to_sample()(row) for row in load_rows(hf_home)]
    return Task(
        name="humaneval",
        version="ptb-pinned-isolated-v1",
        dataset=MemoryDataset(
            samples, name=DATASET, location=f"hf://datasets/{DATASET}@{REVISION}/{SOURCE_FILE}"
        ),
        solver=generate(),
        scorer=native.verify(),
        epochs=1,
        sandbox=sandbox_name,
        metadata=metadata,
    )


def sample_binding(sample):
    """A content binding, not a copy of benchmark text for planner consumption."""
    return canonical_hash(
        {"input": sample.input, "target": sample.target, "metadata": sample.metadata}
    )


def selection_contract(samples):
    rows = [{"id": sample.id, "epoch": 1, "binding": sample_binding(sample)} for sample in samples]
    if not rows or any(type(row["id"]) is not str for row in rows):
        raise ValueError("HumanEval selection requires nonempty string IDs")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("HumanEval selection contains duplicate IDs")
    return {
        "schema_version": 1,
        "task": "humaneval",
        "dataset": DATASET,
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "population": POPULATION,
        "epochs": 1,
        "scorer": "verify",
        "samples": rows,
    }
