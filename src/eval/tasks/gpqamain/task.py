"""Pinned local GPQA Main with native randomization and explicit option lineage.

No real data profile is supplied until the gated source is lawfully accessible.
Missing profile/data fails before task construction; there is no network fallback.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
import re
from pathlib import Path

DATASET = "Idavidrein/gpqa"
REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
CONFIG = "gpqa_main"
SPLIT = "train"  # Benchmark evaluation population, never training permission.
SOURCE_FILE = "gpqa_main.csv"
FIELDS = (
    "Record ID",
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)
NATIVE_HASHES = {
    "inspect_ai.dataset._dataset": "4bac5d707f05c07512a91d04f5c3a19f90b06375276d8f2713b621cba408b36a",
    "inspect_ai.solver._multiple_choice": "8f26c08c70c5d855df5b1d77e7f653db81b8a72f99a7a88e969fe78a19a8c4b1",
    "inspect_ai.scorer._choice": "242fc442a5d112d1500d87516b0ca204f29d275b4f0aa721cc3198e58fbc559b",
    "inspect_ai.model._model_output": "98041a8f343894eab9db745109799dfac5c9dfe5e2fd00136aa79db42a4fd6ae",
    "inspect_ai.model._chat_message": "7d02809ecf40060df2cd8a1ceafefed6df056247825b3f8eccb5e75065899561",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def source_record(row):
    return {
        "id": row["Record ID"],
        "input": row["Question"],
        "choices": [str(row[field]) for field in FIELDS[2:]],
        "target": "A",
    }


def rows_identity(rows):
    return {
        "rows": len(rows),
        "typed_id_epoch_sha256": digest(sorted([["str", row["Record ID"], 1] for row in rows])),
        "ordered_rows_sha256": digest([digest(source_record(row)) for row in rows]),
    }


def validate_profile(profile):
    if type(profile.get("schema_version")) is not int:
        raise ValueError("GPQA profile schema version must be an integer")
    fixed = {
        "schema_version": 1,
        "task": "gpqamain",
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "source_file": SOURCE_FILE,
    }
    if any(profile.get(key) != value for key, value in fixed.items()):
        raise ValueError("GPQA frozen data profile identity mismatch")
    for key in ("source_bytes", "rows"):
        if type(profile.get(key)) is not int or profile[key] <= 0:
            raise ValueError("GPQA data profile requires observed positive counts")
    for key in (
        "source_sha256",
        "typed_id_epoch_sha256",
        "ordered_rows_sha256",
        "reference_sha256",
    ):
        if not isinstance(profile.get(key), str) or not re.fullmatch("[0-9a-f]{64}", profile[key]):
            raise ValueError("GPQA data profile requires frozen content hashes")


def load_profile():
    # Adjacent, source-frozen metadata only. No arbitrary user-path override.
    raw = Path(__file__).with_name("data_provenance.json").read_bytes()

    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate GPQA profile key")
            result[key] = value
        return result

    profile = json.loads(raw, object_pairs_hook=unique_pairs)
    validate_profile(profile)
    return profile, hashlib.sha256(raw).hexdigest()


def parse_csv(raw, profile):
    validate_profile(profile)
    if (
        len(raw) != profile["source_bytes"]
        or hashlib.sha256(raw).hexdigest() != profile["source_sha256"]
    ):
        raise ValueError("GPQA source bytes differ from the frozen profile")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if len(headers) != len(set(headers)) or not set(FIELDS).issubset(headers):
        raise ValueError("GPQA CSV header mismatch or duplicate column")
    rows = []
    for raw_row in reader:
        if None in raw_row or any(
            not isinstance(raw_row.get(key), str) or not raw_row[key].strip() for key in FIELDS
        ):
            raise ValueError("GPQA CSV row has absent/empty required fields")
        rows.append({key: raw_row[key] for key in FIELDS})
    if len({row["Record ID"] for row in rows}) != len(rows):
        raise ValueError("GPQA CSV IDs are not unique")
    observed = rows_identity(rows)
    if any(observed[key] != profile[key] for key in observed):
        raise ValueError("GPQA CSV population or ordered identity mismatch")
    return rows


def load_rows(profile, hf_home=None):
    home = Path(hf_home or os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    path = home / "hub/datasets--Idavidrein--gpqa/snapshots" / REVISION / SOURCE_FILE
    return parse_csv(path.read_bytes(), profile)


def native_modules():
    modules = {name: importlib.import_module(name) for name in NATIVE_HASHES}
    if any(
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != NATIVE_HASHES[name]
        for name, module in modules.items()
    ):
        raise ValueError("GPQA native dataset/solver/scorer source differs from accepted pin")
    return modules


def prepare_samples(rows, *, seed=None):
    """Native position shuffling on unique markers; markers never reach the model.

    The native algorithm permutes positions, independent of option values. A
    shadow dataset lets us preserve the exact original positions even when two
    answer texts are identical. Production keeps native Random(None), one RNG
    per whole dataset. Seed is only a deterministic CPU-test seam, not a CLI flag.
    """
    if seed is not None and type(seed) is not int:
        raise ValueError("GPQA shuffle seed must be None or an explicit test integer")
    native_modules()
    from inspect_ai.dataset import MemoryDataset, Sample

    shadows = MemoryDataset(
        [
            Sample(id=row["Record ID"], input="", target="A", choices=["0", "1", "2", "3"])
            for row in rows
        ]
    )
    shadows.shuffle_choices(seed=seed)
    samples = []
    for row, shadow in zip(rows, shadows):
        original = source_record(row)
        positions = [int(value) for value in shadow.choices]
        samples.append(
            Sample(
                id=original["id"],
                input=original["input"],
                choices=[original["choices"][position] for position in positions],
                target=shadow.target,
                metadata={
                    "gpqa_choice_positions": positions,
                    "gpqa_source_row_sha256": digest(original),
                    "gpqa_shuffle_seed": seed,
                },
            )
        )
    return samples


def make_task():
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.scorer import choice
    from inspect_ai.solver import multiple_choice

    profile, profile_sha = load_profile()
    samples = prepare_samples(load_rows(profile))
    task = Task(
        name="gpqa_main",
        version="ptb-pinned-presentation-v1",
        dataset=MemoryDataset(
            samples, name=DATASET, location=f"hf://datasets/{DATASET}@{REVISION}/{SOURCE_FILE}"
        ),
        solver=multiple_choice(cot=True),
        scorer=choice(),
        epochs=1,
    )
    return task, profile, profile_sha


def selection_contract(samples, profile, profile_sha):
    modules = native_modules()
    from inspect_ai.solver._task_state import Choices

    native_solver = modules["inspect_ai.solver._multiple_choice"]
    seeds = {sample.metadata["gpqa_shuffle_seed"] for sample in samples}
    if not samples or len(seeds) != 1:
        raise ValueError("GPQA selection needs one explicit shared randomization contract")
    entries = [
        {
            "id": sample.id,
            "epoch": 1,
            "binding": digest(
                {
                    "input": sample.input,
                    "choices": sample.choices,
                    "target": sample.target,
                    "metadata": sample.metadata,
                }
            ),
            "source_row_sha256": sample.metadata["gpqa_source_row_sha256"],
            "rendered_prompt_sha256": hashlib.sha256(
                native_solver.prompt(
                    sample.input, Choices(sample.choices), native_solver.SINGLE_ANSWER_TEMPLATE_COT
                ).encode()
            ).hexdigest(),
        }
        for sample in samples
    ]
    return {
        "schema_version": 1,
        "task": "gpqamain",
        "native_task": "gpqa_main",
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": REVISION,
        "profile_sha256": profile_sha,
        "source_sha256": profile["source_sha256"],
        "population": profile["rows"],
        "epochs": 1,
        "scorer": "choice",
        "randomization": {"algorithm": "native-position-shuffle-v1", "seed": seeds.pop()},
        "samples": entries,
    }
