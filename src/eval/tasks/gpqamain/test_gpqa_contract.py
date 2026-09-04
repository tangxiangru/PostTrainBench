"""Invented CSV/choice fixtures; never claim them as GPQA benchmark data."""

import csv
import hashlib
import importlib.util
import io
import json
import os
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        "test_gpqa_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t, e = load("task"), load("official_evidence")


def fake_rows():
    return [
        {
            "Record ID": f"invented-{i}",
            "Question": f"SYNTHETIC question {i}, no benchmark item",
            "Correct Answer": "duplicate" if i == 0 else "right",
            "Incorrect Answer 1": "duplicate" if i == 0 else "wrong-one",
            "Incorrect Answer 2": "wrong-two",
            "Incorrect Answer 3": "wrong-three",
        }
        for i in range(3)
    ]


def csv_and_profile(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=t.FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    profile = {
        "schema_version": 1,
        "task": "gpqamain",
        "dataset": t.DATASET,
        "config": t.CONFIG,
        "split": t.SPLIT,
        "revision": t.REVISION,
        "source_file": t.SOURCE_FILE,
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_sha256": "0" * 64,
        **t.rows_identity(rows),
    }
    return raw, profile


def test_csv_roundtrip_preserves_duplicate_answer_text():
    rows = fake_rows()
    raw, profile = csv_and_profile(rows)
    assert t.parse_csv(raw, profile) == rows


@pytest.mark.parametrize(
    "change", ["bytes", "population", "ids", "order", "revision", "bool", "missing", "duplicate-id"]
)
def test_csv_rejects_incomplete_or_wrong_identity(change):
    rows = fake_rows()
    if change == "missing":
        rows[0]["Correct Answer"] = ""
    if change == "duplicate-id":
        rows[1]["Record ID"] = rows[0]["Record ID"]
    raw, profile = csv_and_profile(rows)
    if change == "bytes":
        raw += b" "
    elif change == "population":
        profile["rows"] += 1
    elif change == "ids":
        profile["typed_id_epoch_sha256"] = "a" * 64
    elif change == "order":
        profile["ordered_rows_sha256"] = "a" * 64
    elif change == "revision":
        profile["revision"] = "wrong"
    elif change == "bool":
        profile["rows"] = True
    with pytest.raises(ValueError):
        t.parse_csv(raw, profile)


def test_missing_profile_fails_before_task_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "__file__", str(tmp_path / "task.py"))
    with pytest.raises(FileNotFoundError):
        t.load_profile()


@pytest.fixture
def fixture():
    rows = fake_rows()
    _, profile = csv_and_profile(rows)
    samples, entries = [], []
    for index, row in enumerate(rows):
        source = t.source_record(row)
        positions = [1, 0, 3, 2]
        content = {
            "input": row["Question"],
            "choices": [source["choices"][i] for i in positions],
            "target": "B",
            "metadata": {
                "gpqa_choice_positions": positions,
                "gpqa_source_row_sha256": t.digest(source),
                "gpqa_shuffle_seed": None,
            },
        }
        rendered = f"invented rendered prompt {index}"
        answer = "B" if index == 0 else "A" if index == 1 else ""
        completion = "ANSWER: " + answer if answer else "No formatted answer."
        assistant = {"role": "assistant", "content": completion}
        output = {
            "model": "synthetic-output-model",
            "choices": [{"message": deepcopy(assistant), "stop_reason": "stop"}],
            "completion": completion,
        }
        samples.append(
            {
                "id": row["Record ID"],
                "epoch": 1,
                **content,
                "messages": [{"role": "user", "content": rendered}, deepcopy(assistant)],
                "output": deepcopy(output),
                "events": [
                    {
                        "event": "model",
                        "model": "mockllm/model",
                        "config": {"max_tokens": 37, "max_connections": 6},
                        "input": [{"role": "user", "content": rendered}],
                        "output": deepcopy(output),
                    }
                ],
                "scores": {
                    "choice": {
                        "answer": answer,
                        "value": "C" if index == 0 else "I",
                        "explanation": completion,
                    }
                },
            }
        )
        entries.append(
            {
                "id": row["Record ID"],
                "epoch": 1,
                "binding": t.digest(content),
                "source_row_sha256": t.digest(source),
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        )
    contract = {
        "schema_version": 1,
        "task": "gpqamain",
        "native_task": "gpqa_main",
        "dataset": t.DATASET,
        "config": t.CONFIG,
        "split": t.SPLIT,
        "revision": t.REVISION,
        "profile_sha256": "fake-profile",
        "source_sha256": profile["source_sha256"],
        "population": 3,
        "epochs": 1,
        "scorer": "choice",
        "samples": entries,
        "randomization": {"algorithm": "native-position-shuffle-v1", "seed": None},
    }
    invocation = {
        "attempt_id": "invented-attempt",
        "model": "mockllm/model",
        "max_tokens": 37,
        "max_connections": 6,
    }
    log = {
        "status": "success",
        "eval": {
            "task": "gpqa_main",
            "model": "mockllm/model",
            "config": {"epochs": 1},
            "model_generate_config": {"max_tokens": 37, "max_connections": 6},
            "metadata": {"ptb_invocation": invocation, "ptb_selection_sha256": t.digest(contract)},
        },
        "samples": samples,
        "results": {
            "total_samples": 3,
            "completed_samples": 3,
            "scores": [
                {
                    "name": "choice",
                    "scorer": "choice",
                    "scored_samples": 3,
                    "unscored_samples": 0,
                    "metrics": {"accuracy": {"value": 1 / 3}, "stderr": {"value": 1 / 3}},
                }
            ],
        },
    }
    return log, contract, invocation, profile


def test_choice_counts_and_full_source_coverage(fixture):
    log, contract, invocation, profile = fixture
    assert e.validate_log(log, contract, invocation)["correct"] == 1
    e.validate_full_contract(contract, profile, "fake-profile")
    contract["samples"].pop()
    with pytest.raises(ValueError):
        e.validate_full_contract(contract, profile, "fake-profile")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.update(status="error"),
        lambda x: x["samples"].pop(),
        lambda x: x["samples"][0].update(epoch=True),
        lambda x: x["samples"][0].update(id="unmatched"),
        lambda x: x["samples"][1].update(id="invented-0"),
        lambda x: x["samples"][0].update(error={"message": "fake error"}),
        lambda x: x["samples"][0]["messages"][0].update(content="different prompt"),
        lambda x: x["samples"][0].update(events=[]),
        lambda x: x["samples"][0]["events"][0]["input"][0].update(
            content="different actual prompt"
        ),
        lambda x: x["samples"][0]["events"][0]["config"].update(max_tokens=99),
        lambda x: x["samples"][0]["events"][0].update(model="wrong-model"),
        lambda x: x["samples"][0].pop("output"),
        lambda x: x["samples"][0]["events"][0].pop("output"),
        lambda x: x["samples"][0]["output"].update(completion="ANSWER: A"),
        lambda x: x["samples"][0]["events"][0]["output"].update(completion="ANSWER: A"),
        lambda x: x["samples"][0]["messages"][1].update(content="ANSWER: A"),
        lambda x: x["samples"][0]["scores"]["choice"].update(explanation="different output"),
        lambda x: x["samples"][0]["choices"].reverse(),
        lambda x: x["samples"][0].update(target="A"),
        lambda x: x["samples"][0]["metadata"].update(gpqa_choice_positions=[0, 1, 2, 3]),
        lambda x: x["samples"][0]["scores"]["choice"].update(answer="A"),
        lambda x: x["samples"][0]["scores"]["choice"].update(value=True),
        lambda x: x["samples"][0]["scores"]["choice"].update(answer="B, C"),
        lambda x: x["eval"].update(task="gpqamain"),
        lambda x: x["eval"]["model_generate_config"].update(max_tokens=38),
        lambda x: x["eval"]["model_generate_config"].update(max_connections=1),
        lambda x: x["results"].update(completed_samples=2),
        lambda x: x["results"]["scores"][0].update(unscored_samples=1),
        lambda x: x["results"]["scores"][0].update(scorer="verify"),
        lambda x: x["results"]["scores"][0]["metrics"]["accuracy"].update(value=1),
        lambda x: x["results"]["scores"][0]["metrics"]["stderr"].update(value=float("nan")),
    ],
)
def test_bad_or_partial_logs_do_not_publish(fixture, mutate):
    log, contract, invocation, _ = fixture
    mutate(log)
    with pytest.raises(e.EvidenceError):
        e.validate_log(log, contract, invocation)


def test_duplicate_choice_text_does_not_confuse_original_correct_position(fixture):
    log, contract, invocation, _ = fixture
    # Move the purported target to the other identical string, even rebinding
    # its content. Option lineage still says the original correct index is at B.
    log["samples"][0]["target"] = "A"
    content = {key: log["samples"][0][key] for key in ("input", "choices", "target", "metadata")}
    contract["samples"][0]["binding"] = t.digest(content)
    log["eval"]["metadata"]["ptb_selection_sha256"] = t.digest(contract)
    with pytest.raises(e.EvidenceError, match="original correct option"):
        e.validate_log(log, contract, invocation)


def test_coherent_output_tampering_cannot_retain_old_score(fixture):
    log, contract, invocation, _ = fixture
    sample = log["samples"][0]
    for output in (sample["output"], sample["events"][0]["output"]):
        output["completion"] = "ANSWER: A"
        output["choices"][0]["message"]["content"] = "ANSWER: A"
    sample["messages"][1]["content"] = "ANSWER: A"
    sample["scores"]["choice"]["explanation"] = "ANSWER: A"
    with pytest.raises(e.EvidenceError, match="native completion parse"):
        e.validate_log(log, contract, invocation)


def test_log_attachment_resolution_is_local_and_score_bound(fixture):
    log, contract, invocation, _ = fixture
    sample = log["samples"][0]
    prompt = sample["events"][0]["input"][0]
    sample["attachments"] = {
        "prompt": prompt["content"],
        "completion": sample["output"]["completion"],
    }
    prompt["content"] = "attachment://prompt"
    for output in (sample["output"], sample["events"][0]["output"]):
        output["completion"] = "attachment://completion"
        output["choices"][0]["message"]["content"] = "attachment://completion"
    sample["messages"][1]["content"] = "attachment://completion"
    sample["scores"]["choice"]["explanation"] = "attachment://completion"
    assert e.validate_log(log, contract, invocation)["correct"] == 1
    sample["attachments"].pop("completion")
    with pytest.raises(e.EvidenceError, match="unresolved log attachment"):
        e.validate_log(log, contract, invocation)


@pytest.mark.parametrize(
    "completion,answer",
    [
        ("ANSWER: A", "A"),
        ("ANSWER: a", ""),
        ("answer: B", "B"),
        ("ANSWER: A, B", ""),
        ("ANSWER: AB", ""),
        ("ANSWER: A. trailing", "A"),
        ("ANSWER: D\nANSWER: A", "D"),
        ("some text ANSWER: C", "C"),
        ("", ""),
        ("ANSWER: E", ""),
        ("ANSWER: 1", ""),
    ],
)
def test_pinned_single_answer_semantics(completion, answer):
    assert e.parse_single_answer(completion) == answer


def test_snapshot_publication_and_no_clobber(fixture, tmp_path):
    log, contract, invocation, _ = fixture
    raw = tmp_path / "inspect.json"
    raw.write_text(json.dumps(log))
    metrics = tmp_path / "metrics.json"
    result = e.publish_metrics(
        log_path=raw, metrics_path=metrics, contract=contract, invocation=invocation
    )
    assert (
        e.validate_published(metrics, expected_contract=contract, expected_invocation=invocation)[
            "correct"
        ]
        == 1
    )
    raw.write_text("overwritten original logger path")
    assert e.validate_published(metrics, expected_contract=contract, expected_invocation=invocation)
    with pytest.raises(e.EvidenceError):
        e.publish_metrics(
            log_path=raw, metrics_path=metrics, contract=contract, invocation=invocation
        )
    snapshot = tmp_path / result["official_evidence"]["raw_log"]
    snapshot.chmod(0o600)
    snapshot.write_text("changed snapshot")
    with pytest.raises(e.EvidenceError):
        e.validate_published(metrics, expected_contract=contract, expected_invocation=invocation)


@pytest.mark.skipif(
    os.environ.get("PTB_GPQA_NATIVE") != "1", reason="requires pinned Inspect image"
)
def test_native_shuffle_equivalence_and_duplicate_positions():
    from inspect_ai.dataset import MemoryDataset, Sample

    rows = fake_rows()
    permutations = set()
    for seed in range(300):
        native = MemoryDataset([Sample(**t.source_record(row)) for row in rows])
        native.shuffle_choices(seed=seed)
        ours = t.prepare_samples(rows, seed=seed)
        assert [(s.choices, s.target) for s in ours] == [(s.choices, s.target) for s in native]
        permutations.add(tuple(ours[0].metadata["gpqa_choice_positions"]))
        assert all(s.metadata["gpqa_shuffle_seed"] == seed for s in ours)
    assert len(permutations) == 24


@pytest.mark.skipif(
    os.environ.get("PTB_GPQA_NATIVE") != "1", reason="requires pinned Inspect image"
)
def test_native_single_answer_projection():
    t.native_modules()
    from inspect_ai.solver._multiple_choice import parse_answers

    for prefix, answer, suffix in product(
        ["ANSWER: ", "answer: ", "Reasoning. ANSWER: ", "ANSWER:\t", "ANSWER\n: ", "No answer "],
        [
            "A",
            "B",
            "C",
            "D",
            "a",
            "b",
            "E",
            "1",
            "A,B",
            "A B",
            "AB",
            "A and B",
            "",
            " A ",
            "A_",
            "Ａ",
        ],
        ["", ".", "\n", " trailing", "\nANSWER: D", "!", ". More text"],
    ):
        completion = prefix + answer + suffix
        state = SimpleNamespace(output=SimpleNamespace(completion=completion), choices=[None] * 4)
        assert e.parse_single_answer(completion) == ", ".join(sorted(parse_answers(state, False)))


def formal_bundle(tmp_path, fixture, monkeypatch, *, node="owned-node", anomaly=False):
    """Freeze an explicitly synthetic tiny PTB tree/result, without real weights/data."""
    log, contract, invocation, profile = deepcopy(fixture)
    source_root = tmp_path / "synthetic-ptb"
    source = source_root / "src/eval/tasks/gpqamain"
    source.mkdir(parents=True)
    real_source = Path(__file__).parent
    for name in ("task.py", "evaluate.py", "official_evidence.py", "evidence_io.py"):
        (source / name).write_bytes((real_source / name).read_bytes())
    reference = b'[{"question":"invented only","answer":"invented only"}]'
    (source / "test_data.json").write_bytes(reference)
    profile["reference_sha256"] = hashlib.sha256(reference).hexdigest()
    profile["synthetic_fixture_only"] = True
    profile_bytes = json.dumps(profile).encode()
    (source / "data_provenance.json").write_bytes(profile_bytes)
    contract["profile_sha256"] = hashlib.sha256(profile_bytes).hexdigest()
    templates = source.parent.parent / "templates"
    templates.mkdir()
    (templates / "gemma3.jinja").write_text(
        "invented artifact-binding template; not model-executed"
    )
    utils = source_root / "src/utils"
    utils.mkdir()
    validator_path = utils / "validate_completed_run.py"
    validator_path.write_bytes(
        (real_source.parents[2] / "utils/validate_completed_run.py").read_bytes()
    )
    monkeypatch.setattr(e, "__file__", str(source / "official_evidence.py"))
    result = tmp_path / "raw/agent/gpqamain_fake_101"
    model = result / "final_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"architectures":["InventedForSyntheticTest"]}')
    (model / "model.safetensors").write_bytes(b"not real weights")
    fingerprint = e.io.model_fingerprint(model)
    for name in (
        "prompt.txt",
        "solve_out.txt",
        "solve_parsed.txt",
        "cli_version.txt",
        "time_taken.txt",
        "system_monitor.log",
        "final_eval_1.txt",
    ):
        (result / name).write_text("synthetic fixture only\n")
    provenance = {
        "experiment": {
            "task": "gpqamain",
            "batch_id": "synthetic-gpqa",
            "cell_id": "q01",
            "run_purpose": "formal",
        },
        "slurm": {"job_id": "101", "node": node},
        "judge_profile": "official",
        "finalized_at": "synthetic-time",
        "evaluation_container": {
            "sha256": "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8"
        },
    }
    provenance_bytes = json.dumps(provenance).encode()
    (result / "runtime_provenance.json").write_bytes(provenance_bytes)
    invocation.update(
        attempt_id="b" * 32,
        attempt_number=1,
        formal=True,
        model=f"vllm/{model}",
        model_sha256=fingerprint["sha256"],
        max_tokens=16000,
        gpu_memory_utilization=0.8,
        runtime_provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
        native_source_sha256=t.NATIVE_HASHES,
        template_name="gemma3.jinja",
        template_sha256=hashlib.sha256((templates / "gemma3.jinja").read_bytes()).hexdigest(),
        source_sha256={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("evaluate.py", "task.py", "official_evidence.py", "evidence_io.py")
        },
    )
    log["eval"].update(model=invocation["model"])
    log["eval"]["metadata"] = {
        "ptb_invocation": invocation,
        "ptb_selection_sha256": t.digest(contract),
    }
    log["eval"]["model_generate_config"]["max_tokens"] = 16000
    for sample in log["samples"]:
        sample["events"][0]["config"]["max_tokens"] = 16000
        sample["events"][0]["model"] = invocation["model"]
    attempt = result / "official_eval" / invocation["attempt_id"]
    attempt.mkdir(parents=True)
    (attempt / "request.json").write_text(
        json.dumps({"contract": contract, "invocation": invocation, "model": fingerprint})
    )
    raw = attempt / "inspect.json"
    raw.write_text(json.dumps(log))
    e.publish_metrics(
        log_path=raw, metrics_path=result / "metrics.json", contract=contract, invocation=invocation
    )
    spec = importlib.util.spec_from_file_location("synthetic_completed_validator", validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    for filename, schema in validator.JUDGEMENT_SCHEMAS.items():
        values = {
            key: False if value_type is bool else "synthetic checked"
            for key, value_type in schema.items()
        }
        if filename == "judgement_general.json":
            values["general_anomaly"] = anomaly
        (result / filename).write_text(json.dumps(values))
    return result, source_root, validator


def test_host_completed_validator_and_tamper_rejection(tmp_path, fixture, monkeypatch):
    result, _, validator = formal_bundle(tmp_path, fixture, monkeypatch)
    assert validator.validate(result, "official", "gpqamain") == []
    data = json.loads((result / "metrics.json").read_text())
    data["accuracy"] = 1.0
    (result / "metrics.json").write_text(json.dumps(data))
    assert any(
        "GPQA official evidence invalid" in issue
        for issue in validator.validate(result, "official", "gpqamain")
    )


@pytest.mark.parametrize(
    "node,anomaly,eligible,clean",
    [
        ("owned-node", False, True, 1),
        ("spill-node", False, False, 0),
        ("owned-node", True, True, 0),
    ],
)
def test_awm_discovery_harvest_and_judge_placement_separation(
    tmp_path, fixture, monkeypatch, node, anomaly, eligible, clean
):
    ptb = pytest.importorskip("awm.ptb_experiments")
    results = pytest.importorskip("awm.ptb_results")
    ops = pytest.importorskip("awm.ptb_ops")
    result, source_root, _ = formal_bundle(
        tmp_path, fixture, monkeypatch, node=node, anomaly=anomaly
    )
    monkeypatch.setattr(ptb, "PTB_ROOT", source_root)
    assert ptb.audit_result(result, expected_task="gpqamain") == []
    receipts = tmp_path / "receipts/synthetic-gpqa"
    receipts.mkdir(parents=True)
    (receipts / "formal.json").write_text(
        json.dumps(
            {
                "batch_id": "synthetic-gpqa",
                "site": {"POST_TRAIN_BENCH_SLURM_NODELIST": "owned-node"},
                "jobs": [{"job_id": "101", "cell_id": "q01"}],
            }
        )
    )
    monkeypatch.setattr(results, "_results_root", lambda: tmp_path / "raw")
    monkeypatch.setattr(results, "_receipts_root", lambda: tmp_path / "receipts")
    manifest = {
        "batch_id": "synthetic-gpqa",
        "_path": "synthetic-only",
        "ownership": {"spec": "synthetic-only"},
        "contract": {"task": "gpqamain"},
        "cells": [
            {
                "id": "q01",
                "replicate": 1,
                "base_model": "synthetic-only",
                "agent": "synthetic-only",
                "agent_model": "synthetic-only",
                "effort": "high",
                "context_tokens": 1000000,
            }
        ],
    }
    report = results.build_report(manifest)
    assert report["complete"] == 1 and report["clean_complete"] == clean
    assert report["rows"][0]["eligible"] is eligible
    status = ops.harvest_job(
        result,
        tmp_path / "bundle",
        batch="synthetic-gpqa",
        cell="q01",
        job_id="101",
        expected_task="gpqamain",
        expected_nodes={"owned-node"},
    )
    assert status["complete"] is True and status["eligible"] is eligible
    assert ("general_anomaly" in status["judge_flags"]) is anomaly
