import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "humaneval_official_evidence", Path(__file__).with_name("official_evidence.py")
)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


@pytest.fixture
def fixture():
    # Invented records only, not HumanEval questions, canonical solutions or tests.
    samples = []
    expected = []
    for index, outcome in enumerate(("C", "I", "C")):
        content = {
            "input": "invented input",
            "target": "invented target",
            "metadata": {
                "prompt": "def invented():\n",
                "test": "# no benchmark tests",
                "entry_point": "invented",
            },
        }
        answer = "    pass\n"
        code = (
            content["metadata"]["prompt"]
            + answer
            + "\n"
            + content["metadata"]["test"]
            + "\ncheck(invented)"
        )
        execution = {
            "schema": "ptb-python-sandbox-execution-v1",
            "started": True,
            "monitor_reaped": True,
            "descendants_reaped": True,
            "cleanup_errors": [],
            "runtime_sha256": "synthetic-runtime",
            "backend_sha256": "synthetic-helper",
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "outcome": "success" if outcome == "C" else "program_failure",
            "program_returncode": 0 if outcome == "C" else 1,
            "limits": {"synthetic": "limits"},
        }
        samples.append(
            {
                "id": f"invented-{index}",
                "epoch": 1,
                **content,
                "scores": {"verify": {"value": outcome, "answer": answer}},
                "store": {"ptb_python_execution": [execution]},
            }
        )
        expected.append({"id": f"invented-{index}", "epoch": 1, "binding": e.digest(content)})
    contract = {"task": "humaneval", "epochs": 1, "scorer": "verify", "samples": expected}
    invocation = {
        "attempt_id": "invented-attempt",
        "model": "mockllm/model",
        "max_tokens": 37,
        "sandbox_runtime_sha256": "synthetic-runtime",
        "sandbox_helper_sha256": "synthetic-helper",
        "sandbox_limits": {"synthetic": "limits"},
    }
    log = {
        "status": "success",
        "eval": {
            "task": "humaneval",
            "model": invocation["model"],
            "config": {"epochs": 1},
            "model_generate_config": {"max_tokens": 37},
            "metadata": {"ptb_invocation": invocation, "ptb_selection_sha256": e.digest(contract)},
        },
        "samples": samples,
        "results": {
            "total_samples": 3,
            "completed_samples": 3,
            "scores": [
                {
                    "name": "verify",
                    "scorer": "verify",
                    "scored_samples": 3,
                    "unscored_samples": 0,
                    "metrics": {"accuracy": {"value": 2 / 3}, "stderr": {"value": 1 / 3}},
                }
            ],
        },
    }
    return log, contract, invocation


def test_recomputes_metrics_and_does_not_mutate(fixture):
    before = deepcopy(fixture)
    result = e.validate_log(*fixture)
    assert result["correct"] == 2 and result["scored_samples"] == 3
    assert fixture == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x.update(status="error"),
        lambda x: x.update(error={"message": "partial"}),
        lambda x: x["samples"].pop(),
        lambda x: x["samples"].append(x["samples"][0]),
        lambda x: x["samples"][0].update(id="unexpected"),
        lambda x: x["samples"][0].update(id=0),
        lambda x: x["samples"][0].update(epoch=True),
        lambda x: x["samples"][1].update(id="invented-0"),
        lambda x: x["samples"][0].update(error={"message": "executor failed"}),
        lambda x: x["samples"][0].update(input="different input"),
        lambda x: x["samples"][0].update(target="different target"),
        lambda x: x["samples"][0].update(metadata={"synthetic": False}),
        lambda x: x["samples"][0].update(scores={"match": {"value": "C"}}),
        lambda x: x["samples"][0]["scores"]["verify"].update(value=True),
        lambda x: x["samples"][0]["scores"]["verify"].update(value="P"),
        lambda x: x["samples"][0].pop("store"),
        lambda x: x["samples"][0]["store"]["ptb_python_execution"][0].update(started=False),
        lambda x: x["samples"][0]["store"]["ptb_python_execution"][0].update(
            descendants_reaped=False
        ),
        lambda x: x["samples"][0]["store"]["ptb_python_execution"][0].update(
            runtime_sha256="wrong"
        ),
        lambda x: x["samples"][0]["store"]["ptb_python_execution"][0].update(code_sha256="wrong"),
        lambda x: x["samples"][0]["store"]["ptb_python_execution"][0].update(
            outcome="dependency_error"
        ),
        lambda x: x["samples"][0]["store"]["ptb_python_execution"][0].update(program_returncode=1),
        lambda x: x["eval"].update(task="gsm8k"),
        lambda x: x["eval"].update(model="wrong-model"),
        lambda x: x["eval"]["config"].update(epochs=5),
        lambda x: x["eval"]["model_generate_config"].update(max_tokens=38),
        lambda x: x["eval"]["metadata"].update(ptb_invocation={}),
        lambda x: x["eval"]["metadata"].update(ptb_selection_sha256="0" * 64),
        lambda x: x["results"].update(total_samples=4),
        lambda x: x["results"].update(completed_samples=2),
        lambda x: x["results"]["scores"][0].update(scorer="match"),
        lambda x: x["results"]["scores"][0].update(reducer="max"),
        lambda x: x["results"]["scores"][0].update(scored_samples=None),
        lambda x: x["results"]["scores"][0].update(unscored_samples=1),
        lambda x: x["results"]["scores"][0]["metrics"].pop("stderr"),
        lambda x: x["results"]["scores"][0]["metrics"]["accuracy"].update(value=1),
        lambda x: x["results"]["scores"][0]["metrics"]["stderr"].update(value=-1),
        lambda x: x["results"]["scores"][0]["metrics"]["accuracy"].update(value=float("nan")),
    ],
)
def test_rejects_partial_wrong_or_unbound_logs(fixture, mutation):
    log, contract, invocation = fixture
    mutation(log)
    with pytest.raises(e.EvidenceError):
        e.validate_log(log, contract, invocation)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', "{"])
def test_strict_json(raw):
    with pytest.raises(e.EvidenceError):
        e.strict_json(raw)


def publish(tmp_path, fixture):
    log, contract, invocation = fixture
    logs = tmp_path / "official_eval" / "invented-attempt"
    logs.mkdir(parents=True)
    raw = logs / "inspect.json"
    raw.write_text(json.dumps(log))
    metrics = tmp_path / "metrics.json"
    result = e.publish_metrics(
        log_path=raw, metrics_path=metrics, contract=contract, invocation=invocation
    )
    return raw, metrics, result


def test_publication_roundtrip_no_clobber_and_changed_raw(tmp_path, fixture):
    raw, metrics, result = publish(tmp_path, fixture)
    assert (
        e.validate_published(metrics, expected_contract=fixture[1], expected_invocation=fixture[2])
        == result["official_evidence"]["validated"]
    )
    old = metrics.read_bytes()
    with pytest.raises(e.EvidenceError, match="already exist"):
        e.publish_metrics(
            log_path=raw, metrics_path=metrics, contract=fixture[1], invocation=fixture[2]
        )
    assert metrics.read_bytes() == old
    # Original logger output is not the atomic publication's authority.
    raw.write_text(raw.read_text() + " ")
    assert e.validate_published(
        metrics, expected_contract=fixture[1], expected_invocation=fixture[2]
    )
    snapshot = tmp_path / result["official_evidence"]["raw_log"]
    snapshot.chmod(0o600)
    snapshot.write_text(snapshot.read_text() + " ")
    with pytest.raises(e.EvidenceError, match="bytes changed"):
        e.validate_published(metrics, expected_contract=fixture[1], expected_invocation=fixture[2])


def test_no_metrics_for_incomplete_attempt(tmp_path, fixture):
    fixture[0]["status"] = "error"
    with pytest.raises(e.EvidenceError):
        publish(tmp_path, fixture)
    assert not (tmp_path / "metrics.json").exists()
    assert (tmp_path / "official_eval/invented-attempt/inspect.json").is_file()


@pytest.mark.parametrize("change", ["path", "metric", "contract", "invocation", "summary"])
def test_published_tampering(tmp_path, fixture, change):
    _raw, metrics, payload = publish(tmp_path, fixture)
    if change == "path":
        payload["official_evidence"]["raw_log"] = "../outside.json"
    elif change == "metric":
        payload["accuracy"] = 1.0
    elif change == "contract":
        payload["official_evidence"]["contract"]["epochs"] = 5
    elif change == "invocation":
        payload["official_evidence"]["invocation"]["attempt_id"] = "other"
    else:
        payload["official_evidence"]["validated"]["correct"] = 3
    # independent pre-eval authority, not the mutable embedded dict
    original = deepcopy(fixture)
    if change in ("contract", "invocation"):
        original[1]["epochs"] = 1
        original[2]["attempt_id"] = "invented-attempt"
    metrics.write_text(json.dumps(payload))
    with pytest.raises(e.EvidenceError):
        e.validate_published(
            metrics, expected_contract=original[1], expected_invocation=original[2]
        )


def test_symlink_raw_denied(tmp_path, fixture):
    real = tmp_path / "real.json"
    real.write_text(json.dumps(fixture[0]))
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(e.EvidenceError):
        e.publish_metrics(
            log_path=link,
            metrics_path=tmp_path / "metrics.json",
            contract=fixture[1],
            invocation=fixture[2],
        )


def test_replaced_original_cannot_swap_published_bytes(tmp_path, fixture, monkeypatch):
    original = e.validate_log

    def replace_after_validation(*args):
        answer = original(*args)
        raw = tmp_path / "official_eval/invented-attempt/inspect.json"
        raw.write_text('{"different":true}')
        return answer

    monkeypatch.setattr(e, "validate_log", replace_after_validation)
    _, metrics, _ = publish(tmp_path, fixture)
    monkeypatch.setattr(e, "validate_log", original)
    assert (
        e.validate_published(metrics, expected_contract=fixture[1], expected_invocation=fixture[2])[
            "correct"
        ]
        == 2
    )


def test_snapshot_directory_synced_before_metrics_commit(tmp_path, fixture, monkeypatch):
    events = []
    sync = e.sync_directory
    link = e.os.link

    def observe_sync(path):
        events.append(("sync", Path(path)))
        sync(path)

    def observe_link(source, destination):
        events.append(("publish", Path(destination)))
        link(source, destination)

    monkeypatch.setattr(e, "sync_directory", observe_sync)
    monkeypatch.setattr(e.os, "link", observe_link)
    _, _, result = publish(tmp_path, fixture)
    snapshot_parent = (tmp_path / result["official_evidence"]["raw_log"]).parent
    publication = next(i for i, item in enumerate(events) if item[0] == "publish")
    assert ("sync", snapshot_parent) in events[:publication]
    assert ("sync", tmp_path) in events[:publication]


@pytest.fixture
def formal(tmp_path, fixture, monkeypatch):
    """Tiny invented source/data/model contract, never a claimed HumanEval result."""
    log, contract, invocation = fixture
    source = tmp_path / "synthetic-source/src/eval/tasks/humaneval"
    source.mkdir(parents=True)
    original = {"profile": "explicitly-nonexecutable-unit-fixture", "files": []}
    source_runtime_sha = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    profile_source = (
        f"FULL_SELECTION_SHA256 = {e.digest(contract)!r}\n"
        f"FORMAL_PUBLIC_RUNTIME_SHA256 = {source_runtime_sha!r}\n"
        "EXECUTION_LIMITS = {'synthetic': 'limits'}\n"
    )
    (source / "task.py").write_text(profile_source)
    for name in ("evaluate.py", "runtime.py", "official_evidence.py"):
        (source / name).write_text(f"# invented source identity: {name}\n")
    helper = source.parent.parent / "ptb_python_sandbox.py"
    helper.write_text("# non-executable helper fixture\n")
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    monkeypatch.setattr(e, "__file__", str(source / "official_evidence.py"))
    result = tmp_path / "result"
    model = result / "final_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"architectures":["InventedForUnitTest"]}')
    (model / "model.safetensors").write_bytes(b"not real model weights")
    fingerprint = e.model_fingerprint(model)
    image = {"sha256": "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8"}
    provenance = {
        "experiment": {"task": "humaneval"},
        "finalized_at": "invented-time",
        "judge_profile": "official",
        "evaluation_container": image,
    }
    provenance_path = result / "runtime_provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    materialization = {
        "source_runtime": {**original, "materialization": None},
        "source_runtime_sha256": source_runtime_sha,
        "source_image": image,
    }
    receipt_sha = hashlib.sha256(
        (json.dumps(materialization, sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()
    runtime = {
        "materialization": {"source_runtime_sha256": source_runtime_sha, "sha256": receipt_sha}
    }
    runtime_sha = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    invocation.update(
        attempt_id="a" * 32,
        attempt_number=1,
        formal=True,
        max_tokens=4000,
        max_connections=1,
        gpu_memory_utilization=0.3,
        model=f"vllm/{model}",
        model_sha256=fingerprint["sha256"],
        sandbox_runtime_sha256=runtime_sha,
        sandbox_helper_sha256=helper_sha,
        runtime_provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        source_sha256={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("evaluate.py", "task.py", "official_evidence.py", "runtime.py")
        },
    )
    log["eval"].update(model=invocation["model"])
    log["eval"]["model_generate_config"]["max_tokens"] = 4000
    for sample in log["samples"]:
        sample["store"]["ptb_python_execution"][0].update(
            runtime_sha256=runtime_sha, backend_sha256=helper_sha
        )
    attempt = result / "official_eval" / invocation["attempt_id"]
    attempt.mkdir(parents=True)
    (attempt / "request.json").write_text(
        json.dumps({"contract": contract, "invocation": invocation, "model": fingerprint})
    )
    runtime_record = {
        "helper_sha256": helper_sha,
        "runtime_sha256": runtime_sha,
        "source_runtime_sha256": source_runtime_sha,
        "runtime": runtime,
        "limits": {"synthetic": "limits"},
        "materialization": materialization,
    }
    (attempt / "python-runtime.json").write_text(json.dumps(runtime_record))
    raw = attempt / "inspect.json"
    raw.write_text(json.dumps(log))
    e.publish_metrics(
        log_path=raw, metrics_path=result / "metrics.json", contract=contract, invocation=invocation
    )
    return result, attempt, source


def test_formal_independent_contract_roundtrip(formal):
    result, _, _ = formal
    assert e.validate_result(result)["scored_samples"] == 3


@pytest.mark.parametrize(
    "change",
    ["model", "source", "provenance", "partial_selection", "runtime_manifest", "limits", "receipt"],
)
def test_formal_detects_identity_and_archive_changes(formal, change):
    result, attempt, source = formal
    if change == "model":
        (result / "final_model/model.safetensors").write_bytes(b"different fake weights")
    elif change == "source":
        (source / "runtime.py").write_text("# altered source\n")
    elif change == "provenance":
        path = result / "runtime_provenance.json"
        path.write_text(path.read_text() + " ")
    elif change == "partial_selection":
        path = attempt / "request.json"
        data = json.loads(path.read_text())
        data["contract"]["samples"].pop()
        path.write_text(json.dumps(data))
    else:
        path = attempt / "python-runtime.json"
        data = json.loads(path.read_text())
        if change == "runtime_manifest":
            data["runtime"] = {}
        elif change == "limits":
            data["limits"] = {"synthetic": "changed"}
        else:
            data["materialization"]["source_runtime"]["files"] = ["changed"]
        path.write_text(json.dumps(data))
    with pytest.raises(e.EvidenceError):
        e.validate_result(result)


def test_provided_frozen_base_alias_without_download(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    monkeypatch.setenv("PTB_BASE_MODEL_ID", "invented/base-model")
    monkeypatch.setenv("PTB_BASE_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("PTB_BASE_MODEL_SNAPSHOT", str(snapshot))
    assert e.resolve_local_model("invented/base-model") == snapshot.resolve()
    with pytest.raises(e.EvidenceError):
        e.resolve_local_model("unprovided/moving-model")
    monkeypatch.setenv("PTB_BASE_MODEL_REVISION", "b" * 40)
    with pytest.raises(e.EvidenceError):
        e.resolve_local_model("invented/base-model")


def test_model_fingerprint_rejects_symlink(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    target = tmp_path / "elsewhere"
    target.write_bytes(b"fake")
    (model / "model.safetensors").symlink_to(target)
    with pytest.raises(e.EvidenceError, match="symlink"):
        e.model_fingerprint(model)
    assert (
        e.model_fingerprint(model, strict=False)["files"][-1]["sha256"]
        == e.hashlib.sha256(b"fake").hexdigest()
    )
    target.write_bytes(b"changed")
    assert (
        e.model_fingerprint(model, strict=False)["files"][-1]["sha256"]
        == e.hashlib.sha256(b"changed").hexdigest()
    )
