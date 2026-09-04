"""Fail-closed HumanEval Inspect evidence validation (stdlib-only).

No model, scorer, dataset loader or code execution is imported here. This is
also usable by the host-side completed-run validator. Publication is a single
no-clobber metrics commit after a complete, bound raw Inspect log is durable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from pathlib import Path


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON object key")
            result[key] = value
        return result

    def constant(_):
        raise EvidenceError("non-finite JSON constant")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError("invalid or incomplete JSON evidence") from exc


def check_count(value, expected, label):
    require(type(value) is int and value == expected, f"{label} mismatch")


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def model_fingerprint(directory):
    """Bind the exact local model files; never import code from the model tree."""
    directory = Path(directory)
    require(
        directory.is_dir() and not directory.is_symlink(), "model must be a local real directory"
    )
    files = []
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "symlink in final model is unsupported")
        if path.is_dir():
            continue
        require(path.is_file(), "nonregular model artifact")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while chunk := stream.read(8 * 1024 * 1024):
                sha.update(chunk)
            after = os.fstat(stream.fileno())
        require(
            (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "model artifact changed during fingerprinting",
        )
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": after.st_size,
                "sha256": sha.hexdigest(),
            }
        )
    require(any(item["path"] == "config.json" for item in files), "model has no config")
    require(
        any(item["path"].endswith((".safetensors", ".bin")) for item in files),
        "model has no weights",
    )
    return {"schema_version": 1, "files": files, "sha256": digest(files)}


def validate_log(log, contract, invocation):
    """Recompute binary accuracy and SEM from every unique expected sample.

    `contract` is built from pinned, selected Samples before eval starts;
    `invocation` is operator-owned and includes an unpredictable attempt ID.
    Neither may be inferred from an already-produced log being validated.
    """
    require(
        isinstance(log, dict) and log.get("status") == "success" and not log.get("error"),
        "Inspect log did not finish successfully",
    )
    require(
        contract.get("task") == "humaneval"
        and contract.get("epochs") == 1
        and contract.get("scorer") == "verify",
        "unsupported task contract",
    )
    require(
        isinstance(invocation.get("attempt_id"), str) and bool(invocation["attempt_id"]),
        "missing independent attempt identity",
    )
    expected = contract.get("samples")
    require(isinstance(expected, list) and bool(expected), "empty expected selection")
    keys = [(type(row["id"]).__name__, row["id"], row["epoch"]) for row in expected]
    require(
        all(kind == "str" and type(epoch) is int and epoch == 1 for kind, _, epoch in keys)
        and len(set(keys)) == len(keys),
        "invalid expected ID/epoch keys",
    )
    bindings = {key: row["binding"] for key, row in zip(keys, expected)}
    spec = log.get("eval", {})
    require(spec.get("task") == "humaneval", "logged task mismatch")
    require(spec.get("model") == invocation["model"], "logged model mismatch")
    check_count(spec.get("config", {}).get("epochs"), 1, "logged epochs")
    require(
        spec.get("model_generate_config", {}).get("max_tokens") == invocation["max_tokens"],
        "logged token cap mismatch",
    )
    require(
        (spec.get("metadata") or {}).get("ptb_invocation") == invocation,
        "logged attempt/source identity mismatch",
    )
    require(
        (spec.get("metadata") or {}).get("ptb_selection_sha256") == digest(contract),
        "logged selected-data binding mismatch",
    )
    samples = log.get("samples")
    require(
        isinstance(samples, list) and len(samples) == len(expected), "logged sample count mismatch"
    )
    seen = set()
    correct = 0
    for sample in samples:
        key = (type(sample.get("id")).__name__, sample.get("id"), sample.get("epoch"))
        require(
            type(sample.get("epoch")) is int and key in bindings and key not in seen,
            "unexpected, duplicate or wrongly typed sample ID/epoch",
        )
        seen.add(key)
        require(not sample.get("error"), "sample has an evaluation error")
        require(
            digest({field: sample.get(field) for field in ("input", "target", "metadata")})
            == bindings[key],
            "logged sample content differs from pinned selection",
        )
        scores = sample.get("scores")
        require(isinstance(scores, dict) and set(scores) == {"verify"}, "sample scorer mismatch")
        value = scores["verify"].get("value")
        require(
            type(value) is str and value in ("C", "I"),
            "sample is not a binary verification outcome",
        )
        observations = (sample.get("store") or {}).get("ptb_python_execution")
        require(
            isinstance(observations, list) and len(observations) == 1,
            "missing unique generated-code execution record",
        )
        execution = observations[0]
        require(
            execution.get("schema") == "ptb-python-sandbox-execution-v1"
            and execution.get("started") is True
            and execution.get("monitor_reaped") is True
            and execution.get("descendants_reaped") is True
            and not execution.get("cleanup_errors"),
            "unadmitted or unclean execution",
        )
        require(
            execution.get("runtime_sha256") == invocation["sandbox_runtime_sha256"]
            and execution.get("backend_sha256") == invocation["sandbox_helper_sha256"],
            "execution runtime/source differs from invocation",
        )
        require(
            execution.get("limits") == invocation["sandbox_limits"],
            "execution limits differ from invocation",
        )
        answer = scores["verify"].get("answer")
        metadata = sample["metadata"]
        require(isinstance(answer, str), "missing extracted verification answer")
        code = (
            metadata["prompt"]
            + answer
            + "\n"
            + metadata["test"]
            + "\n"
            + "check("
            + metadata["entry_point"]
            + ")"
        )
        require(
            execution.get("code_sha256") == hashlib.sha256(code.encode()).hexdigest(),
            "executed code differs from native scorer construction",
        )
        outcome = execution.get("outcome")
        require(
            outcome in ("success", "program_failure", "timeout"),
            "infrastructure/policy failure is not a model-quality outcome",
        )
        if outcome == "timeout":
            require(
                value == "I" and execution.get("error_category") == "wall_timeout",
                "timeout result mapping mismatch",
            )
        else:
            returncode = execution.get("program_returncode")
            require(
                type(returncode) is int
                and ((returncode == 0) == (outcome == "success") == (value == "C")),
                "program exit and verification score disagree",
            )
        correct += value == "C"
    results = log.get("results", {})
    for field in ("total_samples", "completed_samples"):
        check_count(results.get(field), len(expected), field)
    scores = results.get("scores")
    require(isinstance(scores, list) and len(scores) == 1, "aggregate scorer count mismatch")
    score = scores[0]
    require(
        score.get("name") == "verify" and score.get("scorer") == "verify",
        "aggregate scorer identity mismatch",
    )
    require(score.get("reducer") in (None, "mean"), "unexpected epoch reducer")
    # The pinned logger emits these; do not replace missing fields with expected n.
    check_count(score.get("scored_samples"), len(expected), "scored_samples")
    check_count(score.get("unscored_samples"), 0, "unscored_samples")
    metrics = score.get("metrics", {})
    require(set(metrics) == {"accuracy", "stderr"}, "metric family mismatch")
    accuracy = correct / len(expected)
    stderr = (
        math.sqrt(accuracy * (1 - accuracy) / (len(expected) - 1)) if len(expected) > 1 else 0.0
    )
    for name, recomputed in (("accuracy", accuracy), ("stderr", stderr)):
        actual = metrics[name].get("value")
        require(
            finite(actual) and math.isclose(actual, recomputed, rel_tol=1e-10, abs_tol=1e-12),
            f"{name} does not reconcile with binary outcomes",
        )
    return {
        "accuracy": accuracy,
        "stderr": stderr,
        "correct": correct,
        "scored_samples": len(expected),
        "epochs": 1,
        "scorer": "verify",
    }


def publish_metrics(*, log_path, metrics_path, contract, invocation):
    log_path, metrics_path = Path(log_path), Path(metrics_path)
    require(
        log_path.is_file() and not log_path.is_symlink(), "raw log must be a regular local file"
    )
    root = metrics_path.parent.resolve()
    try:
        relative = log_path.resolve().relative_to(root)
    except ValueError as exc:
        raise EvidenceError(
            "raw log must be retained beneath the metrics result directory"
        ) from exc
    require(
        not metrics_path.exists() and not metrics_path.is_symlink(),
        "metrics already exist; refusing stale overwrite",
    )
    # Read a single non-symlink file descriptor; refuse observed in-place races.
    fd = os.open(log_path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read()
        after = os.fstat(stream.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        and len(raw) == before.st_size,
        "raw log changed while being read",
    )
    validated = validate_log(strict_json(raw), contract, invocation)
    # The logger's path may be rewritten later. Publish our own exact validated
    # bytes and bind metrics to that snapshot, never fsync a second pathname read.
    snapshot_dir = Path(tempfile.mkdtemp(prefix=".official-inspect-", dir=root))
    snapshot = snapshot_dir / "inspect.json"
    with snapshot.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    snapshot.chmod(0o400)
    sync_directory(snapshot_dir)
    sync_directory(root)
    evidence = {
        "schema_version": 1,
        "kind": "ptb-humaneval-inspect-v1",
        "raw_log": snapshot.relative_to(root).as_posix(),
        "original_log": relative.as_posix(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "contract": contract,
        "invocation": invocation,
        "validated": validated,
    }
    payload = {
        "accuracy": validated["accuracy"],
        "stderr": validated["stderr"],
        "official_evidence": evidence,
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    fd, staging = tempfile.mkstemp(prefix=".metrics-", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic no-clobber publication; existing valid/invalid results are preserved.
        os.link(staging, metrics_path)
        sync_directory(root)
    finally:
        os.unlink(staging)
    return payload


def validate_published(metrics_path, *, expected_contract, expected_invocation):
    """Caller supplies expected frozen identities; embedded claims are not authority."""
    path = Path(metrics_path)
    require(path.is_file() and not path.is_symlink(), "metrics must be a regular file")
    payload = strict_json(path.read_bytes())
    evidence = payload.get("official_evidence", {})
    require(
        evidence.get("kind") == "ptb-humaneval-inspect-v1" and evidence.get("schema_version") == 1,
        "missing supported official evidence",
    )
    require(
        evidence.get("contract") == expected_contract
        and evidence.get("invocation") == expected_invocation,
        "frozen evidence identity mismatch",
    )
    relative = Path(evidence.get("raw_log", ""))
    require(
        not relative.is_absolute() and relative.parts and ".." not in relative.parts,
        "unsafe raw log relative path",
    )
    raw_path = path.parent / relative
    require(raw_path.is_file() and not raw_path.is_symlink(), "missing regular raw log")
    require(
        raw_path.resolve().is_relative_to(path.parent.resolve()), "raw log escapes result directory"
    )
    raw = raw_path.read_bytes()
    require(
        len(raw) == evidence.get("raw_bytes")
        and hashlib.sha256(raw).hexdigest() == evidence.get("raw_sha256"),
        "raw log bytes changed",
    )
    checked = validate_log(strict_json(raw), expected_contract, expected_invocation)
    require(evidence.get("validated") == checked, "cached validation summary mismatch")
    for name in ("accuracy", "stderr"):
        require(
            finite(payload.get(name)) and payload[name] == checked[name],
            "published metric mismatch",
        )
    return checked


def validate_result(result_dir):
    """Official full-task validation from frozen source, not embedded authority.

    No full benchmark texts are needed on the host: the full selected content
    hash is independently frozen by task.py from the approved pinned parquet.
    """
    root = Path(result_dir).resolve()
    source_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ptb_humaneval_profile", source_dir / "task.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    payload = strict_json((root / "metrics.json").read_bytes())
    recorded = payload.get("official_evidence", {}).get("invocation", {})
    attempt = recorded.get("attempt_id")
    require(
        type(attempt) is str and re.fullmatch("[0-9a-f]{32}", attempt),
        "invalid formal attempt identity",
    )
    attempt_dir = root / "official_eval" / attempt
    require(attempt_dir.resolve().is_relative_to(root), "attempt directory escapes result")
    request = strict_json((attempt_dir / "request.json").read_bytes())
    contract, invocation = request["contract"], request["invocation"]
    require(
        digest(contract) == profile.FULL_SELECTION_SHA256,
        "not the complete frozen HumanEval selection",
    )
    require(
        invocation["attempt_id"] == attempt and invocation.get("formal") is True,
        "not an official invocation",
    )
    require(
        type(invocation.get("attempt_number")) is int and 1 <= invocation["attempt_number"] <= 9,
        "formal attempt is outside the frozen retry budget",
    )
    require(
        invocation.get("max_tokens") in (4000, 3000, 2000)
        and invocation.get("max_connections") == 1
        and invocation.get("gpu_memory_utilization") == 0.3,
        "formal serving configuration differs",
    )
    expected_cap = (
        4000
        if invocation["attempt_number"] <= 4
        else 3000
        if invocation["attempt_number"] <= 7
        else 2000
    )
    require(invocation["max_tokens"] == expected_cap, "formal retry phase/token cap mismatch")
    require(invocation.get("model") == f"vllm/{root / 'final_model'}", "formal model path differs")
    model = model_fingerprint(root / "final_model")
    require(
        request.get("model") == model and invocation.get("model_sha256") == model["sha256"],
        "formal model bytes differ from pre-eval request",
    )
    provenance_bytes = (root / "runtime_provenance.json").read_bytes()
    provenance = strict_json(provenance_bytes)
    require(
        provenance.get("experiment", {}).get("task") == "humaneval"
        and provenance.get("judge_profile") == "official"
        and provenance.get("finalized_at"),
        "formal result provenance is incomplete or not HumanEval",
    )
    require(
        invocation.get("runtime_provenance_sha256") == hashlib.sha256(provenance_bytes).hexdigest(),
        "formal runtime provenance changed",
    )
    require(
        provenance.get("evaluation_container", {}).get("sha256")
        == "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8",
        "formal evaluation image differs",
    )
    expected_sources = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("evaluate.py", "task.py", "official_evidence.py", "runtime.py")
    }
    require(
        invocation.get("source_sha256") == expected_sources, "official evaluator source differs"
    )
    helper = source_dir.parent.parent / "ptb_python_sandbox.py"
    require(
        invocation.get("sandbox_helper_sha256") == hashlib.sha256(helper.read_bytes()).hexdigest(),
        "official Python helper source differs",
    )
    runtime = strict_json((attempt_dir / "python-runtime.json").read_bytes())
    require(
        runtime.get("helper_sha256") == invocation["sandbox_helper_sha256"]
        and runtime.get("runtime_sha256") == invocation["sandbox_runtime_sha256"],
        "archived Python runtime differs from invocation",
    )
    archived_runtime = runtime["runtime"]
    actual_runtime_sha = hashlib.sha256(
        json.dumps(archived_runtime, sort_keys=True).encode()
    ).hexdigest()
    require(
        actual_runtime_sha == runtime["runtime_sha256"], "archived transport manifest hash mismatch"
    )
    materialization = runtime["materialization"]
    original_runtime = dict(materialization["source_runtime"])
    require(
        original_runtime.pop("materialization", None) is None,
        "nested runtime transport unsupported",
    )
    actual_original_sha = hashlib.sha256(
        json.dumps(original_runtime, sort_keys=True).encode()
    ).hexdigest()
    require(
        actual_original_sha
        == profile.FORMAL_PUBLIC_RUNTIME_SHA256
        == runtime["source_runtime_sha256"]
        == materialization["source_runtime_sha256"]
        == archived_runtime["materialization"]["source_runtime_sha256"],
        "archived original public runtime differs from accepted closure",
    )
    receipt_bytes = (json.dumps(materialization, sort_keys=True, indent=2) + "\n").encode()
    require(
        hashlib.sha256(receipt_bytes).hexdigest() == archived_runtime["materialization"]["sha256"],
        "archived transport receipt hash mismatch",
    )
    require(
        runtime.get("limits") == invocation.get("sandbox_limits") == profile.EXECUTION_LIMITS,
        "archived Python limits differ from accepted contract",
    )
    require(
        runtime.get("materialization", {}).get("source_image", {}).get("sha256")
        == provenance["evaluation_container"]["sha256"],
        "public runtime source image differs",
    )
    return validate_published(
        root / "metrics.json", expected_contract=contract, expected_invocation=invocation
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate_result(args.result_dir)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SystemExit(f"HumanEval official evidence rejected: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
