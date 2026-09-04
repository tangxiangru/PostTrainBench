"""Previously tested JSON/model/snapshot primitives; caller supplies task validation.

Copied from the HumanEval IO implementation, not its scorer/execution rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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


def resolve_local_model(requested):
    candidate = Path(requested)
    if candidate.is_dir():
        return candidate.resolve(strict=True)
    snapshot = os.environ.get("PTB_BASE_MODEL_SNAPSHOT", "")
    revision = os.environ.get("PTB_BASE_MODEL_REVISION", "")
    if (
        requested == os.environ.get("PTB_BASE_MODEL_ID")
        and snapshot
        and revision
        and Path(snapshot).name == revision
        and Path(snapshot).is_dir()
    ):
        return Path(snapshot).resolve(strict=True)
    raise EvidenceError(
        "use a local checkpoint or the provided frozen base-model snapshot; no unpinned download fallback"
    )


def model_fingerprint(directory, *, strict=True):
    """Bind the exact local model files; never import code from the model tree."""
    directory = Path(directory)
    require(
        directory.is_dir() and not directory.is_symlink(), "model must be a local real directory"
    )
    files = []
    for path in sorted(directory.rglob("*")):
        require(
            not path.is_symlink() or (not strict and path.resolve().is_file()),
            "symlink in final model is unsupported (development permits file links only)",
        )
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


def publish_metrics(*, log_path, metrics_path, contract, invocation, validator, kind):
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
    validated = validator(strict_json(raw), contract, invocation)
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
        "kind": kind,
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


def validate_published(metrics_path, *, expected_contract, expected_invocation, validator, kind):
    """Caller supplies expected frozen identities; embedded claims are not authority."""
    path = Path(metrics_path)
    require(path.is_file() and not path.is_symlink(), "metrics must be a regular file")
    payload = strict_json(path.read_bytes())
    evidence = payload.get("official_evidence", {})
    require(
        evidence.get("kind") == kind and evidence.get("schema_version") == 1,
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
    checked = validator(strict_json(raw), expected_contract, expected_invocation)
    require(evidence.get("validated") == checked, "cached validation summary mismatch")
    for name in ("accuracy", "stderr"):
        require(
            finite(payload.get(name)) and payload[name] == checked[name],
            "published metric mismatch",
        )
    return checked
