#!/usr/bin/env python3
"""Write immutable PTB runtime provenance and resolve Claude trace metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def command(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def verified_sha256(path: Path, configured_name: str) -> str:
    if not path.is_file():
        raise SystemExit(f"container not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    configured = os.environ.get(configured_name, "")
    if configured and actual != configured:
        raise SystemExit(
            f"container digest mismatch for {path}: actual={actual}, expected={configured_name}={configured}"
        )
    return actual


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_uuids() -> list[str]:
    selector = os.environ.get("POST_TRAIN_BENCH_VISIBLE_GPUS", "")
    args = ["nvidia-smi"]
    if selector:
        args.extend(["-i", selector])
    args.extend(["--query-gpu=uuid", "--format=csv,noheader"])
    raw = command(*args)
    return [line.strip() for line in raw.splitlines() if line.strip() and line != "unknown"]


def load_context_validation(
    requested_model: str, requested_context_tokens: int | None = None
) -> dict[str, Any] | None:
    record_name = os.environ.get("POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD", "")
    required = os.environ.get("POST_TRAIN_BENCH_REQUIRE_CONTEXT_VALIDATION", "0") == "1"
    if not record_name:
        if required:
            raise SystemExit("context validation record is required but unset")
        return None
    path = Path(record_name)
    if not path.is_file():
        raise SystemExit(f"context validation record not found: {path}")
    raw = path.read_bytes()
    actual_digest = hashlib.sha256(raw).hexdigest()
    expected_digest = os.environ.get("POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256", "")
    if required and not expected_digest:
        raise SystemExit("required context validation has no frozen SHA-256 digest")
    if expected_digest and actual_digest != expected_digest:
        raise SystemExit(
            f"context validation digest mismatch: actual={actual_digest}, expected={expected_digest}"
        )
    data = json.loads(raw)
    if data.get("requested_model") != requested_model:
        raise SystemExit(
            f"context validation model mismatch: {data.get('requested_model')!r} != {requested_model!r}"
        )
    if data.get("provider") != "vertex" or data.get("verified") is not True:
        raise SystemExit("context validation must be a verified Vertex provider result")
    if requested_context_tokens is not None:
        if int(data.get("requested_context_tokens", 0)) != requested_context_tokens:
            raise SystemExit("context validation requested context differs from agent profile")
        if int(data.get("resolved_context_tokens", 0)) != requested_context_tokens:
            raise SystemExit("context validation resolved context differs from agent profile")
    return {"path": str(path.resolve()), "sha256": actual_digest, "record": data}


def init(args: argparse.Namespace) -> None:
    repo = Path.cwd()
    image = Path(os.environ["POST_TRAIN_BENCH_CONTAINERS_DIR"]) / (
        os.environ["POST_TRAIN_BENCH_CONTAINER_NAME"] + ".sif"
    )
    judge_image = Path(os.environ["POST_TRAIN_BENCH_CONTAINERS_DIR"]) / os.environ.get(
        "POST_TRAIN_BENCH_OFFICIAL_JUDGE_CONTAINER", "opus_5.sif"
    )
    evaluation_image = (
        Path(os.environ["POST_TRAIN_BENCH_CONTAINERS_DIR"]) / "vllm_debug.sif"
    )
    top = command("git", "rev-parse", "--show-superproject-working-tree", cwd=repo)
    top_path = Path(top) if top and top != "unknown" else repo
    requested_context = os.environ.get("PTB_AGENT_REQUESTED_CONTEXT_TOKENS", "unknown")
    requested_context_tokens = int(requested_context) if requested_context.isdigit() else None
    frozen_context = os.environ.get("POST_TRAIN_BENCH_EXPECTED_CONTEXT_TOKENS", "")
    if frozen_context.isdigit() and requested_context_tokens != int(frozen_context):
        raise SystemExit("agent profile context differs from frozen cell setup")
    context_validation = load_context_validation(args.agent_config, requested_context_tokens)
    base_model_revision = os.environ.get("POST_TRAIN_BENCH_BASE_MODEL_REVISION", "")
    base_model_cache_key = "models--" + args.base_model.replace("/", "--")
    base_model_snapshot = (
        Path(os.environ["HF_HOME"])
        / "hub"
        / base_model_cache_key
        / "snapshots"
        / base_model_revision
    )
    base_model_config = base_model_snapshot / "config.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "batch_id": os.environ.get("POST_TRAIN_BENCH_BATCH_ID", "untracked"),
            "cell_id": os.environ.get("POST_TRAIN_BENCH_CELL_ID", "untracked"),
            "run_purpose": os.environ.get("POST_TRAIN_BENCH_RUN_PURPOSE", "untracked"),
            "spec_path": os.environ.get("POST_TRAIN_BENCH_SPEC_PATH", "untracked"),
            "task": args.task,
            "agent": args.agent,
            "agent_config": args.agent_config,
            "base_model": args.base_model,
            "base_model_revision": base_model_revision or "unfrozen",
            "agent_budget_hours": args.hours,
            "num_gpus": args.num_gpus,
            "experiment_name": os.environ.get("POST_TRAIN_BENCH_EXPERIMENT_NAME", ""),
        },
        "agent_runtime": {
            "provider": os.environ.get("PTB_AGENT_PROVIDER", "unknown"),
            "requested_model": args.agent_config,
            "resolved_model": "pending",
            "requested_context_tokens": int(requested_context)
            if requested_context.isdigit()
            else requested_context,
            "resolved_context_tokens": (
                context_validation["record"]["resolved_context_tokens"]
                if context_validation
                else "unverified"
            ),
            "effort": os.environ.get("PTB_AGENT_EFFORT", "unknown"),
            "context_validation": context_validation,
        },
        "base_model_cache_snapshot": {
            "host_path": str(base_model_snapshot.resolve()),
            "container_path": f"{os.environ.get('HF_HOME_NEW', '/home/ben/hf_cache')}/hub/"
            f"{base_model_cache_key}/snapshots/{base_model_revision}",
            "config_sha256": file_sha256(base_model_config)
            if base_model_config.is_file()
            else "missing",
        },
        "slurm": {
            "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME")
            or os.environ.get("POST_TRAIN_BENCH_SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME") or command("hostname"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "allocated_gpu_ids": os.environ.get("POST_TRAIN_BENCH_ALLOCATED_GPUS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "gpu_uuids": gpu_uuids(),
        },
        "source": {
            "top_branch": os.environ.get("POST_TRAIN_BENCH_FROZEN_TOP_BRANCH")
            or os.environ.get("POST_TRAIN_BENCH_RUN_BRANCH")
            or command("git", "branch", "--show-current", cwd=top_path),
            "top_commit": os.environ.get("POST_TRAIN_BENCH_FROZEN_TOP_COMMIT")
            or command("git", "rev-parse", "HEAD", cwd=top_path),
            "ptb_commit": os.environ.get("POST_TRAIN_BENCH_FROZEN_PTB_COMMIT")
            or command("git", "rev-parse", "HEAD", cwd=repo),
            "top_dirty": False
            if os.environ.get("POST_TRAIN_BENCH_FROZEN_TOP_COMMIT")
            else command("git", "status", "--porcelain", cwd=top_path) != "",
            "ptb_dirty": False
            if os.environ.get("POST_TRAIN_BENCH_FROZEN_PTB_COMMIT")
            else command("git", "status", "--porcelain", cwd=repo) != "",
            "materialization": "git-archive"
            if os.environ.get("POST_TRAIN_BENCH_FROZEN_PTB_COMMIT")
            else "working-tree",
        },
        "container": {
            "path": str(image.resolve()),
            "sha256": verified_sha256(image, "POST_TRAIN_BENCH_CONTAINER_SHA256"),
        },
        "evaluation_container": {
            "path": str(evaluation_image.resolve()),
            "sha256": verified_sha256(
                evaluation_image, "POST_TRAIN_BENCH_EVALUATION_CONTAINER_SHA256"
            ),
        },
        "official_judge_container": {
            "path": str(judge_image.resolve()),
            "sha256": verified_sha256(
                judge_image, "POST_TRAIN_BENCH_OFFICIAL_JUDGE_CONTAINER_SHA256"
            ),
        },
        "judge_profile": os.environ.get("POST_TRAIN_BENCH_JUDGE_PROFILE", "official"),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def finalize(args: argparse.Namespace) -> None:
    output = Path(args.output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    resolved_model = None
    trace = Path(args.trace)
    if trace.is_file():
        for raw in trace.read_text(encoding="utf-8", errors="replace").splitlines():
            start = raw.find("{")
            if start < 0:
                continue
            try:
                event = json.loads(raw[start:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                resolved_model = event.get("model") or event.get("settings", {}).get("model")
                if resolved_model:
                    break
    payload["agent_runtime"]["resolved_model"] = resolved_model or "unknown"
    cli_record = (
        Path(args.cli_version).read_text(encoding="utf-8", errors="replace").strip()
        if Path(args.cli_version).is_file()
        else ""
    )
    version_line = next(
        (line.removeprefix("version:").strip() for line in cli_record.splitlines() if line.startswith("version:")),
        "",
    )
    version_match = re.search(r"\b\d+\.\d+\.\d+\b", version_line)
    payload["agent_runtime"]["cli_version"] = (
        version_match.group(0) if version_match else "unknown"
    )
    payload["agent_runtime"]["cli_version_record"] = cli_record or "missing"
    payload["finalized_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--output", required=True)
    init_parser.add_argument("--task", required=True)
    init_parser.add_argument("--agent", required=True)
    init_parser.add_argument("--agent-config", required=True)
    init_parser.add_argument("--base-model", required=True)
    init_parser.add_argument("--hours", required=True, type=int)
    init_parser.add_argument("--num-gpus", required=True, type=int)
    init_parser.set_defaults(handler=init)
    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--output", required=True)
    final_parser.add_argument("--trace", required=True)
    final_parser.add_argument("--cli-version", required=True)
    final_parser.set_defaults(handler=finalize)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
