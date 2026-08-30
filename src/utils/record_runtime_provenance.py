#!/usr/bin/env python3
"""Write immutable PTB runtime provenance and resolve Claude trace metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def command(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path, configured_name: str) -> str:
    configured = os.environ.get(configured_name, "")
    if configured:
        return configured
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


def load_context_validation(requested_model: str) -> dict[str, Any] | None:
    record_name = os.environ.get("POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD", "")
    required = os.environ.get("POST_TRAIN_BENCH_REQUIRE_CONTEXT_VALIDATION", "0") == "1"
    if not record_name:
        if required:
            raise SystemExit("context validation record is required but unset")
        return None
    path = Path(record_name)
    if not path.is_file():
        raise SystemExit(f"context validation record not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("requested_model") != requested_model:
        raise SystemExit(
            f"context validation model mismatch: {data.get('requested_model')!r} != {requested_model!r}"
        )
    if data.get("provider") != "vertex" or data.get("verified") is not True:
        raise SystemExit("context validation must be a verified Vertex provider result")
    if int(data.get("resolved_context_tokens", 0)) < 1_000_000:
        raise SystemExit("context validation did not resolve a >=1M context window")
    return {"path": str(path.resolve()), "record": data}


def init(args: argparse.Namespace) -> None:
    repo = Path.cwd()
    image = Path(os.environ["POST_TRAIN_BENCH_CONTAINERS_DIR"]) / (
        os.environ["POST_TRAIN_BENCH_CONTAINER_NAME"] + ".sif"
    )
    judge_image = Path(os.environ["POST_TRAIN_BENCH_CONTAINERS_DIR"]) / os.environ.get(
        "POST_TRAIN_BENCH_OFFICIAL_JUDGE_CONTAINER", "gpt_5_5.sif"
    )
    top = command("git", "rev-parse", "--show-superproject-working-tree", cwd=repo)
    top_path = Path(top) if top and top != "unknown" else repo
    requested_context = os.environ.get("PTB_AGENT_REQUESTED_CONTEXT_TOKENS", "unknown")
    context_validation = load_context_validation(args.agent_config)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "task": args.task,
            "agent": args.agent,
            "agent_config": args.agent_config,
            "base_model": args.base_model,
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
        "slurm": {
            "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or command("hostname"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "gpu_uuids": gpu_uuids(),
        },
        "source": {
            "top_commit": command("git", "rev-parse", "HEAD", cwd=top_path),
            "ptb_commit": command("git", "rev-parse", "HEAD", cwd=repo),
            "top_dirty": command("git", "status", "--porcelain", cwd=top_path) != "",
            "ptb_dirty": command("git", "status", "--porcelain", cwd=repo) != "",
        },
        "container": {
            "path": str(image.resolve()),
            "sha256": sha256(image, "POST_TRAIN_BENCH_CONTAINER_SHA256"),
        },
        "official_judge_container": {
            "path": str(judge_image.resolve()),
            "sha256": sha256(
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
    payload["agent_runtime"]["cli_version"] = (
        Path(args.cli_version).read_text(encoding="utf-8", errors="replace").strip()
        if Path(args.cli_version).is_file()
        else "unknown"
    )
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
