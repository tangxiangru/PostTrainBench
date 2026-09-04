#!/usr/bin/env python3
"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022

Based on: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/gpqa/gpqa.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import official_evidence as evidence
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.util._display import init_display_type
from task import NATIVE_HASHES, make_task, selection_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI eval without banners.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model",
        help="Local checkpoint directory or the provided frozen base-model identifier.",
    )
    # this is a good limit for this task, just keep it like that (or use less in case you want faster tests)
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Optional limit for number of samples to evaluate.",
    )
    parser.add_argument(
        "--json-output-file",
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates/",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
    )
    # You can adjust --max-connections this if you want faster tests and don't receive errors
    parser.add_argument(
        "--max-connections",
        type=int,
        default=6,
    )
    parser.add_argument("--formal-provenance", type=Path)
    parser.add_argument("--attempt-number", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_display_type("plain")
    if args.limit != -1 and args.limit < 1:
        raise ValueError("limit must be positive or -1 for the whole task")
    metrics = Path(args.json_output_file).absolute() if args.json_output_file else None
    if metrics and (metrics.exists() or metrics.is_symlink()):
        raise ValueError("metrics output already exists; use a fresh path")
    root = metrics.parent if metrics else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if metrics:
        metrics = root / metrics.name
    attempt_id = uuid.uuid4().hex
    attempt_dir = root / "official_eval" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)

    # A missing lawful/frozen data profile fails here, before model execution.
    task, profile, profile_sha = make_task()
    selected = list(task.dataset) if args.limit == -1 else list(task.dataset)[: args.limit]
    contract = selection_contract(selected, profile, profile_sha)
    task.dataset = MemoryDataset(selected, name=task.dataset.name, location=task.dataset.location)
    formal = args.formal_provenance is not None
    model_path = evidence.io.resolve_local_model(args.model_path)
    args.model_path = str(model_path)
    model_before = evidence.io.model_fingerprint(model_path, strict=formal)
    provenance_sha = None
    if formal:
        evidence.validate_full_contract(contract, profile, profile_sha)
        if args.limit != -1 or args.max_connections != 6 or args.gpu_memory_utilization != 0.8:
            raise ValueError("formal GPQA selection or serving configuration differs")
        if not metrics or metrics.name != "metrics.json" or model_path != root / "final_model":
            raise ValueError("formal model/output must be the canonical result artifacts")
        if args.formal_provenance.resolve() != root / "runtime_provenance.json":
            raise ValueError("formal provenance must belong to this result")
        if not args.attempt_number or not 1 <= args.attempt_number <= 9:
            raise ValueError("formal attempt number must be within the declared retry budget")
        expected_cap = (
            16000 if args.attempt_number <= 4 else 12000 if args.attempt_number <= 7 else 8000
        )
        if args.max_tokens != expected_cap:
            raise ValueError("formal token cap differs from its retry phase")
        raw = args.formal_provenance.read_bytes()
        provenance = evidence.strict_json(raw)
        if (
            provenance.get("experiment", {}).get("task") != "gpqamain"
            or provenance.get("judge_profile") != "official"
            or not provenance.get("finalized_at")
        ):
            raise ValueError("finalized official GPQA provenance is required")
        if (
            provenance.get("evaluation_container", {}).get("sha256")
            != "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8"
        ):
            raise ValueError("formal GPQA evaluation image differs from accepted pin")
        provenance_sha = hashlib.sha256(raw).hexdigest()

    model_args = {"gpu_memory_utilization": args.gpu_memory_utilization, **template_kwargs(args)}
    template_path = Path(model_args["chat_template"]).resolve(strict=True)
    source_dir = Path(__file__).resolve().parent
    invocation = {
        "attempt_id": attempt_id,
        "attempt_number": args.attempt_number,
        "formal": formal,
        "model": f"vllm/{model_path}",
        "model_sha256": model_before["sha256"],
        "max_tokens": args.max_tokens,
        "max_connections": args.max_connections,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "runtime_provenance_sha256": provenance_sha,
        "native_source_sha256": NATIVE_HASHES,
        "template_name": template_path.name,
        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in ("evaluate.py", "task.py", "official_evidence.py", "evidence_io.py")
        },
    }
    task.metadata = {
        "ptb_invocation": invocation,
        "ptb_selection_sha256": evidence.digest(contract),
    }
    with (attempt_dir / "request.json").open("x") as stream:
        json.dump(
            {"contract": contract, "invocation": invocation, "model": model_before},
            stream,
            indent=2,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for directory in (attempt_dir, attempt_dir.parent, root):
        evidence.io.sync_directory(directory)

    outputs = inspect_eval(
        task,
        model=invocation["model"],
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format="json",
        log_samples=True,
        log_buffer=1,
        log_dir=str(attempt_dir / "inspect"),
        fail_on_error=True,
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
    )
    if len(outputs) != 1:
        raise ValueError("expected exactly one Inspect result")
    if evidence.io.model_fingerprint(model_path, strict=formal) != model_before:
        raise ValueError("model bytes changed during evaluation")
    raw_path = Path(outputs[0].location)
    if metrics:
        evidence.publish_metrics(
            log_path=raw_path, metrics_path=metrics, contract=contract, invocation=invocation
        )
    else:
        evidence.validate_log(evidence.strict_json(raw_path.read_bytes()), contract, invocation)


def model_type(args) -> str:
    if "qwen" in args.model_path.lower():
        return "qwen"
    if "llama" in args.model_path.lower():
        return "llama"
    if "gemma" in args.model_path.lower():
        return "gemma"
    if "smollm" in args.model_path.lower():
        return "smollm"

    with open(os.path.join(args.model_path, "config.json"), "r") as f:
        config = json.load(f)
    architecture = config["architectures"][0].lower()
    if "gemma" in architecture:
        return "gemma"
    if "llama" in architecture:
        return "llama"
    if "qwen" in architecture:
        return "qwen"
    if "smollm" in architecture:
        return "smollm"
    raise ValueError(architecture)


def template_kwargs(args) -> dict:
    model_type_str = model_type(args)
    if model_type_str == "qwen":
        template = "qwen3.jinja"
    elif model_type_str == "llama":
        template = "llama3.jinja"
    elif model_type_str == "gemma":
        template = "gemma3.jinja"
    elif model_type_str == "smollm":
        template = "smollm.jinja"
    else:
        raise ValueError(model_type_str)
    return {"chat_template": os.path.join(args.templates_dir, template)}


if __name__ == "__main__":
    main()
