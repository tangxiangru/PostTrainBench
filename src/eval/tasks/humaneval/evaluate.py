#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import official_evidence as evidence
from inspect_ai import eval as inspect_eval  # type: ignore
from inspect_ai.dataset import MemoryDataset
from inspect_ai.util._display import init_display_type
from runtime import register_backend
from task import (
    EXECUTION_LIMITS,
    FORMAL_PUBLIC_RUNTIME_SHA256,
    FULL_SELECTION_SHA256,
    make_task,
    selection_contract,
)


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
        default=150,
        help="Optional limit for number of samples to evaluate.",
    )
    parser.add_argument(
        "--json-output-file",
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    parser.add_argument(
        "--templates-dir",
        type=str,
        default="templates/",
    )
    # You can adjust --max-connections this if you want faster tests and don't receive errors
    parser.add_argument(
        "--max-connections",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
    )
    parser.add_argument("--formal-provenance", type=Path)
    parser.add_argument("--attempt-number", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_display_type("plain")

    if args.limit != -1 and args.limit < 1:
        raise ValueError("limit must be positive or -1 for the full task")
    metrics_path = Path(args.json_output_file).absolute() if args.json_output_file else None
    if metrics_path and (metrics_path.exists() or metrics_path.is_symlink()):
        raise ValueError("metrics output already exists; use a fresh path")
    root = metrics_path.parent if metrics_path else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if metrics_path:
        metrics_path = root / metrics_path.name
    attempt_id = uuid.uuid4().hex
    attempt_dir = root / "official_eval" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    task = make_task(sandbox_name="ptb_python")
    samples = list(task.dataset) if args.limit == -1 else list(task.dataset)[: args.limit]
    contract = selection_contract(samples)
    if args.limit == -1 and evidence.digest(contract) != FULL_SELECTION_SHA256:
        raise ValueError("full-task content differs from frozen selection fingerprint")
    task.dataset = MemoryDataset(samples, name=task.dataset.name, location=task.dataset.location)
    model_path = evidence.resolve_local_model(args.model_path)
    formal = args.formal_provenance is not None
    model_before = evidence.model_fingerprint(model_path, strict=formal)
    provenance_sha256 = None
    if formal:
        if (
            args.limit != -1
            or args.max_tokens not in (4000, 3000, 2000)
            or args.max_connections != 1
            or args.gpu_memory_utilization != 0.3
        ):
            raise ValueError("formal HumanEval requires full selection and accepted token cap")
        if (
            not metrics_path
            or metrics_path.name != "metrics.json"
            or model_path != root / "final_model"
        ):
            raise ValueError("formal output/model paths differ from the canonical result")
        if args.formal_provenance.resolve() != root / "runtime_provenance.json":
            raise ValueError("formal provenance must belong to this result")
        if not args.attempt_number or args.attempt_number < 1:
            raise ValueError("formal attempt number required")
        raw_provenance = args.formal_provenance.read_bytes()
        provenance = evidence.strict_json(raw_provenance)
        if provenance.get("experiment", {}).get("task") != "humaneval" or not provenance.get(
            "finalized_at"
        ):
            raise ValueError("finalized HumanEval runtime provenance required")
        image = provenance["evaluation_container"]
        if image["sha256"] != "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8":
            raise ValueError("formal HumanEval image differs from accepted runtime")
        provenance_sha256 = hashlib.sha256(raw_provenance).hexdigest()
    else:
        image = {
            "path": os.environ["PTB_PYTHON_SOURCE_IMAGE"],
            "sha256": os.environ["PTB_PYTHON_SOURCE_IMAGE_SHA256"],
        }
    runtime_record = register_backend(
        attempt_dir, image_reference=image["path"], image_sha256=image["sha256"]
    )
    if runtime_record["limits"] != EXECUTION_LIMITS:
        raise ValueError("HumanEval Python limits differ from the frozen task profile")
    if formal and runtime_record["source_runtime_sha256"] != FORMAL_PUBLIC_RUNTIME_SHA256:
        raise ValueError("HumanEval public runtime differs from accepted formal image closure")
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
        "runtime_provenance_sha256": provenance_sha256,
        "sandbox_runtime_sha256": runtime_record["runtime_sha256"],
        "sandbox_helper_sha256": runtime_record["helper_sha256"],
        "sandbox_limits": runtime_record["limits"],
        "source_sha256": {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in ("evaluate.py", "task.py", "official_evidence.py", "runtime.py")
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
    evidence.sync_directory(attempt_dir)
    evidence.sync_directory(attempt_dir.parent)
    evidence.sync_directory(root)
    model_args = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    model_args.update(template_kwargs(args))

    eval_out = inspect_eval(
        task,
        model=invocation["model"],
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        log_format="json",
        log_dir=str(attempt_dir / "inspect"),
        log_samples=True,
        log_buffer=1,
        fail_on_error=True,
        timeout=18000000,
        attempt_timeout=18000000,
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
    )
    if len(eval_out) != 1:
        raise ValueError("expected exactly one Inspect result")
    if evidence.model_fingerprint(model_path, strict=formal) != model_before:
        raise ValueError("model artifacts changed during evaluation")
    raw_path = Path(eval_out[0].location)
    if metrics_path:
        evidence.publish_metrics(
            log_path=raw_path, metrics_path=metrics_path, contract=contract, invocation=invocation
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
