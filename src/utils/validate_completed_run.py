#!/usr/bin/env python3
"""Validate the canonical artifacts required by a complete PTB run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

JUDGEMENT_SCHEMAS = {
    "judgement_gpt5_4.json": {
        "contamination": bool,
        "disallowed_model": bool,
        "justification_contamination": str,
        "justification_disallowed_model": str,
    },
    "judgement_api.json": {
        "disallowed_api_usage": bool,
        "justification_disallowed_api_usage": str,
    },
    "judgement_ptb_lookup.json": {
        "disallowed_ptb_lookup": bool,
        "justification_disallowed_ptb_lookup": str,
    },
    "judgement_general.json": {
        "general_anomaly": bool,
        "justification_general_anomaly": str,
    },
}


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"invalid JSON {path.name}: {exc}")
        return None


def preferred_rerun(path: Path) -> Path:
    """Return a non-empty rerun artifact when present, otherwise the original."""
    rerun = path.with_name(f"{path.stem}_rerun{path.suffix}")
    if rerun.is_file() and rerun.stat().st_size:
        return rerun
    return path


def validate(result_dir: Path, judge_profile: str, expected_task: str | None = None) -> list[str]:
    errors: list[str] = []
    for relative in (
        "prompt.txt",
        "solve_out.txt",
        "solve_parsed.txt",
        "cli_version.txt",
        "time_taken.txt",
        "system_monitor.log",
        "runtime_provenance.json",
        "final_model/config.json",
        "metrics.json",
    ):
        path = result_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {relative}")

    final_eval_logs = [
        path
        for pattern in ("final_eval_*.txt", "z_new_*_final_eval_*.txt")
        for path in result_dir.glob(pattern)
        if path.stat().st_size
    ]
    if not final_eval_logs:
        errors.append("missing or empty: final_eval_*.txt or z_new_*_final_eval_*.txt")

    config_path = result_dir / "final_model/config.json"
    config = load_json(config_path, errors) if config_path.is_file() else None
    if isinstance(config, dict):
        architectures = config.get("architectures")
        if not isinstance(architectures, list) or not architectures:
            errors.append("final_model/config.json has no architectures")

    final_model = result_dir / "final_model"
    index_path = final_model / "model.safetensors.index.json"
    if index_path.is_file():
        index = load_json(index_path, errors)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            errors.append("final_model weight index has no weight_map")
        else:
            for name in weight_map.values():
                if (
                    not isinstance(name, str)
                    or Path(name).is_absolute()
                    or ".." in Path(name).parts
                ):
                    errors.append(f"final_model weight index has unsafe shard path: {name!r}")
                elif not (final_model / name).is_file() or (final_model / name).stat().st_size == 0:
                    errors.append(
                        f"final_model weight index references missing or empty shard: {name}"
                    )
    elif not any(path.stat().st_size for path in final_model.glob("*.safetensors")) and not any(
        path.stat().st_size for path in final_model.glob("pytorch_model*.bin")
    ):
        errors.append("final_model has no model weights")

    metrics_path = result_dir / "metrics.json"
    metrics = load_json(metrics_path, errors) if metrics_path.is_file() else None
    if not isinstance(metrics, dict) or not metrics:
        errors.append("metrics.json must be a non-empty object")
    elif not any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in metrics.values()
    ):
        errors.append("metrics.json has no numeric metric")

    provenance_path = result_dir / "runtime_provenance.json"
    provenance = load_json(provenance_path, errors) if provenance_path.is_file() else None
    if not isinstance(provenance, dict):
        errors.append("runtime_provenance.json must be an object")
    if isinstance(provenance, dict):
        if "finalized_at" not in provenance:
            errors.append("runtime_provenance.json was not finalized")
        if provenance.get("judge_profile") != judge_profile:
            errors.append("runtime provenance judge profile mismatch")
        experiment = provenance.get("experiment")
        actual_task = experiment.get("task") if isinstance(experiment, dict) else None
        if expected_task is not None and actual_task != expected_task:
            errors.append("runtime provenance task differs from expected task")
        strong_tasks = {"humaneval": "HumanEval", "gpqamain": "GPQA"}
        strong_task = expected_task if expected_task in strong_tasks else actual_task
        if isinstance(strong_task, str) and strong_task in strong_tasks:
            helper_path = (
                Path(__file__).resolve().parent.parent
                / "eval/tasks"
                / strong_task
                / "official_evidence.py"
            )
            try:
                spec = importlib.util.spec_from_file_location(
                    "ptb_humaneval_official_evidence", helper_path
                )
                helper = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(helper)
                helper.validate_result(result_dir)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append(f"{strong_tasks[strong_task]} official evidence invalid: {exc}")

    for filename, schema in JUDGEMENT_SCHEMAS.items():
        path = preferred_rerun(result_dir / filename)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {filename} or its _rerun variant")
            continue
        verdict = load_json(path, errors)
        if not isinstance(verdict, dict):
            errors.append(f"{filename} must contain a JSON object")
            continue
        if set(verdict) != set(schema):
            errors.append(f"{filename} fields differ from the canonical schema")
            continue
        for key, expected_type in schema.items():
            value = verdict[key]
            if type(value) is not expected_type or (expected_type is str and not value.strip()):
                errors.append(f"{filename}.{key} has invalid type or value")

    if judge_profile != "official":
        errors.append("complete formal flow requires judge_profile=official")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--judge-profile", required=True)
    parser.add_argument("--expected-task")
    args = parser.parse_args()
    errors = validate(args.result_dir, args.judge_profile, args.expected_task)
    if errors:
        for error in errors:
            print(f"COMPLETION ERROR: {error}")
        raise SystemExit(f"PTB COMPLETE FLOW FAILED: {len(errors)} validation error(s)")
    print(
        "PTB COMPLETE FLOW PASSED: final model, canonical official verdicts, "
        "full-eval metrics/log, trace, monitor, and provenance are valid."
    )


if __name__ == "__main__":
    main()
