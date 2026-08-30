#!/usr/bin/env python3
"""Fail closed unless a pinned Hugging Face snapshot has all indexed weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(snapshot: Path) -> list[str]:
    errors: list[str] = []
    config = snapshot / "config.json"
    index = snapshot / "model.safetensors.index.json"
    if not config.is_file() or config.stat().st_size == 0:
        errors.append("missing or empty config.json")
    if not index.is_file() or index.stat().st_size == 0:
        return errors + ["missing or empty model.safetensors.index.json"]
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return errors + [f"invalid model.safetensors.index.json: {exc}"]
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return errors + ["weight_map is missing or empty"]
    names = weight_map.values()
    for name in sorted(names, key=repr):
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            errors.append(f"unsafe weight shard path: {name!r}")
            continue
        shard = snapshot / name
        if not shard.is_file() or shard.stat().st_size == 0:
            errors.append(f"missing or empty weight shard: {name}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    errors = validate(args.snapshot)
    if errors:
        for error in errors:
            print(f"MODEL SNAPSHOT ERROR: {error}")
        raise SystemExit(1)
    print(f"MODEL SNAPSHOT PASSED: {args.snapshot}")


if __name__ == "__main__":
    main()
