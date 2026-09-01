#!/usr/bin/env python3
"""Tests for pinned Hugging Face snapshot validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_model_snapshot import validate


class ValidateModelSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.snapshot = Path(self.temporary.name)
        (self.snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        (self.snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
                    }
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_snapshot_passes(self) -> None:
        self.assertEqual(validate(self.snapshot), [])

    def test_complete_monolithic_snapshot_passes(self) -> None:
        (self.snapshot / "model.safetensors.index.json").unlink()
        (self.snapshot / "model-00001-of-00001.safetensors").unlink()
        (self.snapshot / "model.safetensors").write_bytes(b"weights")
        self.assertEqual(validate(self.snapshot), [])

    def test_empty_monolithic_snapshot_fails(self) -> None:
        (self.snapshot / "model.safetensors.index.json").unlink()
        (self.snapshot / "model-00001-of-00001.safetensors").unlink()
        (self.snapshot / "model.safetensors").write_bytes(b"")
        self.assertIn(
            "missing or empty model.safetensors and model.safetensors.index.json",
            validate(self.snapshot),
        )

    def test_missing_shard_fails(self) -> None:
        (self.snapshot / "model-00001-of-00001.safetensors").unlink()
        self.assertIn(
            "missing or empty weight shard: model-00001-of-00001.safetensors",
            validate(self.snapshot),
        )

    def test_parent_traversal_fails(self) -> None:
        (self.snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "../outside.safetensors"}}),
            encoding="utf-8",
        )
        self.assertIn("unsafe weight shard path: '../outside.safetensors'", validate(self.snapshot))


if __name__ == "__main__":
    unittest.main()
