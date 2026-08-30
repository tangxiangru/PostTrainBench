#!/usr/bin/env python3
"""Focused tests for frozen provider-context evidence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from record_runtime_provenance import load_context_validation


class ContextValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.record = Path(self.temporary.name) / "context.json"
        self.record.write_text(
            json.dumps(
                {
                    "requested_model": "claude-opus-5[1m]",
                    "provider": "vertex",
                    "verified": True,
                    "resolved_context_tokens": 1_000_000,
                }
            ),
            encoding="utf-8",
        )
        self.digest = hashlib.sha256(self.record.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def environment(self, digest: str) -> dict[str, str]:
        return {
            "POST_TRAIN_BENCH_CONTEXT_VALIDATION_RECORD": str(self.record),
            "POST_TRAIN_BENCH_CONTEXT_VALIDATION_SHA256": digest,
            "POST_TRAIN_BENCH_REQUIRE_CONTEXT_VALIDATION": "1",
        }

    def test_matching_frozen_digest_passes(self) -> None:
        with patch.dict(os.environ, self.environment(self.digest), clear=False):
            loaded = load_context_validation("claude-opus-5[1m]")
        self.assertEqual(loaded["sha256"], self.digest)

    def test_changed_record_fails(self) -> None:
        with patch.dict(os.environ, self.environment("0" * 64), clear=False):
            with self.assertRaises(SystemExit):
                load_context_validation("claude-opus-5[1m]")


if __name__ == "__main__":
    unittest.main()
