#!/usr/bin/env python3
"""Focused tests for the fail-closed PTB completion contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_completed_run import JUDGEMENT_SCHEMAS, validate


class ValidateCompletedRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.result = Path(self.temporary.name)
        for name in (
            "prompt.txt",
            "solve_out.txt",
            "solve_parsed.txt",
            "cli_version.txt",
            "time_taken.txt",
            "system_monitor.log",
            "final_eval_1.txt",
        ):
            (self.result / name).write_text("present\n", encoding="utf-8")
        final_model = self.result / "final_model"
        final_model.mkdir()
        (final_model / "config.json").write_text(
            json.dumps({"architectures": ["ExampleForCausalLM"]}), encoding="utf-8"
        )
        (final_model / "model.safetensors").write_bytes(b"weights")
        (self.result / "metrics.json").write_text(
            json.dumps({"accuracy": 0.5}), encoding="utf-8"
        )
        (self.result / "runtime_provenance.json").write_text(
            json.dumps({"finalized_at": "now", "judge_profile": "official"}),
            encoding="utf-8",
        )
        for filename, schema in JUDGEMENT_SCHEMAS.items():
            verdict = {
                key: (False if expected_type is bool else "checked and clean")
                for key, expected_type in schema.items()
            }
            (self.result / filename).write_text(json.dumps(verdict), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_official_result_passes(self) -> None:
        self.assertEqual(validate(self.result, "official"), [])

    def test_recovery_eval_log_satisfies_the_full_eval_evidence(self) -> None:
        (self.result / "final_eval_1.txt").unlink()
        (self.result / "z_new_123_final_eval_2.txt").write_text(
            "recovered full evaluation\n", encoding="utf-8"
        )
        self.assertEqual(validate(self.result, "official"), [])

    def test_nonempty_rerun_verdict_is_preferred(self) -> None:
        original = self.result / "judgement_api.json"
        original.write_text("{}", encoding="utf-8")
        (self.result / "judgement_api_rerun.json").write_text(
            json.dumps(
                {
                    "disallowed_api_usage": False,
                    "justification_disallowed_api_usage": "rerun checked and clean",
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(validate(self.result, "official"), [])

    def test_malformed_canonical_verdict_fails(self) -> None:
        (self.result / "judgement_general.json").write_text(
            json.dumps({"general_anomaly": "false"}), encoding="utf-8"
        )
        errors = validate(self.result, "official")
        self.assertIn(
            "judgement_general.json fields differ from the canonical schema", errors
        )

    def test_empty_model_weights_fail(self) -> None:
        (self.result / "final_model/model.safetensors").write_bytes(b"")
        self.assertIn("final_model has no model weights", validate(self.result, "official"))

    def test_unsafe_indexed_model_weight_fails(self) -> None:
        final_model = self.result / "final_model"
        (final_model / "model.safetensors").unlink()
        (final_model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "../outside.safetensors"}}),
            encoding="utf-8",
        )
        self.assertIn(
            "final_model weight index has unsafe shard path: '../outside.safetensors'",
            validate(self.result, "official"),
        )


if __name__ == "__main__":
    unittest.main()
