"""Native sandbox acceptance with invented programs; no scientist or real model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path


def record_admission(root, event):
    """Record the identity-checked event from the real supervisor process."""
    with (root / "outer-admitted.json").open("x") as stream:
        json.dump({"marker": "PTB_OUTER_ADMITTED_68d273", "event": event,
                   "observed_monotonic": time.monotonic(), "observer": "native-supervisor-admission"}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def programs(outer_timeout=False):
    if outer_timeout:
        return {"invented-outer-timeout":
                "    import time\n    print('PTB_OUTER_ADMITTED_68d273', flush=True)\n"
                "    time.sleep(120)\n    return sorted((a,b))\n"}
    return {
        "invented-good": "    return sorted((a,b))\n",
        "invented-wrong": "    return [a,b]\n",
        "invented-import": "    import nonexistent_invented_module_8421\n",
        "invented-timeout": "    import time\n    time.sleep(40)\n",
        "invented-private": (
            "    import os\n    from pathlib import Path\n"
            "    assert os.getpid()==1\n"
            "    assert os.getenv('PTB_FAKE_SECRET_FOR_TEST') is None\n"
            "    assert not Path('/opt/ptb-probe.py').exists()\n"
            "    assert not Path('/opt/ptb-humaneval/evaluate.py').exists()\n"
            "    assert not Path('/opt/ptb-data.parquet').exists()\n"
            "    assert not Path('/dev/nvidia0').exists()\n"
            "    return sorted((a,b))\n"
        ),
    }


def validate_rows(rows):
    expected = {name: ("C" if name in {"invented-good", "invented-private"} else "I")
                for name in programs()}
    if len(rows) != 5 or {r["id"]: r["score"] for r in rows} != expected:
        raise ValueError("native sandbox outcomes differ from the frozen invented programs")
    if any(r["error"] or r["executions"] != 1 or not r["started"] or not r["cleanup_complete"] for r in rows):
        raise ValueError("native per-sample execution evidence is incomplete")
    timeout_row = next(row for row in rows if row["id"] == "invented-timeout")
    if (timeout_row["outcome"], timeout_row["error_category"]) != ("timeout", "wall_timeout"):
        raise ValueError("ordinary program failure cannot stand in for native timeout")


def run(args):
    sys.path.insert(0, "/opt/ptb-humaneval")
    import official_evidence as evidence
    import runtime
    import task as profile
    import torch
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import generate
    from pyarrow import parquet

    root = Path(args.output_dir)
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if count != 1 or not all("H100" in name for name in names):
        raise ValueError(f"expected one allocated H100, got {count} {names}")
    data = Path("/opt/ptb-data.parquet")
    dataset = {"sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
               "bytes": data.stat().st_size, "rows": parquet.read_metadata(data).num_rows}
    if dataset != {"sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
                   "bytes": 83920, "rows": 164}:
        raise ValueError("HumanEval source bytes/count differ from the frozen profile")
    record = runtime.register_backend(
        root, image_reference=args.image, image_sha256=args.image_sha256,
        **({"on_admission": lambda event: record_admission(root, event)} if args.outer_timeout else {})
    )
    native = importlib.import_module("inspect_evals.humaneval.humaneval")
    cases = programs(args.outer_timeout)
    samples = [Sample(id=case, input=case, target="synthetic target not executed",
                      metadata={"prompt": "def arrange(a,b):\n", "entry_point": "arrange",
                                "test": "def check(fn):\n    assert fn(9,3)==[3,9]\n"})
               for case in cases]
    contract = profile.selection_contract(samples)
    invocation = {"attempt_id": "environment-native-synthetic", "model": "mockllm/model",
                  "max_tokens": 37, "sandbox_runtime_sha256": record["runtime_sha256"],
                  "sandbox_helper_sha256": record["helper_sha256"], "sandbox_limits": record["limits"]}
    (root / "request.json").write_text(json.dumps({"contract": contract, "invocation": invocation}, indent=2))

    def output(messages, *_):
        return ModelOutput.from_content(model="mockllm/model", content=cases[messages[-1].text])

    task = Task(name="humaneval", dataset=MemoryDataset(samples), epochs=1, solver=generate(),
                scorer=native.verify(), sandbox="ptb_python",
                metadata={"ptb_invocation": invocation, "ptb_selection_sha256": evidence.digest(contract)})
    (root / "probe-started.json").write_text(json.dumps({"native_backend_registered": True,
                                                         "outer_timeout": args.outer_timeout}))
    logs = inspect_eval(task, model="mockllm/model", model_args={"custom_outputs": output},
                        max_tokens=37, max_connections=1, max_samples=1, log_dir=str(root / "logs"),
                        log_format="json", log_realtime=False, log_samples=True, log_buffer=1,
                        fail_on_error=True, score_display=False)
    if args.outer_timeout:
        raise RuntimeError("outer timeout probe unexpectedly returned before being interrupted")
    if len(logs) != 1:
        raise ValueError("native probe did not produce exactly one log")
    raw_path = Path(logs[0].location)
    raw = evidence.strict_json(raw_path.read_bytes())
    rows = []
    for sample in raw.get("samples", []):
        executions = (sample.get("store") or {}).get("ptb_python_execution", [])
        execution = executions[0] if len(executions) == 1 else {}
        rows.append({"id": sample["id"], "score": sample.get("scores", {}).get("verify", {}).get("value"),
                     "error": bool(sample.get("error")), "executions": len(executions),
                     "outcome": execution.get("outcome"), "error_category": execution.get("error_category"),
                     "started": execution.get("started"),
                     "cleanup_complete": execution.get("descendants_reaped") is True
                     and execution.get("monitor_reaped") is True and execution.get("cleanup_errors") == []})
    validate_rows(rows)
    published = evidence.publish_metrics(log_path=raw_path, metrics_path=root / "metrics.json",
                                         contract=contract, invocation=invocation)
    checked = evidence.validate_published(root / "metrics.json", expected_contract=contract,
                                          expected_invocation=invocation)
    if published["accuracy"] != 0.4 or checked["scored_samples"] != 5:
        raise ValueError("native publication/revalidation failed")
    result = {"status": "passed", "cuda": {"count": count, "names": names, "runtime_uid": os.getuid()},
              "dataset": dataset, "rows": rows, "real_model_called": False,
              "benchmark_programs_executed": False, "native_revalidated": True,
              "layout_observed": {"home": str(Path.home()), "cwd": str(Path.cwd())}}
    (root / "probe-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-timeout", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
