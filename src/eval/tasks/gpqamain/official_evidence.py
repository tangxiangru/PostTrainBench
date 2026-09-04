"""Task-specific GPQA choice/presentation validation, without loading the dataset."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
from pathlib import Path


def local_module(name):
    spec = importlib.util.spec_from_file_location(
        "ptb_gpqa_" + name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


io = local_module("evidence_io")
EvidenceError, require, digest, strict_json = (
    io.EvidenceError,
    io.require,
    io.digest,
    io.strict_json,
)
KIND = "ptb-gpqa-inspect-v1"


def parse_single_answer(completion):
    """Stdlib projection of pinned native parse_answers(multiple_correct=False).

    The runtime still uses the original solver/scorer. Host revalidation has no
    Inspect dependency; native-image differential tests protect this projection.
    In particular, do NOT uppercase answers or prefer the last matching line.
    """
    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        completion,
        flags=re.MULTILINE,
    )
    if match is None:
        match = re.search(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)", completion)
    if match is None:
        return ""
    matched = match.group(1).strip().rstrip(".")
    return matched if matched in {"A", "B", "C", "D"} else ""


def resolve_text(value, sample):
    if isinstance(value, str) and value.startswith("attachment://"):
        value = (sample.get("attachments") or {}).get(value.removeprefix("attachment://"))
    require(type(value) is str, "missing text or unresolved log attachment")
    return value


def message_text(message, sample):
    require(isinstance(message, dict), "missing output message")
    content = message.get("content")
    if isinstance(content, list):
        require(all(isinstance(part, dict) for part in content), "invalid message content")
        return "\n".join(
            resolve_text(part.get("text"), sample) for part in content if part.get("type") == "text"
        )
    return resolve_text(content, sample)


def output_completion(output, sample):
    require(isinstance(output, dict) and not output.get("error"), "missing or failed model output")
    choices = output.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "expected one completion choice")
    message = choices[0].get("message")
    require(
        isinstance(message, dict) and message.get("role") == "assistant", "invalid assistant output"
    )
    completion = resolve_text(output.get("completion"), sample)
    require(
        message_text(message, sample) == completion, "completion and generated message disagree"
    )
    return completion


def validate_contract(contract):
    require(
        contract.get("task") == "gpqamain"
        and contract.get("native_task") == "gpqa_main"
        and contract.get("epochs") == 1
        and contract.get("scorer") == "choice"
        and isinstance(contract.get("randomization"), dict)
        and contract["randomization"].get("algorithm") == "native-position-shuffle-v1",
        "unsupported GPQA evaluation contract",
    )
    entries = contract.get("samples")
    require(
        type(contract.get("epochs")) is int
        and set(contract["randomization"]) == {"algorithm", "seed"}
        and (
            contract["randomization"]["seed"] is None
            or type(contract["randomization"]["seed"]) is int
        ),
        "invalid epoch/randomization type",
    )
    require(isinstance(entries, list) and bool(entries), "empty GPQA selection")
    require(
        type(contract.get("population")) is int and 0 < len(entries) <= contract["population"],
        "selection/population mismatch",
    )
    keys = [(type(entry["id"]).__name__, entry["id"], entry["epoch"]) for entry in entries]
    require(
        all(
            kind == "str" and bool(identifier) and type(epoch) is int and epoch == 1
            for kind, identifier, epoch in keys
        )
        and len(keys) == len(set(keys)),
        "invalid expected typed ID/epoch keys",
    )
    return dict(zip(keys, entries))


def validate_log(log, contract, invocation):
    expected = validate_contract(contract)
    require(
        log.get("status") == "success" and not log.get("error"),
        "Inspect evaluation did not succeed",
    )
    require(
        isinstance(invocation.get("attempt_id"), str) and bool(invocation["attempt_id"]),
        "missing independent attempt identity",
    )
    spec = log.get("eval", {})
    require(
        spec.get("task") == "gpqa_main" and spec.get("model") == invocation["model"],
        "native task/model mismatch",
    )
    io.check_count(spec.get("config", {}).get("epochs"), 1, "logged epochs")
    generate_config = spec.get("model_generate_config", {})
    for key in ("max_tokens", "max_connections"):
        require(
            generate_config.get(key) == invocation[key],
            "effective generation configuration mismatch",
        )
    metadata = spec.get("metadata") or {}
    require(
        metadata.get("ptb_invocation") == invocation
        and metadata.get("ptb_selection_sha256") == digest(contract),
        "logged invocation/selection differs",
    )
    samples = log.get("samples")
    require(
        isinstance(samples, list) and len(samples) == len(expected), "logged sample count mismatch"
    )
    seen, correct = set(), 0
    for sample in samples:
        key = (type(sample.get("id")).__name__, sample.get("id"), sample.get("epoch"))
        require(
            type(sample.get("epoch")) is int and key in expected and key not in seen,
            "unexpected, duplicate or wrongly typed sample ID/epoch",
        )
        seen.add(key)
        require(not sample.get("error"), "sample evaluation error")
        entry = expected[key]
        messages = sample.get("messages") or []
        user_messages = [message for message in messages if message.get("role") == "user"]
        require(
            len(user_messages) == 1 and type(user_messages[0].get("content")) is str,
            "missing unique rendered user prompt",
        )
        require(
            hashlib.sha256(user_messages[0]["content"].encode()).hexdigest()
            == entry["rendered_prompt_sha256"],
            "rendered model prompt differs from selection",
        )
        calls = [event for event in sample.get("events", []) if event.get("event") == "model"]
        require(
            calls and any(not call.get("error") for call in calls),
            "missing successful native model event",
        )
        for call in calls:
            require(call.get("model") == invocation["model"], "model event identity mismatch")
            require(
                all(
                    call.get("config", {}).get(field) == invocation[field]
                    for field in ("max_tokens", "max_connections")
                ),
                "model call configuration mismatch",
            )
            sent = [message for message in call.get("input", []) if message.get("role") == "user"]
            require(
                len(sent) == 1 and type(sent[0].get("content")) is str,
                "model call user prompt missing",
            )
            sent_text = resolve_text(sent[0]["content"], sample)
            require(
                type(sent_text) is str
                and hashlib.sha256(sent_text.encode()).hexdigest()
                == entry["rendered_prompt_sha256"],
                "actual model-event prompt differs from selection",
            )
        successful = [call for call in calls if not call.get("error")]
        require(
            len(successful) == 1 and successful[0] is calls[-1],
            "expected one final successful native generation",
        )
        completion = output_completion(sample.get("output"), sample)
        require(
            output_completion(successful[0].get("output"), sample) == completion
            and successful[0]["output"].get("model") == sample["output"].get("model"),
            "sample output differs from successful model event",
        )
        assistants = [message for message in messages if message.get("role") == "assistant"]
        require(
            len(assistants) == 1 and message_text(assistants[0], sample) == completion,
            "retained assistant message differs from completion",
        )
        content = {field: sample.get(field) for field in ("input", "choices", "target", "metadata")}
        require(
            digest(content) == entry["binding"],
            "actual question/options/target differ from selection",
        )
        choices, target, details = content["choices"], content["target"], content["metadata"]
        require(
            isinstance(content["input"], str)
            and isinstance(choices, list)
            and len(choices) == 4
            and all(isinstance(option, str) for option in choices),
            "GPQA input/choice schema mismatch",
        )
        positions = details.get("gpqa_choice_positions") if isinstance(details, dict) else None
        require(
            isinstance(details, dict)
            and details.get("gpqa_shuffle_seed") == contract["randomization"].get("seed"),
            "sample randomization contract mismatch",
        )
        require(
            isinstance(positions, list)
            and len(positions) == 4
            and all(type(position) is int for position in positions)
            and sorted(positions) == [0, 1, 2, 3],
            "invalid original-option permutation",
        )
        require(
            target == "ABCD"[positions.index(0)],
            "target does not track the original correct option",
        )
        original_options = [None] * 4
        for presented, original in enumerate(positions):
            original_options[original] = choices[presented]
        source_hash = digest(
            {
                "id": sample["id"],
                "input": sample["input"],
                "choices": original_options,
                "target": "A",
            }
        )
        require(
            source_hash == entry["source_row_sha256"] == details.get("gpqa_source_row_sha256"),
            "presentation does not reconstruct its frozen source row",
        )
        scores = sample.get("scores")
        require(isinstance(scores, dict) and set(scores) == {"choice"}, "sample scorer mismatch")
        score = scores["choice"]
        require(
            score.get("value") in ("C", "I") and type(score.get("value")) is str,
            "non-binary choice score",
        )
        answer = score.get("answer")
        require(
            type(answer) is str and answer in ("", "A", "B", "C", "D"),
            "unsupported single-choice answer",
        )
        require(
            answer == parse_single_answer(completion),
            "choice answer differs from native completion parse",
        )
        require(
            resolve_text(score.get("explanation"), sample) == completion,
            "choice explanation differs from unshuffled native completion",
        )
        require((score["value"] == "C") == (answer == target), "choice answer and score disagree")
        correct += score["value"] == "C"
    results = log.get("results", {})
    for field in ("total_samples", "completed_samples"):
        io.check_count(results.get(field), len(expected), field)
    scores = results.get("scores")
    require(isinstance(scores, list) and len(scores) == 1, "aggregate scorer count mismatch")
    score = scores[0]
    require(
        score.get("name") == "choice"
        and score.get("scorer") == "choice"
        and score.get("reducer") in (None, "mean"),
        "aggregate choice scorer mismatch",
    )
    io.check_count(score.get("scored_samples"), len(expected), "scored_samples")
    io.check_count(score.get("unscored_samples"), 0, "unscored_samples")
    metrics = score.get("metrics", {})
    require(set(metrics) == {"accuracy", "stderr"}, "metric family mismatch")
    accuracy = correct / len(expected)
    stderr = (
        math.sqrt(accuracy * (1 - accuracy) / (len(expected) - 1)) if len(expected) > 1 else 0.0
    )
    for name, value in (("accuracy", accuracy), ("stderr", stderr)):
        actual = metrics[name].get("value")
        require(
            io.finite(actual) and math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-12),
            "metrics do not reconcile with binary choice outcomes",
        )
    return {
        "accuracy": accuracy,
        "stderr": stderr,
        "correct": correct,
        "scored_samples": len(expected),
        "epochs": 1,
        "scorer": "choice",
    }


def publish_metrics(**kwargs):
    return io.publish_metrics(**kwargs, validator=validate_log, kind=KIND)


def validate_published(*args, **kwargs):
    return io.validate_published(*args, **kwargs, validator=validate_log, kind=KIND)


def validate_full_contract(contract, profile, profile_sha):
    validate_contract(contract)
    require(
        contract["randomization"].get("seed") is None,
        "formal GPQA must retain native unseeded randomization",
    )
    fixed = {
        "dataset": profile["dataset"],
        "config": profile["config"],
        "split": profile["split"],
        "revision": profile["revision"],
        "source_sha256": profile["source_sha256"],
        "profile_sha256": profile_sha,
        "population": profile["rows"],
    }
    require(
        all(contract.get(key) == value for key, value in fixed.items()),
        "frozen data profile mismatch",
    )
    samples = contract["samples"]
    require(len(samples) == profile["rows"], "formal selection is not the full population")
    require(
        digest(sorted([["str", sample["id"], sample["epoch"]] for sample in samples]))
        == profile["typed_id_epoch_sha256"],
        "formal ID/epoch identity mismatch",
    )
    require(
        digest([sample["source_row_sha256"] for sample in samples])
        == profile["ordered_rows_sha256"],
        "formal source-row coverage/order mismatch",
    )


def validate_result(result_dir):
    root = Path(result_dir).resolve()
    source_dir = Path(__file__).resolve().parent
    task_profile = local_module("task")
    profile, profile_sha = task_profile.load_profile()
    require(
        hashlib.sha256((source_dir / "test_data.json").read_bytes()).hexdigest()
        == profile["reference_sha256"],
        "frozen contamination reference differs from data profile",
    )
    payload = strict_json((root / "metrics.json").read_bytes())
    attempt = payload.get("official_evidence", {}).get("invocation", {}).get("attempt_id")
    require(
        type(attempt) is str and re.fullmatch("[0-9a-f]{32}", attempt),
        "invalid formal attempt identity",
    )
    directory = root / "official_eval" / attempt
    require(directory.resolve().is_relative_to(root), "attempt evidence escapes result directory")
    request = strict_json((directory / "request.json").read_bytes())
    contract, invocation = request["contract"], request["invocation"]
    validate_full_contract(contract, profile, profile_sha)
    require(
        invocation.get("attempt_id") == attempt and invocation.get("formal") is True,
        "not a formal invocation",
    )
    number = invocation.get("attempt_number")
    require(type(number) is int and 1 <= number <= 9, "attempt exceeds frozen retry budget")
    cap = 16000 if number <= 4 else 12000 if number <= 7 else 8000
    require(
        invocation.get("max_tokens") == cap
        and invocation.get("max_connections") == 6
        and invocation.get("gpu_memory_utilization") == 0.8,
        "formal serving phase/configuration mismatch",
    )
    require(invocation.get("model") == f"vllm/{root / 'final_model'}", "formal model path differs")
    model = io.model_fingerprint(root / "final_model")
    require(
        request.get("model") == model and invocation.get("model_sha256") == model["sha256"],
        "model artifacts differ from pre-eval fingerprint",
    )
    provenance_raw = (root / "runtime_provenance.json").read_bytes()
    provenance = strict_json(provenance_raw)
    require(
        provenance.get("experiment", {}).get("task") == "gpqamain"
        and provenance.get("judge_profile") == "official"
        and provenance.get("finalized_at"),
        "formal runtime provenance is incomplete or not GPQA",
    )
    require(
        invocation.get("runtime_provenance_sha256") == hashlib.sha256(provenance_raw).hexdigest(),
        "formal runtime provenance changed",
    )
    require(
        provenance.get("evaluation_container", {}).get("sha256")
        == "72748f77f9fe5a1abe925bb532c1da64d80b1dcce7849179c9546700099448f8",
        "formal image differs",
    )
    expected_sources = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("evaluate.py", "task.py", "official_evidence.py", "evidence_io.py")
    }
    require(
        invocation.get("source_sha256") == expected_sources
        and invocation.get("native_source_sha256") == task_profile.NATIVE_HASHES,
        "official task/native source differs",
    )
    template = invocation.get("template_name")
    require(
        template in ("gemma3.jinja", "qwen3.jinja", "llama3.jinja", "smollm.jinja"),
        "unsupported template",
    )
    expected_template = source_dir.parent.parent / "templates" / template
    require(
        hashlib.sha256(expected_template.read_bytes()).hexdigest()
        == invocation.get("template_sha256"),
        "formal template source differs",
    )
    return validate_published(
        root / "metrics.json", expected_contract=contract, expected_invocation=invocation
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate_result(args.result_dir)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SystemExit(f"GPQA official evidence rejected: {exc}") from exc
    print(io.json.dumps(result, sort_keys=True))
