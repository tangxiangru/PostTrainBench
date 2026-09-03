#!/usr/bin/env python3
"""Take a graded read of a checkpoint, or refuse -- but never render a failed read
as a number.

Every cell writes its own evaluation wrapper, and in the twelve-cell gsm8k arm
89727/89809/89810 two of them re-invented the same defect: a readout produced by a step
whose exit code nobody checked. Both are on disk, and both are near-miss *silent wrong
answers* -- not crashes, not empty files, but plausible numbers attached to the wrong
thing.

**89727_g5.** ``code/chain_stage05_final.sh`` ran a read and then called
``eval_readout.py --latest`` unconditionally. The read failed (vLLM would not start: the
container-root overlay was full and scipy's bytecode cache had been truncated mid-write),
so ``--latest`` resolved to *the previous inspect log on disk* and the script wrote
``outputs/results/readout_grpo_k10_1319_r2.json``. That file is now a retraction, and it
says what it used to hold, verbatim:

    A copy of the n=500 screening read of ``runs/grpo_main_k0/final`` (inspect log
    ``2026-09-02T15-31-04+00-00_gsm8k_UnHx5jvLysmSfysSX7pp7u.json``, 500 rows, 79.2%).
    Note that this is not even the right MODEL: the label names ``runs/grpo_h8_k10/final``
    and the underlying log is a read of a different checkpoint.

Re-read here from the archived cell: that log carries ``total_samples`` 500,
``completed_samples`` 500, ``config.limit`` 500, ``model`` ``vllm/runs/grpo_main_k0/final``
and accuracy 0.792. Every one of the three facts a reader needed -- *how many rows*, *which
model*, *did the command succeed* -- was sitting in the log the whole time, and nothing
compared them against what the filename claimed.

**89810_g7.** ``code/run_arm.sh`` lines 83-107 run ``ship_rule.py``, take ``SHIP=$?``, and
branch ``if [ $SHIP -eq 0 ]`` to ship and *everything else* to HOLD. ``ship_rule.py``'s
ENOENT exit-2 is therefore indistinguishable from its deliberate HOLD exit-10. The cell's
own report says a missing-file error rendered as a verdict would have silently withheld
the best checkpoint of the run.

So the durable fix is not a rule in the prompt -- an accelerator-budget skill was pinned
by name in all twelve ``run_config.json`` of this arm and changed nothing measurable -- it
is a helper that is *easier to call than to re-write*, and that cannot return a number it
did not verify.

**What "verified" means here.** Five checks, each with its own exit code, so a wrapper
reading ``$?`` can tell them apart and none of them can collapse into "0":

===== ======================= =====================================================
 code  kind                    what it means
===== ======================= =====================================================
   0  ``verified``            all checks passed; the metrics file is a real read
   3  ``command_failed``      ``evaluate.py`` exited nonzero
   4  ``output_missing``      it exited 0 and wrote no metrics file, or no log
   5  ``output_unreadable``   metrics/log unparseable, ambiguous, or disagreeing
   6  ``row_count_mismatch``  the log ran a different number of rows than requested
   8  ``checkpoint_mismatch`` the log is a read of a different model path
===== ======================= =====================================================

Codes start at 3 on purpose. ``1`` is what CPython exits on an uncaught traceback and
``2`` is what :mod:`argparse` exits on a bad flag, so leaving both free keeps "graded_read
decided something" distinguishable from "graded_read itself broke" -- which is the same
distinction ``run_arm.sh`` lost.

**Why the row count is checked against the log and not against the flag.** The obvious
field is the wrong one. ``eval.dataset.samples`` in an inspect log is the size of the
*dataset*, not of the read: in the four gsm8k logs archived under
``claude_vertex_claude-opus-5_10h/gsm8k_Qwen_Qwen3-1.7B-Base_84729_g7/task/logs`` it reads
1319 in all four, including the three that ran 150 rows. A check built on it passes on
exactly the reads it exists to catch. The fields that move are ``results.total_samples``,
``results.completed_samples`` and each scorer's ``scored_samples``; all three are compared,
because a read that was cut short has ``completed_samples`` below ``total_samples`` and
still produces an accuracy.

**Why nothing verified is left where a glob can find it.** On any refusal the metrics file
this run produced is renamed to ``*.UNVERIFIED``. The invariant an audit can then rely on
is a filename one: nothing matching ``<out-dir>/*.metrics.json`` was ever written by a read
this helper refused. 89727_g5's defect was a *filename* that outranked its contents.

**Two benchmarks have no log to check, and refusing them was worse than useless.** Six of
the eight tasks in ``src/eval/tasks`` run ``inspect_ai`` and leave an eval log. healthbench
and arenahardwriting do not: their ``evaluate.py`` generates, judges, and writes a metrics
dict, and that is the whole artefact. Run against those, the strict path above found no log,
refused every single time -- and, because a refusal quarantines, *renamed their perfectly
good metrics file to* ``*.UNVERIFIED``. A checker that destroys the only copy of the result
on a task it was never able to check is not conservative, it is a data-loss bug wearing a
refusal's clothes.

So there are two decision bases, and every record names the one it used in ``decided_on``:

``inspect_log``
    the five checks in the table above.
``exit_code+metrics_file``
    the command exited 0, and this run wrote a metrics file that parses to a non-empty
    JSON object carrying a numeric ``accuracy``. The row count is checked only if the
    metrics dict carries one (healthbench's ``n_examples``); when it does not
    (arenahardwriting) the record says ``rows_verified: false`` and gives the reason.
    The checkpoint identity is not checked at all, and the record does not pretend it was.

Which basis applies is decided by *reading the evaluation script that is about to run*, not
by a task-name list that has to be maintained beside the tasks: ``evaluate_uses_inspect()``
below looks for ``inspect_ai`` in the file named by ``--evaluate``. ``--expect-inspect-log
yes|no`` overrides it, and ``yes`` is the way to keep a gsm8k read strict if that file is
ever restructured. ``src/eval/general/get_prompt.py`` imports the same predicate to decide
which of the two the prompt bullet describes, so the tool and the sentence advertising it
cannot disagree.

That prediction is checked against the read rather than trusted: in ``auto``, "no log
expected" only takes effect if the read really did write no log, and a read that writes one
anyway is verified against it under ``inspect_log``. A prediction is allowed to *add*
checks, never to remove one that turned out to be available -- which is what keeps a stale
substring from silently disarming the row-count and checkpoint checks on a task that still
writes logs.

Dependency-free by construction: standard library only. It runs inside the agent sandbox,
where an import that needs installing is a helper that gets skipped, and it must keep
working on the day the read it is wrapping fails because the environment is broken.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

#: Version tag on every record. Bump it if a field changes meaning, never if one is added:
#: the records outlive the run and an auditor reading a directory of them months later has
#: no other way to know which reader to use.
RECORD_SCHEMA = "ptb.graded_read/1"

EXIT_OK = 0
EXIT_COMMAND_FAILED = 3
EXIT_OUTPUT_MISSING = 4
EXIT_OUTPUT_UNREADABLE = 5
EXIT_ROW_COUNT_MISMATCH = 6
#: 7 is the setup refusal (bad checkpoint path, missing evaluate.py) -- a refusal made
#: before anything was spent, kept apart from the five that describe a read that ran.
EXIT_SETUP_REFUSED = 7
EXIT_CHECKPOINT_MISMATCH = 8

#: The two decision bases, spelled the way they appear in ``record["decided_on"]`` and on
#: the stdout line. Constants and not literals because a reader grepping a directory of
#: records for "which of these were only exit-code checked" needs the string to be one
#: string.
DECIDED_ON_INSPECT_LOG = "inspect_log"
DECIDED_ON_EXIT_AND_METRICS = "exit_code+metrics_file"

#: Keys a non-inspect metrics dict may use for "how many items were graded". healthbench
#: writes ``n_examples``; arenahardwriting writes none of these, and that case is reported
#: rather than assumed away. Checked in order, first hit wins.
ROW_COUNT_KEYS = ("n_examples", "num_examples", "n_samples", "total_samples", "n")

#: Tolerance on the metrics-file-vs-log accuracy cross-check. Measured to be unnecessary
#: and kept anyway: on 89727_g5's ``logs/grpo_main_1319_r1.json`` against its inspect log
#: ``2026-09-02T15-45-19+00-00_gsm8k_7eHarPmzTi8ghGw2EKzTLv.json`` the two dicts compare
#: ``==`` exactly (0.7808946171341926 and 0.011393706634978006 both round-trip through
#: ``json.dump`` at full repr precision), because ``evaluate.py`` copies the float out of
#: the same log object. The tolerance is here only so that an ``evaluate.py`` which starts
#: rounding its output does not turn this cross-check into a permanent refusal; it is far
#: tighter than any real disagreement, which is a whole different read.
METRIC_AGREEMENT_TOLERANCE = 1e-9

#: inspect prints the log it wrote as a trailing ``Log: <path>`` line -- verbatim from a
#: real run: ``Log: logs/2026-08-31T18-55-37+00-00_gsm8k_T4XeLgGJcBTxyK6eWsmWMf.json``.
#: Parsed as one of three independent ways to find the log, never as the only one, because
#: a display change upstream would otherwise silently turn the row check off.
_LOG_LINE_RE = re.compile(r"^\s*Log:\s+(\S+\.(?:json|eval))\s*$")


class Refusal(Exception):
    """A read that will not be reported, and the exit code that says which kind.

    An exception rather than a return value because every check below is a place where
    the honest thing is to stop, and the alternative shape -- a status threaded through
    ten call sites -- is how a check ends up with a caller that forgets to look at it.
    That is the defect this file exists to prevent, and it would be embarrassing to
    reproduce it internally.
    """

    def __init__(self, code: int, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.kind = kind
        self.detail = detail


def _json_or_refuse(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a truncated file raises many things
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"{what} at {path} is not readable JSON: {type(exc).__name__}: {exc}",
        ) from exc


def describe_checkpoint(model_path: str) -> dict:
    """What was actually read, in enough detail to catch a swap after the fact.

    The resolved path alone is not enough. 89727_g5's bad readout named
    ``runs/grpo_h8_k10/final`` while the log underneath it was a read of
    ``runs/grpo_main_k0/final``, and both paths existed -- the two are told apart by their
    *contents*. Size and newest-mtime over the weight shards are cheap (a stat per file,
    no read) and change on every save, so two records that quote the same bytes and mtime
    are reads of the same weights.
    """
    resolved = os.path.realpath(model_path)
    info: dict = {"argument": model_path, "resolved": resolved}
    p = Path(resolved)
    if not p.is_dir():
        info["exists"] = False
        return info
    info["exists"] = True

    shards = sorted(
        [f for pat in ("*.safetensors", "*.bin") for f in p.glob(pat)],
        key=lambda f: f.name,
    )
    total = 0
    newest = 0.0
    for f in shards:
        try:
            st = f.stat()
        except OSError:
            continue
        total += st.st_size
        newest = max(newest, st.st_mtime)
    info["weight_files"] = [f.name for f in shards]
    info["weight_bytes"] = total
    info["weight_mtime"] = newest or None

    # config.json is read for identity only: model_type and architectures are what
    # model_identity_check.py compares, so quoting them here lets the two agree or
    # visibly disagree instead of each having its own private idea of what was scored.
    cfg = p / "config.json"
    if cfg.is_file():
        try:
            conf = json.loads(cfg.read_text(encoding="utf-8"))
            info["model_type"] = conf.get("model_type")
            info["architectures"] = conf.get("architectures")
        except Exception:  # noqa: BLE001 - an unreadable config is a fact, not a crash
            info["config_unreadable"] = True

    # The decode the grader will actually use. evaluate.py passes no temperature, so vLLM
    # reads this file; recording it means a record can be re-read later to answer "was
    # this number a greedy read?" without the checkpoint still being on disk.
    gen = p / "generation_config.json"
    if gen.is_file():
        try:
            conf = json.loads(gen.read_text(encoding="utf-8"))
            info["generation_config"] = {
                k: conf.get(k) for k in ("do_sample", "temperature", "top_p", "top_k")
                if k in conf
            }
        except Exception:  # noqa: BLE001
            info["generation_config_unreadable"] = True
    return info


def find_inspect_log(
    *,
    child_output: str,
    private_log_dir: Path,
    fallback_log_dir: Path,
    pre_existing: set[str],
    started_at: float,
) -> tuple[Path, list[str]]:
    """The one log this read wrote, or a refusal -- and never "the newest one".

    Three independent rules, because each can fail on its own and a single rule that
    fails silently turns every check downstream into a formality:

    1. the ``Log:`` line inspect prints, which breaks if the display format changes;
    2. anything in the private ``INSPECT_LOG_DIR`` handed to the child, which breaks on
       an inspect old enough not to read that variable;
    3. anything under the conventional ``logs/`` that was not there before we started,
       which breaks if the child writes somewhere else entirely.

    Candidates are de-duplicated on ``realpath`` and the result must be exactly one. Zero
    is :data:`EXIT_OUTPUT_MISSING`; two or more is :data:`EXIT_OUTPUT_UNREADABLE`, not a
    coin-flip -- resolving ambiguity by taking the most recent is precisely
    ``eval_readout.py --latest``, and it is what wrote 89727_g5's fabricated readout.

    Note what rule 3 is *not*: it is "new since this read started", not "newest on disk".
    The stale log that poisoned 89727_g5 predated its read and could not have passed it.
    """
    found: dict[str, list[str]] = {}

    def offer(path: Path, how: str) -> None:
        try:
            if not path.is_file():
                return
            key = os.path.realpath(path)
        except OSError:
            return
        found.setdefault(key, []).append(how)

    for line in child_output.splitlines():
        m = _LOG_LINE_RE.match(line)
        if m:
            candidate = Path(m.group(1))
            # Only if it is this read's log. A `Log:` line naming a file that predates the
            # run is exactly the stale-log case, and it must not be adopted.
            try:
                if candidate.is_file() and candidate.stat().st_mtime >= started_at - 1.0:
                    offer(candidate, "log-line")
            except OSError:
                pass

    if private_log_dir.is_dir():
        for f in sorted(private_log_dir.glob("*.json")):
            offer(f, "private-log-dir")
        for f in sorted(private_log_dir.glob("*.eval")):
            offer(f, "private-log-dir")

    if fallback_log_dir.is_dir():
        for f in sorted(fallback_log_dir.glob("*.json")):
            if os.path.realpath(f) not in pre_existing:
                offer(f, "new-in-log-dir")
        for f in sorted(fallback_log_dir.glob("*.eval")):
            if os.path.realpath(f) not in pre_existing:
                offer(f, "new-in-log-dir")

    if not found:
        raise Refusal(
            EXIT_OUTPUT_MISSING,
            "output_missing",
            "the command exited 0 but wrote no inspect log this helper could find "
            f"(looked for a 'Log:' line, in {private_log_dir}, and for new files in "
            f"{fallback_log_dir}). Nothing here is a measurement.",
        )
    if len(found) > 1:
        listing = ", ".join(f"{k} [{'+'.join(v)}]" for k, v in sorted(found.items()))
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"{len(found)} candidate inspect logs for one read, which one is the "
            f"measurement is not decidable: {listing}",
        )

    key, hows = next(iter(found.items()))
    path = Path(key)
    if path.suffix == ".eval":
        # The binary .eval bundle carries the same results block, but reading it here
        # would be a guess at a third-party archive layout that nothing in this repo can
        # check. Refusing with the one-line fix is honest; parsing it blind is how a row
        # check quietly starts returning whatever it happens to find.
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"the inspect log at {path} is the binary .eval format, whose row counts this "
            "helper does not read. Pass log_format='json' in evaluate.py (the six "
            "inspect-based tasks already do) or set INSPECT_LOG_FORMAT=json.",
        )
    return path, sorted(set(hows))


def check_rows(log: dict, rows_requested: int) -> dict:
    """Compare what the log says it ran against what was asked for.

    Deliberately does not consult ``eval.dataset.samples``: see the module docstring, it
    reads 1319 on 150-row reads and would make this function vacuous.
    """
    results = log.get("results")
    if not isinstance(results, dict):
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            "the inspect log has no 'results' block, so the read produced no scores",
        )

    config = (log.get("eval") or {}).get("config") or {}
    observed = {
        "status": log.get("status"),
        "total_samples": results.get("total_samples"),
        "completed_samples": results.get("completed_samples"),
        "scored_samples": {
            (s.get("name") or s.get("scorer") or f"scorer{i}"): s.get("scored_samples")
            for i, s in enumerate(results.get("scores") or [])
        },
        "unscored_samples": {
            (s.get("name") or s.get("scorer") or f"scorer{i}"): s.get("unscored_samples")
            for i, s in enumerate(results.get("scores") or [])
        },
        # Recorded, not checked. epochs>1 multiplies total_samples, so it is the first
        # thing that explains a mismatch of an exact multiple -- and a reader who has the
        # number does not have to re-open the log to find it.
        "epochs": config.get("epochs"),
        "limit": config.get("limit"),
        "dataset_samples": ((log.get("eval") or {}).get("dataset") or {}).get("samples"),
    }

    if log.get("status") != "success":
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"inspect recorded status={log.get('status')!r}, not 'success'; a run that "
            "did not finish still writes an accuracy over the rows it managed",
        )

    mismatches = []
    for field in ("total_samples", "completed_samples"):
        if observed[field] != rows_requested:
            mismatches.append(f"{field}={observed[field]}")
    for name, n in observed["scored_samples"].items():
        if n != rows_requested:
            mismatches.append(f"scores[{name}].scored_samples={n}")

    if mismatches:
        raise Refusal(
            EXIT_ROW_COUNT_MISMATCH,
            "row_count_mismatch",
            f"asked for {rows_requested} rows, the log says " + ", ".join(mismatches)
            + f" (epochs={observed['epochs']}, config.limit={observed['limit']}). "
            "This is the 89727_g5 defect: a readout labelled with one n carrying "
            "another n's accuracy.",
        )
    return observed


def check_checkpoint(log: dict, model_path: str) -> str:
    """Refuse a log that is a read of some other checkpoint.

    ``evaluate.py`` passes ``model=f"vllm/{args.model_path}"``, so the log's ``eval.model``
    is the literal argument with a ``vllm/`` prefix -- verified against a real archived
    log, which reads ``vllm/runs/grpo_main_k0/final``. Comparing realpaths rather than
    strings so that ``./runs/x`` and ``runs/x`` agree, while a genuinely different
    checkpoint does not. Half of 89727_g5's bad readout was exactly this: the right-looking
    filename over a read of a different model.
    """
    logged = ((log.get("eval") or {}).get("model") or "")
    stripped = logged.split("/", 1)[1] if "/" in logged else logged
    if os.path.realpath(stripped) != os.path.realpath(model_path):
        raise Refusal(
            EXIT_CHECKPOINT_MISMATCH,
            "checkpoint_mismatch",
            f"asked to read {model_path!r} (-> {os.path.realpath(model_path)}) but the "
            f"inspect log is a read of {logged!r} (-> {os.path.realpath(stripped)})",
        )
    return logged


def check_metrics(metrics_path: Path, log: dict) -> dict:
    """The metrics file, cross-checked against the log it claims to summarise.

    ``evaluate.py`` writes this file by flattening one scorer's metrics out of the very
    log object it is about, so the two agree bit for bit or the metrics file is left over
    from an earlier read -- the stale-artefact half of the same defect. Verified on a real
    pair: ``{'accuracy': 0.7808946171341926, 'stderr': 0.011393706634978006}`` compares
    ``==`` against the log's flattened metrics.
    """
    if not metrics_path.is_file():
        raise Refusal(
            EXIT_OUTPUT_MISSING,
            "output_missing",
            f"the command exited 0 and wrote no metrics file at {metrics_path}",
        )
    metrics = _json_or_refuse(metrics_path, "the metrics file")
    if not isinstance(metrics, dict) or not metrics:
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"the metrics file at {metrics_path} is not a non-empty JSON object: "
            f"{metrics!r}",
        )

    from_log: dict = {}
    for scorer in ((log.get("results") or {}).get("scores") or []):
        for name, metric in (scorer.get("metrics") or {}).items():
            from_log.setdefault(name, metric.get("value"))

    agreed = []
    for name, value in metrics.items():
        if name not in from_log:
            continue
        theirs = from_log[name]
        if not isinstance(value, (int, float)) or not isinstance(theirs, (int, float)):
            continue
        if not math.isclose(float(value), float(theirs), rel_tol=0.0,
                            abs_tol=METRIC_AGREEMENT_TOLERANCE):
            raise Refusal(
                EXIT_OUTPUT_UNREADABLE,
                "output_unreadable",
                f"the metrics file and the inspect log disagree on {name}: "
                f"{value} vs {theirs}. One of the two is left over from another read.",
            )
        agreed.append(name)

    if not agreed:
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"nothing in the metrics file {sorted(metrics)} could be matched against the "
            f"log's metrics {sorted(from_log)}, so the metrics file is uncorroborated",
        )
    if "accuracy" not in metrics:
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"the metrics file has no 'accuracy': {sorted(metrics)}",
        )
    return {"metrics": metrics, "corroborated": sorted(agreed)}


def evaluate_uses_inspect(evaluate_path) -> bool:
    """Does this evaluation script produce an inspect log?

    Answered by reading the file that is about to run, because that is the only form of
    the question that stays true when a task is added, renamed, or given an
    ``evaluate_openrouter.py`` variant. The six inspect-based tasks all
    ``import inspect_ai`` (``from inspect_ai import eval as inspect_eval`` in
    ``src/eval/tasks/<task>/evaluate.py``); healthbench and arenahardwriting mention it
    nowhere, in either of their two evaluate variants.

    A substring test and not an AST walk on purpose: this file is stdlib-only and runs
    inside a sandbox on the day the environment is broken, and the failure mode of the
    substring test is one-directional in the safe direction. A script that names
    ``inspect_ai`` anywhere -- even in a comment -- is treated as producing a log, which
    keeps the strict checks on; the mistake it cannot make is quietly *downgrading* a task
    that really does write one.

    Raises ``FileNotFoundError`` rather than defaulting, because "I could not read the
    grader" is not a fact about which mode to use.
    """
    return "inspect_ai" in Path(evaluate_path).read_text(encoding="utf-8", errors="replace")


def check_metrics_without_log(metrics_path: Path, rows_requested: int) -> dict:
    """Everything that can honestly be checked when the task writes no inspect log.

    Which is: the file is here, this run wrote it (``main`` deletes any earlier one under
    the same label before starting, so its presence means this read produced it), it parses
    to a non-empty JSON object, and it carries a numeric ``accuracy``. Plus the row count
    *if the metrics dict happens to carry one* -- and an explicit
    ``rows_verified: False`` with a reason when it does not, because "the check did not run"
    and "the check passed" have to be different words in the record. arenahardwriting's
    ``_compute_metrics`` returns ``{"accuracy": ..., "stderr": ...}`` and nothing else;
    healthbench's returns ``n_examples`` as well.
    """
    if not metrics_path.is_file():
        raise Refusal(
            EXIT_OUTPUT_MISSING,
            "output_missing",
            f"the command exited 0 and wrote no metrics file at {metrics_path}. This "
            "benchmark's grading writes no inspect log either, so there is nothing left "
            "that could be a measurement.",
        )
    metrics = _json_or_refuse(metrics_path, "the metrics file")
    if not isinstance(metrics, dict) or not metrics:
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"the metrics file at {metrics_path} is not a non-empty JSON object: {metrics!r}",
        )
    if "accuracy" not in metrics:
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"the metrics file has no 'accuracy': {sorted(metrics)}",
        )
    if not isinstance(metrics["accuracy"], (int, float)) or isinstance(metrics["accuracy"], bool):
        raise Refusal(
            EXIT_OUTPUT_UNREADABLE,
            "output_unreadable",
            f"the metrics file's 'accuracy' is not a number: {metrics['accuracy']!r}",
        )

    for key in ROW_COUNT_KEYS:
        value = metrics.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            if value != rows_requested:
                raise Refusal(
                    EXIT_ROW_COUNT_MISMATCH,
                    "row_count_mismatch",
                    f"the metrics file reports {key}={value} but this read asked for "
                    f"{rows_requested} rows. Whatever produced that number is not the read "
                    "that was requested.",
                )
            return {"metrics": metrics, "rows_verified": True,
                    "rows_check": f"metrics['{key}'] == {rows_requested}"}

    return {
        "metrics": metrics,
        "rows_verified": False,
        "rows_check": (
            f"NOT CHECKED: no inspect log for this benchmark, and its metrics file carries "
            f"none of {list(ROW_COUNT_KEYS)}, so nothing here corroborates --rows "
            f"{rows_requested}"
        ),
    }


def build_command(args, passthrough: list[str]) -> list[str]:
    """The evaluate.py invocation, in one place so the record can quote what ran.

    ``--limit`` is ``-1`` under ``--full`` and the requested row count otherwise. The two
    are the same read for gsm8k, whose test split is 1319 rows, and they are not the same
    *claim*: ``--full --rows 1319`` says "give me the whole split, and I assert it is 1319
    rows", which is the assertion that fails loudly on the day the split changes size.
    """
    limit = "-1" if args.full else str(args.rows)
    return [
        args.python, args.evaluate,
        "--model-path", args.model_path,
        "--limit", limit,
        "--json-output-file", str(args.metrics_path),
        *passthrough,
    ]


def run_child(cmd: list[str], transcript: Path, echo: bool, env: dict) -> tuple[int, str]:
    """Run the read, tee its output, and return its exit code -- to a caller that looks.

    Output goes to the transcript file and to *stderr*, never stdout: stdout carries only
    what was verified, so ``ACC=$(python graded_read.py --print accuracy ...)`` is a number
    or nothing, and cannot be a progress bar.

    ``env`` is passed explicitly rather than by mutating :data:`os.environ`, so the log-dir
    pin below applies to the child and to nothing else -- a helper that edits its own
    process environment changes the behaviour of whatever the agent runs next.
    """
    lines: list[str] = []
    with transcript.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace", env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            fh.write(line)
            if echo:
                sys.stderr.write(line)
        proc.wait()
    return proc.returncode, "".join(lines)


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog="graded_read.py",
        description=(
            "Run evaluate.py against a checkpoint and report the score only if the row "
            "count, the checkpoint identity and the exit code all check out. Writes a "
            "record beside the result in every case, including refusals."
        ),
        epilog=(
            "example: python graded_read.py --model-path runs/grpo_main/final "
            "--rows 1319 --full --label grpo_main_r1"
        ),
    )
    parser.add_argument("--model-path", required=True,
                        help="checkpoint directory to read (as evaluate.py --model-path)")
    parser.add_argument("--rows", required=True, type=int,
                        help="how many rows this read must cover; refused if the log "
                             "disagrees. The gsm8k test split is 1319.")
    parser.add_argument("--full", action="store_true",
                        help="pass --limit -1 to evaluate.py (whole split) while still "
                             "asserting --rows against what the log reports")
    parser.add_argument("--label", default=None,
                        help="name for the output files; defaults to the checkpoint "
                             "directory name plus the row count")
    parser.add_argument("--out-dir", default="graded_reads",
                        help="where the metrics, record and transcript are written "
                             "(default: graded_reads)")
    parser.add_argument("--evaluate", default="evaluate.py",
                        help="the evaluation script to run (default: evaluate.py)")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter used to run it (default: this one)")
    parser.add_argument("--log-dir", default="logs",
                        help="where inspect writes its logs by default, scanned for a "
                             "log that is new since this read started (default: logs)")
    parser.add_argument("--expect-inspect-log", default="auto", choices=("auto", "yes", "no"),
                        help="whether this benchmark's grading writes an inspect log. "
                             "'auto' (default) decides by reading the --evaluate script for "
                             "'inspect_ai', and still uses a log if the read writes one "
                             "unexpectedly; 'no' checks only the exit code and the metrics "
                             "file (healthbench / arenahardwriting write no log); 'yes' "
                             "forces the full row-count and checkpoint checks")
    parser.add_argument("--print", dest="print_what", default="line",
                        choices=("line", "accuracy", "record", "none"),
                        help="what a verified read puts on stdout (default: line)")
    parser.add_argument("--quiet", action="store_true",
                        help="do not tee the child's output to stderr; it still goes to "
                             "the transcript file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Split on a literal `--` by hand rather than with argparse.REMAINDER, which is
    # documented as unstable and silently swallows flags it does not expect. Everything
    # after it is forwarded to evaluate.py untouched (e.g. --max-connections 1).
    passthrough: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    args = parse_args(argv)

    label = args.label or f"{Path(args.model_path.rstrip('/')).name}_n{args.rows}"
    out_dir = Path(args.out_dir)
    args.metrics_path = out_dir / f"{label}.metrics.json"
    record_path = out_dir / f"{label}.graded_read.json"
    transcript_path = out_dir / f"{label}.transcript.txt"
    private_log_dir = out_dir / f"{label}.inspectlog"
    fallback_log_dir = Path(args.log_dir)

    record: dict = {
        "schema": RECORD_SCHEMA,
        "label": label,
        "verdict": "refused",
        "rows_requested": args.rows,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    exit_code = EXIT_SETUP_REFUSED

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        private_log_dir.mkdir(parents=True, exist_ok=True)

        if args.rows <= 0:
            raise Refusal(EXIT_SETUP_REFUSED, "setup_refused",
                          f"--rows must be a positive count, got {args.rows}")
        if not Path(args.evaluate).is_file():
            raise Refusal(EXIT_SETUP_REFUSED, "setup_refused",
                          f"no evaluation script at {args.evaluate} (cwd {os.getcwd()})")

        # Decided before the read, and recorded whether the read succeeds or not, so a
        # directory of records answers "which of these were only exit-code checked" without
        # re-deriving anything.
        if args.expect_inspect_log == "auto":
            expect_log = evaluate_uses_inspect(Path(args.evaluate))
            because = (f"auto: {args.evaluate} "
                       f"{'mentions' if expect_log else 'does not mention'} inspect_ai")
        else:
            expect_log = args.expect_inspect_log == "yes"
            because = f"--expect-inspect-log {args.expect_inspect_log}"
        record["decided_on"] = (DECIDED_ON_INSPECT_LOG if expect_log
                                else DECIDED_ON_EXIT_AND_METRICS)
        record["decided_on_because"] = because

        record["checkpoint"] = describe_checkpoint(args.model_path)
        if not record["checkpoint"]["exists"]:
            raise Refusal(EXIT_SETUP_REFUSED, "setup_refused",
                          f"--model-path {args.model_path!r} is not a directory")

        # Anything left from a previous read under this label is removed, not overwritten:
        # if this read dies before evaluate.py writes, a leftover metrics file would be
        # cross-checked against this run's log and could pass. Removing it makes "the file
        # is here" mean "this run wrote it".
        for stale in (args.metrics_path, out_dir / f"{label}.metrics.json.UNVERIFIED"):
            if stale.exists():
                stale.unlink()
                record.setdefault("removed_stale", []).append(str(stale))
        for stale in private_log_dir.glob("*"):
            if stale.is_file():
                stale.unlink()

        cmd = build_command(args, passthrough)
        record["command"] = cmd
        record["command_str"] = " ".join(shlex.quote(c) for c in cmd)

        pre_existing = set()
        if fallback_log_dir.is_dir():
            pre_existing = {
                os.path.realpath(f)
                for f in list(fallback_log_dir.glob("*.json"))
                + list(fallback_log_dir.glob("*.eval"))
            }

        env = dict(os.environ)
        # A private log dir per label removes the ambiguity at the source instead of
        # resolving it afterwards. INSPECT_LOG_FORMAT is set for the two tasks whose
        # evaluate.py does not pass log_format='json'; the six that do pass it win, which
        # is the same value anyway.
        env["INSPECT_LOG_DIR"] = str(private_log_dir.resolve())
        env.setdefault("INSPECT_LOG_FORMAT", "json")
        record["child_env_pins"] = {k: env[k]
                                    for k in ("INSPECT_LOG_DIR", "INSPECT_LOG_FORMAT")}

        started_at = time.time()
        t0 = time.monotonic()
        rc, output = run_child(cmd, transcript_path, echo=not args.quiet, env=env)
        record["wall_seconds"] = round(time.monotonic() - t0, 3)
        record["exit_code"] = rc
        record["transcript"] = str(transcript_path)

        if rc != 0:
            tail = "".join(output.splitlines(keepends=True)[-15:]).strip()
            raise Refusal(
                EXIT_COMMAND_FAILED, "command_failed",
                f"{args.evaluate} exited {rc}. Nothing below this line is a measurement. "
                f"Last lines follow (full output in {transcript_path}):\n{tail}",
            )

        # `expect_log` is a prediction made before the read, off the text of the grader.
        # In `auto` it is only allowed to *weaken* the checks if the read then really does
        # produce no log: predicting "no log" and finding one means the prediction was
        # wrong, and the strong checks are available, so they run. That ordering matters
        # more than it looks -- the substring predicate is the kind of thing that goes
        # stale silently (a task switches to an `inspect_evals` import, a wrapper stops
        # naming the package) and this is what stops a stale predicate from turning the
        # row-count and checkpoint checks off on a task that could have had them.
        # `--expect-inspect-log no` is an explicit instruction and is not second-guessed.
        found_log = None
        if expect_log:
            found_log = find_inspect_log(
                child_output=output,
                private_log_dir=private_log_dir,
                fallback_log_dir=fallback_log_dir,
                pre_existing=pre_existing,
                started_at=started_at,
            )
        elif args.expect_inspect_log == "auto":
            try:
                found_log = find_inspect_log(
                    child_output=output,
                    private_log_dir=private_log_dir,
                    fallback_log_dir=fallback_log_dir,
                    pre_existing=pre_existing,
                    started_at=started_at,
                )
            except Refusal as probe:
                # "no log at all" confirms the prediction; anything else (two candidates, a
                # binary .eval) is a real ambiguity about a log that does exist, and is not
                # something to fall back past.
                if probe.kind != "output_missing":
                    raise
                record["decided_on_because"] += ", and this read wrote none"
            else:
                expect_log = True
                record["decided_on"] = DECIDED_ON_INSPECT_LOG
                record["decided_on_because"] = (
                    f"auto: {args.evaluate} does not mention inspect_ai, but this read "
                    f"wrote an inspect log anyway, so it is checked against it"
                )

        if found_log is not None:
            log_path, resolved_by = found_log
            record["inspect_log"] = str(log_path)
            record["inspect_log_resolved_by"] = resolved_by

            log = _json_or_refuse(log_path, "the inspect log")
            record["logged_model"] = check_checkpoint(log, args.model_path)
            record["rows_found"] = check_rows(log, args.rows)
            checked = check_metrics(args.metrics_path, log)
            record["corroborated_metrics"] = checked["corroborated"]
            record["rows_verified"] = True
        else:
            checked = check_metrics_without_log(args.metrics_path, args.rows)
            record["rows_verified"] = checked["rows_verified"]
            record["rows_check"] = checked["rows_check"]
            # Absent on purpose, and the absence is the honest part: nothing here read a
            # log, so there is no `logged_model` to compare the checkpoint against and no
            # `inspect_log` to point at. A null would read as "checked, came back empty".
            record["checkpoint_identity_checked"] = False

        record["verdict"] = "verified"
        record["metrics"] = checked["metrics"]
        # `accuracy` is a top-level key ONLY on a verified record. A reader that does
        # `record["accuracy"]` gets a KeyError on every refusal instead of a default, and
        # `record.get("accuracy", 0)` -- the shape that produces silent zeros -- has
        # nothing to find. This absence is load-bearing; do not add a null here.
        record["accuracy"] = checked["metrics"]["accuracy"]
        record["metrics_file"] = str(args.metrics_path)
        exit_code = EXIT_OK

    except Refusal as refusal:
        record["failure"] = {"code": refusal.code, "kind": refusal.kind,
                             "detail": refusal.detail}
        exit_code = refusal.code
    except Exception as exc:  # noqa: BLE001 - a crash must still leave a record
        record["failure"] = {"code": 1, "kind": "graded_read_crashed",
                             "detail": f"{type(exc).__name__}: {exc}"}
        exit_code = 1
    finally:
        record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if record["verdict"] != "verified" and getattr(args, "metrics_path", None):
            # Quarantine, not delete: the numbers are evidence for the post-mortem, and a
            # deleted file is a defect nobody can reconstruct. Renaming is what keeps the
            # invariant that `*.metrics.json` means verified.
            if args.metrics_path.is_file():
                quarantined = Path(str(args.metrics_path) + ".UNVERIFIED")
                args.metrics_path.replace(quarantined)
                record["quarantined_metrics_file"] = str(quarantined)
        # Written last and atomically, so a record that exists is a record that is
        # complete -- a half-written record read by an audit is the same class of bug as
        # a half-checked read.
        tmp = Path(str(record_path) + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n",
                       encoding="utf-8")
        tmp.replace(record_path)

    if exit_code == EXIT_OK:
        if args.print_what == "accuracy":
            print(record["accuracy"])
        elif args.print_what == "record":
            print(record_path)
        elif args.print_what == "line":
            # `rows=N` on its own would claim a check that the weaker basis cannot make, so
            # an unverified count says so in the same breath as the number it qualifies.
            rows = f"rows={args.rows}" if record.get("rows_verified") \
                else f"rows={args.rows}(UNVERIFIED)"
            print(f"graded_read: VERIFIED {label} {rows} "
                  f"accuracy={record['accuracy']} "
                  f"decided_on={record.get('decided_on')} record={record_path}")
        return EXIT_OK

    failure = record.get("failure", {})
    sys.stderr.write(
        f"\ngraded_read: REFUSED [{failure.get('kind')}] {label}\n"
        f"  {failure.get('detail')}\n"
        f"  decided_on={record.get('decided_on')} ({record.get('decided_on_because')})\n"
        f"  no number was produced; record written to {record_path}\n"
        f"  exit {exit_code}\n"
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
