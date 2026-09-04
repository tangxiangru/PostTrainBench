#!/usr/bin/env python3
"""Make the answer-key audit operative: mark READ cells so nothing scores them.

`ptb_ops/answer_key_audit.py` grades every cell READ / PATH / NAME and writes a
manifest. Knowing which cells are contaminated changes nothing on its own -- the
aggregators walk the results tree and read whatever `metrics.json` they find. This
script closes that gap by acting on the manifest:

  * writes `<cell>/VOIDED_ANSWER_KEY.json` -- the marker every reader checks, and
    the record of *why*, including the audit that produced it and the score being
    withdrawn;
  * renames `metrics.json` to `metrics.json.VOID_ANSWER_KEY`.

Quarantine, not delete: the numbers are evidence, and the rename is what keeps the
invariant that a `metrics.json` on disk is a score anyone may quote. Same
convention as `src/utils/graded_read.py`'s `*.metrics.json.UNVERIFIED`.

The exclusion list is *generated*, never typed here. A hand-maintained list beside
a detector drifts from it the first time either changes, and then the thing that
finds contamination and the thing that acts on it disagree silently.

    ptb_ops/void_cells.py [RESULTS_ROOT] [--manifest PATH] [--dry-run]
    ptb_ops/void_cells.py [RESULTS_ROOT] --undo        # restore, e.g. false positive

Exit 0 if the tree now matches the manifest, 1 if anything could not be applied.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_key_audit import default_manifest_path  # noqa: E402

MARKER = "VOIDED_ANSWER_KEY.json"
VOID_SUFFIX = ".VOID_ANSWER_KEY"
REASON = ("read RTT1/posttrainbench-gsm8k-recipes, the shipped-recipe corpus for the "
          "exact task this cell is graded on")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_atomic(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    tmp.replace(path)


def void_cell(cell: Path, rec: dict, audit: dict, dry: bool) -> tuple[bool, str]:
    """-> (changed, one-line description)."""
    marker = cell / MARKER
    metrics = cell / "metrics.json"
    voided = Path(str(metrics) + VOID_SUFFIX)

    if marker.exists() and not metrics.exists():
        return False, "already voided"

    withdrawn = None
    if metrics.is_file():
        try:
            withdrawn = json.loads(metrics.read_text())
        except (OSError, json.JSONDecodeError) as e:
            withdrawn = {"unreadable": str(e)}
    elif voided.is_file():
        # Marker was removed by hand but the score is still quarantined.
        try:
            withdrawn = json.loads(voided.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    doc = {
        "schema": 1,
        "voided_at": _now(),
        "reason": REASON,
        "verdict": rec.get("verdict"),
        "evidence": rec.get("evidence", []),
        "executed_readers": rec.get("executed_readers", []),
        "audit": {
            "generated_at": audit.get("generated_at"),
            "detector": audit.get("detector", {}),
            "manifest": audit.get("_manifest_path"),
        },
        "withdrawn_metrics": withdrawn,
        "restore": (f"ptb_ops/void_cells.py {cell.parent.parent} --undo  "
                    f"# only if the audit is wrong, and say so in the note"),
    }
    if dry:
        return True, f"would void (accuracy={_acc(withdrawn)})"
    _write_atomic(marker, doc)
    if metrics.is_file():
        metrics.replace(voided)
    return True, f"voided (accuracy={_acc(withdrawn)} withdrawn)"


def unvoid_cell(cell: Path, dry: bool) -> tuple[bool, str]:
    marker = cell / MARKER
    metrics = cell / "metrics.json"
    voided = Path(str(metrics) + VOID_SUFFIX)
    if not marker.exists() and not voided.exists():
        return False, "not voided"
    if metrics.exists() and voided.exists():
        return False, ("REFUSED: both metrics.json and its voided copy exist -- "
                       "something rescored this cell; resolve by hand")
    if dry:
        return True, "would restore"
    if voided.is_file():
        voided.replace(metrics)
    marker.unlink(missing_ok=True)
    return True, "restored"


def _acc(m) -> str:
    if isinstance(m, dict) and isinstance(m.get("accuracy"), (int, float)):
        return f"{m['accuracy']:.4f}"
    return "n/a"


def _flag_value(name: str, argv: list[str]) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def main() -> int:
    argv = sys.argv[1:]
    dry = "--dry-run" in argv or "-n" in argv
    undo = "--undo" in argv
    positional, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--manifest":
            skip = True
            continue
        if not a.startswith("-"):
            positional.append(a)
    root = Path(positional[0]) if positional else Path(
        os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "/rmeng_data/robtang/ptb-results"))
    if not root.is_dir():
        print(f"no such results root: {root}", file=sys.stderr)
        return 2

    manifest_arg = _flag_value("--manifest", argv)
    manifest = Path(manifest_arg) if manifest_arg else default_manifest_path(root)

    if undo:
        # Undo works off the tree, not the manifest: the reason to undo is usually
        # that the manifest is wrong.
        n = 0
        for cell in sorted(root.glob(f"*/*/{MARKER}")):
            changed, msg = unvoid_cell(cell.parent, dry)
            print(f"  {cell.parent.parent.name}/{cell.parent.name}: {msg}")
            n += changed
        print(f"{'would restore' if dry else 'restored'} {n} cell(s)")
        return 0

    if not manifest.is_file():
        print(f"no manifest at {manifest} -- run ptb_ops/answer_key_audit.py first",
              file=sys.stderr)
        return 2
    audit = json.loads(manifest.read_text())
    audit["_manifest_path"] = str(manifest)

    read_cells = [r for r in audit.get("cells", []) if r.get("verdict") == "READ"]
    print(f"manifest {manifest}\n  generated {audit.get('generated_at')} "
          f"by {audit.get('detector', {}).get('commit') or 'unknown'}; "
          f"{len(read_cells)} READ of {audit.get('counts', {}).get('scanned', '?')} scanned")

    rc, changed = 0, 0
    listed = set()
    for rec in read_cells:
        cell = Path(rec["path"])
        listed.add(str(cell))
        if not cell.is_dir():
            print(f"  MISSING {cell}", file=sys.stderr)
            rc = 1
            continue
        did, msg = void_cell(cell, rec, audit, dry)
        changed += did
        print(f"  {cell.parent.name}/{cell.name}: {msg}")

    # A cell marked in a previous run that the current audit no longer calls READ is
    # not silently released -- that would let a detector regression un-void a real
    # leak. Report it and make the operator choose.
    for marker in sorted(root.glob(f"*/*/{MARKER}")):
        if str(marker.parent) not in listed:
            print(f"  STALE {marker.parent.parent.name}/{marker.parent.name}: voided on "
                  f"disk but not READ in this manifest -- use --undo if that is right",
                  file=sys.stderr)
            rc = 1

    print(f"{'would change' if dry else 'changed'} {changed} cell(s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
