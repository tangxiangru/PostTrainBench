#!/usr/bin/env python3
"""Check that a judge actually returned a verdict before it is stored.

`collect_judge_output` used to `cp` whatever `judgement.json` the sandbox left
behind, unconditionally. Three things can leave a file there that is not a
verdict:

  * the codex CLI dies mid-run -- 401, usage limit, container fault -- after the
    model has written a partial file. Neither `set -e` nor the exit status
    catches it, because the exec is the left side of a `| tee` pipeline;
  * the model writes the *example* from the prompt instead of its own judgement,
    which is a well-formed object full of `false` and the sample justification;
  * the model writes `{"contamination": "false"}` -- a string, not a bool. The
    downstream reader does `if flag:` and a non-empty string is truthy, so this
    inverts the verdict silently.

A verdict that is missing is a visible hole. A verdict that is wrong is a number
in a table. So the file is validated against the fields the judge declares in its
judge.conf, and anything that fails is preserved as evidence but never installed.

    validate_judgement.py <judgement.json> --fields "contamination disallowed_model"
                          [--min-justification N]

Exit 0 if it is a usable verdict, 1 otherwise (reasons on stderr).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Long enough to exclude "n/a", "none", "ok" and an empty string; short enough
# that a terse but real one-sentence finding passes.
DEFAULT_MIN_JUSTIFICATION = 40


def validate(doc, fields: list[str], min_just: int) -> list[str]:
    """-> list of problems; empty means the verdict is usable."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return [f"top level is {type(doc).__name__}, expected an object"]

    for f in fields:
        if f not in doc:
            problems.append(f"missing field {f!r}")
            continue
        v = doc[f]
        if not isinstance(v, bool):
            # The string "false" is truthy downstream. This is the failure that
            # would flip a verdict rather than lose it.
            problems.append(
                f"{f!r} is {type(v).__name__} ({v!r}), expected a JSON boolean"
            )
        j = f"justification_{f}"
        jv = doc.get(j)
        if jv is None:
            problems.append(f"missing field {j!r}")
        elif not isinstance(jv, str):
            problems.append(f"{j!r} is {type(jv).__name__}, expected a string")
        elif len(jv.strip()) < min_just:
            problems.append(
                f"{j!r} is {len(jv.strip())} chars, under the {min_just}-char floor "
                f"({jv.strip()[:60]!r})"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--fields", required=True,
                    help="space-separated boolean verdict fields the judge declares")
    ap.add_argument("--min-justification", type=int, default=DEFAULT_MIN_JUSTIFICATION)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    fields = args.fields.split()
    if not fields:
        print("validate_judgement: --fields is empty; nothing to check", file=sys.stderr)
        return 1

    try:
        raw = args.path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"validate_judgement: cannot read {args.path}: {e}", file=sys.stderr)
        return 1
    if not raw.strip():
        print(f"validate_judgement: {args.path} is empty", file=sys.stderr)
        return 1
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"validate_judgement: {args.path} is not JSON: {e}", file=sys.stderr)
        return 1

    problems = validate(doc, fields, args.min_justification)
    if problems:
        print(f"validate_judgement: {args.path} is not a usable verdict:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(" ".join(f"{f}={json.dumps(doc[f])}" for f in fields))
    return 0


if __name__ == "__main__":
    sys.exit(main())
