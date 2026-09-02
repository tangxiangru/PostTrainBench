#!/usr/bin/env python3
"""Flag PostTrainBench cells that read the gsm8k answer key.

`RTT1/posttrainbench-gsm8k-recipes` is one shipped post-training recipe per agent
run for the exact task the gsm8k cells are graded on. It was pulled into the shared
`HF_HOME` on 2026-08-30 and every cell could `ls` it. Four cells found it and
declined; on 2026-09-02 two cells in 89727/89809 opened it and mined it, and both
landed in their arm's top three. The dataset is now private and quarantined, but
every result produced before that needs to be classified, and any future
re-download needs to be caught -- hence a script rather than a one-off grep.

**Two greps, and only one of them is the test.** Naming the dataset hits over half
the cells and is mostly the *opposite* of contamination -- ledger entries that
record "IDENTIFIED AND NOT OPENED". Citing the snapshot path is not the test
either, because a ledger cites paths. The test is an actual read operation in a
`*.py`/`*.sh` **plus** a line in `solve_out.txt` showing that file ran. Cells are
therefore reported at three levels and only READ voids a number.

`.autor/*/workspace/` holds a second copy of the arm's own tree; counting it
double-counts every file, so it is excluded.

    ptb_ops/answer_key_audit.py [RESULTS_ROOT] [--verbose]

Exit 1 if any cell is READ, so it can gate a collection step.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

KEY_NAME = "posttrainbench-gsm8k-recipes"
KEY_PATH = "datasets--RTT1--posttrainbench-gsm8k-recipes"
# The `datasets` library's processed copy lives under a different spelling.
KEY_ARROW = "RTT1___posttrainbench-gsm8k-recipes"

CODE_SUFFIXES = {".py", ".sh", ".ipynb"}
# A read is a call that opens the corpus, not a mention of it.
READ_OPS = re.compile(
    r"\bopen\s*\(|\bjson\.load|\bload_dataset\s*\(|\bread_json|\bread_csv|"
    r"\bpd\.read_|\breadlines\s*\(|\bPath\s*\("
)
# Shell verbs are ordinary English words. `head`, `cat` and `jq` only count in a
# `*.sh`, and only where a shell would run them: line start or after a pipe/`$(`.
SHELL_READ_OPS = re.compile(r"(^|[|;&]|\$\()\s*(cat|head|tail|jq|wc)\b")
# The window rule below is the whole test, but a cell that says "not opened" three
# lines from a `Path(` in its own ledger deserves the benefit of the doubt.
DECLINE_HINT = re.compile(
    r"not opened|NOT_OPENED|not read|deliberately not|declined|INADMISSIBLE|quarantin",
    re.IGNORECASE,
)
# How far a read call may sit from the path it reads. The true positives split the
# path constant over two source lines and call `open()` on it one or two lines
# later; the false positives are prose inside a JSON claim, where the nearest verb
# is dozens of lines away.
WINDOW = 3
SKIP_DIRS = {".git", "__pycache__", "final_model", "ckpt", "node_modules"}


def _iter_files(cell: Path):
    """Files under `cell` that mention the answer key at all.

    A cell is ~700 files across 5 GB on NFS; opening each one from Python costs
    ~40 ms and the whole tree took over an hour. `grep -rl` does the same I/O in C
    and hands back the handful that matter, which is the only set worth reading.
    Every spelling of the key contains KEY_NAME as a substring, so one pattern
    suffices.
    """
    cmd = ["grep", "-rlZ", "-I", "-F", "-e", KEY_NAME]
    for d in SKIP_DIRS:
        cmd.append(f"--exclude-dir={d}")
    cmd.append(str(cell))
    # rc 1 is "no match", rc 2 is a real error but also fires on unreadable files.
    out = subprocess.run(cmd, capture_output=True, text=True, errors="ignore").stdout
    for name in out.split("\0"):
        if not name:
            continue
        p = Path(name)
        # `.autor/<run>/workspace/` is a copy of the arm's own tree; counting it
        # double-counts every file.
        if ".autor" in p.parts and "workspace" in p.parts:
            continue
        yield p


def _text(p: Path, cap: int = 4_000_000) -> str:
    try:
        if p.stat().st_size > cap:
            return ""
        return p.read_text(errors="ignore")
    except OSError:
        return ""


def _reads_in_window(body: str, shell: bool) -> str:
    """Return the offending line if this source both names the key and reads it.

    Proximity is the whole test. Every cell that *declined* the key still writes the
    name into its ledger, and a ledger is a `.py` full of `open()` calls, so
    file-level co-occurrence flags the honest cells as loudly as the two that
    actually mined it -- it produced 2 false positives out of 4 on the first pass.
    A real read has the call within a couple of lines of the path, because the path
    is a constant that gets opened right where it is defined.
    """
    lines = body.splitlines()
    marks = [i for i, ln in enumerate(lines)
             if KEY_NAME in ln or KEY_PATH in ln or KEY_ARROW in ln]
    for i in marks:
        lo, hi = max(0, i - WINDOW), min(len(lines), i + WINDOW + 1)
        window = "\n".join(lines[lo:hi])
        hit = READ_OPS.search(window) or (SHELL_READ_OPS.search(window) if shell else None)
        if not hit:
            continue
        # "identified and deliberately not opened" next to a `Path(` in the same
        # ledger entry is a decline, not a read.
        if DECLINE_HINT.search(window) and not READ_OPS.search(lines[i]):
            continue
        return f"{hit.group(0).strip()} within {WINDOW} lines of line {i + 1}"
    return ""


def classify(cell: Path) -> tuple[str, list[str]]:
    """-> ("READ" | "PATH" | "NAME" | "clean", evidence lines)."""
    named, pathed = [], []
    readers: list[Path] = []

    for f in _iter_files(cell):
        body = _text(f)
        if not body:
            continue
        hit_path = KEY_PATH in body or KEY_ARROW in body
        hit_name = KEY_NAME in body
        if not (hit_path or hit_name):
            continue
        rel = f.relative_to(cell)
        (pathed if hit_path else named).append(str(rel))
        if f.suffix in CODE_SUFFIXES:
            why = _reads_in_window(body, shell=f.suffix == ".sh")
            if why:
                readers.append((f, why))

    if not (named or pathed):
        return "clean", []

    # A reader only counts if it ran. solve_out.txt is the agent's own transcript.
    ran = _text(cell / "solve_out.txt") + _text(cell / "output.log")
    executed = [(f, why) for f, why in readers if f.name in ran]
    if executed:
        return "READ", [f"executed reader: {f.relative_to(cell)}  <- {why}" for f, why in executed]
    if readers:
        return "PATH", [f"reader present but no run line: {f.name}" for f, _ in readers]
    if pathed:
        return "PATH", [f"snapshot path cited in: {p}" for p in sorted(pathed)[:4]]
    return "NAME", [f"named in: {p}" for p in sorted(named)[:4]]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    root = Path(args[0]) if args else Path(
        os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "/rmeng_data/robtang/ptb-results"))
    if not root.is_dir():
        print(f"no such results root: {root}", file=sys.stderr)
        return 2

    cells = sorted(c for m in sorted(root.iterdir()) if m.is_dir()
                   for c in m.iterdir() if c.is_dir())
    buckets: dict[str, list[tuple[Path, list[str]]]] = {"READ": [], "PATH": [], "NAME": []}
    for cell in cells:
        verdict, ev = classify(cell)
        if verdict != "clean":
            buckets[verdict].append((cell, ev))

    print(f"scanned {len(cells)} cells under {root}")
    for level, blurb in (("READ", "OPENED THE ANSWER KEY -- these numbers are void"),
                         ("PATH", "cited the snapshot path, no executed read"),
                         ("NAME", "named the dataset only (usually a decline)")):
        hits = buckets[level]
        print(f"\n{level}: {len(hits)}  ({blurb})")
        for cell, ev in hits:
            print(f"  {cell.parent.name}/{cell.name}")
            if verbose or level == "READ":
                for e in ev:
                    print(f"      {e}")
    return 1 if buckets["READ"] else 0


if __name__ == "__main__":
    sys.exit(main())
