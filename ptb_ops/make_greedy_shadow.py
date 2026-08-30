#!/usr/bin/env python3
"""Build a symlink shadow of a checkpoint whose only difference is the decode.

`src/eval/tasks/gsm8k/evaluate.py` passes `max_tokens` and `max_connections` to
inspect_ai and no temperature, so vLLM falls back to the checkpoint's own
generation_config.json.  Qwen3-1.7B-Base ships `"do_sample": false` with no
temperature, and `do_sample` is a transformers field vLLM ignores -- so a cell
that writes no temperature is sampled at the library default of 1.0, states a
correct `ANSWER: n`, misses the stop token and runs on until the 4000-token cap.
The task scores with `match(numeric=True)`, which reads the *last* number.

Two of the twenty-three cells in the one-hour board happened to write
`"temperature": 0.0`.  They are the only two above 0.5.  That makes "which arm"
and "did this cell notice the harness's sampling default" the same column, and
no arm comparison survives it.

This equalises the column without retraining anything and without touching the
board: the shadow directory symlinks every file of the original `final_model/`
and substitutes one rewritten generation_config.json.

The rewrite is deliberately minimal -- it copies the original file and overrides
only the four sampling fields.  eos/bos/pad token ids are left exactly as the
cell shipped them, because those decide where generation *stops* and differ
across cells (some set eos to [151645, 151643], some to 151645 alone).
Normalising those too would swap one confound for another.

Usage:  make_greedy_shadow.py <src_model_dir> <shadow_model_dir>
"""
from __future__ import annotations

import json
import os
import sys

# temperature 0.0 is what actually forces greedy: vLLM's SamplingParams switches to
# greedy when temperature < _SAMPLING_EPS and ignores top_p/top_k entirely from there.
# The other three are written for the record, so the config states the intended decode
# rather than leaving it to be inferred. top_k is -1 (vLLM's "off") rather than HF's 0,
# since the two libraries disagree about which value disables it.
GREEDY = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
}


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <src_model_dir> <shadow_model_dir>", file=sys.stderr)
        return 2
    src, dst = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])

    if not os.path.isfile(os.path.join(src, "config.json")):
        print(f"FATAL: no config.json under {src}", file=sys.stderr)
        return 1

    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        link = os.path.join(dst, name)
        if os.path.lexists(link):
            os.unlink(link)
        if name == "generation_config.json":
            continue
        # Absolute targets: the shadow tree and the board are bound into the container
        # at their host paths, so an absolute link resolves on both sides.
        os.symlink(os.path.join(src, name), link)

    original = os.path.join(src, "generation_config.json")
    cfg: dict = {}
    if os.path.isfile(original):
        with open(original) as f:
            cfg = json.load(f)
    was = {k: cfg.get(k, "<unset>") for k in GREEDY}
    cfg.update(GREEDY)
    with open(os.path.join(dst, "generation_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"shadow={dst}")
    print(f"  src={src}")
    print(f"  generation_config.json: {'rewritten from original' if original else 'created (cell shipped none)'}")
    print(f"  was={was}")
    print(f"  now={GREEDY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
