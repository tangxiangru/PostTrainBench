#!/bin/bash
# Prove the GRPO path works end to end before spending an hour on it.
#
# The failure this is aimed at is not "the loss is bad", it is "the run dies at the
# first checkpoint save" -- which is where 89809_g3, 89727_g1, 89810_g3, 89810_g7 and
# 89727_g7 each lost their first RL attempt. So this trains for two steps with
# --save-steps 1 and then re-reads what was written: a checkpoint that exists but whose
# generation_config cannot be round-tripped is the exact defect, and it is invisible
# until something tries to save it a second time.
#
# Usage: bash reference/smoke.sh [output_dir] [extra train_grpo.py flags...]
#        (needs one GPU; ~3-5 min)
#
# Everything after the optional output dir is forwarded verbatim to train_grpo.py, and a
# first argument that starts with `-` is a flag rather than an output dir. That matters
# because this script no longer knows which model it is smoke-testing: train_grpo.py reads
# the base model from $MODEL_TO_TRAIN, which the harness sets inside the sandbox, and
# outside it you need `bash reference/smoke.sh /tmp/out --model <org>/<model>`. Passing no
# way to say that was the whole defect -- the flags below are two-step overrides, not a
# model choice, and hardcoding a model here would silently smoke-test the wrong one on
# three of the four base models this benchmark is swept over.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    OUT="$1"; shift
else
    OUT="${TMPDIR:-/tmp}/grpo_smoke.$$"
fi

python3 "$HERE/train_grpo.py" --output-dir "$OUT" \
    --fewshot 0 --max-train-samples 32 --max-steps 2 --save-steps 1 \
    --num-generations 4 --per-device-train-batch-size 4 --gradient-accumulation-steps 1 \
    --max-prompt-length 512 --max-completion-length 64 --save-total-limit 2 \
    "$@"

# `|| true`: without it `set -e` aborts on the empty glob and swallows the message below.
CKPT="$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
[ -n "$CKPT" ] || { echo "FAIL: train_grpo.py wrote no checkpoint under $OUT" >&2; exit 1; }

python3 - "$CKPT" <<'PY'
import sys, tempfile
from transformers import AutoModelForCausalLM, GenerationConfig
ckpt = sys.argv[1]
gc = GenerationConfig.from_pretrained(ckpt)
assert gc.temperature is not None, f"{ckpt} has no temperature: vLLM will sample it at 1.0"
with tempfile.TemporaryDirectory() as d:
    gc.save_pretrained(d)  # the five-cell ValueError fires on a save, not on a load
AutoModelForCausalLM.from_pretrained(ckpt, dtype="bfloat16")
print(f"PASS {ckpt} loads; do_sample={gc.do_sample} temperature={gc.temperature} eos={gc.eos_token_id}")
PY

echo "SMOKE PASSED -- now beat it (see reference/README.md)"
