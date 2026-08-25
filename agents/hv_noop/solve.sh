#!/bin/bash
# hv smoke agent: no LLM in the loop. Does literally what POST_TRAIN_BENCH_PROMPT=mock_test
# asks ("Just store {model} to final_model/ and exit") so that run_task.sh's plumbing --
# apptainer sandbox, check_cuda, timer, system_monitor, trace parsing, final_model
# collection, evaluate.py -> metrics.json -- can be exercised without an agent or any
# API credentials. Analogous to Harbor's `oracle` agent.
set -x
echo "hv_noop agent starting"
echo "PROMPT<<<"; echo "$PROMPT"; echo ">>>"
pwd; ls -la
bash timer.sh || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# The base model to copy is named in the prompt (mock_test.txt is exactly
# "Just store \`{model}\` to final_model/ and exit."), so read it from there.
python <<'PY'
import os, re, shutil, sys
from huggingface_hub import snapshot_download
prompt = os.environ.get("PROMPT", "")
m = re.search(r"`([^`]+/[^`]+)`", prompt)
model_id = m.group(1) if m else "Qwen/Qwen3-1.7B-Base"
print("model_id from prompt:", model_id)
src = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
dst = os.path.join(os.getcwd(), "final_model")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=False)
print("copied", src, "->", dst)
print(sorted(os.listdir(dst)))
PY
ls -la final_model
echo "hv_noop agent done"
