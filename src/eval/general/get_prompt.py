#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

INSPECT_EVALS = [
    "aime2025",
    "bfcl",
    "gpqamain",
    "gsm8k",
    "humaneval",
    "humanevalplus",
]

#: Where run_task.sh's `--home`/`--pwd` put the job directory inside the agent sandbox.
#: Defaults only: run_task.sh passes its own SANDBOX_HOME / SANDBOX_TASK_DIR on the command
#: line, and tests/test_graded_read.sh [14] asserts these two constants still equal the
#: values that file binds. Kept here as well so that running get_prompt.py by hand -- which
#: is how the prompt is read during development -- renders the same paths the agent sees.
DEFAULT_SANDBOX_HOME = "/home/ben"
DEFAULT_SANDBOX_TASK_DIR = "/home/ben/task"

#: The one predicate for "does this benchmark's grading write an inspect log", imported from
#: the helper whose behaviour depends on the answer. Two copies of this question -- one in
#: the prompt that announces the tool and one in the tool -- is a prompt that promises the
#: wrong mode, and the failure is invisible: the bullet reads fine either way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "utils"))
from graded_read import evaluate_uses_inspect  # noqa: E402


def task_evaluate_script(benchmark_id: str) -> Path:
    """The evaluate.py that run_task.sh will copy into this benchmark's sandbox.

    ``EVAL_SCRIPT`` is set by run_task.sh (``evaluate.py``, or ``evaluate_openrouter.py``
    when the judge backend falls back to OpenRouter); read it rather than assume, because
    the copied file is always named ``evaluate.py`` inside the sandbox and the source name
    is the only thing that distinguishes the two.
    """
    script = os.environ.get("EVAL_SCRIPT", "evaluate.py") or "evaluate.py"
    return Path("src/eval/tasks") / benchmark_id / script

def read_benchmark_name(benchmark_id: str) -> str:
    """Resolve the human-readable benchmark name from the benchmark_id."""
    bench_file = Path("src/eval/tasks") / benchmark_id / "benchmark.txt"
    if not bench_file.is_file():
        raise FileNotFoundError(f"Benchmark file not found for id '{benchmark_id}': {bench_file}")
    return bench_file.read_text(encoding="utf-8").strip()

def read_required_api_keys(benchmark_id: str) -> list[str]:
    """Read the benchmark's required third-party API keys from info.json.

    These keys are provisioned into the agent sandbox solely so the benchmark's
    own grading (evaluate.py) can run. Defaults to none if the field is absent.
    """
    info_file = Path("src/eval/tasks") / benchmark_id / "info.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"Info file not found for id '{benchmark_id}': {info_file}")
    info = json.loads(info_file.read_text(encoding="utf-8"))
    return info.get("required_api_keys", [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--model-to-train', type=str, required=True)
    parser.add_argument('--benchmark-id', type=str, required=True)
    parser.add_argument('--num-hours', type=str, required=True)
    parser.add_argument('--num-gpus', type=int, default=1)
    # Absolute, and supplied by the caller that owns the bind. See the two module
    # constants above for why the prompt may not use a relative path here.
    parser.add_argument('--sandbox-home-dir', type=str, default=DEFAULT_SANDBOX_HOME)
    parser.add_argument('--sandbox-task-dir', type=str, default=DEFAULT_SANDBOX_TASK_DIR)

    args = parser.parse_args()

    home = args.sandbox_home_dir.rstrip('/')
    task = args.sandbox_task_dir.rstrip('/')
    for name, value in (('--sandbox-home-dir', home), ('--sandbox-task-dir', task)):
        if not value.startswith('/'):
            raise SystemExit(
                f"{name}={value!r} must be an absolute path. The agent's working directory "
                "is not the same on every arm -- the AutoR operator runs its stages two "
                "levels below the task root -- so a relative path here advertises a file "
                "that only one arm can open."
            )

    benchmark_name = read_benchmark_name(args.benchmark_id)

    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')

    template_path = f'src/eval/general/{base_prompt}.txt'

    with open(template_path, 'r') as f:
        template = f.read()

    datetime = subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip()

    if args.num_gpus == 1:
        gpu_info = "- The machine is equipped with an Nvidia H100 GPU."
    else:
        gpu_info = f"- The machine is equipped with {args.num_gpus} Nvidia H100 GPUs."

    result = template.replace('{model}', args.model_to_train)
    result = result.replace('{benchmark}', benchmark_name)
    result = result.replace('{num_hours}', args.num_hours)
    result = result.replace('{gpu_info}', gpu_info)

    if args.benchmark_id in INSPECT_EVALS:
        result = result.replace('{setup_other}', "- A final note: the evaluate.py script sometimes outputs ERROR warnings. Do not be alarmed by this, this is normal behavior for inspect-ai. Also if you run into issues with the evaluate.py script, this is likely due to memory constraints on the GPU. In this case please decrease --max-connections or --max-tokens.\n")
    else:
        result = result.replace('{setup_other}', "")

    # Benchmarks that ship a test_data.json get a self-decontamination section:
    # run_task.sh copies the contamination judge's n-gram checker plus the test
    # set into the sandbox home, so the agent can screen its own training data
    # with the exact tool the judge uses afterwards. The prompt must explain the
    # tool and restate that the test-set copy is checker input only.
    test_data_file = Path("src/eval/tasks") / args.benchmark_id / "test_data.json"
    if test_data_file.is_file():
        result = result.replace('{decontamination_tool}', (
            "\n## Decontamination Tool\n"
            f"- A copy of the {benchmark_name} test set is available at \\`{home}/test_data.json\\`, "
            f"together with the contamination checker \\`{home}/contamination_check.py\\`\n"
            "- You can check data which you want to use for training, e.g.:\n"
            f"  \\`python {home}/contamination_check.py --reference {home}/test_data.json --input train_data.jsonl\\`\n"
            "  The input is JSONL with one document per line (pass \\`--input-format text\\` for "
            "plain text lines, or pipe documents via stdin). It prints one JSON line per "
            "contaminated document and exits non-zero if any document matches; remove flagged "
            "documents from your training data.\n"
            "- Reading and running these two files in the home directory is an allowed exception "
            "to the working-directory rule below.\n"
            "- The test-set copy is provided ONLY as reference input for this checker. Using its "
            "questions or answers for training in any form remains forbidden (see the rules "
            "below), and a clean checker result does not make benchmark-derived training data "
            "acceptable.\n"
        ))
    else:
        result = result.replace('{decontamination_tool}', "")

    # Benchmarks that ship a reference/ directory get one bullet announcing it. Conditional
    # and not a literal line in the template for the same reason the decontamination
    # section is: only gsm8k has the directory today, run_task.sh copies it with an
    # `if [ -d ... ]` guard, and a literal sentence would promise every other benchmark's
    # agent a file that is not in its sandbox. Placed before {decontamination_tool} rather
    # than after it because {decontamination_tool} opens a "## Decontamination Tool"
    # heading -- anything appended after that renders under the wrong section instead of
    # under "## Information on the Setup" where this bullet belongs.
    #
    # Every path below is ABSOLUTE. The bullet used to say `reference/train_grpo.py`, which
    # resolves only for an agent whose working directory is the task root. It is not for the
    # claude_autor arm, whose operator runs each stage with cwd
    # <task>/.autor/<stamp>/ -- two levels down, where the relative name resolves to
    # nothing and the whole intervention is silently absent from the one arm it was written
    # for. Nothing would have reported that: a file the agent never finds produces no error
    # anywhere, only a run that looks like it ignored the advice.
    reference_dir = Path("src/eval/tasks") / args.benchmark_id / "reference"
    if reference_dir.is_dir():
        result = result.replace('{reference_script}', (
            f"- A reference GRPO training script is provided at \\`{task}/reference/train_grpo.py\\`, "
            f"together with \\`{task}/reference/README.md\\` and \\`{task}/reference/smoke.sh\\`, "
            "which trains "
            "two steps on a tiny subset and re-loads the checkpoint it wrote. It is "
            "deliberately unremarkable: correct, not tuned. It exists only to pre-empt setup "
            "failures that have cost previous runs their first hour of RL, and it is NOT a "
            f"required entrypoint and NOT the intended solution. Read \\`{task}/reference/README.md\\` "
            "before you use it, and treat any score it produces as a floor to beat rather "
            "than a recipe to follow. It reads the base model id from the \\`MODEL_TO_TRAIN\\` "
            "environment variable, so it trains the model this run is about without being "
            "told which one that is.\n"
        ))
    else:
        result = result.replace('{reference_script}', "")

    # graded_read.py is copied into EVERY sandbox, so the bullet is unconditional in that
    # sense -- but what the helper can check is not. On a benchmark whose evaluate.py runs
    # inspect_ai it verifies the row count and the checkpoint identity against the eval log;
    # on healthbench and arenahardwriting there is no such log to read and it can only check
    # the exit code and the metrics file. Promising the strong version on the two tasks that
    # cannot deliver it is how a helper gets called once, refuses, and is worked around for
    # the rest of the run. The predicate is graded_read.py's own, imported above, so the two
    # cannot disagree about which mode a task is in.
    if evaluate_uses_inspect(task_evaluate_script(args.benchmark_id)):
        graded_read_bullet = (
            f"- \\`{task}/graded_read.py\\` is an optional wrapper around evaluate.py: it refuses to "
            "report a score unless the command exited cleanly and the evaluation log confirms the "
            "row count you asked for and the checkpoint you named, and it writes a JSON record of "
            "every read including the refused ones. It is not a required entrypoint; see "
            f"\\`python {task}/graded_read.py --help\\`.\n"
        )
    else:
        graded_read_bullet = (
            f"- \\`{task}/graded_read.py\\` is an optional wrapper around evaluate.py: it refuses to "
            "report a score unless the command exited cleanly and wrote a metrics file this run can "
            "account for, and it writes a JSON record of every read including the refused ones. This "
            "benchmark's evaluate.py writes no inspect log, so the wrapper says so on every record "
            "(\\`decided_on\\`) and does not claim to have checked the row count or the checkpoint "
            f"identity. It is not a required entrypoint; see \\`python {task}/graded_read.py --help\\`.\n"
        )
    result = result.replace('{graded_read}', graded_read_bullet)

    # The template's rule 9 is a general prohibition on using any environment API
    # key to call an external LLM API directly. Benchmarks whose own grading needs a
    # provider key (e.g. OPENAI_API_KEY for healthbench/arenahardwriting) get a rule 10
    # carving out that key as an exception, permitted for evaluation only.
    required_api_keys = read_required_api_keys(args.benchmark_id)
    if required_api_keys:
        keys_str = ", ".join(f"\\`{k}\\`" for k in required_api_keys)
        if len(required_api_keys) == 1:
            exc, prov, use = "is an exception", "it is provided", "it"
        else:
            exc, prov, use = "are exceptions", "they are provided", "them"
        result = result.replace('{eval_api_note}', (
            f"10. The {keys_str} in your environment {exc} to the previous rule: {prov} so that "
            f"you can run this benchmark's grading via evaluate.py. Use {use} for that evaluation "
            f"only, and never to generate training data or for any other purpose.\n"
        ))
    else:
        result = result.replace('{eval_api_note}', "")

    result = result.replace('{datetime}', datetime)

    if 'claude' in args.agent:
        result += """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""
    print(result)

if __name__ == '__main__':
    main()
