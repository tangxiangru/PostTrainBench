"""Receipt-backed node admission; no benchmark model, provider call or scientist."""

from __future__ import annotations

import hashlib
import json
import math
import os
import select
import signal
import subprocess
import time
from pathlib import Path

DATA_RELATIVE = "hub/datasets--openai--openai_humaneval/snapshots/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval/test-00000-of-00001.parquet"
DATA_SHA256 = "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6"
BWRAP_SHA256 = "d78807229d616606e339c5988392b9e0ab4a6a6998fa51e4590837f426a12fca"


def fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def publish(path, value):
    temporary = path.with_name(".acceptance.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def start_ticks(pid):
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def descendants(pid):
    """Read only this producer's current descendant graph, including thread children."""
    found, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        if parent in found:
            continue
        found.add(parent)
        try:
            for path in (Path("/proc") / str(parent) / "task").glob("*/children"):
                pending.extend(int(value) for value in path.read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def exited(fd):
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    return bool(poll.poll(0))


def execute(command, output, seconds, *, expect_timeout=False, ready_file=None):
    """Observe exact descendants; arm the outer deadline after native admission.

    The actual trusted supervisor supplies identity-checked readiness. GNU timeout receives
    SIGALRM on its own pidfd three seconds later, the same signal as its timer.
    The outer test uses a five-second escalation grace so its cleanup must
    finish before the independent 30-second program timeout can mask failure.
    A startup-only timeout, dead sandbox or emergency cleanup cannot pass.
    """
    handles, sandbox_handles, supervisor_handles, observations = {}, set(), {}, []
    observed_sandbox = False
    admitted_at = None
    alarm_sent = False
    live_at_alarm = False
    supervisor_live_at_alarm = False
    admission_event = None
    process = None
    root_key = None

    def observe(pid):
        nonlocal observed_sandbox
        fd = None
        try:
            ticks = start_ticks(pid)
            key = (pid, ticks)
            if key in sandbox_handles or key in supervisor_handles:
                return key
            if key not in handles:
                fd = os.pidfd_open(pid)
                if ticks != start_ticks(pid):
                    os.close(fd)
                    return None
                handles[key] = fd
                fd = None
            args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            if b'--unshare-all' in args and b'--as-pid-1' in args:
                sandbox_handles.add(key)
                observed_sandbox = True
            if b'--supervise' in args and any(arg.endswith(b'ptb_python_sandbox.py') for arg in args):
                status = (Path('/proc') / str(pid) / 'status').read_text()
                ns_line = next(line for line in status.splitlines() if line.startswith('NSpid:'))
                supervisor_handles[key] = int(ns_line.split()[-1])
            return key
        except (FileNotFoundError, ProcessLookupError):
            if fd is not None:
                os.close(fd)
            return None
        except OSError as exc:
            if fd is not None:
                os.close(fd)
            observations.append(f'{pid}: {type(exc).__name__}')
            return None

    def terminate_known():
        for sig, delay in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
            for fd in handles.values():
                if not exited(fd):
                    try:
                        signal.pidfd_send_signal(fd, sig)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + delay
            while any(not exited(fd) for fd in handles.values()) and time.monotonic() < deadline:
                time.sleep(.05)

    try:
        with output.open('xb') as log:
            process = subprocess.Popen(
                ['timeout', '--signal=TERM', '--kill-after=5s' if expect_timeout else '--kill-after=30s',
                 f'{seconds}s', *command],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            root_key = observe(process.pid)
            absolute_deadline = time.monotonic() + seconds + 35
            while process.poll() is None:
                for pid in descendants(process.pid):
                    observe(pid)
                if expect_timeout and ready_file is not None and admitted_at is None:
                    try:
                        ready = json.loads(ready_file.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        ready = {}
                    event = ready.get('event') or {}
                    if (ready.get('marker') == 'PTB_OUTER_ADMITTED_68d273'
                            and ready.get('observer') == 'native-supervisor-admission'
                            and event.get('schema') == 'ptb-python-admission-v1'
                            and event.get('started') is True):
                        stamp = event.get('admitted_monotonic')
                        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                                or not 0 <= time.monotonic() - stamp <= seconds):
                            raise ValueError('invalid native admission clock')
                        admitted_at = stamp
                        admission_event = event
                if admitted_at is not None and not alarm_sent and time.monotonic() >= admitted_at + 3:
                    live_at_alarm = any(not exited(handles[key]) for key in sandbox_handles)
                    supervisor_live_at_alarm = any(
                        namespace_pid == admission_event.get('supervisor_pid') and not exited(handles[key])
                        for key, namespace_pid in supervisor_handles.items())
                    if root_key is None or exited(handles[root_key]):
                        raise RuntimeError('outer timer handle unavailable at deadline')
                    signal.pidfd_send_signal(handles[root_key], signal.SIGALRM)
                    alarm_sent = True
                if time.monotonic() > absolute_deadline:
                    raise TimeoutError('outer timeout supervisor exceeded its bound')
                time.sleep(.1)
            returncode = process.wait()
            log.flush()
            os.fsync(log.fileno())
        deadline = time.monotonic() + 3
        while any(not exited(fd) for fd in handles.values()) and time.monotonic() < deadline:
            time.sleep(.05)
        survivors = [pid for (pid, _), fd in handles.items() if not exited(fd)]
        evidence = {'returncode': returncode, 'timed_out': returncode in {124, 137},
                    'cleanup_complete': not survivors, 'observed_native_sandbox': observed_sandbox,
                    'admitted_before_timeout': admitted_at is not None,
                    'admitted_sandbox_live_at_timeout': live_at_alarm, 'alarm_sent': alarm_sent,
                    'admission_supervisor_live_at_timeout': supervisor_live_at_alarm,
                    'admission_event': admission_event,
                    'elapsed_since_admission': None if admitted_at is None else time.monotonic() - admitted_at,
                    'termination_grace_seconds': 5 if expect_timeout else 30,
                    'observation_errors': observations,
                    'observed_processes': [{'pid': pid, 'start_ticks': ticks, 'terminal': exited(fd)}
                                           for (pid, ticks), fd in sorted(handles.items())],
                    'survivors': survivors}
        if survivors:
            terminate_known()
        evidence['passed'] = (not survivors and not observations and observed_sandbox and
                              (evidence['timed_out'] and alarm_sent and live_at_alarm and supervisor_live_at_alarm
                               and evidence['elapsed_since_admission'] < 30
                               if expect_timeout else returncode == 0))
        return evidence
    except BaseException:
        terminate_known()
        if process is not None:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        raise
    finally:
        for fd in handles.values():
            os.close(fd)


def image_command(source, image, digest, work, scratch, bwrap, dataset, *, outer=False):
    """Use the production scientist/evaluator home and cwd shapes, public fixtures only."""
    work.mkdir(parents=True)
    scratch.mkdir(parents=True)
    home = work / 'home'
    home.mkdir()
    if image.name == 'opus_5.sif':
        task_output = home / 'task'
        task_output.mkdir()
        home_target = '/home/ben'
        cwd_target = '/home/ben/task'
        output_target = cwd_target
        role = 'scientist'
    else:
        task_output = work / 'result'
        task_output.mkdir()
        home_target = str(Path.home())
        cwd_target = str(source / 'src/eval/tasks/humaneval')
        output_target = str(task_output)
        role = 'official-evaluator'
    command = [os.environ.get('POST_TRAIN_BENCH_APPTAINER_BIN', 'apptainer'), 'exec', '--nv',
               '-c', '--cleanenv', '--pid', '--no-init', '--writable-tmpfs',
               '--home', f'{home}:{home_target}', '--pwd', cwd_target,
               '--bind', f'{scratch}:/tmp', '--bind', f'{source}:{source}:ro',
               '--bind', f'{work}:{work}',
               '--bind', f"{source / 'src/eval/humaneval_environment_probe.py'}:/opt/ptb-probe.py:ro",
               '--bind', f"{source / 'src/eval/tasks/humaneval'}:/opt/ptb-humaneval:ro",
               '--bind', f"{source / 'src/eval/ptb_python_sandbox.py'}:/opt/ptb-python/ptb_python_sandbox.py:ro",
               '--bind', f'{bwrap}:/opt/ptb-python/bwrap:ro',
               '--bind', f'{dataset}:/opt/ptb-data.parquet:ro',
               '--env', 'PTB_FAKE_SECRET_FOR_TEST=must-not-enter-generated-code',
               '--env', 'PYTHONNOUSERSITE=1', '--env', 'TMPDIR=/tmp',
               '--env', 'VLLM_API_KEY=inspectai',
               '--env', f"CUDA_VISIBLE_DEVICES={os.environ.get('POST_TRAIN_BENCH_VISIBLE_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES', ''))}"]
    if role == 'official-evaluator':
        for key, value in {'XDG_CACHE_HOME': '.cache', 'XDG_CONFIG_HOME': '.config',
                           'VLLM_CACHE_ROOT': '.cache/vllm', 'TORCHINDUCTOR_CACHE_DIR': '.cache/torchinductor',
                           'TRITON_CACHE_DIR': '.cache/triton'}.items():
            command += ['--env', f'{key}={home_target}/{value}']
    command += [str(image), 'python', '/opt/ptb-probe.py', '--image', image.name,
                '--image-sha256', digest, '--output-dir', output_target]
    if outer:
        command.append('--outer-timeout')
    return command, task_output, {'home': home_target, 'cwd': cwd_target, 'role': role,
                                  'fixture_binds': 'public-data-and-invented-programs-only'}


def main():
    if os.getuid() == 0:
        raise ValueError("environment acceptance must use the scientist's non-root runtime identity")
    spec = json.loads(os.environ["POST_TRAIN_BENCH_ENVIRONMENT_ACCEPTANCE_SPEC"])
    operation = spec["operation"]
    if operation["kind"] != "environment-acceptance" or operation["target"] != "humaneval":
        raise ValueError("unsupported environment operation")
    if os.environ.get("POST_TRAIN_BENCH_RUN_PURPOSE") != "environment-acceptance":
        raise ValueError("environment probe must not masquerade as a formal model run")
    source = Path(__file__).resolve().parents[3]
    batch = os.environ["POST_TRAIN_BENCH_BATCH_ID"]
    cell = os.environ["POST_TRAIN_BENCH_CELL_ID"]
    job = os.environ["SLURM_JOB_ID"]
    for value in (batch, cell, job):
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError("unsafe environment result identity")
    destination = Path(os.environ["POST_TRAIN_BENCH_ENVIRONMENT_ACCEPTANCE_OUTPUT_ROOT"]) / cell / job
    if not destination.is_absolute():
        raise ValueError("environment output root must be an absolute frozen path")
    destination.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "ptb-humaneval-environment-v1", "operation": "environment-acceptance",
              "target": "humaneval", "scientific_result": False, "status": "running", "errors": [],
              "batch_id": batch, "cell_id": cell, "job_id": job, "node": os.environ["SLURMD_NODENAME"],
              "uid": os.getuid(), "source": {
                  "top_commit": os.environ["POST_TRAIN_BENCH_FROZEN_TOP_COMMIT"],
                  "ptb_commit": os.environ["POST_TRAIN_BENCH_FROZEN_PTB_COMMIT"]},
              "images": {}}
    publish(destination / "acceptance.json", report)
    try:
        sources = {"probe": source / "src/eval/humaneval_environment_probe.py",
                   "runner": Path(__file__).resolve(), "helper": source / "src/eval/ptb_python_sandbox.py"}
        report["probe_sources"] = {key: fingerprint(path)["sha256"] for key, path in sources.items()}
        for key, actual in report["probe_sources"].items():
            if actual != operation[key + "_sha256"]:
                raise ValueError(f"frozen {key} bytes changed")
        bwrap = Path(os.environ["POST_TRAIN_BENCH_PYTHON_BWRAP"])
        if fingerprint(bwrap)["sha256"] != BWRAP_SHA256:
            raise ValueError("unexpected bubblewrap bytes")
        dataset = Path(os.environ["HF_HOME"]) / DATA_RELATIVE
        if fingerprint(dataset) != {"sha256": DATA_SHA256, "bytes": 83920}:
            raise ValueError("HumanEval data bytes differ")
        containers = Path(os.environ["POST_TRAIN_BENCH_CONTAINERS_DIR"])
        if set(spec["images"]) != {"opus_5.sif", "vllm_debug.sif"}:
            raise ValueError("acceptance must exercise both frozen images")
        for name, digest in spec["images"].items():
            image = containers / name
            image_report = {"sha256": fingerprint(image)["sha256"]}
            report["images"][name] = image_report
            if image_report["sha256"] != digest:
                raise ValueError(f"image changed: {name}")
            work = destination / "official_eval" / name / "normal"
            temporary = Path(os.environ["POST_TRAIN_BENCH_SCRATCH_DIR"]) / "acceptance" / name / "normal"
            normal_command, task_output, layout = image_command(source, image, digest, work, temporary, bwrap, dataset)
            normal = execute(normal_command, work / "container.log", 180)
            image_report["normal_execution"] = normal
            if not normal["passed"]:
                raise ValueError(f"native probe or cleanup failed: {name}")
            observed = json.loads((task_output / "probe-result.json").read_text())
            if observed["layout_observed"] != {key: layout[key] for key in ("home", "cwd")}:
                raise ValueError(f"actual home/cwd differs from production layout: {name}")
            image_report["layout"] = layout
            image_report["layout_observed"] = observed["layout_observed"]
            image_report.update({"cuda": observed["cuda"], "rows": observed["rows"],
                                 "native_revalidated": observed["native_revalidated"]})
            report["dataset"] = observed["dataset"]
            outer_work = destination / "official_eval" / name / "outer-timeout"
            outer_tmp = temporary.parent / "outer-timeout"
            outer_command, outer_output, _ = image_command(source, image, digest, outer_work, outer_tmp, bwrap, dataset, outer=True)
            image_report["outer_timeout"] = execute(outer_command, outer_work / "container.log", 180,
                                                      expect_timeout=True, ready_file=outer_output / "outer-admitted.json")
            if not image_report["outer_timeout"]["passed"]:
                raise ValueError(f"outer timeout/owned cleanup not established: {name}")
            image_report["raw_files"] = [
                {"path": path.relative_to(destination).as_posix(), **fingerprint(path)}
                for path in sorted((destination / "official_eval" / name).rglob("*")) if path.is_file()
            ]
            publish(destination / "acceptance.json", report)
        report["status"] = "passed"
    except BaseException as exc:  # noqa: BLE001 - persist failed admission and return nonzero
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    publish(destination / "acceptance.json", report)
    print(json.dumps({"report": str(destination / "acceptance.json"), "status": report["status"],
                      "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    def interrupted(signum, _frame):
        signal.signal(signum, signal.SIG_DFL)
        raise RuntimeError(f"environment acceptance interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
