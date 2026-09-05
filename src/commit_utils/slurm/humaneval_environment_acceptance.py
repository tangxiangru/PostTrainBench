"""Receipt-backed node admission; no benchmark model, provider call or scientist."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import select
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
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


def execute(command, output, seconds, *, expect_timeout=False, ready_file=None,
            require_sandbox=True):
    """Run one container as a child subreaper and clean only its owned descendants.

    The production scientist/evaluator wrapper and admission probe use this exact
    lifecycle. Native admission arms the probe's three-second outer deadline;
    five-second escalation plus bounded cleanup must finish before the independent
    thirty-second inner program timeout. PID handles, never names/users, authorize
    cleanup. This foreground supervisor must start without other children.
    """
    if descendants(os.getpid()) != {os.getpid()}:
        raise RuntimeError('container supervisor must start without existing children')
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot read child-subreaper state')
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot become child subreaper')
    handles, records, sandbox_handles, supervisor_handles, observations = {}, {}, set(), {}, []
    actions = []
    observed_sandbox = False
    admitted_at = None
    alarm_sent = live_at_alarm = supervisor_live_at_alarm = False
    admission_event = None
    process = None
    root_key = None

    def observe(pid):
        nonlocal observed_sandbox
        fd = None
        try:
            ticks = start_ticks(pid)
            key = (pid, ticks)
            if records.get(key, {}).get('terminal'):
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
                return key
            if key not in handles:
                fd = os.pidfd_open(pid)
                # Revalidate ancestry after pinning: a scanned PID may have been
                # recycled before pidfd_open. A stable identity alone is not ownership.
                if ((not (pid == process.pid and process.returncode is None)
                     and pid not in descendants(os.getpid()))
                        or ticks != start_ticks(pid)):
                    os.close(fd)
                    return None
                handles[key] = fd
                fd = None
                records[key] = {'pid': pid, 'start_ticks': ticks, 'terminal': False,
                                'comm': (Path('/proc') / str(pid) / 'comm').read_text().strip()}
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

    def reap_finished():
        for key, fd in list(handles.items()):
            if not exited(fd):
                continue
            records.setdefault(key, {'pid': key[0], 'start_ticks': key[1]})['terminal'] = True
            if process is not None and key[0] == process.pid:
                process.poll()  # Preserve Popen's native return code.
            else:
                try:
                    os.waitpid(key[0], os.WNOHANG)
                except ChildProcessError:
                    pass  # Descendant still belongs to its living parent.
            os.close(fd)
            del handles[key]

    def scan_owned():
        # Orphans from setsid/double-fork are adopted here, even if their original
        # ancestor exited between polls. There were no unrelated initial children.
        for pid in descendants(os.getpid()) - {os.getpid()}:
            observe(pid)
        reap_finished()

    def terminate_owned():
        for sig, delay in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
            deadline = time.monotonic() + delay
            sent = set()
            while True:
                try:
                    scan_owned()
                except Exception as exc:  # noqa: BLE001 - retain failure, still stop pinned handles
                    observations.append(f'cleanup observation: {type(exc).__name__}')
                for key, fd in list(handles.items()):
                    if key not in sent and not exited(fd):
                        try:
                            signal.pidfd_send_signal(fd, sig)
                            actions.append({'pid': key[0], 'start_ticks': key[1], 'signal': sig.name})
                        except ProcessLookupError:
                            pass
                        sent.add(key)
                reap_finished()
                if not handles or time.monotonic() >= deadline:
                    break
                time.sleep(.05)
        scan_owned()

    try:
        context = output.open('xb') if output is not None else nullcontext(None)
        with context as log:
            launch = command if seconds is None else [
                'timeout', '--signal=TERM', '--kill-after=5s' if expect_timeout else '--kill-after=30s',
                f'{seconds}s', *command]
            process = subprocess.Popen(launch, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT if log is not None else None,
                                       start_new_session=True)
            root_key = observe(process.pid)
            absolute_deadline = None if seconds is None else time.monotonic() + seconds + 35
            while process.poll() is None:
                scan_owned()
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
                        admitted_at, admission_event = stamp, event
                if admitted_at is not None and not alarm_sent and time.monotonic() >= admitted_at + 3:
                    live_at_alarm = any(key in handles and not exited(handles[key]) for key in sandbox_handles)
                    supervisor_live_at_alarm = any(
                        namespace_pid == admission_event.get('supervisor_pid')
                        and key in handles and not exited(handles[key])
                        for key, namespace_pid in supervisor_handles.items())
                    if root_key not in handles or exited(handles[root_key]):
                        raise RuntimeError('outer timer handle unavailable at deadline')
                    signal.pidfd_send_signal(handles[root_key], signal.SIGALRM)
                    alarm_sent = True
                if absolute_deadline is not None and time.monotonic() > absolute_deadline:
                    raise TimeoutError('outer timeout supervisor exceeded its bound')
                time.sleep(.1)
            returncode = process.wait()
            if log is not None:
                log.flush()
                os.fsync(log.fileno())
        scan_owned()
        before_cleanup = [pid for (pid, _), fd in handles.items() if not exited(fd)]
        terminate_owned()
        survivors = sorted({pid for (pid, _), fd in handles.items() if not exited(fd)}
                           | (descendants(os.getpid()) - {os.getpid()}))
        evidence = {'returncode': returncode,
                    'timed_out': returncode in {124, 137} or (returncode == -signal.SIGKILL and alarm_sent),
                    'cleanup_complete': not survivors, 'observed_native_sandbox': observed_sandbox,
                    'admitted_before_timeout': admitted_at is not None,
                    'admitted_sandbox_live_at_timeout': live_at_alarm, 'alarm_sent': alarm_sent,
                    'admission_supervisor_live_at_timeout': supervisor_live_at_alarm,
                    'admission_event': admission_event,
                    'elapsed_since_admission': None if admitted_at is None else time.monotonic() - admitted_at,
                    'termination_grace_seconds': 5 if expect_timeout else 30,
                    'observation_errors': observations, 'cleanup_actions': actions,
                    'survivors_before_cleanup': before_cleanup,
                    'observed_processes': [records[key] for key in sorted(records)],
                    'survivors': survivors, 'lifecycle': 'owned-subreaper-v1'}
        evidence['passed'] = (not survivors and not observations and (observed_sandbox or not require_sandbox)
                              and (evidence['timed_out'] and alarm_sent and live_at_alarm and supervisor_live_at_alarm
                                   and evidence['elapsed_since_admission'] < 30
                                   if expect_timeout else returncode == 0))
        return evidence
    except BaseException:
        try:
            terminate_owned()
        finally:
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        raise
    finally:
        for fd in handles.values():
            os.close(fd)
        libc.prctl(36, previous.value, 0, 0, 0)


def container_main(argv):
    parser = argparse.ArgumentParser(description='Run one owned container and reap its descendants')
    parser.add_argument('--timeout-seconds', type=int)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or (args.timeout_seconds is not None and args.timeout_seconds <= 0):
        parser.error('a command and a positive optional timeout are required')
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    # Reserve an immutable record before execution, without storing credential-bearing argv.
    with args.journal.open('x') as journal:
        try:
            result = execute(command, None, args.timeout_seconds, require_sandbox=False)
        except BaseException as exc:  # noqa: BLE001 - retain lifecycle failure and fail the command
            result = {'passed': False, 'error': type(exc).__name__, 'returncode': 1}
        result['command_sha256'] = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        json.dump(result, journal, indent=2)
        journal.write('\n')
        journal.flush()
        os.fsync(journal.fileno())
    code = result['returncode']
    if code == 0 and not result['passed']:
        return 1
    return 128 - code if code < 0 else code


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
    raise SystemExit(container_main(sys.argv[2:]) if sys.argv[1:2] == ['--container-command'] else main())
