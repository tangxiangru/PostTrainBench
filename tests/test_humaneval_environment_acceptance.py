"""CPU-only validation of the admission harness, never a node/GPU acceptance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("environment_runner", ROOT / "src/commit_utils/slurm/humaneval_environment_acceptance.py")
probe = load("environment_probe", ROOT / "src/eval/humaneval_environment_probe.py")


def rows():
    result = []
    for name in probe.programs():
        good = name in {"invented-good", "invented-private"}
        timeout = name == "invented-timeout"
        result.append({"id": name, "score": "C" if good else "I", "error": False,
                       "executions": 1, "started": True, "cleanup_complete": True,
                       "outcome": "success" if good else "timeout" if timeout else "program_failure",
                       "error_category": None if good else "wall_timeout" if timeout else "program_exit"})
    return result


def test_program_failure_cannot_pass_timeout_acceptance():
    values = rows()
    probe.validate_rows(values)
    timeout = next(row for row in values if row["id"] == "invented-timeout")
    timeout.update(outcome="program_failure", error_category="program_exit")
    with pytest.raises(ValueError, match="cannot stand in"):
        probe.validate_rows(values)


@pytest.mark.parametrize("key,value", [("executions", 0), ("error", True), ("started", False),
                                        ("cleanup_complete", False)])
def test_missing_native_evidence_is_rejected(key, value):
    values = rows()
    values[0][key] = value
    with pytest.raises(ValueError):
        probe.validate_rows(values)


def test_admission_callback_crosses_the_fresh_supervisor_process(tmp_path, monkeypatch):
    # This validates transport across a real -I/-S exec, not a native sandbox.
    from dataclasses import dataclass

    helper = load('environment_transport_helper', ROOT / 'src/eval/ptb_python_sandbox.py')
    child = tmp_path / 'transport_child.py'
    child.write_text("""import hashlib,json,os,sys,time
request=json.load(sys.stdin)
if 'admission_fd' in request:
    event={'schema':'ptb-python-admission-v1','started':True,'supervisor_pid':os.getpid(),'admitted_monotonic':time.monotonic(),
           'attested_namespace_pid':1,'code_sha256':hashlib.sha256(request['code'].encode()).hexdigest()}
    encoded=json.dumps(event).encode()+b'\\n'
    os.write(request['admission_fd'],encoded[:15]);time.sleep(.08)
    os.write(request['admission_fd'],encoded[15:]);os.close(request['admission_fd'])
time.sleep(.08)
print(json.dumps({'kind':'execution','value':{'success':True,'returncode':0,'stdout':'',
                  'stderr':'','evidence':{'synthetic_transport':True}}}))
""")
    monkeypatch.setattr(helper, '__file__', str(child))

    @dataclass
    class TransportFixture:
        synthetic: bool = True

    events = []
    result = helper.execute_python('pass', TransportFixture(), on_admission=events.append)
    assert result.success and len(events) == 1
    assert events[0]['schema'] == 'ptb-python-admission-v1'
    assert events[0]['supervisor_pid'] != __import__('os').getpid()
    # The unchanged default path creates no observer pipe and still completes.
    assert helper.execute_python('pass', TransportFixture()).success


def test_invalid_admission_channel_cannot_start_a_sandbox():
    helper = load('environment_invalid_channel', ROOT / 'src/eval/ptb_python_sandbox.py')
    with pytest.raises(helper.SandboxInfrastructureError, match='inherited pipe'):
        helper._execute_python('pass', None, admission_fd=1)


@pytest.mark.parametrize("image", ["opus_5.sif", "vllm_debug.sif"])
def test_commands_follow_production_home_and_cwd_shapes(tmp_path, image):
    command, output, layout = runner.image_command(
        ROOT, Path("/images") / image, "a" * 64, tmp_path / "work", tmp_path / "scratch",
        Path("/public/bwrap"), Path("/public/data.parquet"))
    assert command[command.index("--pwd") + 1] == layout["cwd"]
    assert "--nv" in command and "--pid" in command and "--no-init" in command
    home = command[command.index("--home") + 1]
    if image == "opus_5.sif":
        assert home.endswith(":/home/ben") and layout["cwd"] == "/home/ben/task"
        assert output == tmp_path / "work/home/task"
    else:
        assert home.endswith(":" + str(Path.home()))
        assert layout["cwd"] == str(ROOT / "src/eval/tasks/humaneval")
        assert output == tmp_path / "work/result"
    assert not any("auth.json" in arg or "context_probe" in arg for arg in command)


def fake_backend(source, *args):
    return [sys.executable, "-c", source, *args, "--unshare-all", "--as-pid-1", "--supervise", "ptb_python_sandbox.py"]


def test_real_local_process_exit_is_observed_and_reaped(tmp_path):
    result = runner.execute(fake_backend("import time; time.sleep(.4)"), tmp_path / "normal.log", 3)
    assert result["passed"] and result["cleanup_complete"]
    assert all(p["terminal"] for p in result["observed_processes"])


def test_startup_timeout_cannot_count_as_admitted_program_timeout(tmp_path):
    result = runner.execute(fake_backend("import time; time.sleep(10)"), tmp_path / "startup.log", 1,
                            expect_timeout=True, ready_file=tmp_path / "missing.json")
    assert result["timed_out"] and not result["passed"]
    assert not result["admitted_before_timeout"]


def test_outer_deadline_is_armed_only_after_ready_and_checks_live_handle(tmp_path):
    ready = tmp_path / "ready.json"
    source = (
        "import json,sys,time,os; time.sleep(.3); "
        "open(sys.argv[1],'w').write(json.dumps({'marker':'PTB_OUTER_ADMITTED_68d273', "
        "'observer':'native-supervisor-admission','event':{'schema':'ptb-python-admission-v1',"
        "'started':True,'supervisor_pid':os.getpid(),'admitted_monotonic':time.monotonic()}})); "
        "time.sleep(30)"
    )
    result = runner.execute(fake_backend(source, str(ready)), tmp_path / "outer.log", 10,
                            expect_timeout=True, ready_file=ready)
    assert result["passed"] and result["alarm_sent"] and result["timed_out"]
    assert result["admitted_sandbox_live_at_timeout"] and result["cleanup_complete"]
    assert result["admission_supervisor_live_at_timeout"]


def test_early_program_exit_cannot_pass_outer_timeout(tmp_path):
    ready = tmp_path / "ready.json"
    ready.write_text('{"marker":"PTB_OUTER_ADMITTED_68d273"}')
    result = runner.execute(fake_backend("import time; time.sleep(.4)"), tmp_path / "early.log", 5,
                            expect_timeout=True, ready_file=ready)
    assert not result["passed"] and not result["alarm_sent"]


def test_observation_exception_still_stops_the_owned_producer(tmp_path, monkeypatch):
    real_popen = runner.subprocess.Popen
    created = []

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        created.append(process)
        return process

    real_descendants = runner.descendants

    def fail_descendants(pid):
        if not created:
            return real_descendants(pid)
        raise RuntimeError("synthetic observer failure")

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "descendants", fail_descendants)
    with pytest.raises(RuntimeError, match="synthetic observer"):
        runner.execute(fake_backend("import time; time.sleep(30)"), tmp_path / "failure.log", 35)
    assert created[0].poll() is not None


def test_production_wrapper_cleans_detached_child_without_touching_other_process(tmp_path):
    import json
    import os
    import subprocess

    child_file = tmp_path / 'detached-pid'
    journal = tmp_path / 'lifecycle.json'
    # This unrelated process belongs to the test, outside the wrapper ancestry.
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    child = f"import os,signal,time,pathlib; os.setsid(); signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(child_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(.5)"
    try:
        completed = subprocess.run(
            [sys.executable, runner.__file__, '--container-command', '--journal', str(journal),
             '--', sys.executable, '-c', parent], timeout=12, check=False)
        result = json.loads(journal.read_text())
        detached_pid = int(child_file.read_text())
        assert completed.returncode == 0 and result['cleanup_complete']
        assert detached_pid in result['survivors_before_cleanup']
        assert any(x['pid'] == detached_pid and x['signal'] == 'SIGKILL' for x in result['cleanup_actions'])
        assert not Path(f'/proc/{detached_pid}').exists()  # Reaped, not just a zombie.
        assert unrelated.poll() is None
        assert all(x['terminal'] for x in result['observed_processes'])
        assert os.getpid() not in [x['pid'] for x in result['observed_processes']]
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_production_wrapper_preserves_command_failure(tmp_path):
    import json
    import subprocess

    journal = tmp_path / 'failed.json'
    result = subprocess.run([sys.executable, runner.__file__, '--container-command', '--journal', str(journal),
                             '--', sys.executable, '-c', 'raise SystemExit(7)'], check=False, timeout=5)
    assert result.returncode == 7
    assert json.loads(journal.read_text())['passed'] is False


def test_adopted_zombie_is_reaped_after_original_parent_exits(tmp_path):
    import os

    code = "import os,time; pid=os.fork(); time.sleep(.6) if pid else None; os._exit(0)"
    result = runner.execute([sys.executable, '-c', code], tmp_path / 'zombie.log', 3, require_sandbox=False)
    assert result['passed'] and result['cleanup_complete']
    assert runner.descendants(os.getpid()) == {os.getpid()}
