"""Synthetic CPU-only boundary tests; no benchmark items, model, or network I/O."""

import base64
import dataclasses
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import ptb_python_sandbox as sandbox
import pytest

NATIVE = os.environ.get("PTB_SANDBOX_NATIVE") == "1"
native = pytest.mark.skipif(
    not NATIVE,
    reason="real namespace tests require explicit PTB_SANDBOX_NATIVE=1 admission",
)
PINNED = Path(
    "/tmp/exp-protocol-save-runtime.JEZlHo/rootfs/usr/local/lib/python3.10/dist-packages"
)


@pytest.fixture(scope="module")
def runtime():
    if not NATIVE:
        pytest.skip("native admission is required")
    return sandbox.build_runtime(
        bwrap="/home/robtang_google_com/.local/bin/bwrap",
        python="/usr/bin/python3.10",
        stdlib="/usr/lib/python3.10",
        library_dirs=["/lib/x86_64-linux-gnu", "/lib64"],
        public_packages=[
            PINNED / "numpy",
            PINNED / "numpy.libs",
            PINNED / "numpy-2.2.6.dist-info",
        ],
    )


@pytest.mark.parametrize(
    "change",
    [
        {"wall_seconds": 31},
        {"new_tasks": 257},
        {"scratch_bytes": 0},
        {"cpu_seconds": True},
        {"open_files": 1.5},
        {"wall_seconds": float("nan")},
        {"output_bytes": 10 * 1024 * 1024 + 1},
    ],
)
def test_limit_rejection(change):
    with pytest.raises(ValueError):
        sandbox.Limits(**change)


def test_profile_cannot_expose_arbitrary_host_destination(tmp_path):
    asset = tmp_path / "public"
    asset.write_text("synthetic")
    value = sandbox.Runtime(
        str(asset),
        "/usr/bin/python3.10",
        ((str(asset), "/home/secret", sandbox._sha(asset)),),
        sandbox._sha(asset),
    )
    with pytest.raises(sandbox.SandboxInfrastructureError, match="destination"):
        value.verify()


@pytest.mark.parametrize(
    "name",
    [
        "local",
        "docker",
        "default",
        "inspect_ai/local",
        "other/ptb_python",
        "",
        " bad",
        "Local",
    ],
)
def test_backend_name_rejected_before_registration(name):
    with pytest.raises(sandbox.SandboxInfrastructureError):
        sandbox.register_inspect(None, name=name)


def test_output_capture_bounds_raw_bytes_and_incomplete_decoding():
    capture = sandbox._OutputCapture(1)
    assert capture.feed("é".encode())
    evidence = capture.snapshot()
    assert len(capture.data) == 1
    assert base64.b64decode(evidence["bytes_base64"]) == b"\xc3"
    assert evidence["truncated"] and not evidence["eof"]
    assert evidence["coverage"] == "truncated"
    assert evidence["decode_error"]["reason"] == "unexpected end of data"
    assert evidence["observed_bytes_lower_bound"] == 2
    partial = sandbox._OutputCapture(10)
    partial.feed(b"ok")
    assert partial.snapshot()["coverage"] == "incomplete"
    partial.eof = True
    assert partial.snapshot()["coverage"] == "complete"


@pytest.fixture
def public_transport_manifest(tmp_path):
    # Filesystem/manifest unit fixtures only: deliberately not executable Python
    # or libseccomp. Actual executable closures are tested natively below/in-image.
    root = tmp_path / "public-source"
    root.mkdir()
    interpreter = root / "synthetic-interpreter"
    interpreter.write_bytes(b"synthetic public executable bytes\n")
    interpreter.chmod(0o700)
    library = root / "synthetic-library"
    library.write_bytes(b"synthetic public library bytes\n")
    return sandbox.Runtime(
        str(interpreter),
        "/usr/bin/python3.10",
        (
            (str(interpreter), "/usr/bin/python3.10", sandbox._sha(interpreter)),
            (
                str(library),
                "/lib/x86_64-linux-gnu/libseccomp.so.2",
                sandbox._sha(library),
            ),
        ),
        sandbox._sha(interpreter),
        "non-executable-manifest-unit-fixture",
    )


def materialize_fixture(runtime, destination):
    return sandbox.materialize_runtime(
        runtime,
        directory=destination,
        source_image_reference="synthetic manifest fixture, not an image acceptance",
        source_image_sha256="0" * 64,
    )


def test_materialization_preserves_original_identity_and_exact_files(
    public_transport_manifest, tmp_path
):
    original = public_transport_manifest
    legacy = dataclasses.asdict(original)
    legacy.pop("materialization")
    import hashlib

    assert (
        original.identity
        == hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    )
    staged = materialize_fixture(original, tmp_path / "transport")
    assert staged.identity != original.identity
    assert staged.materialization["source_runtime_sha256"] == original.identity
    assert [entry[1:] for entry in staged.files] == [
        entry[1:] for entry in original.files
    ]
    for old, new in zip(original.files, staged.files):
        assert Path(old[0]).read_bytes() == Path(new[0]).read_bytes()
    receipt = json.loads(Path(staged.materialization["receipt"]).read_text())
    assert (
        receipt["source_image"]["verification"] == "caller_provenance_not_rehashed_here"
    )
    assert receipt["state"] == "complete"
    staged.verify()


@pytest.mark.parametrize(
    "change",
    [
        "copy_bytes",
        "copy_symlink",
        "source_bytes",
        "source_symlink",
        "source_mode",
        "receipt_bytes",
        "source_path_reference",
    ],
)
def test_materialization_rejects_mutation(public_transport_manifest, tmp_path, change):
    original = public_transport_manifest
    staged = materialize_fixture(original, tmp_path / "transport")
    target = Path(staged.files[0][0])
    source = Path(original.files[0][0])
    receipt_path = Path(staged.materialization["receipt"])
    if change == "copy_bytes":
        target.chmod(0o600)
        target.write_bytes(b"different")
    elif change in {"copy_symlink", "source_symlink"}:
        changed = target if change == "copy_symlink" else source
        backup = changed.with_name(changed.name + ".saved")
        changed.rename(backup)
        changed.symlink_to(backup)
    elif change == "source_bytes":
        source.write_bytes(b"changed original public source")
    elif change == "source_mode":
        source.chmod(0o500)
    elif change == "receipt_bytes":
        receipt_path.write_text("{}")
    else:
        receipt = json.loads(receipt_path.read_text())
        receipt["source_runtime"]["files"][0][0] = str(target)
        receipt_path.write_text(json.dumps(receipt))
        staged = dataclasses.replace(
            staged,
            materialization={
                **staged.materialization,
                "sha256": sandbox._sha(receipt_path),
            },
        )
    with pytest.raises(sandbox.SandboxInfrastructureError):
        staged.verify()


def test_materialization_never_overwrites_existing_destination(
    public_transport_manifest, tmp_path
):
    destination = tmp_path / "incumbent"
    destination.mkdir()
    marker = destination / "keep"
    marker.write_text("owned synthetic incumbent")
    with pytest.raises(sandbox.SandboxInfrastructureError, match="fresh path"):
        materialize_fixture(public_transport_manifest, destination)
    assert marker.read_text() == "owned synthetic incumbent"
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(sandbox.SandboxInfrastructureError):
        materialize_fixture(public_transport_manifest, alias / "new")


def test_materialization_keeps_failed_partial_copy(
    public_transport_manifest, tmp_path, monkeypatch
):
    real_copy = sandbox.shutil.copyfileobj
    calls = 0

    def failing_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic copy interruption")
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(sandbox.shutil, "copyfileobj", failing_copy)
    root = tmp_path / "partial"
    with pytest.raises(sandbox.SandboxInfrastructureError) as caught:
        materialize_fixture(public_transport_manifest, root)
    receipt = json.loads((root / "materialization.json").read_text())
    assert receipt["state"] == "failed"
    assert receipt["error"]["message"] == "synthetic copy interruption"
    assert len(receipt["assets"]) == 1 and (root / "files" / "000000").is_file()
    assert caught.value.report["error_category"] == "runtime_materialization"
    public_transport_manifest.verify()


@native
def test_materialized_native_execution_preserves_source_binding(runtime, tmp_path):
    staged = sandbox.materialize_runtime(
        runtime,
        directory=tmp_path / "native-public",
        source_image_reference="mixed host/extracted public CPU test runtime (not image acceptance)",
        source_image_sha256="0" * 64,
    )
    result = sandbox.execute_python(
        "import numpy; print(int(numpy.arange(4).sum()))", staged
    )
    assert result.success and result.stdout == "6\n"
    assert (
        result.evidence["runtime_materialization"]["source_runtime_sha256"]
        == runtime.identity
    )
    target = Path(staged.files[0][0])
    target.chmod(0o600)
    target.write_bytes(b"mutated transport")
    with pytest.raises(sandbox.SandboxInfrastructureError) as caught:
        sandbox.execute_python("raise AssertionError('must not launch')", staged)
    assert not caught.value.report.get("started")


@native
def test_native_registry_reserved_duplicate_foreign_and_concurrent(runtime):
    sys.path.insert(0, str(PINNED))
    from inspect_ai.util._sandbox import registry

    local = registry.registry_find_sandboxenv("local")
    for reserved in ("local", "docker", "default"):
        with pytest.raises(sandbox.SandboxInfrastructureError, match="reserved"):
            sandbox.register_inspect(runtime, name=reserved)
    assert registry.registry_find_sandboxenv("local") is local

    name = "ptb_registry_owned_regression"
    first = sandbox.register_inspect(runtime, name=name)
    for other_runtime, other_limits in (
        (runtime, sandbox.DEFAULT_LIMITS),
        (
            dataclasses.replace(runtime, profile="synthetic-distinct-profile"),
            sandbox.DEFAULT_LIMITS,
        ),
        (runtime, sandbox.Limits(cpu_seconds=29)),
    ):
        with pytest.raises(
            sandbox.SandboxInfrastructureError, match="already registered"
        ):
            sandbox.register_inspect(other_runtime, limits=other_limits, name=name)
        assert registry.registry_find_sandboxenv(name) is first

    class Foreign(first):
        pass

    registry.sandboxenv_register(
        Foreign, "synthetic_extension/ptb_registry_foreign_regression"
    )
    with pytest.raises(sandbox.SandboxInfrastructureError, match="already registered"):
        sandbox.register_inspect(runtime, name="ptb_registry_foreign_regression")
    assert (
        registry.registry_find_sandboxenv("ptb_registry_foreign_regression") is Foreign
    )

    barrier = threading.Barrier(2)

    def register_once(_):
        barrier.wait()
        try:
            return sandbox.register_inspect(
                runtime, name="ptb_registry_race_regression"
            )
        except sandbox.SandboxInfrastructureError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(register_once, range(2)))
    winners = [result for result in results if isinstance(result, type)]
    assert len(winners) == 1
    assert (
        registry.registry_find_sandboxenv("ptb_registry_race_regression") is winners[0]
    )
    assert registry.registry_find_sandboxenv("local") is local


@native
def test_real_python_success_and_failure(runtime):
    code = "import collections,math,itertools,statistics,decimal,fractions,re,sqlite3,ssl,bz2,lzma,ctypes;\ndef f(x): return math.factorial(x)\ndef check(fn): assert fn(5)==120\ncheck(f)\nprint('synthetic-ok')"
    result = sandbox.execute_python(code, runtime)
    assert result.success and result.stdout == "synthetic-ok\n"
    assert result.evidence["started"] and result.evidence["monitor_reaped"]
    failed = sandbox.execute_python("assert 2+2 == 5", runtime)
    assert (
        not failed.success
        and failed.returncode == 1
        and "AssertionError" in failed.stderr
    )
    assert failed.evidence["outcome"] == "program_failure"
    assert failed.evidence["error_category"] == "program_exit"
    assert "AssertionError" in failed.evidence["output"]["stderr"]["text"]


@native
def test_public_numpy_dependency_is_real(runtime):
    result = sandbox.execute_python(
        "import numpy as np; print(np.__version__); assert np.linalg.det(np.eye(3)) == 1",
        runtime,
    )
    assert result.success and "2.2.6" in result.stdout


@native
def test_isolation_environment_files_devices_processes(runtime, tmp_path, monkeypatch):
    secret = tmp_path / "synthetic-secret"
    secret.write_text("not-a-real-credential")
    monkeypatch.setenv("SYNTHETIC_PARENT_SECRET", "must-not-be-inherited")
    code = f"""import os,json,socket
assert 'SYNTHETIC_PARENT_SECRET' not in os.environ
assert not os.path.exists({str(secret)!r})
for path in ['/rmeng_data','/root','/sys','/dev/nvidia0','/dev/dri','/home/ben/task','/etc/passwd']:
    assert not os.path.exists(path), path
assert os.getpid()==1 and os.getppid()==0
assert set(x for x in os.listdir('/proc') if x.isdigit())=={{'1'}}
assert set(os.listdir('/dev'))=={{'null','zero','random','urandom','shm'}}
assert set(x[1] for x in socket.if_nameindex())=={{'lo'}}
assert '00000000\t00000000' not in open('/proc/net/route').read()
assert os.read(0,1)==b''
print(json.dumps({{'env':dict(os.environ),'pid':os.getpid()}}))
"""
    result = sandbox.execute_python(code, runtime)
    assert result.success, result.stderr
    assert "must-not-be-inherited" not in result.stdout


@native
def test_files_private_and_total_tmpfs_bounded(runtime):
    limits = sandbox.Limits(scratch_bytes=1024 * 1024, file_bytes=1024 * 1024)
    code = "import os; s=os.statvfs('/work'); assert s.f_blocks*s.f_frsize==1048576; open('/work/first','wb').write(b'x'*700000); open('/work/second','wb').write(b'x'*700000)"
    result = sandbox.execute_python(code, runtime, limits=limits)
    assert not result.success and result.returncode == 1
    assert "OSError: [Errno 28]" in result.evidence["stderr_diagnostics"]["matches"]
    assert sandbox.execute_python(
        "import os; assert not os.path.exists('/work/first')", runtime
    ).success


@native
def test_normal_threads_fork_and_subprocess(runtime):
    code = """import threading,os,subprocess,sys
values=[]
worker=threading.Thread(target=lambda: values.append(7)); worker.start(); worker.join()
assert values==[7]
pid=os.fork()
if pid==0: os._exit(0)
assert os.waitpid(pid,0)[1]==0
assert subprocess.check_output([sys.executable,'-c','print(19)']).strip()==b'19'
"""
    result = sandbox.execute_python(code, runtime)
    assert result.success, result.stderr
    assert 3 <= result.evidence["new_task_permissions"] <= 8


@native
def test_multiprocessing_pool_and_bounded_posix_shm(runtime):
    code = """import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
def square(x): return x*x
with mp.get_context('fork').Pool(2) as pool:
    assert pool.map(square,[2,3,4])==[4,9,16]
memory=SharedMemory(create=True,size=32)
memory.buf[0]=7
assert memory.buf[0]==7
memory.close(); memory.unlink()
"""
    result = sandbox.execute_python(code, runtime)
    assert result.success, result.stderr


@native
def test_cpu_and_file_limits(runtime):
    with pytest.raises(sandbox.SandboxDependencyError):
        sandbox.execute_python(
            "while True: pass", runtime, limits=sandbox.Limits(cpu_seconds=4)
        )
    result = sandbox.execute_python(
        "with open('large','wb') as stream:\n stream.write(b'x'*5000)",
        runtime,
        limits=sandbox.Limits(file_bytes=1000),
    )
    assert not result.success and result.returncode == 1
    assert "OSError: [Errno 27]" in result.evidence["stderr_diagnostics"]["matches"]


@native
def test_non_utf8_output_is_not_fabricated_success(runtime):
    with pytest.raises(
        sandbox.SandboxInfrastructureError, match="UnicodeDecodeError"
    ) as caught:
        sandbox.execute_python("import os; os.write(1,bytes([255]))", runtime)
    report = caught.value.report
    assert report["error_category"] == "output_decode_error"
    assert report["error"]["type"] == "UnicodeDecodeError"
    assert report["output"]["stdout"]["coverage"] == "complete"
    assert base64.b64decode(report["output"]["stdout"]["bytes_base64"]) == b"\xff"
    assert report["output"]["stdout"]["decode_error"] is not None


@native
def test_lifetime_process_budget_is_not_a_fork_bomb(runtime):
    # At most four sequential, immediately reaped children; no exponential fork.
    code = "import os\nfor i in range(4):\n p=os.fork()\n if p==0: os._exit(0)\n os.waitpid(p,0)"
    with pytest.raises(sandbox.SandboxDependencyError) as caught:
        sandbox.execute_python(code, runtime, limits=sandbox.Limits(new_tasks=2))
    assert "lifetime_task_budget" in caught.value.report["policy_denials"]
    assert caught.value.report["new_task_permissions"] == 2


@native
def test_denied_kernel_namespace_is_not_a_model_failure(runtime):
    code = "import ctypes,sys; print('policy-before',flush=True); print('policy-stderr',file=sys.stderr,flush=True); ctypes.CDLL(None).unshare(0x10000000)"
    with pytest.raises(sandbox.SandboxDependencyError) as caught:
        sandbox.execute_python(code, runtime)
    assert "unshare" in caught.value.report["policy_denials"]
    assert caught.value.report["error_category"] == "policy_denied"
    assert caught.value.report["output"]["stdout"]["text"] == "policy-before\n"
    assert "policy-stderr" in caught.value.report["output"]["stderr"]["text"]


@native
def test_missing_module_preserves_exit_scoring_and_partial_output(runtime):
    result = sandbox.execute_python(
        "print('review_partial_marker',flush=True)\nimport harmless_absent_public_dependency_review_9723",
        runtime,
    )
    assert not result.success and result.returncode == 1
    report = result.evidence
    assert report["outcome"] == "program_failure"
    assert report["error_category"] == "program_exit"
    assert report["stderr_diagnostics"] == {
        "source": "untrusted_program_stderr",
        "matches": ["ModuleNotFoundError"],
        "affects_outcome": False,
    }
    assert report["output"]["stdout"]["text"] == "review_partial_marker\n"
    assert (
        base64.b64decode(report["output"]["stdout"]["bytes_base64"])
        == b"review_partial_marker\n"
    )
    assert (
        "harmless_absent_public_dependency_review_9723"
        in report["output"]["stderr"]["text"]
    )
    assert "Traceback" in report["output"]["stderr"]["text"]
    assert all(
        report["output"][name]["coverage"] == "complete"
        for name in ("stdout", "stderr")
    )


@native
def test_output_bound(runtime):
    with pytest.raises(sandbox.SandboxDependencyError, match="output bound") as caught:
        sandbox.execute_python(
            "print('x'*9000)", runtime, limits=sandbox.Limits(output_bytes=1000)
        )
    evidence = caught.value.report["output"]["stdout"]
    assert evidence["retained_bytes"] == 1000
    assert len(base64.b64decode(evidence["bytes_base64"])) == 1000
    assert evidence["truncated"] and evidence["coverage"] == "truncated"
    assert evidence["observed_bytes_lower_bound"] > 1000
    assert caught.value.report["error_category"] == "output_limit"


@native
def test_memory_limit(runtime):
    result = sandbox.execute_python(
        "blob=bytearray(400*1024*1024)",
        runtime,
        limits=sandbox.Limits(address_space_bytes=128 * 1024 * 1024),
    )
    assert not result.success and result.returncode == 1
    assert "MemoryError" in result.evidence["stderr_diagnostics"]["matches"]


@native
def test_timeout_kills_setsid_descendant(runtime):
    code = "import os,time,sys\nprint('timeout-before',flush=True)\nprint('timeout-stderr',file=sys.stderr,flush=True)\np=os.fork()\nif p==0: os.setsid()\nwhile True: time.sleep(0.05)"
    with pytest.raises(TimeoutError) as caught:
        sandbox.execute_python(code, runtime, limits=sandbox.Limits(wall_seconds=0.25))
    report = caught.value.report
    assert report["started"] and report["monitor_reaped"]
    assert report["new_task_permissions"] == 1
    assert report["observed_namespace_children"]
    assert report["descendants_reaped"]
    assert report["outcome"] == "timeout" and report["error_category"] == "wall_timeout"
    assert report["error"]["type"] == "TimeoutError"
    assert report["output"]["stdout"]["text"] == "timeout-before\n"
    assert "timeout-stderr" in report["output"]["stderr"]["text"]
    assert not Path(f"/proc/{report['namespace_child_pid']}").exists()
    assert all(
        not Path(f"/proc/{pid}").exists()
        for pid in report["observed_namespace_children"]
    )


@native
def test_cancellation_is_bounded(runtime):
    cancel = threading.Event()
    timer = threading.Timer(6, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(sandbox.SandboxCancelled) as caught:
            sandbox.execute_python(
                "import os,time,sys\nprint('cancel-before',flush=True)\nprint('cancel-stderr',file=sys.stderr,flush=True)\np=os.fork()\nif p==0: os.setsid()\ntime.sleep(20)",
                runtime,
                cancel=cancel,
            )
    finally:
        timer.cancel()
    assert time.monotonic() - started < 15
    assert caught.value.report["started"]
    assert caught.value.report["descendants_reaped"]
    assert caught.value.report["outcome"] == "cancelled"
    assert caught.value.report["error_category"] == "caller_cancellation"
    assert caught.value.report["output"]["stdout"]["text"] == "cancel-before\n"
    assert "cancel-stderr" in caught.value.report["output"]["stderr"]["text"]
    assert (
        caught.value.report["supervisor_termination"]["error"]["type"]
        == "KeyboardInterrupt"
    )
    assert all(
        not Path(f"/proc/{pid}").exists()
        for pid in caught.value.report["observed_namespace_children"]
    )


@native
def test_injected_cleanup_error_preserves_primary_timeout_and_output(runtime):
    # Real namespace/sleep execution, only the trusted supervisor's pidfd-send
    # is fault-injected. Run in its own process because it owns subreaper state.
    script = """import json,sys
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import ptb_python_sandbox as s
r=s.Runtime(**json.load(sys.stdin))
with patch.object(s.signal,'pidfd_send_signal',side_effect=PermissionError('synthetic pidfd denial')):
    try:
        s._execute_python("print('cleanup-before',flush=True); import time; time.sleep(10)",r,limits=s.Limits(wall_seconds=0.2))
    except s.SandboxInfrastructureError as exc:
        print(json.dumps({'type':type(exc).__name__,'cause':type(exc.__cause__).__name__,'report':exc.report}))
    else:
        raise AssertionError('cleanup fault was concealed')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(sandbox.__file__).parent)],
        input=json.dumps(dataclasses.asdict(runtime)),
        capture_output=True,
        check=False,
        text=True,
        timeout=25,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    report = observed["report"]
    assert observed["cause"] == "TimeoutError"
    assert report["outcome"] == "infrastructure_error"
    assert report["error_category"] == "cleanup_failure"
    assert report["primary_outcome"] == "timeout"
    assert report["primary_error_category"] == "wall_timeout"
    assert report["error"]["type"] == "TimeoutError"
    assert report["cleanup_errors"][0]["message"] == "synthetic pidfd denial"
    assert report["output"]["stdout"]["text"] == "cleanup-before\n"
    assert report["monitor_reaped"] and report["descendants_reaped"]
    assert not Path(f"/proc/{report['namespace_child_pid']}").exists()


@native
def test_bootstrap_failure_is_infrastructure(runtime):
    # A real bwrap invocation cannot start its missing interpreter. Never a score.
    broken = dataclasses.replace(
        runtime,
        files=tuple(
            entry
            for entry in runtime.files
            if not entry[1].endswith("/encodings/__init__.py")
        ),
    )
    with pytest.raises(sandbox.SandboxInfrastructureError) as caught:
        sandbox.execute_python("assert False", broken)
    assert not caught.value.report.get("started")
    assert caught.value.report["error_category"] == "bootstrap_failure"
    assert caught.value.report["output"]["stderr"]["text"]


@native
def test_manifest_change_is_infrastructure(runtime, tmp_path):
    copy = tmp_path / "fake"
    copy.write_text("old")
    changed = dataclasses.replace(
        runtime,
        files=runtime.files + ((str(copy), "/usr/lib/changed", sandbox._sha(copy)),),
    )
    copy.write_text("new")
    with pytest.raises(sandbox.SandboxInfrastructureError, match="changed"):
        sandbox.execute_python("assert False", changed)


@native
def test_pinned_inspect_adapter_without_benchmark_data(runtime):
    # Import installed library code only. No Task/dataset/model is constructed.
    sys.path.insert(0, str(PINNED))
    import anyio
    from inspect_ai.util._sandbox.registry import registry_find_sandboxenv

    cls = sandbox.register_inspect(runtime, name="ptb_python_synthetic_test")
    assert registry_find_sandboxenv("ptb_python_synthetic_test") is cls

    async def scenario():
        envs = await cls.sample_init("synthetic", None, {})
        env = envs["default"]
        result = await env.exec(["python", "-c", "print(27)"], timeout=30)
        assert result.success and result.stdout == "27\n"
        with pytest.raises(sandbox.SandboxInfrastructureError):
            await env.exec(["sh", "-c", "true"])
        missing = await env.exec(["python", "-c", "import not_installed_synthetic_942"])
        assert not missing.success and missing.returncode == 1
        assert (
            "not_installed_synthetic_942"
            in env.execution_evidence[-1]["output"]["stderr"]["text"]
        )
        with anyio.move_on_after(6) as scope:
            await env.exec(["python", "-c", "import time; time.sleep(15)"], timeout=30)
        assert scope.cancel_called
        assert env.execution_evidence[-1]["descendants_reaped"]

        # Saturate AnyIO's worker pool: cancellation must not wait for a job
        # which never started and therefore never set its completion event.
        limiter = anyio.to_thread.current_default_thread_limiter()
        old_tokens = limiter.total_tokens
        limiter.total_tokens = 1
        await limiter.acquire()
        try:
            with anyio.move_on_after(0.05) as early:
                await env.exec(["python", "-c", "print(1)"], timeout=30)
            assert early.cancel_called
        finally:
            limiter.release()
            limiter.total_tokens = old_tokens
        await cls.sample_cleanup("synthetic", None, envs, True)
        with pytest.raises(sandbox.SandboxInfrastructureError):
            await env.exec(["python", "-c", "print(1)"])

    anyio.run(scenario)


@native
def test_original_scorer_and_actual_30_second_timeout_on_synthetic_code(runtime):
    sys.path.insert(0, str(PINNED))
    import anyio
    from inspect_ai.util._sandbox.context import (
        sandbox_default_context_var,
        sandbox_environments_context_var,
    )

    humaneval = importlib.import_module("inspect_evals.humaneval.humaneval")
    assert (
        sandbox._sha(humaneval.__file__)
        == "e7bd00d5002afa39f8c42ac5183bcb65feb7692f069eca41a32bad44024f6aaa"
    )
    assert humaneval.VERIFY_TIMEOUT == 30
    cls = sandbox.register_inspect(runtime, name="ptb_python_synthetic_scorer")

    async def scenario():
        envs = await cls.sample_init("synthetic-scorer-only", None, {})
        token = sandbox_environments_context_var.set(envs)
        default = sandbox_default_context_var.set("default")
        try:
            state = SimpleNamespace(
                output=SimpleNamespace(
                    completion="```python\ndef synthetic_sum(a,b):\n    return a+b\n```"
                ),
                metadata={
                    "prompt": "def synthetic_sum(a,b):\n",
                    "test": "def check(fn):\n    assert fn(17,25)==42\n",
                    "entry_point": "synthetic_sum",
                },
            )
            scorer = humaneval.verify()
            assert (await scorer(state, None)).value == "C"
            state.output.completion = "    return 0\n"
            assert (await scorer(state, None)).value == "I"
            # These are ordinary admitted rc=1 programs, not trusted signals of
            # evaluator failure. The scorer and captured diagnostics stay real.
            for marker, completion, expected_matches in (
                (
                    "missing",
                    "    import nonexistent_synthetic_dependency_873\n",
                    ["ModuleNotFoundError"],
                ),
                (
                    "forged",
                    "    import sys\n    print('ModuleNotFoundError ImportError MemoryError',file=sys.stderr)\n    raise RuntimeError('invented student failure')\n",
                    ["ImportError", "MemoryError", "ModuleNotFoundError"],
                ),
                ("value", "    raise ValueError('invented student failure')\n", []),
            ):
                state.output.completion = (
                    f"    print('{marker}-partial',flush=True)\n" + completion
                )
                assert (await scorer(state, None)).value == "I"
                evidence = envs["default"].execution_evidence[-1]
                assert evidence["outcome"] == "program_failure"
                assert evidence["program_returncode"] == 1
                assert evidence["output"]["stdout"]["text"] == f"{marker}-partial\n"
                assert "Traceback" in evidence["output"]["stderr"]["text"]
                assert evidence["stderr_diagnostics"]["matches"] == expected_matches
                assert evidence["stderr_diagnostics"]["affects_outcome"] is False
            state.output.completion = "    import sys\n    print('ImportError MemoryError',file=sys.stderr)\n    return a+b\n"
            assert (await scorer(state, None)).value == "C"
            assert envs["default"].execution_evidence[-1]["stderr_diagnostics"][
                "matches"
            ] == ["ImportError", "MemoryError"]
            state.output.completion = (
                "    import time\n    time.sleep(35)\n    return a+b\n"
            )
            started = time.monotonic()
            result = await scorer(state, None)
            elapsed = time.monotonic() - started
            assert (
                result.value == "I" and "Verification timed out" in result.explanation
            )
            assert 30 <= elapsed < 45
        finally:
            sandbox_default_context_var.reset(default)
            sandbox_environments_context_var.reset(token)
            await cls.sample_cleanup("synthetic-scorer-only", None, envs, False)

    anyio.run(scenario)
