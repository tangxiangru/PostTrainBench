"""Opt-in Linux execution boundary for Inspect's unchanged python -c scorer.

No automatic registration/fallback: call register_inspect(runtime) explicitly.
The runtime manifest is trusted operator configuration, never model input.
"""

from __future__ import annotations

import array
import base64
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import selectors
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


class SandboxInfrastructureError(RuntimeError):
    """Not an incorrect model answer; abort/quarantine the evaluation attempt."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = (
            report
            if report is not None
            else {
                "outcome": "infrastructure_error",
                "error_category": "configuration_or_preparation",
                "error": {"type": type(self).__name__, "message": message},
                "output": {
                    name: {
                        "coverage": "unavailable",
                        "reason": "no execution report",
                        "eof": False,
                    }
                    for name in ("stdout", "stderr")
                },
            }
        )


class SandboxDependencyError(SandboxInfrastructureError):
    """Policy/resource failure observed independently of program stderr text."""


class SandboxCancelled(SandboxInfrastructureError):
    pass


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 30
    startup_seconds: float = 10
    address_space_bytes: int = 1024 * 1024 * 1024
    cpu_seconds: int = 30
    scratch_bytes: int = 64 * 1024 * 1024
    scratch_inodes: int = 4096
    file_bytes: int = 16 * 1024 * 1024
    output_bytes: int = 10 * 1024 * 1024  # each stream, as in Inspect
    open_files: int = 128
    new_tasks: int = 32  # lifetime successful *permissions*, not concurrent slots
    code_bytes: int = 120 * 1024  # Linux's per-argv-string limit is 128 KiB

    def __post_init__(self):
        for key, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"invalid positive limit: {key}")
            if key not in {"wall_seconds", "startup_seconds"} and not isinstance(
                value, int
            ):
                raise ValueError(f"integer limit required: {key}")
        if (
            self.wall_seconds > 30
            or self.new_tasks > 256
            or self.code_bytes > 120 * 1024
            or self.output_bytes > 10 * 1024 * 1024
        ):
            raise ValueError(
                "profile permits at most 30 seconds, 256 new tasks, 120 KiB code, 10 MiB per output stream"
            )


DEFAULT_LIMITS = Limits()
_REGISTRATION_LOCK = threading.Lock()
_RESERVED_BACKENDS = frozenset({"local", "docker", "default"})


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _regular(path):
    path = Path(path).resolve(strict=True)
    if not stat.S_ISREG(path.stat().st_mode):
        raise SandboxInfrastructureError(f"not a regular runtime file: {path}")
    return path


@dataclass(frozen=True)
class Runtime:
    """Only enumerated public files are mounted; no directory-wide host binds."""

    bwrap: str
    interpreter: str
    files: tuple[tuple[str, str, str], ...]  # source, jail destination, SHA256
    bwrap_sha256: str
    profile: str = "ptb-python-bwrap-v1"
    materialization: dict | None = None

    def verify(self):
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise SandboxInfrastructureError("only Linux x86_64 is implemented")
        if _sha(self.bwrap) != self.bwrap_sha256:
            raise SandboxInfrastructureError("bubblewrap identity changed")
        destinations = set()
        for source, destination, digest in self.files:
            if (
                not destination.startswith(
                    ("/usr/", "/lib/", "/lib64/", "/runtime/site-packages/")
                )
                or ".." in Path(destination).parts
            ):
                raise SandboxInfrastructureError("invalid runtime destination")
            resolved_source = _regular(source)
            if resolved_source != Path(source):
                raise SandboxInfrastructureError(
                    "runtime source path replaced by a symlink/noncanonical path"
                )
            if destination in destinations or _sha(resolved_source) != digest:
                raise SandboxInfrastructureError("duplicate/changed runtime asset")
            destinations.add(destination)
        if (
            self.interpreter not in destinations
            or "/lib/x86_64-linux-gnu/libseccomp.so.2" not in destinations
        ):
            raise SandboxInfrastructureError("interpreter or libseccomp missing")
        if self.materialization is not None:
            _verify_materialization(self)

    @property
    def identity(self):
        payload = asdict(self)
        if self.materialization is None:
            # Existing unstaged manifests retain their original content identity.
            payload.pop("materialization")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _verify_materialization(runtime):
    binding = runtime.materialization
    receipt_path = _regular(binding["receipt"])
    if _sha(receipt_path) != binding["sha256"]:
        raise SandboxInfrastructureError(
            "public runtime materialization receipt changed"
        )
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "ptb-public-runtime-materialization-v1"
        or receipt.get("state") != "complete"
    ):
        raise SandboxInfrastructureError(
            "public runtime materialization is incomplete/unsupported"
        )
    original = Runtime(**receipt["source_runtime"])
    if original.materialization is not None:
        raise SandboxInfrastructureError(
            "nested public runtime materializations unsupported"
        )
    original.verify()  # Do not silently detach from the actual source image files.
    if (
        original.identity != binding["source_runtime_sha256"]
        or original.identity != receipt["source_runtime_sha256"]
    ):
        raise SandboxInfrastructureError("original public runtime identity differs")
    if (
        runtime.bwrap != original.bwrap
        or runtime.bwrap_sha256 != original.bwrap_sha256
        or runtime.interpreter != original.interpreter
        or runtime.profile != original.profile
    ):
        raise SandboxInfrastructureError(
            "materialization changed execution configuration"
        )
    root = Path(receipt["directory"])
    if (
        root.resolve(strict=True) != root
        or receipt_path != root / "materialization.json"
    ):
        raise SandboxInfrastructureError(
            "materialization directory/receipt binding differs"
        )
    original_files = {
        destination: (source, digest) for source, destination, digest in original.files
    }
    if len(receipt["assets"]) != len(original_files) or len(runtime.files) != len(
        original_files
    ):
        raise SandboxInfrastructureError("materialization file coverage differs")
    expected_files = []
    for index, item in enumerate(receipt["assets"]):
        staged = root / "files" / f"{index:06d}"
        source, digest = original_files.pop(item["destination"])
        if (
            item["source"] != source
            or item["sha256"] != digest
            or item["staged"] != str(staged)
        ):
            raise SandboxInfrastructureError(
                "materialization source/file identity differs"
            )
        if stat.S_IMODE(Path(source).stat().st_mode) != item["source_mode"]:
            raise SandboxInfrastructureError(
                "original public runtime file mode changed"
            )
        if (
            staged.is_symlink()
            or not staged.is_file()
            or staged.stat().st_nlink != 1
            or stat.S_IMODE(staged.stat().st_mode) != item["staged_mode"]
        ):
            raise SandboxInfrastructureError(
                "materialized asset mode/type/link changed"
            )
        expected_files.append((str(staged), item["destination"], digest))
    if [tuple(item) for item in runtime.files] != expected_files or original_files:
        raise SandboxInfrastructureError("materialization execution file set differs")


def materialize_runtime(
    runtime, *, directory, source_image_reference, source_image_sha256
):
    """Copy only the verified public closure to a fresh ordinary-FS/tmpfs directory.

    Needed for Apptainer roots marked unbindable. The original Runtime identity
    and every file hash are retained; the image digest is caller provenance,
    explicitly not a claim that the whole image was rehashed inside the jail.
    No directory-wide runtime bind, reuse, overwrite, or recursive cleanup.
    """
    runtime.verify()
    if runtime.materialization is not None:
        raise SandboxInfrastructureError(
            "materialize the original runtime, not another copy"
        )
    if (
        not isinstance(source_image_reference, str)
        or not source_image_reference.strip()
        or not isinstance(source_image_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_image_sha256)
    ):
        raise SandboxInfrastructureError(
            "explicit source image reference and SHA256 required"
        )
    root = Path(directory).absolute()
    if (
        root.parent.resolve(strict=True) != root.parent
        or root.exists()
        or root.is_symlink()
        or root == Path("/")
    ):
        raise SandboxInfrastructureError(
            "materialization requires a fresh path under an existing canonical directory"
        )
    root.mkdir(mode=0o700, exist_ok=False)
    files_dir = root / "files"
    files_dir.mkdir(mode=0o700)
    receipt_path = root / "materialization.json"
    receipt = {
        "schema": "ptb-public-runtime-materialization-v1",
        "state": "preparing",
        "directory": str(root),
        "source_runtime": asdict(runtime),
        "source_runtime_sha256": runtime.identity,
        "source_image": {
            "reference": source_image_reference,
            "sha256": source_image_sha256,
            "verification": "caller_provenance_not_rehashed_here",
        },
        "assets": [],
    }

    def write_receipt():
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")

    write_receipt()
    started = time.monotonic()
    try:
        copied = []
        for index, (source, destination, digest) in enumerate(runtime.files):
            source_path = _regular(source)
            source_mode = stat.S_IMODE(source_path.stat().st_mode)
            staged = files_dir / f"{index:06d}"
            with source_path.open("rb") as incoming, staged.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            staged_mode = 0o500 if source_mode & 0o111 else 0o400
            staged.chmod(staged_mode)
            if _sha(staged) != digest or _sha(source_path) != digest:
                raise SandboxInfrastructureError(
                    "public source/copy bytes differ during materialization"
                )
            receipt["assets"].append(
                {
                    "source": source,
                    "destination": destination,
                    "staged": str(staged),
                    "sha256": digest,
                    "bytes": staged.stat().st_size,
                    "source_mode": source_mode,
                    "staged_mode": staged_mode,
                }
            )
            copied.append((str(staged), destination, digest))
        runtime.verify()
        receipt["state"] = "complete"
        receipt["copy_seconds"] = time.monotonic() - started
        write_receipt()
        result = Runtime(
            runtime.bwrap,
            runtime.interpreter,
            tuple(copied),
            runtime.bwrap_sha256,
            runtime.profile,
            {
                "receipt": str(receipt_path),
                "sha256": _sha(receipt_path),
                "source_runtime_sha256": runtime.identity,
            },
        )
        result.verify()
        return result
    except BaseException as exc:
        receipt["state"] = "failed"
        receipt["error"] = _exception_record(exc)
        try:
            write_receipt()
        except OSError as record_error:
            receipt["receipt_write_error"] = _exception_record(record_error)
        # Keep the bounded public partial copy for inspection; never merge/delete.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            exc.materialization_report = receipt
            raise
        raise SandboxInfrastructureError(
            "public runtime materialization failed; partial copy retained",
            {
                "outcome": "infrastructure_error",
                "error_category": "runtime_materialization",
                "error": _exception_record(exc),
                "materialization": receipt,
            },
        ) from exc


def build_runtime(*, bwrap, python, stdlib, library_dirs, public_packages=()):
    """Freeze a trusted CPython 3.10/public-package ELF closure, without imports.

    readelf parses ELF metadata, never executes inspected files. public_packages
    are explicit package directories/files, including e.g. numpy AND numpy.libs;
    no site-packages root, .pth, sitecustomize, benchmark/evaluator package binds.
    Store/review the resulting full manifest before relying on evaluations.
    """
    python, stdlib = _regular(python), Path(stdlib).resolve(strict=True)
    if stdlib.name != "python3.10":
        raise SandboxInfrastructureError("only explicit CPython 3.10 layout supported")
    libraries = [Path(p).resolve(strict=True) for p in library_dirs]
    files = {}
    pending = []

    def add(source, destination):
        resolved = _regular(source)
        if destination in files:
            if files[destination][0] != str(resolved):
                raise SandboxInfrastructureError(f"ambiguous library: {destination}")
            return
        files[destination] = (str(resolved), destination, _sha(resolved))
        with resolved.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                pending.append((resolved, destination))

    def tree(source, target, skip=()):
        for root, dirs, names in os.walk(source, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in skip and d != "__pycache__")
            for name in dirs:
                if (Path(root) / name).is_symlink():
                    raise SandboxInfrastructureError(
                        "symlinked package directories unsupported"
                    )
            for name in sorted(names):
                path = Path(root) / name
                if path.suffix == ".pyc":
                    continue
                # A source symlink is resolved once and individually bound; no
                # host-path symlink is exposed to generated code.
                add(path, str(Path(target) / path.relative_to(source)))

    interpreter = "/usr/bin/python3.10"
    add(python, interpreter)
    tree(stdlib, "/usr/lib/python3.10", ("site-packages", "dist-packages"))
    for package in public_packages:
        package = Path(package).resolve(strict=True)
        if (
            package.name
            in {"site-packages", "dist-packages", "inspect_ai", "inspect_evals"}
            or package.name.startswith(".")
            or package.name in {"sitecustomize.py", "usercustomize.py"}
            or package.suffix == ".pth"
        ):
            raise SandboxInfrastructureError(
                "whole site/benchmark/startup-hook mounts forbidden"
            )
        target = f"/runtime/site-packages/{package.name}"
        tree(package, target) if package.is_dir() else add(package, target)
    for required in ("libseccomp.so.2",):
        matches = [d / required for d in libraries if (d / required).exists()]
        if not matches:
            raise SandboxInfrastructureError(
                f"missing required public library: {required}"
            )
        add(matches[0], "/lib/x86_64-linux-gnu/" + required)
    processed = set()
    while pending:
        source, _destination = pending.pop()
        if str(source) in processed:
            continue
        processed.add(str(source))
        try:
            info = subprocess.run(
                ["/usr/bin/readelf", "-l", "-d", str(source)],
                check=True,
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SandboxInfrastructureError(
                "readelf dependency inspection failed"
            ) from exc
        for loader in re.findall(r"Requesting program interpreter: ([^\]]+)", info):
            name = Path(loader).name
            candidates = [d / name for d in libraries if (d / name).exists()]
            if not candidates:
                raise SandboxInfrastructureError(f"ELF loader missing: {name}")
            add(candidates[0], loader)
        for needed in re.findall(r"\(NEEDED\).*\[([^\]]+)\]", info):
            # Bundled public wheel libraries are already bound at their RPATH.
            if any(Path(dest).name == needed for dest in files):
                continue
            matches = [d / needed for d in libraries if (d / needed).exists()]
            if not matches:
                raise SandboxInfrastructureError(
                    f"unresolved public ELF dependency: {needed}"
                )
            add(matches[0], "/lib/x86_64-linux-gnu/" + needed)
    result = Runtime(
        str(_regular(bwrap)),
        interpreter,
        tuple(sorted(files.values(), key=lambda x: x[1])),
        _sha(bwrap),
    )
    result.verify()
    return result


# Trusted bootstrap runs before any generated bytes. It performs privileged
# namespace-local tmpfs setup, then irreversibly drops caps and installs seccomp.
# The control socket/listener is closed before the actual interpreter exec.
_BOOTSTRAP = r"""
import array, ctypes, json, os, resource, socket, sys
channel = socket.socket(fileno=0)
header = channel.recv(4, socket.MSG_WAITALL)
if len(header) != 4: raise RuntimeError('missing payload header')
size = int.from_bytes(header, 'big')
if size > 2*1024*1024: raise RuntimeError('oversized bootstrap payload')
payload = bytearray()
while len(payload) < size:
    block = channel.recv(min(65536, size-len(payload)))
    if not block: raise RuntimeError('truncated bootstrap payload')
    payload.extend(block)
spec = json.loads(payload)
limits = spec['limits']
libc = ctypes.CDLL(None, use_errno=True)
def checked(rc, label):
    if rc != 0: raise OSError(ctypes.get_errno(), label)
checked(libc.mount(b'tmpfs', b'/work', b'tmpfs', 6,
    ('size=%d,nr_inodes=%d,mode=0700' % (limits['scratch_bytes'], limits['scratch_inodes'])).encode()), 'sized tmpfs')
os.mkdir('/work/tmp')
os.mkdir('/work/shm')
os.chdir('/work')
for kind, value in [(resource.RLIMIT_AS, limits['address_space_bytes']),
    (resource.RLIMIT_CPU, limits['cpu_seconds']), (resource.RLIMIT_FSIZE, limits['file_bytes']),
    (resource.RLIMIT_NOFILE, limits['open_files']), (resource.RLIMIT_CORE, 0)]:
    resource.setrlimit(kind, (value, value))
# Drop bounding set while CAP_SETPCAP is still available, then all live caps.
for cap in range(64):
    rc = libc.prctl(24, cap, 0, 0, 0)
    if rc != 0 and ctypes.get_errno() != 22: checked(rc, 'drop capability bound')
class Header(ctypes.Structure): _fields_ = [('version',ctypes.c_uint32),('pid',ctypes.c_int)]
class Caps(ctypes.Structure): _fields_ = [('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]
checked(libc.capset(ctypes.byref(Header(0x20080522,0)), (Caps*2)()), 'drop live capabilities')
checked(libc.prctl(38, 1, 0, 0, 0), 'no_new_privs')
sec = ctypes.CDLL('/lib/x86_64-linux-gnu/libseccomp.so.2', use_errno=True)
sec.seccomp_init.argtypes=[ctypes.c_uint32]; sec.seccomp_init.restype=ctypes.c_void_p
sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p]; sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
sec.seccomp_load.argtypes=[ctypes.c_void_p]
sec.seccomp_notify_fd.argtypes=[ctypes.c_void_p]
sec.seccomp_release.argtypes=[ctypes.c_void_p]
ctx=sec.seccomp_init(0x7fff0000)
if not ctx: raise RuntimeError('seccomp_init failed')
names = ['clone','clone3','fork','vfork','unshare','setns','mount','umount2','pivot_root','chroot',
    'ptrace','process_vm_readv','process_vm_writev','bpf','perf_event_open','userfaultfd',
    'io_uring_setup','open_by_handle_at','name_to_handle_at','keyctl','add_key','request_key',
    'reboot','kexec_load','kexec_file_load','init_module','finit_module','delete_module',
    'swapon','swapoff','iopl','ioperm','seccomp','prctl','memfd_create','mknod','mknodat',
    'shmget','msgget','semget']
numbers={}
for name in names:
    nr=sec.seccomp_syscall_resolve_name(name.encode())
    if nr < 0: raise RuntimeError('unresolved seccomp syscall: '+name)
    numbers[str(nr)]=name
    rc=sec.seccomp_rule_add(ctx, 0x7fc00000, nr, 0) # USER_NOTIF
    if rc: raise RuntimeError('seccomp rule failed: '+name)
if sec.seccomp_load(ctx): raise RuntimeError('seccomp notify load failed')
listener=sec.seccomp_notify_fd(ctx)
if listener < 0: raise RuntimeError('seccomp listener missing')
status=open('/proc/self/status').read()
for field in ['CapInh','CapPrm','CapEff','CapBnd','CapAmb']:
    if int(next(x.split()[1] for x in status.splitlines() if x.startswith(field+':')),16):
        raise RuntimeError('capabilities remain: '+field)
ready={'state':'ready','numbers':numbers,'pid':os.getpid(),'tmpfs_bytes':os.statvfs('/work').f_blocks*os.statvfs('/work').f_frsize}
channel.sendmsg([json.dumps(ready).encode()],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[listener]))])
if channel.recv(1) != b'G': raise RuntimeError('supervisor did not admit execution')
os.close(listener)
channel.close()
fd=os.open('/dev/null',os.O_RDONLY)
os.dup2(fd,0)
os.set_inheritable(0,True)
if fd != 0: os.close(fd)
for entry in os.listdir('/proc/self/fd'):
    if int(entry) > 2:
        try: os.close(int(entry))
        except OSError: pass
os.execv(sys.executable, [sys.executable, '-S', '-c', spec['code']])
"""


class _Data(ctypes.Structure):
    _fields_ = [
        ("nr", ctypes.c_int),
        ("arch", ctypes.c_uint32),
        ("ip", ctypes.c_uint64),
        ("args", ctypes.c_uint64 * 6),
    ]


class _Notification(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("pid", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("data", _Data),
    ]


class _Response(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("value", ctypes.c_int64),
        ("error", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


def _ioctl(fd, operation, value):
    library = ctypes.CDLL(None, use_errno=True)
    request = 0xC0000000 | (ctypes.sizeof(value) << 16) | (ord("!") << 8) | operation

    # A pending notification can disappear after poll (thread exits). The
    # kernel RECV ioctl can then block even on an O_NONBLOCK fd. This runs only
    # in our dedicated trusted supervisor's main thread, not evaluator threads.
    def interrupted(signum, frame):
        raise InterruptedError(errno.EINTR, "notification receive interrupted")

    previous = None
    if operation == 0:
        previous = signal.signal(signal.SIGALRM, interrupted)
        signal.setitimer(signal.ITIMER_REAL, 0.05)
    try:
        rc = library.ioctl(fd, ctypes.c_ulong(request), ctypes.byref(value))
    finally:
        if operation == 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    if rc != 0:
        raise OSError(ctypes.get_errno(), "seccomp notification ioctl")


@dataclass
class Execution:
    success: bool
    returncode: int
    stdout: str
    stderr: str
    evidence: dict = field(default_factory=dict)


def _exception_record(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def _record_failure(report, exc, category=None):
    report["outcome"] = (
        "cancelled"
        if isinstance(exc, (SandboxCancelled, KeyboardInterrupt))
        else "timeout"
        if isinstance(exc, TimeoutError)
        else "dependency_error"
        if isinstance(exc, SandboxDependencyError)
        else "infrastructure_error"
    )
    report["error_category"] = (
        category
        or report.get("error_category")
        or (
            "bootstrap_failure"
            if report.get("started") is False
            else "supervisor_failure"
        )
    )
    report["error"] = _exception_record(exc)


@dataclass
class _OutputCapture:
    """Keep a byte-exact bounded prefix; EOF and truncation are independent."""

    limit: int
    data: bytearray = field(default_factory=bytearray)
    observed_bytes: int = 0
    eof: bool = False
    read_error: dict | None = None

    def feed(self, block):
        self.observed_bytes += len(block)
        self.data.extend(block[: max(0, self.limit - len(self.data))])
        return self.observed_bytes > self.limit

    def drain(self, stream):
        # Called after tree termination; still bounded if cleanup was imperfect.
        deadline = time.monotonic() + 0.25
        try:
            os.set_blocking(stream.fileno(), False)
            while not self.eof and time.monotonic() < deadline:
                block = os.read(stream.fileno(), 65536)
                if not block:
                    self.eof = True
                    break
                self.feed(block)
        except BlockingIOError:
            pass  # no EOF observed: explicitly incomplete, never assumed empty
        except (OSError, ValueError) as exc:
            self.read_error = _exception_record(exc)

    def snapshot(self):
        raw = bytes(self.data)
        truncated = self.observed_bytes > len(raw)
        decode_error = None
        try:
            text = raw.decode("utf-8")
            decoding = "utf8_exact"
        except UnicodeDecodeError as exc:
            text = raw.decode("utf-8", errors="replace")
            decoding = "replacement_display_raw_bytes_retained"
            decode_error = {"start": exc.start, "end": exc.end, "reason": exc.reason}
        return {
            "text": text,
            "bytes_base64": base64.b64encode(raw).decode("ascii"),
            "retained_bytes": len(raw),
            "observed_bytes_lower_bound": self.observed_bytes,
            "limit_bytes": self.limit,
            "eof": self.eof,
            "truncated": truncated,
            "coverage": "truncated"
            if truncated
            else "complete"
            if self.eof and self.read_error is None
            else "incomplete",
            "decoding": decoding,
            "decode_error": decode_error,
            "read_error": self.read_error,
        }


def _command(runtime):
    command = [
        runtime.bwrap,
        "--unshare-all",
        "--as-pid-1",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "CAP_SYS_ADMIN",
        "--cap-add",
        "CAP_SETPCAP",
        "--clearenv",
    ]
    for source, destination, _ in runtime.files:
        command += ["--ro-bind", source, destination]
    command += ["--proc", "/proc", "--remount-ro", "/proc", "--dir", "/dev"]
    for name in ("null", "zero", "random", "urandom"):
        command += ["--dev-bind", "/dev/" + name, "/dev/" + name]
    command += [
        "--dir",
        "/work",
        "--symlink",
        "/work/tmp",
        "/tmp",
        "--symlink",
        "/work/shm",
        "/dev/shm",
        "--symlink",
        "/work",
        "/home",
        "--symlink",
        "python3.10",
        "/usr/bin/python",
        "--symlink",
        "python3.10",
        "/usr/bin/python3",
        "--remount-ro",
        "/",
    ]
    environment = {
        "PATH": "/usr/bin",
        "HOME": "/work",
        "TMPDIR": "/work/tmp",
        "LANG": "C.UTF-8",
        "PYTHONPATH": "/runtime/site-packages",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for key, value in environment.items():
        command += ["--setenv", key, value]
    return command + [runtime.interpreter, "-I", "-S", "-c", _BOOTSTRAP]


def _execute_python(code, runtime, *, limits=DEFAULT_LIMITS, cancel=None, admission_fd=None):
    """Execute precisely one generated python -c, with no retries/fallback.

    TimeoutError means an admitted program exceeded its wall deadline.
    Infrastructure/dependency failures never become Execution(success=False).
    """
    if admission_fd is not None:
        if type(admission_fd) is not int or admission_fd <= 2 or not stat.S_ISFIFO(os.fstat(admission_fd).st_mode):
            raise SandboxInfrastructureError("admission channel must be an inherited pipe")
        os.set_inheritable(admission_fd, False)
    if (
        not isinstance(code, str)
        or len(code.encode()) > limits.code_bytes
        or "\0" in code
    ):
        raise SandboxInfrastructureError("unsupported command payload")
    runtime.verify()
    # This is a dedicated trusted helper, not Inspect itself. Adopt/reap the
    # owned namespace descendants if the bubblewrap monitor exits abnormally.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise SandboxInfrastructureError("subreaper setup failed")
    cancel = cancel or threading.Event()
    report = {
        "schema": "ptb-python-sandbox-execution-v1",
        "backend_sha256": _sha(__file__),
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "runtime_sha256": runtime.identity,
        "limits": asdict(limits),
        "started": False,
        "new_task_permissions": 0,
        "policy_denials": [],
        "namespace_child_pid": None,
    }
    if runtime.materialization is not None:
        report["runtime_materialization"] = runtime.materialization
    started = time.monotonic()
    parent, child = socket.socketpair()
    read_status, write_status = os.pipe()
    process = None
    listener = None
    child_pidfd = None
    captures = {
        name: _OutputCapture(limits.output_bytes) for name in ("stdout", "stderr")
    }
    buffers = {
        **{name: capture.data for name, capture in captures.items()},
        "status": bytearray(),
    }
    selector = selectors.DefaultSelector()
    error = None
    try:
        command = _command(runtime)
        command[1:1] = ["--json-status-fd", str(write_status)]
        process = subprocess.Popen(
            command,
            stdin=child,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            pass_fds=(write_status,),
            start_new_session=True,
        )
        child.close()
        os.close(write_status)
        write_status = None
        payload = json.dumps({"code": code, "limits": asdict(limits)}).encode()
        parent.sendall(len(payload).to_bytes(4, "big") + payload)
        for stream, name in (
            (process.stdout, "stdout"),
            (process.stderr, "stderr"),
            (read_status, "status"),
            (parent, "control"),
        ):
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = started + limits.startup_seconds
        while True:
            if cancel.is_set():
                report["error_category"] = "caller_cancellation"
                raise SandboxCancelled("execution cancelled", report)
            if time.monotonic() > deadline:
                if report["started"]:
                    report["error_category"] = "wall_timeout"
                    exc = TimeoutError("Verification timed out.")
                    exc.report = report
                    raise exc
                report["error_category"] = "startup_timeout"
                raise SandboxInfrastructureError("sandbox bootstrap timed out", report)
            for key, _ in selector.select(
                min(0.05, max(0, deadline - time.monotonic()))
            ):
                name = key.data
                if name == "control":
                    data, ancillary, _, _ = parent.recvmsg(16384, socket.CMSG_SPACE(4))
                    selector.unregister(parent)
                    if not data:
                        continue
                    message = json.loads(data)
                    received = array.array("i")
                    for level, kind, value in ancillary:
                        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                            received.frombytes(
                                value[: len(value) - len(value) % received.itemsize]
                            )
                    if (
                        message.get("state") != "ready"
                        or message.get("pid") != 1
                        or len(received) != 1
                        or message.get("tmpfs_bytes") > limits.scratch_bytes
                    ):
                        report["error_category"] = "bootstrap_attestation"
                        for fd in received:
                            os.close(fd)
                        raise SandboxInfrastructureError(
                            "invalid bootstrap attestation", report
                        )
                    listener = received[0]
                    numbers = message["numbers"]
                    selector.register(listener, selectors.EVENT_READ, "notify")
                    parent.sendall(b"G")
                    parent.close()
                    report["started"] = True
                    report["startup_seconds"] = time.monotonic() - started
                    deadline = time.monotonic() + limits.wall_seconds
                    if admission_fd is not None:
                        # This optional trusted-parent pipe is never inherited
                        # by the untrusted command. Notify only after attestation
                        # and the actual permission to execute, not at bootstrap.
                        event = {"schema": "ptb-python-admission-v1", "started": True,
                                 "supervisor_pid": os.getpid(),
                                 "attested_namespace_pid": message["pid"],
                                 "admitted_monotonic": time.monotonic(),
                                 "code_sha256": report["code_sha256"]}
                        os.write(admission_fd, json.dumps(event).encode() + b"\n")
                        os.close(admission_fd)
                        admission_fd = None
                elif name == "notify":
                    if process.poll() is not None:
                        selector.unregister(listener)
                        continue
                    notification = _Notification()
                    try:
                        _ioctl(listener, 0, notification)
                    except OSError as exc:
                        if exc.errno in (errno.ENOENT, errno.EINTR):
                            continue
                        raise
                    syscall = numbers.get(str(notification.data.nr), "unknown")
                    response = _Response(notification.id, 0, -errno.EPERM, 0)
                    if syscall == "clone3":
                        response.error = (
                            -errno.ENOSYS
                        )  # safe glibc fallback; never dereference user pointers
                    elif syscall in ("clone", "fork", "vfork"):
                        # clone flags are scalar registers: no pointer TOCTOU.
                        namespaces = 0x7E020080
                        if (
                            syscall == "clone"
                            and notification.data.args[0] & namespaces
                        ):
                            report["policy_denials"].append("clone_namespace")
                        elif report["new_task_permissions"] >= limits.new_tasks:
                            report["policy_denials"].append("lifetime_task_budget")
                            response.error = -errno.EAGAIN
                        else:
                            report["new_task_permissions"] += 1
                            response.error, response.flags = 0, 1
                    else:
                        report["policy_denials"].append(syscall)
                    try:
                        _ioctl(listener, 1, response)
                    except OSError as exc:
                        if exc.errno not in (errno.ENOENT, errno.EINTR):
                            raise
                    if report["policy_denials"]:
                        report["error_category"] = (
                            "task_budget_exhausted"
                            if "lifetime_task_budget" in report["policy_denials"]
                            else "policy_denied"
                        )
                        raise SandboxDependencyError(
                            "program requires denied syscall or exhausted task budget",
                            report,
                        )
                else:
                    fd = key.fd
                    block = os.read(fd, 65536)
                    if not block:
                        if name in captures:
                            captures[name].eof = True
                        selector.unregister(key.fileobj)
                        continue
                    if name in captures:
                        over_limit = captures[name].feed(block)
                    else:
                        over_limit = len(buffers[name]) + len(block) > 16384
                        buffers[name].extend(
                            block[: max(0, 16384 - len(buffers[name]))]
                        )
                    if over_limit:
                        report["error_category"] = (
                            "output_limit"
                            if name in captures
                            else "supervisor_status_limit"
                        )
                        error_class = (
                            SandboxDependencyError
                            if name in captures
                            else SandboxInfrastructureError
                        )
                        raise error_class(f"{name} output bound exceeded", report)
                    if name == "status" and child_pidfd is None:
                        first = bytes(buffers[name]).split(b"\n", 1)[0]
                        if b"\n" in buffers[name]:
                            status_message = json.loads(first)
                            child_pid = status_message.get("child-pid")
                            if isinstance(child_pid, int) and child_pid > 1:
                                report["namespace_child_pid"] = child_pid
                                try:
                                    child_pidfd = os.pidfd_open(child_pid)
                                except ProcessLookupError:
                                    pass
            if process.poll() is not None and not any(
                key.data in {"stdout", "stderr"} for key in selector.get_map().values()
            ):
                break
        if not report["started"]:
            report["error_category"] = "bootstrap_failure"
            raise SandboxInfrastructureError(
                "bubblewrap/bootstrap failed: "
                + buffers["stderr"].decode(errors="replace"),
                report,
            )
        stdout, stderr = (buffers[key].decode("utf-8") for key in ("stdout", "stderr"))
        if process.returncode in {
            -signal.SIGKILL,
            128 + signal.SIGKILL,
            -signal.SIGXCPU,
            128 + signal.SIGXCPU,
            -signal.SIGXFSZ,
            128 + signal.SIGXFSZ,
            -signal.SIGSYS,
            128 + signal.SIGSYS,
        }:
            report["error_category"] = "resource_or_signal"
            raise SandboxDependencyError(
                "resource-bound or external signal termination", report
            )
        # The program controls stderr: keywords cannot establish an actual
        # dependency/isolation failure or override the original exit-code score.
        report["stderr_diagnostics"] = {
            "source": "untrusted_program_stderr",
            "matches": sorted(
                set(
                    re.findall(
                        r"(?:ModuleNotFoundError|ImportError|MemoryError|OSError: \[Errno (?:12|24|27|28)\])",
                        stderr,
                    )
                )
            ),
            "affects_outcome": False,
        }
        report["outcome"] = "success" if process.returncode == 0 else "program_failure"
        report["error_category"] = None if process.returncode == 0 else "program_exit"
        report["error"] = None
        report["program_returncode"] = process.returncode
        return Execution(
            process.returncode == 0, process.returncode, stdout, stderr, report
        )
    except BaseException as exc:
        error = exc
        _record_failure(
            report,
            exc,
            "output_decode_error" if isinstance(exc, UnicodeDecodeError) else None,
        )
        if isinstance(exc, (SandboxInfrastructureError, TimeoutError)):
            raise
        wrapped = SandboxInfrastructureError(
            f"sandbox supervisor failure: {type(exc).__name__}: {exc}", report
        )
        report["raised_exception"] = _exception_record(wrapped)
        raise wrapped from exc
    finally:
        # Killing namespace PID 1 kills every descendant, including setsid/fork.
        # pidfds avoid PID-reuse mistakes; die-with-parent also covers supervisor death.
        cleanup_errors = []
        report["cleanup_errors"] = cleanup_errors

        def safely(label, operation):
            try:
                operation()
            except ProcessLookupError:
                pass
            except BaseException as exc:  # noqa: BLE001 -- retain secondary failures, continue cleanup/output drain
                cleanup_errors.append({"operation": label, **_exception_record(exc)})

        report["observed_namespace_children"] = []
        main_pid = report.get("namespace_child_pid")
        if main_pid is not None and child_pidfd is not None:
            try:
                for task in (Path("/proc") / str(main_pid) / "task").iterdir():
                    report["observed_namespace_children"].extend(
                        int(pid) for pid in (task / "children").read_text().split()
                    )
            except (FileNotFoundError, ProcessLookupError):
                pass
            except OSError as exc:
                report["child_observation_error"] = _exception_record(exc)
        if child_pidfd is not None:
            safely(
                "kill namespace pidfd",
                lambda: signal.pidfd_send_signal(child_pidfd, signal.SIGKILL),
            )
        if process is not None:
            report["monitor_reaped"] = False
            report["descendants_reaped"] = False

            def stop_monitor():
                if process.poll() is None and child_pidfd is None:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                report["monitor_reaped"] = True

            safely("stop/reap monitor", stop_monitor)
            report["monitor_returncode"] = process.returncode

            def reap_descendants():
                reap_deadline = time.monotonic() + 2
                while time.monotonic() < reap_deadline:
                    try:
                        reaped, _ = os.waitpid(-1, os.WNOHANG)
                    except ChildProcessError:
                        report["descendants_reaped"] = True
                        break
                    if reaped == 0:
                        time.sleep(0.01)

            safely("reap adopted descendants", reap_descendants)
            for name in ("stdout", "stderr"):
                stream = getattr(process, name)
                captures[name].drain(stream)
                safely(f"close {name}", stream.close)
        report["output"] = {
            name: capture.snapshot() for name, capture in captures.items()
        }
        safely("close selector", selector.close)
        for fd in (listener, child_pidfd, read_status, write_status):
            if fd is not None:
                safely(f"close owned fd {fd}", lambda fd=fd: os.close(fd))
        safely("close parent control", parent.close)
        safely("close child control", child.close)
        report["elapsed_seconds"] = time.monotonic() - started
        if cleanup_errors or (
            process is not None
            and not (report.get("monitor_reaped") and report.get("descendants_reaped"))
        ):
            report["primary_outcome"] = report.get("outcome")
            report["primary_error_category"] = report.get("error_category")
            report["outcome"] = "infrastructure_error"
            report["error_category"] = "cleanup_failure"
            report["cleanup_unconfirmed"] = True
            cleanup_error = SandboxInfrastructureError(
                "namespace descendant cleanup unconfirmed", report
            )
            report["raised_exception"] = _exception_record(cleanup_error)
            raise cleanup_error from error


def execute_python(code, runtime, *, limits=DEFAULT_LIMITS, cancel=None, on_admission=None):
    """Run the trusted supervisor separately so notification waits are cancellable.

    Generated code only runs inside bubblewrap. The outer helper receives a
    clean environment and owns its own signal timer, independent of Inspect.
    """
    cancel = cancel or threading.Event()
    if on_admission is not None and not callable(on_admission):
        raise TypeError("on_admission must be a trusted caller callback")
    request = {"code": code, "runtime": asdict(runtime), "limits": asdict(limits)}
    admission_read = admission_write = None
    admission_buffer = bytearray()
    admission_reported = False
    if on_admission is not None:
        admission_read, admission_write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
        request["admission_fd"] = admission_write
    payload = json.dumps(request).encode()
    try:
        helper = subprocess.Popen(
            [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--supervise"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"}, cwd="/", start_new_session=True,
            pass_fds=() if admission_write is None else (admission_write,),
        )
    except BaseException:
        if admission_read is not None:
            os.close(admission_read)
        raise
    finally:
        if admission_write is not None:
            os.close(admission_write)

    def read_admission():
        nonlocal admission_reported
        if admission_read is None or admission_reported:
            return
        try:
            admission_buffer.extend(os.read(admission_read, 4096))
        except BlockingIOError:
            return
        if len(admission_buffer) > 4096:
            raise SandboxInfrastructureError("oversized supervisor admission event")
        if b"\n" not in admission_buffer:
            return
        event = json.loads(admission_buffer.split(b"\n", 1)[0])
        if (event.get("schema") != "ptb-python-admission-v1" or event.get("started") is not True
                or event.get("supervisor_pid") != helper.pid
                or event.get("code_sha256") != hashlib.sha256(code.encode()).hexdigest()
                or event.get("attested_namespace_pid") != 1):
            raise SandboxInfrastructureError("supervisor admission identity differs")
        admission_reported = True
        on_admission(event)
    deadline = time.monotonic() + limits.startup_seconds + limits.wall_seconds + 15
    pending_error = None
    report_received = False
    try:
        first = True
        while True:
            read_admission()
            if cancel.is_set():
                raise SandboxCancelled("execution cancelled")
            if time.monotonic() > deadline:
                raise SandboxInfrastructureError(
                    "trusted supervisor exceeded cleanup deadline"
                )
            try:
                stdout, stderr = helper.communicate(
                    input=payload if first else None, timeout=0.05
                )
                read_admission()
                break
            except subprocess.TimeoutExpired:
                first = False
        if helper.returncode != 0:
            raise SandboxInfrastructureError(
                "trusted supervisor failed: " + stderr.decode(errors="replace")
            )
        message = json.loads(stdout)
        report_received = True
        if message["kind"] == "execution":
            return Execution(**message["value"])
        report = message.get("report", {})
        if message["kind"] == "timeout":
            exc = TimeoutError(message["message"])
            exc.report = report
            raise exc
        cls = (
            SandboxCancelled
            if message["kind"] == "cancelled"
            else SandboxDependencyError
            if message["kind"] == "dependency"
            else SandboxInfrastructureError
        )
        raise cls(message["message"], report)
    except BaseException as exc:
        pending_error = exc
        raise
    finally:
        if helper.poll() is None or (pending_error is not None and not report_received):
            recovered_report = None
            helper_cleanup_error = None
            try:
                if helper.poll() is None:
                    helper.send_signal(signal.SIGINT)
                cleanup_stdout, _ = helper.communicate(timeout=10)
                recovered_report = json.loads(cleanup_stdout).get("report")
            except (
                subprocess.TimeoutExpired,
                OSError,
                ValueError,
                UnicodeError,
            ) as exc:
                helper_cleanup_error = _exception_record(exc)
                if helper.poll() is None:
                    try:
                        helper.kill()
                        helper.communicate(timeout=3)
                    except (subprocess.TimeoutExpired, OSError) as secondary:
                        helper_cleanup_error["kill_error"] = _exception_record(
                            secondary
                        )
            if pending_error is not None:
                if recovered_report:
                    recovered_report["supervisor_termination"] = {
                        key: recovered_report.get(key)
                        for key in (
                            "outcome",
                            "error_category",
                            "error",
                            "raised_exception",
                        )
                    }
                    pending_error.report = recovered_report
                else:
                    pending_error.report = getattr(
                        pending_error,
                        "report",
                        SandboxInfrastructureError(
                            "supervisor report unavailable"
                        ).report,
                    )
                    pending_error.report["cleanup_unconfirmed"] = True
                if helper_cleanup_error:
                    pending_error.report["helper_cleanup_error"] = helper_cleanup_error
                _record_failure(
                    pending_error.report,
                    pending_error,
                    "caller_cancellation"
                    if isinstance(pending_error, (SandboxCancelled, KeyboardInterrupt))
                    else "supervisor_failure",
                )
        for stream in (helper.stdin, helper.stdout, helper.stderr):
            stream.close()
        if admission_read is not None:
            os.close(admission_read)


def register_inspect(runtime, *, limits=DEFAULT_LIMITS, name="ptb_python", on_admission=None):
    """Register once under a fresh name; all duplicates deliberately fail."""
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise SandboxInfrastructureError(
            "backend name must be an unqualified lowercase identifier"
        )
    if name in _RESERVED_BACKENDS:
        raise SandboxInfrastructureError(f"reserved backend name: {name}")
    import anyio
    import inspect_ai.util._sandbox.environment as environment_module
    import inspect_ai.util._sandbox.registry as registry_module
    import inspect_ai.util._subprocess as subprocess_module
    from inspect_ai.util._sandbox.environment import SandboxEnvironment
    from inspect_ai.util._sandbox.registry import sandboxenv
    from inspect_ai.util._subprocess import ExecResult

    runtime.verify()
    pinned = {
        environment_module: "c386719cb676030695447c5e4f589319be52bdc80451ab8ec80b90b8ef9367ef",
        registry_module: "990c16c2acda097175159d9f715ba3a69dd6300df91969e1d70e374a82090153",
        subprocess_module: "21d236056a82984788d2a92f6a6b798350e982d665521136e287e4ee7d9363bc",
    }
    if any(_sha(module.__file__) != digest for module, digest in pinned.items()):
        raise SandboxInfrastructureError(
            "Inspect sandbox API source differs from the tested pin"
        )

    class PTBPythonSandboxEnvironment(SandboxEnvironment):
        def __init__(self):
            self._closed = False
            self._running = {}
            self.execution_evidence = []

        @classmethod
        def default_concurrency(cls):
            return 1

        @classmethod
        async def sample_init(cls, task_name, config, metadata):
            if config is not None:
                raise SandboxInfrastructureError(
                    "runtime is explicitly registered; per-sample config unsupported"
                )
            return {"default": cls()}

        @classmethod
        async def sample_cleanup(cls, task_name, config, environments, interrupted):
            for environment in environments.values():
                environment._closed = True
                pending = list(environment._running.values())
                for cancel, _ in pending:
                    cancel.set()
                with anyio.CancelScope(shield=True):
                    while any(not done.is_set() for _, done in pending):
                        await anyio.sleep(0.025)

        async def exec(
            self,
            cmd,
            input=None,
            cwd=None,
            env=None,
            user=None,
            timeout=None,
            timeout_retry=True,
            concurrency=True,
        ):
            if (
                self._closed
                or not isinstance(cmd, list)
                or len(cmd) != 3
                or cmd[:2] != ["python", "-c"]
                or input is not None
                or cwd is not None
                or env
                or user is not None
            ):
                raise SandboxInfrastructureError(
                    "this backend supports the HumanEval python -c invocation only"
                )
            selected = Limits(
                **{
                    **asdict(limits),
                    "wall_seconds": limits.wall_seconds if timeout is None else timeout,
                }
            )
            cancel, done = threading.Event(), threading.Event()
            outcome = {}

            def run():
                try:
                    outcome["result"] = execute_python(
                        cmd[2], runtime, limits=selected, cancel=cancel,
                        **({"on_admission": on_admission} if on_admission is not None else {})
                    )
                except BaseException as exc:  # noqa: BLE001 -- preserve worker exception in awaiting task
                    outcome["error"] = exc
                finally:
                    done.set()

            # Own the worker from before the first cancellation checkpoint.
            # A queued/abandoned AnyIO thread-pool job might never set `done`.
            worker = threading.Thread(target=run, name="ptb-python-supervision")
            worker.start()
            self._running[id(done)] = (cancel, done)
            try:
                while not done.is_set():
                    await anyio.sleep(0.025)
            finally:
                cancel.set()
                with anyio.CancelScope(shield=True):
                    while not done.is_set():
                        await anyio.sleep(0.025)
                worker.join()
                self._running.pop(id(done), None)
                observed = outcome.get("result")
                self.execution_evidence.append(
                    observed.evidence
                    if observed is not None
                    else getattr(outcome.get("error"), "report", {})
                )
            if "error" in outcome:
                raise outcome["error"]
            result = outcome["result"]
            return ExecResult(
                result.success, result.returncode, result.stdout, result.stderr
            )

        async def write_file(self, file, contents):
            raise SandboxInfrastructureError(
                "persistent sample filesystem is unsupported"
            )

        async def read_file(self, file, text=True):
            raise SandboxInfrastructureError(
                "persistent sample filesystem is unsupported"
            )

    # Serialize this helper's registrations. Setup must not race unrelated
    # third-party direct mutations of Inspect's process-global registry.
    with _REGISTRATION_LOCK:
        if registry_module.registry_has_sandboxenv(name):
            raise SandboxInfrastructureError(
                f"backend already registered: {name}; reuse its class"
            )
        return sandboxenv(name=name)(PTBPythonSandboxEnvironment)


if __name__ == "__main__":
    if sys.argv[1:] != ["--supervise"]:
        raise SystemExit("internal supervisor only; use the Python API")
    request = json.load(sys.stdin)
    try:
        outcome = _execute_python(
            request["code"],
            Runtime(**request["runtime"]),
            limits=Limits(**request["limits"]),
            **({"admission_fd": request["admission_fd"]} if "admission_fd" in request else {}),
        )
        message = {"kind": "execution", "value": asdict(outcome)}
    except BaseException as exc:  # noqa: BLE001 -- serialize failure after owned-tree cleanup, including SIGINT
        report = getattr(exc, "report", None)
        if report is None:
            report = SandboxInfrastructureError(str(exc)).report
            _record_failure(report, exc, "supervisor_preparation")
        message = {
            "kind": "timeout"
            if isinstance(exc, TimeoutError)
            else "cancelled"
            if isinstance(exc, SandboxCancelled)
            else "dependency"
            if isinstance(exc, SandboxDependencyError)
            else "infrastructure",
            "message": str(exc),
            "report": report,
        }
    print(json.dumps(message))
