#!/bin/bash
# Refuse to start unless PYTHONPYCACHEPREFIX points at a real, bound, writable
# filesystem -- i.e. anywhere except the 64 MiB container root.
#
# The failure this exists to prevent. Every apptainer exec in this repo runs
# --writable-tmpfs, so the container root is a fuse-overlayfs capped by
# `sessiondir max size` in apptainer.conf -- 64 on this cluster, measured as
# exactly 65536 1-KiB blocks by `df -P -k /` inside vllm_debug.sif. Anything a
# run writes outside a bind lands there, and the biggest single writer is
# CPython bytecode, because none of it is prebuilt: containers/*.def install
# with `uv pip install --system --no-cache`, uv does not byte-compile, and
# /usr/local/lib/python3.10/dist-packages holds 24602 .py against 886 .pyc in
# both opus_5.sif and vllm_debug.sif. Measured on 2026-09-02 with the prefix
# bound out to a host directory: `import transformers, torch` writes 25 MiB in
# 1294 .pyc, and adding `import vllm` takes it to 41 MiB in 2121 .pyc. One
# python process therefore claims two thirds of the whole 64 MiB before the
# scorer has read a single row.
#
# What that looks like when it goes wrong is never "disk full". ENOSPC part-way
# through writing a .pyc leaves a TRUNCATED one, and every later import of that
# module dies on it, so the visible symptom is vLLM failing to start on a node
# with terabytes free -- the same family as the
#
#     torch._inductor.exc.InductorError: OSError: [Errno 28] No space left
#
# that cost job 81521 an hour of opus and nine dead evaluation attempts. Eight
# of the twelve gsm8k cells in jobs 89727/89809/89810 lost time to this class,
# ~1.5-2 h across the arm.
#
# Why an assertion and not just the variable. Setting PYTHONPYCACHEPREFIX to a
# path that is not bound in moves the bytecode from one place on the overlay to
# another place on the overlay: the run still dies, the variable still looks
# set, and nothing warns. Two shapes of that were measured on the real images,
# and only one of them is caught by the obvious check:
#
#   * no bind at all, exec without -c: the prefix directory is created on the
#     container root, `stat -c %d` equals `stat -c %d /`, `df` says
#     fuse-overlayfs / 65536 blocks. Caught by the st_dev rule below.
#   * no bind, exec WITH -c (the agent's shape): apptainer supplies its own
#     session tmpfs at /tmp, so the prefix has a DIFFERENT st_dev from / and
#     the st_dev rule passes -- but that tmpfs is drawn from the same 64 MiB
#     session budget and `df` reports the same 65536 blocks. Only the capacity
#     rule catches it. An assertion with just the st_dev half would have been
#     green on exactly the exec that runs for ten hours.
#
# So both rules are here, and both are read off the filesystem inside the
# container rather than off the variable. A future bind-list edit that breaks
# the pair then fails in the first seconds of the exec, with the path and the
# numbers, instead of three hours in as a truncated import.
#
# Not fixed by enlarging the overlay: `sessiondir max size` lives in
# apptainer.conf, which is root-owned and not even present on the login node,
# so raising it is not an action this repo can take on the nodes it runs on.
# Writing into binds is, and that is what this checks.
#
# Runs inside the container, at the head of the exec that does the real work,
# so it sees the exact bind list that exec has. Usage: `bash check_pycache.sh`,
# no arguments; it reads PYTHONPYCACHEPREFIX out of the environment.
set -u

#: The apptainer session-dir cap, in 1-KiB blocks. Not a guess: apptainer.conf
#: line 180 on this cluster reads `sessiondir max size = 64`, and both the root
#: fuse-overlayfs and the -c session /tmp report exactly this from `df -P -k`.
#: Used only in the diagnostic text -- the rule below is deliberately not "is it
#: exactly the cap", because a site that raises the knob to 256 would then pass
#: while still being far too small.
PYCACHE_SESSION_CAP_KIB=65536

#: Floor on the total capacity of the filesystem holding the prefix, in 1-KiB
#: blocks (4 GiB). Chosen to sit in the wide gap between the two measured
#: populations rather than near either edge: 64x the 41 MiB one `import vllm`
#: actually writes (so a real scratch dir this size is genuinely enough, with
#: room for the agent's own trl/peft/datasets imports on top), 64x the 64 MiB
#: session cap it exists to reject, and 48x below the smallest real filesystem
#: any of these execs can land on -- the 193 GiB boot disk that backs host /tmp
#: here, with /mnt/localssd at 5.9 TiB and /rmeng_data at 100 TiB above it.
#: Nothing in that gap is a plausible deployment, so the rule cannot be tripped
#: by a legitimate site and cannot be satisfied by a session tmpfs.
PYCACHE_MIN_TOTAL_KIB=4194304

d="${PYTHONPYCACHEPREFIX:-}"
if [ -z "$d" ]; then
    echo "check_pycache: PYTHONPYCACHEPREFIX is unset inside the container." >&2
    echo "  Every .pyc this run writes will land on the ${PYCACHE_SESSION_CAP_KIB} KiB container root." >&2
    exit 71
fi

# mkdir here as well as on the host: the host side creates the bind source, but
# a prefix nested one level deeper inside a bind (or a bind added later) still
# has to be creatable from in here, and CPython will not create it for us if the
# parent is read-only.
mkdir -p "$d" 2>/dev/null
_probe="${d}/.check_pycache.$$"
if ! : > "$_probe" 2>/dev/null; then
    echo "check_pycache: PYTHONPYCACHEPREFIX=${d} is not writable inside the container." >&2
    echo "  Bind a writable node-local directory at that exact path." >&2
    exit 72
fi
rm -f "$_probe"

_root_dev="$(stat -c %d / 2>/dev/null)"
_pfx_dev="$(stat -c %d "$d" 2>/dev/null)"
if [ -n "$_root_dev" ] && [ "$_pfx_dev" = "$_root_dev" ]; then
    echo "check_pycache: PYTHONPYCACHEPREFIX=${d} is ON THE CONTAINER ROOT (st_dev ${_pfx_dev} == st_dev of /)." >&2
    echo "  That is the --writable-tmpfs overlay, ${PYCACHE_SESSION_CAP_KIB} KiB total; a single import of" >&2
    echo "  torch+transformers+vllm writes ~41 MiB of bytecode into it and the run dies on a" >&2
    echo "  truncated .pyc. The --env and the --bind for this path have drifted apart." >&2
    exit 73
fi

# `df -P` is one line per filesystem by contract, so the fields are positional:
# 1 device, 2 total 1-KiB blocks, 3 used, 4 available, 5 capacity, 6 mountpoint.
_dfline="$(df -P -k "$d" 2>/dev/null | tail -1)"
# shellcheck disable=SC2086
set -- $_dfline
_fs="${1:-unknown}"
_total_kib="${2:-0}"
case "$_total_kib" in ''|*[!0-9]*) _total_kib=0 ;; esac
if [ "$_total_kib" -lt "$PYCACHE_MIN_TOTAL_KIB" ]; then
    echo "check_pycache: PYTHONPYCACHEPREFIX=${d} is on a ${_total_kib} KiB filesystem (${_fs})." >&2
    echo "  Floor is ${PYCACHE_MIN_TOTAL_KIB} KiB. A filesystem this small is an apptainer session" >&2
    echo "  dir, not a bind -- note that a session tmpfs has its own st_dev, so it passes the" >&2
    echo "  overlay check above and would otherwise look fine." >&2
    exit 74
fi

# The directory being right is not the same as python using it. -B, -E and
# -X pycache_prefix= all beat the environment variable, and sys.pycache_prefix
# only exists from 3.8; this asks the interpreter that will do the work.
if command -v python3 >/dev/null 2>&1; then
    if ! python3 -c 'import os, sys
want = os.environ.get("PYTHONPYCACHEPREFIX")
got = getattr(sys, "pycache_prefix", None)
sys.exit(0 if got == want else 1)' 2>/dev/null; then
        echo "check_pycache: python3 in this container does not honour PYTHONPYCACHEPREFIX=${d}" >&2
        echo "  (sys.pycache_prefix disagrees). A -B/-E/-X pycache_prefix wrapper, or python < 3.8." >&2
        exit 75
    fi
fi

echo "check_pycache: ok — PYTHONPYCACHEPREFIX=${d} on ${_fs}, $((_total_kib / 1048576)) GiB total, st_dev ${_pfx_dev} (/ is ${_root_dev})"
