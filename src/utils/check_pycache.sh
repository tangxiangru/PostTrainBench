#!/bin/bash
# Keep run-generated CPython bytecode off the 64 MiB container root -- by
# REPAIRING the environment where it can, and by warning loudly where it
# cannot. It kills the exec only when the container has no writable scratch at
# all, which is a container that cannot produce a scoreable model anyway.
#
# HOW TO USE IT.  Source it, do not run it:
#
#     . /path/to/check_pycache.sh || exit 1
#
# The repair is `export PYTHONPYCACHEPREFIX=<somewhere that works>` or
# `export PYTHONDONTWRITEBYTECODE=1`, and an export from a child process
# reaches nothing. Run as `bash check_pycache.sh` it still prints the same
# verdict and still returns the same status -- useful for a human diagnosing a
# node -- but the python that runs afterwards is not fixed. Every call site in
# this repo sources it; tests/test_container_pycache.sh [7] asserts that.
#
# ---------------------------------------------------------------------------
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
# container rather than off the variable.
#
# Not fixed by enlarging the overlay: `sessiondir max size` lives in
# apptainer.conf, which is root-owned and not even present on the login node,
# so raising it is not an action this repo can take on the nodes it runs on.
# Writing into binds is, and that is what this checks.
#
# ---------------------------------------------------------------------------
# WHY ALMOST NOTHING HERE IS FATAL ANY MORE.
#
# The first version of this file exited 71..75 and every call site ran it as
# `bash check_pycache.sh || exit 1`. That turns a bad mount into a dead cell in
# the first seconds -- and a ten-hour 8xH100 gsm8k cell is ~80 GPU-hours, so the
# capacity rule tripping on one node whose scratch fell back to something small
# costs more than every hour this file has ever saved. The rule was written to
# stop a SLOWDOWN (1.5-2 h across an arm). Paying for it with the whole arm is
# the more expensive mistake, and it is the one nobody would see coming, because
# the trigger is a filesystem the harness does not choose.
#
# The honest disposition falls straight out of asking, for each rule, "can this
# cell still produce a scoreable model?":
#
#   unset / container-root / capacity / full / not-honoured
#       Yes. All five say the same thing -- bytecode is about to be written
#       somewhere too small -- and all five have a repair that costs time and
#       nothing else. Ladder: relocate the prefix to a directory that passes the
#       same predicate, and if there is no such directory, set
#       PYTHONDONTWRITEBYTECODE=1 so the interpreter writes no .pyc at all.
#       Slower imports, identical results, no truncated .pyc possible. All five
#       are therefore a loud WARNING and exit 0.
#
#       Note which way the safety runs: an unwritable or absent cache is SAFE
#       (CPython swallows the write error and carries on interpreting); a
#       writable cache that is too small is the dangerous one, because that is
#       where a half-written .pyc comes from. The rule that reads like "merely a
#       small disk" is precisely the dangerous case -- which is why it degrades
#       to no-bytecode rather than being dropped.
#
#   no writable scratch anywhere
#       No. If neither the configured prefix nor /var/tmp nor $HOME nor /tmp can
#       take a single zero-byte file, then tempfile, torch's inductor cache,
#       HF's downloads and vLLM's compile cache have nowhere to go either. That
#       is a broken mount table, not a small disk, and no filesystem SIZE can
#       produce it. This one stays fatal (exit 70): dying in second one with the
#       paths printed beats dying in hour three inside somebody's traceback.
#
# So: no configuration of the filesystem -- any size, any fill level, bound or
# unbound, session tmpfs or overlay -- can now stop a cell that could otherwise
# have produced a scoreable model.
# ---------------------------------------------------------------------------

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
#: Overridable now that tripping it only degrades: a site with genuinely small
#: node-local scratch should lower it rather than run without a bytecode cache.
PYCACHE_MIN_TOTAL_KIB="${PYCACHE_MIN_TOTAL_KIB:-4194304}"

#: Floor on FREE space, 256 MiB -- 6x the measured 41 MiB working set. A 100 %
#: full 5.9 TiB localssd passes the capacity rule and is the exact ENOSPC that
#: writes the truncated .pyc, so total capacity alone is not the question.
PYCACHE_MIN_AVAIL_KIB="${PYCACHE_MIN_AVAIL_KIB:-262144}"

#: Colon-separated extra candidates, tried before the built-in ones. Exists so
#: a caller that knows where its node-local scratch is can say so, and so the
#: tests can drive the relocation ladder hermetically.
PYCACHE_FALLBACK_DIRS="${PYCACHE_FALLBACK_DIRS:-}"

# Sourced or executed? `return` is only legal in the sourced case, and only the
# sourced case can repair anything. Asking bash itself, rather than comparing
# BASH_SOURCE[0] with $0: the comparison is wrong for `bash -c '. "$0"' <path>`,
# where the two are equal and the file IS sourced -- which is one keystroke away
# from the scorer's own call site.
_cpc_sourced=0
if (return 0 2>/dev/null); then _cpc_sourced=1; fi

# _cpc_probe <dir>
#   0  -> the directory is usable as a bytecode cache
#   1  -> not usable; _cpc_why names the rule
# Side effects (read by the caller for the diagnostics): _cpc_fs, _cpc_total,
# _cpc_avail, _cpc_dev, _cpc_rootdev, and _cpc_writable=1 if a file could be
# created in it -- which is what separates "small" from "broken mount table".
_cpc_probe() {
    local d="${1:-}" probe
    _cpc_why=""; _cpc_fs="unknown"; _cpc_total=0; _cpc_avail=0; _cpc_dev=""
    _cpc_writable=0
    _cpc_rootdev="$(stat -c %d / 2>/dev/null)"

    [ -n "$d" ] || { _cpc_why="unset"; return 1; }
    # A relative prefix is resolved against each process's cwd, so it scatters
    # bytecode through the run's working directories instead of collecting it.
    case "$d" in /*) ;; *) _cpc_why="relative"; return 1 ;; esac

    # mkdir here as well as on the host: the host side creates the bind source,
    # but a prefix nested one level deeper inside a bind (or a bind added later)
    # still has to be creatable from in here, and CPython will not create it for
    # us if the parent is read-only.
    mkdir -p "$d" 2>/dev/null
    probe="${d}/.check_pycache.$$"
    # The braces matter: `: > "$probe" 2>/dev/null` applies the redirections
    # left to right, so the failing one reports "No such file or directory" on
    # the still-unredirected stderr and every rejected candidate prints a line
    # of noise that looks like an error the run should care about.
    if ! { : > "$probe"; } 2>/dev/null; then _cpc_why="unwritable"; return 1; fi
    rm -f "$probe"
    _cpc_writable=1

    _cpc_dev="$(stat -c %d "$d" 2>/dev/null)"
    # `df -P` is one line per filesystem by contract, so the fields are
    # positional: 1 device, 2 total 1-KiB blocks, 3 used, 4 available,
    # 5 capacity, 6 mountpoint.
    local _dfline
    _dfline="$(df -P -k "$d" 2>/dev/null | tail -1)"
    # shellcheck disable=SC2086
    set -- $_dfline
    _cpc_fs="${1:-unknown}"
    _cpc_total="${2:-0}"
    _cpc_avail="${4:-0}"
    case "$_cpc_total" in ''|*[!0-9]*) _cpc_total=0 ;; esac
    case "$_cpc_avail" in ''|*[!0-9]*) _cpc_avail=0 ;; esac

    if [ -n "${_cpc_rootdev:-}" ] && [ "$_cpc_dev" = "$_cpc_rootdev" ]; then
        _cpc_why="container-root"; return 1
    fi
    if [ "$_cpc_total" -lt "$PYCACHE_MIN_TOTAL_KIB" ]; then _cpc_why="capacity"; return 1; fi
    if [ "$_cpc_avail" -lt "$PYCACHE_MIN_AVAIL_KIB" ]; then _cpc_why="full"; return 1; fi
    return 0
}

# The relocation ladder, most-node-local first.
#   /var/tmp   under -c this is a session tmpfs and the capacity rule rejects it;
#              without -c it is the host's, which is what the scorer wants.
#   $HOME      the agent exec is --home "${JOB_DIR}:/home/ben", i.e. node-local
#              scratch, and nothing copies $HOME/.ptb_pycache into the results
#              (run_task.sh copies ${JOB_DIR}/task, not ${JOB_DIR}).
#   /tmp       last, because in the shape this file exists to catch /tmp IS the
#              small thing; the same predicate filters it when it is.
# Per-uid so a shared /var/tmp does not hand one user another user's directory.
_cpc_candidates() {
    local c oldifs="$IFS"
    IFS=':'
    for c in ${PYCACHE_FALLBACK_DIRS:-}; do
        if [ -n "$c" ]; then printf '%s\n' "$c"; fi
    done
    IFS="$oldifs"
    printf '%s\n' "/var/tmp/.ptb_pycache.${UID:-0}"
    if [ -n "${HOME:-}" ]; then printf '%s\n' "${HOME}/.ptb_pycache"; fi
    printf '%s\n' "/tmp/.ptb_pycache.${UID:-0}"
}

_cpc_main() {
    local configured="${PYTHONPYCACHEPREFIX:-}"
    local chosen="" rule="" any_writable=0 cand
    local cfg_fs cfg_total cfg_avail cfg_dev cfg_root

    if _cpc_probe "$configured"; then
        chosen="$configured"
    else
        rule="$_cpc_why"
        if [ "$_cpc_writable" = 1 ]; then any_writable=1; fi
        cfg_fs="$_cpc_fs"; cfg_total="$_cpc_total"; cfg_avail="$_cpc_avail"
        cfg_dev="$_cpc_dev"; cfg_root="$_cpc_rootdev"
        while IFS= read -r cand; do
            [ -n "$cand" ] || continue
            if [ "$cand" = "$configured" ]; then continue; fi
            if _cpc_probe "$cand"; then chosen="$cand"; break; fi
            if [ "$_cpc_writable" = 1 ]; then any_writable=1; fi
        done < <(_cpc_candidates)
    fi

    # ---- the good case -----------------------------------------------------
    if [ -n "$chosen" ] && [ "$chosen" = "$configured" ]; then
        echo "check_pycache: verdict=ok prefix=${chosen} fs=${_cpc_fs}" \
             "total_kib=${_cpc_total} avail_kib=${_cpc_avail}" \
             "st_dev=${_cpc_dev} (/ is ${_cpc_rootdev})"
        _cpc_verify "$chosen"
        return 0
    fi

    # ---- nothing writable anywhere: the only fatal ------------------------
    if [ -z "$chosen" ] && [ "$any_writable" = 0 ]; then
        {
            echo "check_pycache: verdict=fatal rule=no-writable-scratch prefix=${configured:-<unset>}"
            echo "  Neither PYTHONPYCACHEPREFIX nor any of"
            _cpc_candidates | sed 's/^/    /'
            echo "  can take a single zero-byte file. This is not a small disk, it is a broken"
            echo "  mount table: tempfile, torch's inductor cache, HF downloads and vLLM's"
            echo "  compile cache all need one of these, so nothing this exec is about to run"
            echo "  can finish. Refusing now, with the paths above, rather than in hour three"
            echo "  inside somebody else's traceback."
        } >&2
        echo "check_pycache: verdict=fatal rule=no-writable-scratch"
        return 70
    fi

    # ---- degraded: relocate, or stop writing bytecode ----------------------
    local action
    if [ -n "$chosen" ]; then
        action="relocate"
        export PYTHONPYCACHEPREFIX="$chosen"
    else
        action="no-bytecode"
        export PYTHONDONTWRITEBYTECODE=1
    fi

    {
        echo "check_pycache: WARNING -- the configured bytecode cache is not usable, and this"
        echo "  run has repaired itself rather than dying. It is SLOWER than a healthy cell"
        echo "  (every import recompiles) and its RESULTS ARE UNAFFECTED. Fix the mount and"
        echo "  the warning goes away."
        case "$rule" in
            unset)          echo "  rule=unset: PYTHONPYCACHEPREFIX is not set inside the container, so every" ;;
            relative)       echo "  rule=relative: PYTHONPYCACHEPREFIX=${configured} is not absolute, so every" ;;
            unwritable)     echo "  rule=unwritable: PYTHONPYCACHEPREFIX=${configured} cannot be written to, so no" ;;
            container-root) echo "  rule=container-root: PYTHONPYCACHEPREFIX=${configured} is ON THE CONTAINER ROOT" ;;
            capacity)       echo "  rule=capacity: PYTHONPYCACHEPREFIX=${configured} is on a ${cfg_total:-0} KiB filesystem" ;;
            full)           echo "  rule=full: PYTHONPYCACHEPREFIX=${configured} has only ${cfg_avail:-0} KiB free" ;;
        esac
        case "$rule" in
            unset|relative)
                echo "    .pyc would land next to its .py inside the image, i.e. on the"
                echo "    ${PYCACHE_SESSION_CAP_KIB} KiB --writable-tmpfs overlay." ;;
            unwritable)
                echo "    bytecode would be cached at all (CPython swallows the error). Safe, but"
                echo "    it means the bind for this path is missing." ;;
            container-root)
                echo "    (st_dev ${cfg_dev:-?} == st_dev of / ${cfg_root:-?}) -- the --writable-tmpfs"
                echo "    overlay, ${PYCACHE_SESSION_CAP_KIB} KiB total. The --env and the --bind for this path"
                echo "    have drifted apart." ;;
            capacity)
                echo "    (${cfg_fs:-?}); the floor is ${PYCACHE_MIN_TOTAL_KIB} KiB. A filesystem this small is an"
                echo "    apptainer session dir, not a bind -- and a session tmpfs has its own"
                echo "    st_dev, so it passes the overlay check and would otherwise look fine." ;;
            full)
                echo "    (${cfg_fs:-?}, ${cfg_total:-0} KiB total); the floor is ${PYCACHE_MIN_AVAIL_KIB} KiB. A full"
                echo "    filesystem is exactly where a half-written .pyc comes from." ;;
        esac
        if [ "$action" = "relocate" ]; then
            echo "  action=relocate: PYTHONPYCACHEPREFIX is now ${chosen}"
            echo "    (${_cpc_fs}, ${_cpc_total} KiB total, ${_cpc_avail} KiB free, st_dev ${_cpc_dev})."
        else
            echo "  action=no-bytecode: no candidate directory passed the same test, so"
            echo "    PYTHONDONTWRITEBYTECODE=1 is exported and this run writes no .pyc at all."
            echo "    That is the safe end of the trade: a missing cache costs import time, a"
            echo "    half-written one kills the run. Candidates tried:"
            _cpc_candidates | sed 's/^/      /'
        fi
        echo "check_pycache: verdict=degraded rule=${rule} action=${action}"
    } >&2

    echo "check_pycache: verdict=degraded rule=${rule} action=${action} prefix=${chosen:-<none>}"
    _cpc_verify "$chosen"
    return 0
}

# Ask the interpreter that will do the work, once the repair above is in force,
# whether any of it took. -B, -E and -X pycache_prefix= all beat the environment
# variable, and sys.pycache_prefix only exists from 3.8. Note the asymmetry:
# sys.dont_write_bytecode being on is a PASS -- no .pyc is written, which is the
# outcome the whole file is trying to reach -- so a `python -B` wrapper is not a
# defect here even though it makes sys.pycache_prefix disagree.
_cpc_verify() {
    local want="${1:-}" out dwb eff
    command -v python3 >/dev/null 2>&1 || {
        echo "check_pycache: note -- no python3 on PATH in this container; nothing verified."
        return 0
    }
    out="$(python3 -c 'import sys
print(1 if sys.dont_write_bytecode else 0)
print(getattr(sys, "pycache_prefix", None) or "")' 2>/dev/null)"
    dwb="$(printf '%s\n' "$out" | sed -n 1p)"
    eff="$(printf '%s\n' "$out" | sed -n 2p)"
    if [ "$dwb" = "1" ]; then
        echo "check_pycache: python3 writes no bytecode at all (sys.dont_write_bytecode)."
        return 0
    fi
    [ -n "$want" ] || return 0
    if [ "$eff" != "$want" ]; then
        {
            echo "check_pycache: WARNING -- python3 in this container does not honour"
            echo "  PYTHONPYCACHEPREFIX=${want} (sys.pycache_prefix=${eff:-<none>}), and it ignored"
            echo "  PYTHONDONTWRITEBYTECODE too. That is a -B/-E/-X pycache_prefix wrapper or a"
            echo "  python < 3.8, none of which this script can reach through the environment."
            echo "  Bytecode will be written wherever that wrapper says. NOT fatal: it is not a"
            echo "  filesystem fault and the run may well be fine."
            echo "check_pycache: verdict=degraded rule=not-honoured action=none"
        } >&2
        echo "check_pycache: verdict=degraded rule=not-honoured action=none"
    fi
    return 0
}

_cpc_main
_cpc_rc=$?
unset -f _cpc_main _cpc_probe _cpc_candidates _cpc_verify
unset -v _cpc_why _cpc_fs _cpc_total _cpc_avail _cpc_dev _cpc_rootdev _cpc_writable
if [ "${_cpc_sourced}" = 1 ]; then
    unset -v _cpc_sourced
    return $_cpc_rc
fi
exit $_cpc_rc
