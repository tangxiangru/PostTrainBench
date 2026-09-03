#!/bin/bash
# Drive the bytecode-prefix plumbing that keeps run-generated .pyc off the
# 64 MiB container root.
#
# Worth having as a file, and worth having in this shape, because every cheap
# version of this check is green on a broken harness:
#
#   * grepping for PYTHONPYCACHEPREFIX proves a variable is set, not that the
#     path it names is bound. A prefix pointing at an unbound path moves the
#     bytecode from one place on the overlay to another place on the overlay.
#   * asserting `st_dev(prefix) != st_dev(/)` inside the container looks like
#     the real test and is not. Measured on opus_5.sif: with `-c` and no /tmp
#     bind, apptainer supplies a session tmpfs whose st_dev differs from / and
#     whose `df` total is the same 65536 blocks. The exec that runs for ten
#     hours is exactly the one with `-c`.
#   * counting the execs that carry the variable goes stale the moment someone
#     adds a fifth. So section [2] scans for every `apptainer exec` block in
#     the owned files and requires all of them to comply, with no inventory and
#     no allowlist to fall out of date.
#
# Two halves, deliberately:
#   sections [1]-[4] extract the shipped code -- the functions out of
#     src/run_task.sh with sed, exactly as tests/test_final_model_snapshot.sh
#     does, and the exec blocks out of the two sbatch files -- and run them with
#     `apptainer` stubbed to print its argv, so what is asserted is the argument
#     vector the harness really builds, not a copy of it in this file;
#   section [5] runs the shipped src/utils/check_pycache.sh inside the real
#     container in three mount shapes and checks it returns the right verdict
#     for each. Skipped loudly, never silently, when apptainer or the .sif is
#     not reachable from this machine.
#
# Usage: bash tests/test_container_pycache.sh
set -u

# PTB_TEST_REPO_ROOT lets this file be pointed at a copy of the tree. That is not
# a convenience: it is how "this test fails before the change" is demonstrated
# rather than asserted -- copy the tree, revert the shipped files in the copy,
# run this same test against the copy. Defaults to the checkout this file is in.
REPO_ROOT="${PTB_TEST_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-pycachetest.XXXXXX")"
# Section [1] must create directories on filesystems OTHER than $WORK's -- the
# whole subject is a prefix that is off the container root -- so those are
# tracked by name and removed here.
CLEANUP_DIRS=()
# The checker's own relocation ladder creates real directories outside $WORK when
# a case makes it relocate with no PYCACHE_FALLBACK_DIRS set (section [5c]). Only
# the ones that did not exist when this run started are removed, so a concurrent
# job's cache is never deleted out from under it.
LADDER_DIRS=("/var/tmp/.ptb_pycache.${UID:-0}" "${HOME}/.ptb_pycache" "/tmp/.ptb_pycache.${UID:-0}")
LADDER_PREEXISTING=()
for d in "${LADDER_DIRS[@]}"; do
    [ -e "$d" ] && LADDER_PREEXISTING+=("$d")
done
cleanup() {
    chmod -R u+w "$WORK" 2>/dev/null
    rm -rf "$WORK"
    local d p keep
    for d in "${CLEANUP_DIRS[@]:-}"; do
        [ -n "$d" ] || continue
        case "$d" in *ptb-pycachedir.*) rm -rf "$d" ;; esac
    done
    for d in "${LADDER_DIRS[@]}"; do
        keep=0
        for p in "${LADDER_PREEXISTING[@]:-}"; do [ "$p" = "$d" ] && keep=1; done
        [ "$keep" = 1 ] && continue
        case "$d" in */.ptb_pycache|*/.ptb_pycache.*) rm -rf "$d" ;; esac
    done
}
trap cleanup EXIT

RUN_TASK="$REPO_ROOT/src/run_task.sh"
RESCORE="$REPO_ROOT/ptb_ops/ptb_rescore.sbatch"
GREEDY="$REPO_ROOT/ptb_ops/ptb_greedy_board.sbatch"
JUDGE_LIB="$REPO_ROOT/src/judges/judge_lib.sh"
CHECKER="$REPO_ROOT/src/utils/check_pycache.sh"

# The sandbox path constants are READ OUT of run_task.sh, never restated here.
# Restating them would let this harness agree with itself while the shipped file
# said something else -- and half of what sections [2b], [3] and [6] assert is
# that two files still name the same string.
eval "$(grep -E '^(SANDBOX_(HOME|TASK_DIR|TMP|PYCACHE)|PYCACHE_BASENAME)=' "$RUN_TASK")"
# A name run_task.sh does not define gets a sentinel rather than staying unset:
# under `set -u` an absent constant would abort this file at the first reference
# and the run would report "3 sections passed" instead of "the assertion that
# would have caught this never ran". Every later comparison fails on the
# sentinel, which is the answer we want.
for v in SANDBOX_HOME SANDBOX_TASK_DIR SANDBOX_TMP SANDBOX_PYCACHE PYCACHE_BASENAME; do
    [ -n "${!v:-}" ] || eval "$v=__UNDEFINED_${v}__"
done

fail=0
skipped=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }
note() { echo "  NOTE $1"; }
skip() { echo "  SKIP $1"; skipped=$((skipped + 1)); }

# ---------------------------------------------------------------- stubs ----
# `apptainer` prints one argv entry per line so the assertions can read the
# vector the shipped code built. `timeout` drops its own three arguments and
# runs the rest, which is what the agent exec is wrapped in.
mkdir -p "$WORK/bin"
cat > "$WORK/bin/apptainer" <<'STUB'
#!/bin/bash
printf '%s\n' "$@"
STUB
cat > "$WORK/bin/timeout" <<'STUB'
#!/bin/bash
shift 3
exec "$@"
STUB
for _s in sleep fuse-overlayfs fusermount mountpoint nvidia-smi tree; do
    printf '#!/bin/bash\nexit 0\n' > "$WORK/bin/$_s"
done
chmod +x "$WORK/bin"/*
export PATH="$WORK/bin:$PATH"

# argv_get_prefix <argv_file> -> the PYTHONPYCACHEPREFIX value, or empty
argv_get_prefix() { sed -n 's/^PYTHONPYCACHEPREFIX=//p' "$1" | head -1; }

# argv_prefix_is_bound <argv_file>: true when some --bind in the SAME exec has a
# destination equal to the prefix or an ancestor of it. This is the assertion
# that a prefix pointing at an unbound path cannot pass.
argv_prefix_is_bound() {
    local f="$1" pfx dst
    pfx="$(argv_get_prefix "$f")"
    [ -n "$pfx" ] || return 1
    case "$pfx" in /*) ;; *) return 1 ;; esac   # empty-variable expansion -> relative
    while IFS= read -r dst; do
        [ -n "$dst" ] || continue
        [ "$dst" = "$pfx" ] && return 0
        case "$pfx" in "$dst"/*) return 0 ;; esac
    done < <(awk '/^--bind$/ {getline spec; sub(/^[^:]*:/, "", spec); print spec}' "$f")
    return 1
}

# Pull the whole VLLM_ATTENTION_BACKEND array construction out of a shipped
# file, so the "absent when unset / present when set" test drives the real
# lines rather than a restatement of them here.
#
# The empty-extraction case is a hard error and not a silent "" on purpose.
# Mutation-tested: replacing `ATTN_BACKEND_ENV=()` with a hardcoded
# `ATTN_BACKEND_ENV=(--env VLLM_ATTENTION_BACKEND=TRITON_ATTN)` -- the exact
# mistake requirement 3 forbids -- makes the anchor stop matching, and the
# earlier version of this function answered "" to that, which drove an exec with
# no backend in it and read as "absent when unset: PASS". An empty extraction is
# a check that never ran.
extract_attn_block() {
    local out
    out="$(awk -v name="$2" '
        !p && index($0, name "=()") { p = 1 }
        p { print }
        p && index($0, "POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND}\")") { exit }
    ' "$1")"
    if ! grep -q "${2}=()" <<<"$out" \
       || ! grep -q "POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND:-" <<<"$out" \
       || ! grep -q "POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND}\")" <<<"$out"; then
        echo "EXTRACTION_FAILED: no unset-by-default ${2} construction in $1" >&2
        return 1
    fi
    printf '%s\n' "$out"
}

# Pull the Nth `apptainer exec` continuation block out of a shipped file.
extract_exec_block() {
    awk -v want="$2" '
        /apptainer exec \\$/ { c++; if (c == want) inb = 1 }
        inb { print; if ($0 !~ /\\$/) exit }
    ' "$1"
}

count_exec_blocks() { grep -c 'apptainer exec \\$' "$1"; }

echo "[0] the shipped files still parse"
for f in "$RUN_TASK" "$CHECKER" "$RESCORE" "$GREEDY"; do
    chk "bash -n $(basename "$f")" "bash -n '$f'"
done

# ---------------------------------------------------------------------------
echo "[1] src/utils/check_pycache.sh: the verdict, the repair, and the one fatal"
# What this section asserts changed when the checker stopped being fail-closed,
# and it changed for a measured reason. The first version exited 71..75 and every
# call site ran it as `bash check_pycache.sh || exit 1`, so a node whose scratch
# fell back to something small killed a ten-hour 8xH100 cell in its first seconds
# -- ~80 GPU-hours spent to avoid the ~2 h of extra imports the rule exists to
# prevent. It now REPAIRS: relocate the prefix to a directory that passes the same
# predicate, or export PYTHONDONTWRITEBYTECODE=1 when nothing does. Only a
# container with no writable scratch anywhere is still fatal, and no filesystem
# SIZE can produce that.
#
# So every case below asserts three things -- what it printed, what rc it
# returned, and WHAT ENVIRONMENT IT LEFT BEHIND. The third is not decoration: the
# repair *is* an export, so a test that reads only $? cannot tell a repair from a
# shrug, and a call site that runs the script instead of sourcing it gets the
# shrug. Section [7] closes that loop.

# run_checker <outfile> <VAR=VAL ...> -- source the SHIPPED script in a throwaway
# shell, then append RC/PREFIX/DWB so both halves of the contract are readable.
run_checker() {
    local out="$1"; shift
    env "$@" bash -c '
        . "$0" > "$1" 2>&1
        rc=$?
        {   echo "RC=${rc}"
            echo "PREFIX=${PYTHONPYCACHEPREFIX-<unset>}"
            echo "DWB=${PYTHONDONTWRITEBYTECODE-<unset>}"
        } >> "$1"
    ' "$CHECKER" "$out"
}
ck_field() { sed -n "s/^${2}=//p" "$1" | tail -1; }
# Every degraded verdict is one line of the form `verdict=... rule=... action=...`.
ck_says() { grep -q "verdict=${2}" "$1"; }

# Two real filesystems, both measured, neither assumed: BIG to relocate ONTO, and
# SMALL (smallest total, still off the root device and writable) to be rejected by
# the capacity rule. Faking these with a `df` stub would test this file's idea of
# df output instead of the predicate; section [5] runs the same script in the real
# container, where the 64 MiB session tmpfs is not a fixture at all.
ROOTDEV="$(stat -c %d / 2>/dev/null)"
BIG=""; BIG_TOTAL=0; BIG_AVAIL=0
SMALL=""; SMALL_TOTAL=0; SMALL_AVAIL=0
for cand in "${PTB_TEST_BIG_DIR:-}" /dev/shm "$HOME" /rmeng_data/robtang /var/tmp /tmp; do
    [ -n "$cand" ] && [ -d "$cand" ] && [ -w "$cand" ] || continue
    [ "$(stat -c %d "$cand" 2>/dev/null)" != "$ROOTDEV" ] || continue
    # shellcheck disable=SC2046
    set -- $(df -P -k "$cand" 2>/dev/null | tail -1)
    _t="${2:-0}"; _a="${4:-0}"
    case "${_t}${_a}" in ''|*[!0-9]*) continue ;; esac
    if [ "$_t" -gt "$BIG_TOTAL" ]; then BIG="$cand"; BIG_TOTAL="$_t"; BIG_AVAIL="$_a"; fi
    if [ -z "$SMALL" ] || [ "$_t" -lt "$SMALL_TOTAL" ]; then
        SMALL="$cand"; SMALL_TOTAL="$_t"; SMALL_AVAIL="$_a"
    fi
done

if [ -z "$BIG" ]; then
    skip "[1] found no writable directory off the root device -- the whole section"
    skip "  needs one to relocate onto, so NONE of the verdicts below were checked."
else
    echo "  BIG=$BIG (${BIG_TOTAL} KiB total, ${BIG_AVAIL} free)  SMALL=$SMALL (${SMALL_TOTAL} KiB total, ${SMALL_AVAIL} free)"
    GOOD="$(mktemp -d "$BIG/ptb-pycachedir.XXXXXX")";     CLEANUP_DIRS+=("$GOOD")
    FALLBACK="$(mktemp -d "$BIG/ptb-pycachedir.XXXXXX")"; CLEANUP_DIRS+=("$FALLBACK")
    # PYCACHE_FALLBACK_DIRS is set on every case below so the ladder is
    # deterministic AND so a test run never writes into the real $HOME.

    # --- the healthy case ---------------------------------------------------
    run_checker "$WORK/v_ok.txt" PYTHONPYCACHEPREFIX="$GOOD" PYCACHE_FALLBACK_DIRS="$FALLBACK"
    chk "ok: bound, big, writable prefix -> verdict=ok"  "ck_says '$WORK/v_ok.txt' ok"
    chk "ok: rc 0"                                       "[ \"\$(ck_field '$WORK/v_ok.txt' RC)\" = 0 ]"
    chk "ok: leaves PYTHONPYCACHEPREFIX where it was"    "[ \"\$(ck_field '$WORK/v_ok.txt' PREFIX)\" = '$GOOD' ]"
    chk "ok: does not disable bytecode"                  "[ \"\$(ck_field '$WORK/v_ok.txt' DWB)\" = '<unset>' ]"

    # --- unset (was a fatal 71) --------------------------------------------
    run_checker "$WORK/v_unset.txt" PYTHONPYCACHEPREFIX="" PYCACHE_FALLBACK_DIRS="$FALLBACK"
    chk "unset: relocates instead of dying" "grep -q 'rule=unset action=relocate' '$WORK/v_unset.txt'"
    chk "unset: rc 0, NOT the old 71"       "[ \"\$(ck_field '$WORK/v_unset.txt' RC)\" = 0 ]"
    chk "unset: the export actually happened" \
        "[ \"\$(ck_field '$WORK/v_unset.txt' PREFIX)\" = '$FALLBACK' ]"

    # --- unwritable (was a fatal 72) ----------------------------------------
    mkdir -p "$WORK/ro"; chmod 555 "$WORK/ro"
    run_checker "$WORK/v_unwritable.txt" PYTHONPYCACHEPREFIX="$WORK/ro/sub" PYCACHE_FALLBACK_DIRS="$FALLBACK"
    chmod 755 "$WORK/ro"
    chk "unwritable: relocates instead of dying" \
        "grep -q 'rule=unwritable action=relocate' '$WORK/v_unwritable.txt'"
    chk "unwritable: rc 0, NOT the old 72" "[ \"\$(ck_field '$WORK/v_unwritable.txt' RC)\" = 0 ]"
    chk "unwritable: prints no shell-level 'No such file or directory' noise" \
        "! grep -q 'No such file or directory' '$WORK/v_unwritable.txt'"

    # --- on the container root (was a fatal 73) -----------------------------
    # Needs a writable directory that IS on the root device. $WORK usually is;
    # guarded so the case is skipped rather than mis-reported where it is not.
    if [ "$(stat -c %d "$WORK" 2>/dev/null)" = "$ROOTDEV" ]; then
        mkdir -p "$WORK/onroot"
        run_checker "$WORK/v_root.txt" PYTHONPYCACHEPREFIX="$WORK/onroot" PYCACHE_FALLBACK_DIRS="$FALLBACK"
        chk "container-root: relocates instead of dying" \
            "grep -q 'rule=container-root action=relocate' '$WORK/v_root.txt'"
        chk "container-root: rc 0, NOT the old 73" "[ \"\$(ck_field '$WORK/v_root.txt' RC)\" = 0 ]"
        chk "container-root: relocated onto the fallback" \
            "[ \"\$(ck_field '$WORK/v_root.txt' PREFIX)\" = '$FALLBACK' ]"
    else
        skip "container-root case: \$WORK is not on the root filesystem here"
    fi

    # --- too small (was a fatal 74: THE ONE THAT KILLS THE TEN-HOUR CELL) ---
    if [ -n "$SMALL" ] && [ "$SMALL_TOTAL" -lt "$BIG_TOTAL" ]; then
        SMALLDIR="$(mktemp -d "$SMALL/ptb-pycachedir.XXXXXX")"; CLEANUP_DIRS+=("$SMALLDIR")
        run_checker "$WORK/v_cap.txt" PYTHONPYCACHEPREFIX="$SMALLDIR" \
            PYCACHE_FALLBACK_DIRS="$FALLBACK" PYCACHE_MIN_TOTAL_KIB="$((SMALL_TOTAL + 1))"
        chk "capacity: relocates instead of dying" \
            "grep -q 'rule=capacity action=relocate' '$WORK/v_cap.txt'"
        chk "capacity: rc 0, NOT the old 74 -- the cell survives" \
            "[ \"\$(ck_field '$WORK/v_cap.txt' RC)\" = 0 ]"
        chk "capacity: relocated onto the fallback" \
            "[ \"\$(ck_field '$WORK/v_cap.txt' PREFIX)\" = '$FALLBACK' ]"
    else
        skip "capacity case: found only one writable filesystem off the root device"
    fi

    # --- a full filesystem: big enough, no room. Same ENOSPC, new rule. -----
    if [ -n "$SMALL" ] && [ "$SMALL_AVAIL" -lt "$BIG_AVAIL" ] && [ "$SMALL_TOTAL" -ge 4194304 ]; then
        SMALLDIR2="$(mktemp -d "$SMALL/ptb-pycachedir.XXXXXX")"; CLEANUP_DIRS+=("$SMALLDIR2")
        run_checker "$WORK/v_full.txt" PYTHONPYCACHEPREFIX="$SMALLDIR2" \
            PYCACHE_FALLBACK_DIRS="$FALLBACK" PYCACHE_MIN_AVAIL_KIB="$((SMALL_AVAIL + 1))"
        chk "full: a fs over the size floor with no free space is caught" \
            "grep -q 'rule=full action=relocate' '$WORK/v_full.txt'"
        chk "full: rc 0" "[ \"\$(ck_field '$WORK/v_full.txt' RC)\" = 0 ]"
    else
        skip "full case: needs two off-root filesystems with different free space"
    fi

    # --- nowhere to put it: the no-bytecode fallback ------------------------
    # A floor above every filesystem in existence rejects the prefix AND every
    # candidate, which is the shape of a node where all the scratch is small.
    run_checker "$WORK/v_nobc.txt" PYTHONPYCACHEPREFIX="$GOOD" \
        PYCACHE_FALLBACK_DIRS="$FALLBACK" PYCACHE_MIN_TOTAL_KIB=9999999999999
    chk "no candidate passes: falls back to writing no bytecode at all" \
        "grep -q 'action=no-bytecode' '$WORK/v_nobc.txt'"
    chk "no candidate passes: rc 0 -- still not a dead cell" \
        "[ \"\$(ck_field '$WORK/v_nobc.txt' RC)\" = 0 ]"
    chk "no candidate passes: PYTHONDONTWRITEBYTECODE=1 is exported" \
        "[ \"\$(ck_field '$WORK/v_nobc.txt' DWB)\" = 1 ]"
    chk "no candidate passes: and python3 confirms it took" \
        "grep -q 'writes no bytecode at all' '$WORK/v_nobc.txt'"

    # --- a python that ignores the variable (was a fatal 75) ----------------
    # -X pycache_prefix beats the environment variable. The agent can do this to
    # itself at any point, so the assertion can only speak for the interpreter it
    # asks -- and it must not kill the cell over it.
    cat > "$WORK/bin/python3" <<STUB
#!/bin/bash
exec /usr/bin/python3 -X pycache_prefix="$WORK/elsewhere" "\$@"
STUB
    chmod +x "$WORK/bin/python3"
    run_checker "$WORK/v_honour.txt" PYTHONPYCACHEPREFIX="$GOOD" PYCACHE_FALLBACK_DIRS="$FALLBACK" PATH="$PATH"
    rm -f "$WORK/bin/python3"
    chk "not-honoured: reported" "grep -q 'rule=not-honoured' '$WORK/v_honour.txt'"
    chk "not-honoured: rc 0, NOT the old 75" "[ \"\$(ck_field '$WORK/v_honour.txt' RC)\" = 0 ]"

    # --- THE BAR ------------------------------------------------------------
    # "No configuration of the filesystem should be able to kill a cell that
    # could otherwise have produced a scoreable model." Every case above is a
    # filesystem configuration; every one of them must have returned 0. Asserted
    # as a sweep over whatever cases actually ran, so a case added later is
    # covered without being listed here.
    _nonzero=0
    for _v in "$WORK"/v_*.txt; do
        [ -e "$_v" ] || continue
        [ "$(ck_field "$_v" RC)" = 0 ] || { echo "  ---> nonzero rc from $(basename "$_v")"; _nonzero=1; }
    done
    chk "no filesystem shape in this section returned a fatal rc" "[ $_nonzero -eq 0 ]"
fi

# ---------------------------------------------------------------------------
echo "[2] every apptainer exec in the owned files carries the pair"
# No inventory and no count: whatever blocks are in the files today are the
# blocks that must comply, so a fifth exec added tomorrow fails here by default
# instead of quietly inheriting an exemption.
for f in "$RUN_TASK" "$RESCORE" "$GREEDY"; do
    n="$(count_exec_blocks "$f")"
    chk "$(basename "$f"): found $n apptainer exec block(s)" "[ $n -ge 1 ]"
    for ((b = 1; b <= n; b++)); do
        blk="$WORK/blk.txt"
        extract_exec_block "$f" "$b" > "$blk"
        line="$(grep -n 'apptainer exec \\$' "$f" | sed -n "${b}p" | cut -d: -f1)"
        chk "$(basename "$f"):$line sets PYTHONPYCACHEPREFIX" \
            "grep -q 'PYTHONPYCACHEPREFIX=' '$blk'"
        chk "$(basename "$f"):$line runs check_pycache.sh in-exec" \
            "grep -q 'check_pycache.sh' '$blk'"
        # `bash check_pycache.sh` would print the warning and change nothing:
        # the repair is an export, and an export from a child reaches nobody.
        chk "$(basename "$f"):$line SOURCES it rather than running it" \
            "grep -qE '(^|[^[:alnum:]_.])[.] \"?[^\" ]*check_pycache[.]sh' '$blk'"
    done
done

echo "[2b] the judge exec (src/judges/judge_lib.sh, not owned by this change)"
chk "run_task.sh puts the prefix in JUDGE_EXTRA_APPTAINER_ARGS" \
    "awk '/^JUDGE_EXTRA_APPTAINER_ARGS=\\(/,/^\\)/' '$RUN_TASK' | grep -q 'PYTHONPYCACHEPREFIX=\"\${SANDBOX_PYCACHE}\"'"
# judge_lib.sh is the one file in the pair this change does not own, so its bind
# destination is asserted to still EQUAL SANDBOX_TMP rather than assumed to. If
# someone moves the sandbox /tmp, this is where the judges stop being covered.
chk "the judge exec binds job_tmp at SANDBOX_TMP, so SANDBOX_PYCACHE resolves there" \
    "extract_exec_block '$JUDGE_LIB' 2 | grep -q -- '--bind \"\${job_tmp}:$SANDBOX_TMP\"'"
chk "the judge exec expands JUDGE_EXTRA_APPTAINER_ARGS" \
    "extract_exec_block '$JUDGE_LIB' 2 | grep -q 'JUDGE_EXTRA_APPTAINER_ARGS\\[@\\]'"
note "judge_lib.sh:138 (the pinned-codex npm install) is NOT covered -- the"
note "assertion has to be inside the exec and that file is out of scope here."

# ---------------------------------------------------------------------------
echo "[3] the argv the shipped code actually builds"

# --- 3a: the agent exec, by sourcing solve_task out of run_task.sh ----------
sed -n '/^solve_task() {/,/^}$/p' "$RUN_TASK" > "$WORK/solve_task.sh"
chk "extracted solve_task from src/run_task.sh" "grep -q 'apptainer exec' '$WORK/solve_task.sh'"
# shellcheck disable=SC1090
source "$WORK/solve_task.sh"

# The constants themselves were eval'd out of run_task.sh at the top of this file;
# what [3d] below turns on is that the prompt's absolute paths and this bind are
# the same two strings.
for v in SANDBOX_HOME SANDBOX_TASK_DIR SANDBOX_TMP SANDBOX_PYCACHE PYCACHE_BASENAME; do
    chk "run_task.sh defines $v" "grep -qE '^$v=' '$RUN_TASK'"
done

FIXTURE_MODEL="Qwen/Qwen3-4B-Base"

drive_agent() {
    local AGENT_AUTH_SRC="" CURSOR_AUTH_SRC="" GROK_AUTH_SRC=""
    local NUM_HOURS=10 NUM_GPUS=1 PROMPT="p" AGENT_CONFIG="c"
    local HF_HOME_NEW="/home/ben/hf_cache"
    local JOB_DIR="$WORK/job" JOB_TMP="$WORK/job/tmp" HF_MERGED="$WORK/merged"
    local POST_TRAIN_BENCH_CONTAINERS_DIR="$WORK" POST_TRAIN_BENCH_CONTAINER_NAME="fake"
    local SOLVE_OUT="$1"
    local MODEL_TO_TRAIN="$FIXTURE_MODEL"
    local API_KEY_ENV_ARGS=() AGENT_ENV_ARGS=()
    mkdir -p "$JOB_DIR"
    solve_task
}
drive_agent "$WORK/argv_agent.txt"
chk "agent: prefix is set"              "[ -n \"\$(argv_get_prefix '$WORK/argv_agent.txt')\" ]"
chk "agent: prefix is SANDBOX_PYCACHE"  "[ \"\$(argv_get_prefix '$WORK/argv_agent.txt')\" = '$SANDBOX_PYCACHE' ]"
chk "agent: prefix is under a --bind"   "argv_prefix_is_bound '$WORK/argv_agent.txt'"
chk "agent: check_pycache.sh is sourced first, ahead of the CUDA checks" \
    "grep -q '[.] /home/ben/check_pycache.sh && python /home/ben/check_cuda.py' '$WORK/argv_agent.txt'"
chk "agent: no VLLM_ATTENTION_BACKEND when the harness var is unset" \
    "! grep -q 'VLLM_ATTENTION_BACKEND' '$WORK/argv_agent.txt'"
POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND=TRITON_ATTN drive_agent "$WORK/argv_agent_attn.txt"
chk "agent: VLLM_ATTENTION_BACKEND=TRITON_ATTN when it is set" \
    "grep -qx 'VLLM_ATTENTION_BACKEND=TRITON_ATTN' '$WORK/argv_agent_attn.txt'"

# --- 3c: the base model reaches the sandbox on EVERY arm --------------------
# reference/train_grpo.py refuses to guess a base model and reads $MODEL_TO_TRAIN. That
# variable used to be inside AGENT_CONTEXT_ENV, which run_task.sh only applies when
# "${JOB_DIR}/agent" exists -- i.e. on the payload arms and not on the control arm. Same
# file, same prompt, and the reference script would have exited on one arm and trained on
# the other. drive_agent above creates JOB_DIR with NO agent/ subdirectory, so this is
# the control-arm shape specifically.
chk "agent: JOB_DIR has no agent/ payload, so this is the control-arm shape" \
    "[ ! -d '$WORK/job/agent' ]"
chk "agent: MODEL_TO_TRAIN is in the argv" \
    "grep -qx 'MODEL_TO_TRAIN=$FIXTURE_MODEL' '$WORK/argv_agent.txt'"
chk "agent: MODEL_TO_TRAIN is passed by --env, not just present as a string" \
    "grep -B1 -x 'MODEL_TO_TRAIN=$FIXTURE_MODEL' '$WORK/argv_agent.txt' | grep -qx -- '--env'"

# --- 3d: the sandbox paths the prompt hands the agent are the ones bound ----
# get_prompt.py renders ABSOLUTE paths for reference/, graded_read.py and the
# decontamination tools, because the claude_autor operator runs its stages two levels below
# the task root and a relative path resolves to nothing there. Absolute is only right if it
# is the SAME absolute path this exec binds; these two checks are the join.
chk "agent: --home binds JOB_DIR at SANDBOX_HOME" \
    "grep -A1 -x -- '--home' '$WORK/argv_agent.txt' | grep -qx '$WORK/job:$SANDBOX_HOME'"
chk "agent: --pwd is SANDBOX_TASK_DIR" \
    "grep -A1 -x -- '--pwd' '$WORK/argv_agent.txt' | grep -qx '$SANDBOX_TASK_DIR'"
chk "agent: SANDBOX_TASK_DIR is under SANDBOX_HOME" \
    "case '$SANDBOX_TASK_DIR' in '$SANDBOX_HOME'/*) true ;; *) false ;; esac"
for _pair in "SANDBOX_HOME:--sandbox-home-dir" "SANDBOX_TASK_DIR:--sandbox-task-dir"; do
    _v="${_pair%%:*}"; _flag="${_pair##*:}"
    chk "agent: run_task.sh passes $_flag to get_prompt.py" \
        "grep -q -- '$_flag \"\${$_v}\"' '$RUN_TASK'"
done

# --- 3b: the scorer exec, by sourcing run_evaluation out of run_task.sh -----
sed -n '/^run_evaluation() {/,/^}$/p' "$RUN_TASK" > "$WORK/run_evaluation.sh"
chk "extracted run_evaluation from src/run_task.sh" "grep -q 'apptainer exec' '$WORK/run_evaluation.sh'"
# shellcheck disable=SC1090
source "$WORK/run_evaluation.sh"
reap_gpu_processes() { :; }
with_huggingface_overlay() { "$@"; }

drive_scorer() {
    local EVAL_DIR="$WORK/eval$1" REPO_ROOT="$REPO_ROOT"
    local TMP_HF_CACHE="/tmp/hf_cache_90afd0" HF_MERGED="$WORK/merged"
    local SCORER_PYCACHE="$WORK/scratch/pycache_scorer"
    local EVALUATION_TASK="gsm8k" EVAL_SCRIPT="evaluate.py"
    local POST_TRAIN_BENCH_CONTAINERS_DIR="$WORK"
    local OPENAI_API_KEY="" OPENROUTER_API_KEY=""
    mkdir -p "$EVAL_DIR"
    run_evaluation "" 1
    cp "$EVAL_DIR/final_eval_1.txt" "$2"
}
drive_scorer a "$WORK/argv_scorer.txt"
chk "scorer: prefix is set"                "[ -n \"\$(argv_get_prefix '$WORK/argv_scorer.txt')\" ]"
chk "scorer: prefix is absolute and node-local, not '/pycache'" \
    "[ \"\$(argv_get_prefix '$WORK/argv_scorer.txt')\" = '$WORK/scratch/pycache_scorer' ]"
chk "scorer: prefix is under a --bind"     "argv_prefix_is_bound '$WORK/argv_scorer.txt'"
chk "scorer: check_pycache.sh runs in-exec" "grep -q 'check_pycache.sh' '$WORK/argv_scorer.txt'"
chk "scorer: evaluate.py is still the command" "grep -qx 'evaluate.py' '$WORK/argv_scorer.txt'"
chk "scorer: no VLLM_ATTENTION_BACKEND when unset" \
    "! grep -q 'VLLM_ATTENTION_BACKEND' '$WORK/argv_scorer.txt'"
POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND=TRITON_ATTN drive_scorer b "$WORK/argv_scorer_attn.txt"
chk "scorer: VLLM_ATTENTION_BACKEND=TRITON_ATTN when set" \
    "grep -qx 'VLLM_ATTENTION_BACKEND=TRITON_ATTN' '$WORK/argv_scorer_attn.txt'"

# The trap this whole section exists for: SCORER_PYCACHE has to survive
# run_evaluation_with_retry's `bash -c "$(declare -f ...)"`, and only exported
# scalars do. Unexported it expands to "" and the pair silently becomes
# --bind ":" plus PYTHONPYCACHEPREFIX=/pycache, i.e. the container root.
chk "SCORER_PYCACHE is exported at file scope" \
    "grep -qE '^export SCORER_PYCACHE=' '$RUN_TASK'"
chk "JOB_PYCACHE and SCORER_PYCACHE are created before any exec" \
    "grep -q 'mkdir -p \"\${JOB_PYCACHE}\"' '$RUN_TASK' && grep -q 'mkdir -p \"\${SCORER_PYCACHE}\"' '$RUN_TASK'"

# --- 3c/3d: the two ptb_ops execs, by extracting the blocks -----------------
drive_sbatch_block() {
    local file="$1" out="$2" attn_name="$3"
    local body="$WORK/sbatch_drive.sh"
    # Never leave the previous invocation's argv in place: a drive that dies
    # before the redirect would otherwise be asserted against stale, passing
    # output from the run before it.
    rm -f "$out" "$WORK/shadow/eval_greedy.txt" "$WORK/eval_sb/final_eval_rescore_1.txt"
    extract_attn_block "$file" "$attn_name" >/dev/null || return 1
    {
        # set -u so an array the extraction failed to produce is a hard error
        # here rather than an empty expansion that looks like "not pinned".
        echo "set -u"
        echo "_drive() {"
        echo "  local TMP_HF_CACHE=/tmp/hf_cache_rescore"
        echo "  local EVAL_DIR=$WORK/eval_sb REPO_ROOT=$REPO_ROOT"
        echo "  local HF_MERGED=$WORK/merged merged=$WORK/merged"
        echo "  local PYCACHE=$WORK/scratch/rescore/pycache pycache=$WORK/scratch/greedy/pycache"
        echo "  local TASK=gsm8k POST_TRAIN_BENCH_CONTAINERS_DIR=$WORK"
        echo "  local MODEL_PATH=$WORK/model model=$WORK/model"
        echo "  local SLURM_JOB_ID=1 gpu=0 incache=/tmp/hf_cache_rescore"
        echo "  local out=$WORK/metrics.json shadow=$WORK/shadow"
        echo "  local CACHE_BIND=() cache_bind=() extra_bind=()"
        extract_attn_block "$file" "$attn_name" | sed 's/^/  /'
        extract_exec_block "$file" 1 | sed 's/^/  /'
        echo "}"
        echo "_drive"
    } > "$body"
    mkdir -p "$WORK/shadow" "$WORK/eval_sb"
    bash "$body" >/dev/null 2>&1
    cp "$WORK/shadow/eval_greedy.txt" "$out" 2>/dev/null \
        || cp "$WORK/eval_sb/final_eval_rescore_1.txt" "$out" 2>/dev/null
    rm -f "$WORK/shadow/eval_greedy.txt" "$WORK/eval_sb/final_eval_rescore_1.txt"
    [ -s "$out" ]
}

for spec in "ptb_rescore.sbatch|$RESCORE|ATTN_BACKEND_ENV" "ptb_greedy_board.sbatch|$GREEDY|attn_backend_env"; do
    label="${spec%%|*}"; rest="${spec#*|}"; file="${rest%%|*}"; aname="${rest##*|}"
    chk "$label: the exec and its unset-by-default backend array both drive" \
        "drive_sbatch_block '$file' '$WORK/argv_sb.txt' '$aname'"
    chk "$label: prefix is set"              "[ -n \"\$(argv_get_prefix '$WORK/argv_sb.txt')\" ]"
    chk "$label: prefix is under a --bind"   "argv_prefix_is_bound '$WORK/argv_sb.txt'"
    chk "$label: check_pycache.sh runs in-exec" "grep -q 'check_pycache.sh' '$WORK/argv_sb.txt'"
    chk "$label: evaluate.py is still the command" "grep -qx 'evaluate.py' '$WORK/argv_sb.txt'"
    chk "$label: no VLLM_ATTENTION_BACKEND when unset" \
        "! grep -q 'VLLM_ATTENTION_BACKEND' '$WORK/argv_sb.txt'"
    chk "$label: the exec drives again with the backend set" \
        "POST_TRAIN_BENCH_VLLM_ATTENTION_BACKEND=TRITON_ATTN drive_sbatch_block '$file' '$WORK/argv_sb_attn.txt' '$aname'"
    chk "$label: VLLM_ATTENTION_BACKEND=TRITON_ATTN when set" \
        "grep -qx 'VLLM_ATTENTION_BACKEND=TRITON_ATTN' '$WORK/argv_sb_attn.txt'"
done

# ---------------------------------------------------------------------------
echo "[4] the prefix is never pointed at something that gets deleted mid-run"
# with_huggingface_overlay rm -r's the merged mount after every call, and both
# sbatch jobs rm -rf their TMP_HF_CACHE source. A prefix under either is a cache
# that is wiped between the nine evaluation attempts.
chk "run_task.sh: prefix is not under HF_MERGED/TMP_HF_CACHE" \
    "! grep -qE 'PYTHONPYCACHEPREFIX=\"\\\$\\{(HF_MERGED|TMP_HF_CACHE)' '$RUN_TASK'"
chk "ptb_rescore.sbatch: prefix is not under TMP_HF_CACHE" \
    "! grep -qE 'PYTHONPYCACHEPREFIX=\"\\\$\\{TMP_HF_CACHE' '$RESCORE'"
chk "ptb_greedy_board.sbatch: prefix is per-worker, not the shared incache" \
    "! grep -qE 'PYTHONPYCACHEPREFIX=\"\\\$\\{incache' '$GREEDY'"

# ---------------------------------------------------------------------------
echo "[6] the host path and the in-sandbox path are two names for ONE directory"
# JOB_PYCACHE is the directory the job creates on the node; the --env inside the
# agent and judge sandboxes has to name the SAME directory as seen through
# --bind "${JOB_TMP}:${SANDBOX_TMP}". Those used to be written out twice -- once
# as "${JOB_TMP}/pycache" here, once as a literal "/tmp/pycache" in each exec --
# with nothing joining them. Renaming either half would have left the execs
# pointing at a directory nobody creates: not an error anywhere, just bytecode
# quietly back on the 64 MiB overlay with a variable set to make it look fixed.

sed -n '/^pycache_paths_agree() {/,/^}$/p' "$RUN_TASK" > "$WORK/agree.sh"
chk "extracted pycache_paths_agree from src/run_task.sh" \
    "grep -q 'SANDBOX_PYCACHE' '$WORK/agree.sh'"
# shellcheck disable=SC1090
source "$WORK/agree.sh" 2>/dev/null || true
if ! declare -F pycache_paths_agree >/dev/null 2>&1; then
    chk "pycache_paths_agree is defined in run_task.sh" "false"
    pycache_paths_agree() { return 0; }
fi
chk "run_task.sh runs the assertion at startup, before any exec" \
    "grep -qx 'pycache_paths_agree || exit 1' '$RUN_TASK'"
chk "both names are derived from one basename" \
    "grep -q 'JOB_PYCACHE=\"\${JOB_TMP}/\${PYCACHE_BASENAME}\"' '$RUN_TASK' \
     && grep -q 'SANDBOX_PYCACHE=\"\${SANDBOX_TMP}/\${PYCACHE_BASENAME}\"' '$RUN_TASK'"

# Driven in a subshell so each case gets its own four values and none leak.
drive_agree() (
    JOB_TMP="$1"; JOB_PYCACHE="$2"; SANDBOX_TMP="$3"; SANDBOX_PYCACHE="$4"
    pycache_paths_agree >/dev/null 2>&1
)
chk "agrees on the four values run_task.sh actually ships" \
    "drive_agree /scr/job/tmp '/scr/job/tmp/${PYCACHE_BASENAME}' '$SANDBOX_TMP' '$SANDBOX_PYCACHE'"
chk "catches the HOST name drifting" \
    "! drive_agree /scr/job/tmp /scr/job/tmp/pycache_v2 /tmp /tmp/pycache"
chk "catches the SANDBOX name drifting" \
    "! drive_agree /scr/job/tmp /scr/job/tmp/pycache /tmp /tmp/bytecode"
chk "catches a sandbox prefix that the bind cannot reach at all" \
    "! drive_agree /scr/job/tmp /scr/job/tmp/pycache /tmp /var/pycache"
chk "no value of JOB_TMP can trip it -- only an edit can" \
    "drive_agree /any/where/at/all /any/where/at/all/pycache /tmp /tmp/pycache"

# The other half: the exec must FOLLOW the variables, not merely agree with them
# today. Move both and the argv has to move too; a literal would not.
( SANDBOX_TMP=/zztmp; SANDBOX_PYCACHE=/zztmp/zzcache; drive_agent "$WORK/argv_probe.txt" )
chk "agent: --env PYTHONPYCACHEPREFIX follows SANDBOX_PYCACHE" \
    "[ \"\$(argv_get_prefix '$WORK/argv_probe.txt')\" = /zztmp/zzcache ]"
chk "agent: the JOB_TMP bind destination follows SANDBOX_TMP" \
    "grep -qx '$WORK/job/tmp:/zztmp' '$WORK/argv_probe.txt'"
chk "agent: the pair still agrees after moving both" \
    "argv_prefix_is_bound '$WORK/argv_probe.txt'"
chk "run_task.sh has no literal /tmp/pycache left outside comments" \
    "! grep -v '^[[:space:]]*#' '$RUN_TASK' | grep -q '/tmp/pycache'"

# ---------------------------------------------------------------------------
echo "[7] every call site SOURCES the checker, and sourcing is what makes it work"
# The checker's remedy for a cache it cannot use is `export PYTHONPYCACHEPREFIX=`
# or `export PYTHONDONTWRITEBYTECODE=1`. Run as `bash check_pycache.sh` those
# exports die with the child: the warning still prints, the log still looks like
# something was done, and the python that follows is unrepaired. This is a
# green-looking no-op, so it is asserted twice -- by form, and by behaviour.
for f in "$RUN_TASK" "$RESCORE" "$GREEDY"; do
    b="$(basename "$f")"
    n_bad="$(grep -cE 'bash[[:space:]]+"?[^[:space:];|&]*check_pycache[.]sh' "$f" || true)"
    n_src="$(grep -cE '(^|[^[:alnum:]_.])[.][[:space:]]+"?[^[:space:];|&]*check_pycache[.]sh' "$f" || true)"
    want_src="$(grep -c 'apptainer exec \\$' "$f")"
    chk "$b: no call site runs it as a child (found $n_bad)" "[ '$n_bad' -eq 0 ]"
    chk "$b: one sourced call per apptainer exec (found $n_src, execs $want_src)" \
        "[ '$n_src' -ge '$want_src' ]"
done

if [ -n "${FALLBACK:-}" ]; then
    # Same script, same broken prefix, the two call forms side by side.
    env PYTHONPYCACHEPREFIX="" PYCACHE_FALLBACK_DIRS="$FALLBACK" \
        bash -c 'bash "$0" >/dev/null 2>&1; echo "PREFIX=${PYTHONPYCACHEPREFIX}"' \
        "$CHECKER" > "$WORK/form_child.txt" 2>&1
    env PYTHONPYCACHEPREFIX="" PYCACHE_FALLBACK_DIRS="$FALLBACK" \
        bash -c '. "$0" >/dev/null 2>&1; echo "PREFIX=${PYTHONPYCACHEPREFIX}"' \
        "$CHECKER" > "$WORK/form_source.txt" 2>&1
    chk "run as a child: the repair does NOT reach the caller (the old no-op)" \
        "grep -qx 'PREFIX=' '$WORK/form_child.txt'"
    chk "sourced: the repair reaches the caller, which is what python inherits" \
        "grep -qx 'PREFIX=$FALLBACK' '$WORK/form_source.txt'"
    # And the shape the scorer exec uses verbatim: `bash -c '. "$0" ...' <path>`,
    # where $0 EQUALS the script path -- the case a BASH_SOURCE[0]-vs-$0 test for
    # "am I sourced" gets wrong, silently, by exiting the caller's shell.
    env PYTHONPYCACHEPREFIX="" PYCACHE_FALLBACK_DIRS="$FALLBACK" \
        bash -c '. "$0" >/dev/null 2>&1 || exit 1; echo "SURVIVED PREFIX=${PYTHONPYCACHEPREFIX}"' \
        "$CHECKER" > "$WORK/form_dollar0.txt" 2>&1
    chk "sourced as \$0 (the scorer's exact form): the caller shell survives" \
        "grep -qx 'SURVIVED PREFIX=$FALLBACK' '$WORK/form_dollar0.txt'"
else
    skip "[7] behavioural half: section [1] found no filesystem to relocate onto"
fi

# ---------------------------------------------------------------------------
echo "[5] the predicate inside the real container, in five mount shapes"
# Sections [1]-[4] drive the script on host filesystems and read the shipped argv.
# This section is the deployment half: the same file, inside the same .sif, under
# the mount shapes a real cell actually gets. Every case here used to be an exit
# code that killed the job; the assertions are now on the verdict AND on rc, so a
# regression to "fatal" is caught by name and not just by an integer.
APPTAINER_BIN=""
export LD_LIBRARY_PATH="/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for c in "${PTB_TEST_APPTAINER:-}" \
         "$(PATH="${PATH#"$WORK/bin":}" command -v apptainer 2>/dev/null || true)" \
         /rmeng_data/robtang/tools/apt-root/usr/bin/apptainer; do
    [ -n "$c" ] && [ -x "$c" ] && { APPTAINER_BIN="$c"; break; }
done
SIF=""
for c in "${PTB_TEST_SIF:-}" "${POST_TRAIN_BENCH_CONTAINERS_DIR:-}/opus_5.sif" \
         /rmeng_data/robtang/ptb-containers/opus_5.sif; do
    [ -n "$c" ] && [ -f "$c" ] && { SIF="$c"; break; }
done
if [ -z "$APPTAINER_BIN" ] || [ -z "$SIF" ]; then
    skip "container round-trip: apptainer=[${APPTAINER_BIN:-none}] sif=[${SIF:-none}]"
    skip "  -- sections [1]-[4],[6],[7] ran; the deployment half of this test did NOT."
else
    echo "  using $APPTAINER_BIN and $SIF"
    mkdir -p "$WORK/ctmp/pycache" "$WORK/ro"
    # Sourced, exactly as the shipped execs call it, and then a real import --
    # so each case answers "did the cell survive?", not only "what was the rc?".
    cat > "$WORK/inner.sh" <<'INNER_EOF'
#!/bin/bash
. /opt/check_pycache.sh
rc=$?
echo "RC=${rc}"
echo "PREFIX=${PYTHONPYCACHEPREFIX-<unset>}"
echo "DWB=${PYTHONDONTWRITEBYTECODE-<unset>}"
if [ "$rc" -eq 0 ]; then
    python3 - <<'PYEOF'
import argparse, dataclasses, json, sys
print("IMPORT_OK dwb=%s prefix=%s" % (sys.dont_write_bytecode, sys.pycache_prefix))
PYEOF
    echo "PY_RC=$?"
fi
exit "$rc"
INNER_EOF
    cbind=(--bind "$CHECKER:/opt/check_pycache.sh" --bind "$WORK/inner.sh:/opt/inner.sh")
    c_field() { sed -n "s/^${2}=//p" "$WORK/$1.txt" | tail -1; }

    # (a) the agent's real shape: -c --cleanenv, JOB_TMP bound at /tmp.
    "$APPTAINER_BIN" exec -c --cleanenv --writable-tmpfs \
        --bind "$WORK/ctmp:/tmp" --env PYTHONPYCACHEPREFIX="/tmp/pycache" \
        "${cbind[@]}" "$SIF" bash /opt/inner.sh >"$WORK/c_ok.txt" 2>&1; rc=$?
    chk "(a) bound /tmp, -c --cleanenv -> rc 0 (got $rc)" "[ $rc -eq 0 ]"
    chk "(a) verdict=ok"        "grep -q 'verdict=ok' '$WORK/c_ok.txt'"
    chk "(a) leaves the configured prefix alone" "[ \"\$(c_field c_ok PREFIX)\" = /tmp/pycache ]"
    chk "(a) bytecode stays ON" "[ \"\$(c_field c_ok DWB)\" = '<unset>' ]"
    chk "(a) python still imports" "grep -q 'IMPORT_OK' '$WORK/c_ok.txt'"

    # (b) the false green, and the case that used to kill the cell: -c and NO
    # bind. The session tmpfs has its own st_dev so the overlay rule misses it;
    # only the capacity rule sees the 64 MiB. This exited 74 before.
    "$APPTAINER_BIN" exec -c --cleanenv --writable-tmpfs \
        --env PYTHONPYCACHEPREFIX="/tmp/pycache" \
        "${cbind[@]}" "$SIF" bash /opt/inner.sh >"$WORK/c_sess.txt" 2>&1; rc=$?
    chk "(b) unbound /tmp under -c -> rc 0, NOT the old 74 (got $rc)" "[ $rc -eq 0 ]"
    chk "(b) rule=capacity is still detected, just not fatally" \
        "grep -q 'rule=capacity' '$WORK/c_sess.txt'"
    chk "(b) it warns loudly" "grep -q 'WARNING' '$WORK/c_sess.txt'"
    chk "(b) a repair is actually in force, not just a message" \
        "[ \"\$(c_field c_sess DWB)\" = 1 ] || [ \"\$(c_field c_sess PREFIX)\" != /tmp/pycache ]"
    chk "(b) THE CELL SURVIVES: python imports after the degrade" \
        "grep -q 'IMPORT_OK' '$WORK/c_sess.txt'"
    chk "(b) and the interpreter confirms it writes no bytecode" \
        "grep -q 'IMPORT_OK dwb=True' '$WORK/c_sess.txt'"

    # (c) straight onto the container root overlay -- host /var/tmp is visible
    # here (no -c), so this is the shape where relocation has somewhere to go.
    "$APPTAINER_BIN" exec --writable-tmpfs \
        --env PYTHONPYCACHEPREFIX="/opt/pycache_on_overlay" \
        "${cbind[@]}" "$SIF" bash /opt/inner.sh >"$WORK/c_root.txt" 2>&1; rc=$?
    chk "(c) prefix on the container root -> rc 0, NOT the old 73 (got $rc)" "[ $rc -eq 0 ]"
    chk "(c) rule=container-root is still detected" \
        "grep -q 'rule=container-root' '$WORK/c_root.txt'"
    chk "(c) action=relocate: it moved the cache off the overlay" \
        "grep -q 'action=relocate' '$WORK/c_root.txt'"
    chk "(c) the exported prefix is the relocated one" \
        "[ \"\$(c_field c_root PREFIX)\" != /opt/pycache_on_overlay ]"
    chk "(c) and python honoured it" \
        "grep -q \"IMPORT_OK dwb=False prefix=\$(c_field c_root PREFIX)\" '$WORK/c_root.txt'"

    # (d) the one fatal: a broken mount table. No writable directory anywhere --
    # read-only /tmp and /var/tmp, no home, no writable container root. Nothing
    # this exec is about to run could finish, so 70 is the honest answer.
    "$APPTAINER_BIN" exec -c --cleanenv --no-home \
        --bind "$WORK/ro:/tmp:ro" --bind "$WORK/ro:/var/tmp:ro" \
        --env PYTHONPYCACHEPREFIX="/tmp/pycache" \
        "${cbind[@]}" "$SIF" bash /opt/inner.sh >"$WORK/c_dead.txt" 2>&1; rc=$?
    chk "(d) no writable scratch anywhere -> rc 70 (got $rc)" "[ $rc -eq 70 ]"
    chk "(d) verdict=fatal rule=no-writable-scratch" \
        "grep -q 'verdict=fatal rule=no-writable-scratch' '$WORK/c_dead.txt'"
    chk "(d) it names every path it tried" \
        "grep -q '.ptb_pycache' '$WORK/c_dead.txt'"

    # (e) the same broken shape with ONE writable directory added back. This is
    # the line between the two answers: a bad cache degrades, only a node with
    # nowhere to write at all is fatal.
    "$APPTAINER_BIN" exec -c --cleanenv --no-home \
        --bind "$WORK/ro:/tmp:ro" --bind "$WORK/ctmp:/var/tmp" \
        --env PYTHONPYCACHEPREFIX="/tmp/pycache" \
        "${cbind[@]}" "$SIF" bash /opt/inner.sh >"$WORK/c_one.txt" 2>&1; rc=$?
    chk "(e) one writable dir is enough to stop being fatal -> rc 0 (got $rc)" "[ $rc -eq 0 ]"
    chk "(e) verdict=degraded, not fatal" "grep -q 'verdict=degraded' '$WORK/c_one.txt'"
    chk "(e) THE CELL SURVIVES" "grep -q 'IMPORT_OK' '$WORK/c_one.txt'"

    for t in c_ok c_sess c_root c_dead c_one; do
        echo "    --- $t"; sed 's/^/    | /' "$WORK/$t.txt"
    done
fi
echo
[ "$skipped" = 0 ] || echo "$skipped case(s) SKIPPED -- read them, a skip is not a pass."
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
