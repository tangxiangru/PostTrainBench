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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-pycachetest.XXXXXX")"
trap 'chmod -R u+w "$WORK" 2>/dev/null; rm -rf "$WORK"' EXIT

RUN_TASK="$REPO_ROOT/src/run_task.sh"
RESCORE="$REPO_ROOT/ptb_ops/ptb_rescore.sbatch"
GREEDY="$REPO_ROOT/ptb_ops/ptb_greedy_board.sbatch"
JUDGE_LIB="$REPO_ROOT/src/judges/judge_lib.sh"
CHECKER="$REPO_ROOT/src/utils/check_pycache.sh"

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
echo "[1] src/utils/check_pycache.sh, driven directly on host filesystems"
# These exercise the predicate's plumbing -- variable handling, ordering of the
# rules, exit codes. Section [5] exercises it where it actually runs.
PYTHONPYCACHEPREFIX="" bash "$CHECKER" >/dev/null 2>&1; rc=$?
chk "unset prefix -> 71 (got $rc)" "[ $rc -eq 71 ]"

mkdir -p "$WORK/ro"; chmod 555 "$WORK/ro"
PYTHONPYCACHEPREFIX="$WORK/ro/sub" bash "$CHECKER" >/dev/null 2>&1; rc=$?
chk "unwritable prefix -> 72 (got $rc)" "[ $rc -eq 72 ]"
chmod 755 "$WORK/ro"

# $WORK is under $TMPDIR, and on this cluster /tmp and / are the same device --
# which is what the rule is about: a prefix on the root filesystem of the thing
# you are running in. Guarded so the case is skipped rather than mis-reported
# where /tmp is its own mount.
if [ "$(stat -c %d "$WORK")" = "$(stat -c %d /)" ]; then
    mkdir -p "$WORK/onroot"
    PYTHONPYCACHEPREFIX="$WORK/onroot" bash "$CHECKER" >/dev/null 2>&1; rc=$?
    chk "prefix on the same st_dev as / -> 73 (got $rc)" "[ $rc -eq 73 ]"
else
    skip "same-st_dev-as-/ case: \$TMPDIR is not on the root filesystem here"
fi

# A directory that satisfies both rules: different device, comfortably over the
# 4 GiB floor. Picked by measurement rather than hardcoded.
BIG=""
for cand in "${PTB_TEST_BIG_DIR:-}" /dev/shm "$HOME" /rmeng_data/robtang; do
    [ -n "$cand" ] && [ -d "$cand" ] && [ -w "$cand" ] || continue
    [ "$(stat -c %d "$cand")" != "$(stat -c %d /)" ] || continue
    set -- $(df -P -k "$cand" 2>/dev/null | tail -1)
    [ "${2:-0}" -ge 4194304 ] 2>/dev/null || continue
    BIG="$cand"; break
done
if [ -n "$BIG" ]; then
    OKDIR="$(mktemp -d "$BIG/ptb-pycacheok.XXXXXX")"
    PYTHONPYCACHEPREFIX="$OKDIR" bash "$CHECKER" >/dev/null 2>&1; rc=$?
    chk "bound, big, writable prefix -> 0 (got $rc, on $BIG)" "[ $rc -eq 0 ]"

    # -X pycache_prefix beats the environment variable. The agent can do this to
    # itself at any point, so the assertion can only speak for the interpreter
    # it asks -- but a wrapper that does it globally must not read as fine.
    cat > "$WORK/bin/python3" <<STUB
#!/bin/bash
exec /usr/bin/python3 -X pycache_prefix="$WORK/elsewhere" "\$@"
STUB
    chmod +x "$WORK/bin/python3"
    PYTHONPYCACHEPREFIX="$OKDIR" bash "$CHECKER" >/dev/null 2>&1; rc=$?
    chk "python3 that ignores the variable -> 75 (got $rc)" "[ $rc -eq 75 ]"
    rm -f "$WORK/bin/python3"
    rm -rf "$OKDIR"
else
    skip "ok/75 cases: found no writable dir off the root fs with >= 4 GiB"
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
    done
done

echo "[2b] the judge exec (src/judges/judge_lib.sh, not owned by this change)"
chk "run_task.sh puts the prefix in JUDGE_EXTRA_APPTAINER_ARGS" \
    "awk '/^JUDGE_EXTRA_APPTAINER_ARGS=\\(/,/^\\)/' '$RUN_TASK' | grep -q 'PYTHONPYCACHEPREFIX=\"/tmp/pycache\"'"
chk "the judge exec binds job_tmp at /tmp, so /tmp/pycache resolves there" \
    "extract_exec_block '$JUDGE_LIB' 2 | grep -q -- '--bind \"\${job_tmp}:/tmp\"'"
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

# The two sandbox path constants are READ OUT of run_task.sh rather than restated here.
# Restating them would make this harness agree with itself while the shipped file said
# something else -- and the whole point of [3d] below is that the prompt's absolute paths
# and this bind are the same two strings.
eval "$(grep -E '^SANDBOX_(HOME|TASK_DIR)=' "$RUN_TASK")"
chk "run_task.sh defines SANDBOX_HOME"      '[ -n "${SANDBOX_HOME:-}" ]'
chk "run_task.sh defines SANDBOX_TASK_DIR"  '[ -n "${SANDBOX_TASK_DIR:-}" ]'

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
chk "agent: prefix is /tmp/pycache"     "[ \"\$(argv_get_prefix '$WORK/argv_agent.txt')\" = /tmp/pycache ]"
chk "agent: prefix is under a --bind"   "argv_prefix_is_bound '$WORK/argv_agent.txt'"
chk "agent: check_pycache.sh runs first" \
    "grep -q 'bash /home/ben/check_pycache.sh && python /home/ben/check_cuda.py' '$WORK/argv_agent.txt'"
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
echo "[5] the predicate inside the real container, in three mount shapes"
# The stub directory is first on PATH, so ask the rest of it. The apt-root copy
# is the one every ptb_ops job prepends -- there is no apptainer in the login
# node's default PATH, and looking only there is how this section turns into a
# permanent silent skip.
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
    skip "  -- sections [1]-[4] ran; the deployment half of this test did NOT."
else
    echo "  using $APPTAINER_BIN and $SIF"
    mkdir -p "$WORK/ctmp/pycache"
    # (a) the agent's real shape: -c --cleanenv, JOB_TMP bound at /tmp.
    "$APPTAINER_BIN" exec -c --cleanenv --writable-tmpfs \
        --bind "$WORK/ctmp:/tmp" --env PYTHONPYCACHEPREFIX="/tmp/pycache" \
        --bind "$CHECKER:/opt/check_pycache.sh" \
        "$SIF" bash /opt/check_pycache.sh >"$WORK/c_ok.txt" 2>&1; rc=$?
    chk "bound /tmp, -c --cleanenv -> 0 (got $rc)" "[ $rc -eq 0 ]"
    # (b) the false green: -c and NO bind. Different st_dev from /, still 64 MiB.
    "$APPTAINER_BIN" exec -c --cleanenv --writable-tmpfs \
        --env PYTHONPYCACHEPREFIX="/tmp/pycache" \
        --bind "$CHECKER:/opt/check_pycache.sh" \
        "$SIF" bash /opt/check_pycache.sh >"$WORK/c_sess.txt" 2>&1; rc=$?
    chk "unbound /tmp under -c (session tmpfs) -> 74 (got $rc)" "[ $rc -eq 74 ]"
    # (c) straight onto the container root overlay.
    "$APPTAINER_BIN" exec --writable-tmpfs \
        --env PYTHONPYCACHEPREFIX="/opt/pycache_on_overlay" \
        --bind "$CHECKER:/opt/check_pycache.sh" \
        "$SIF" bash /opt/check_pycache.sh >"$WORK/c_root.txt" 2>&1; rc=$?
    chk "prefix on the container root -> 73 (got $rc)" "[ $rc -eq 73 ]"
    for t in c_ok c_sess c_root; do sed 's/^/    | /' "$WORK/$t.txt"; done
fi

echo
[ "$skipped" = 0 ] || echo "$skipped case(s) SKIPPED -- read them, a skip is not a pass."
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
