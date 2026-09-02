#!/bin/bash
# Runs claude_vertex_high_awm/solve.sh outside a sandbox against stubs: a fake
# checkout whose `awm sandbox setup` records its arguments, a fake claude that
# records its arguments and stdin, and a fake update_agent_cli.sh. Also checks
# the extra-binds block of run_task.sh in isolation.
#
#   bash agents/claude_vertex_high_awm/test_solve.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf -- "$T"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# a fake sandbox home with a fake checkout
mkdir -p "$T/home/task" "$T/home/awm/awm" "$T/bin"
cat > "$T/home/awm/awm/__init__.py" <<'PY'
PY
cat > "$T/home/awm/awm/cli.py" <<'PY'
import os, sys
def main():
    if sys.argv[1:3] == ["sandbox", "setup"]:
        with open(os.path.join(os.getcwd(), "setup_args.txt"), "w") as f:
            f.write(" ".join(sys.argv[3:]) + "\n")
        if "--exp-protocol" in sys.argv[3:]:
            skill = os.path.join(os.getcwd(), ".claude", "skills", "exp_protocol")
            os.makedirs(skill, exist_ok=True)
            with open(os.path.join(skill, "SKILL.md"), "w") as f:
                f.write("# Experiment protocol\n")
        return 0
    return 2
if __name__ == "__main__":
    raise SystemExit(main())
PY
cat > "$T/home/update_agent_cli.sh" <<'SH'
#!/bin/bash
echo "version: stub" > "$HOME/cli_version.txt"
SH
cat > "$T/bin/claude" <<'SH'
#!/bin/bash
printf '%s\n' "$@" > "${CLAUDE_STUB_ARGS}"
cat > "${CLAUDE_STUB_STDIN}"
echo '{"type":"result","stub":true}'
SH
chmod +x "$T/bin/claude"

export PATH="$T/bin:$PATH"
export HOME="$T/home"
export PTB_SANDBOX_HOME="$T/home"
export CLAUDE_STUB_ARGS="$T/claude.args" CLAUDE_STUB_STDIN="$T/claude.stdin"
export CLAUDE_CODE_USE_VERTEX=1 ANTHROPIC_VERTEX_PROJECT_ID=p ANTHROPIC_VERTEX_REGION=r
export AGENT_CONFIG="claude-opus-5[1m]" PROMPT="post-train the model"
export AWM_SANDBOX_SETUP="--exp-protocol --tool claude" AWM_CHECKOUT_SHA="0123abcd"

# happy path
(cd "$T/home/task" && bash "$HERE/solve.sh" > "$T/solve.out" 2>&1) || { cat "$T/solve.out"; fail "solve.sh exited non-zero"; }
grep -q -- "--target $T/home/task --sha 0123abcd --exp-protocol --tool claude" "$T/home/task/setup_args.txt" \
    || fail "setup step did not receive the launcher's arguments: $(cat "$T/home/task/setup_args.txt")"
grep -qx -- "--setting-sources" "$T/claude.args" && grep -qx -- "project" "$T/claude.args" \
    || fail "claude was not started with --setting-sources project"
grep -qx -- "claude-opus-5\[1m\]" "$T/claude.args" || fail "claude did not get --model \$AGENT_CONFIG"
grep -qx -- "--effort" "$T/claude.args" && grep -qx -- "high" "$T/claude.args" || fail "claude did not get --effort high"
grep -q "MANDATORY SCIENTIST BOOTSTRAP" "$T/claude.stdin" || fail "protocol bootstrap was not prepended"
grep -q 'first tool/skill action MUST be to invoke the `exp_protocol` skill' "$T/claude.stdin" \
    || fail "bootstrap does not require exp_protocol as the first action"
grep -q 'successful `awm exp_protocol lock`' "$T/claude.stdin" \
    || fail "bootstrap does not gate training on a successful lock"
grep -q "PTB TASK:" "$T/claude.stdin" && grep -q "post-train the model" "$T/claude.stdin" \
    || fail "the original PTB prompt did not follow the bootstrap"
[ -x "$T/home/.local/bin/awm" ] || fail "awm entry script not written"
grep -q "sha=0123abcd" "$T/solve.out" || fail "checkout sha not echoed"
grep -q "version: stub" "$T/home/cli_version.txt" || fail "update_agent_cli.sh was not run"

# matched null control: same scaffold without an installed protocol must not
# mention the protocol or alter PTB's prompt.
rm -rf -- "$T/home/task/.claude"
export AWM_SANDBOX_SETUP="--tool claude"
: > "$T/claude.stdin"
(cd "$T/home/task" && bash "$HERE/solve.sh" > "$T/null.out" 2>&1) \
    || { cat "$T/null.out"; fail "null-control solve.sh exited non-zero"; }
[ "$(cat "$T/claude.stdin")" = "post-train the model" ] \
    || fail "null-control prompt was altered: $(cat "$T/claude.stdin")"
! grep -q "exp_protocol" "$T/claude.stdin" || fail "null control was told about exp_protocol"

# restore the protocol setup for the refusal cases below.
export AWM_SANDBOX_SETUP="--exp-protocol --tool claude"

# refusals
( unset AWM_SANDBOX_SETUP; cd "$T/home/task" && bash "$HERE/solve.sh" >/dev/null 2>&1 ) && fail "ran without AWM_SANDBOX_SETUP" || true
( export PTB_SANDBOX_HOME="$T/nowhere"; mkdir -p "$T/nowhere/task"; cd "$T/nowhere/task" && bash "$HERE/solve.sh" >/dev/null 2>&1 ) && fail "ran without a checkout" || true

# the run_task.sh extra-binds block, extracted and run on its own
bash -n "$REPO_ROOT/src/run_task.sh" || fail "run_task.sh does not parse"
BLOCK="$(sed -n '/# --- extra binds (begin) ---/,/# --- extra binds (end) ---/p' "$REPO_ROOT/src/run_task.sh")"
[ -n "$BLOCK" ] || fail "extra-binds block not found in run_task.sh"
grep -q '"${EXTRA_BIND_ARGS\[@\]}"' "$REPO_ROOT/src/run_task.sh" || fail "EXTRA_BIND_ARGS not spliced into apptainer exec"
mkdir -p "$T/src1" "$T/src2"
# the block ends in a comment line, so the probe goes on a line of its own
OUT="$(POST_TRAIN_BENCH_EXTRA_BINDS="$T/src1:/home/ben/awm:ro,$T/src2:/home/ben/data" bash -c "${BLOCK}"$'\n'"printf '%s\n' \"\${EXTRA_BIND_ARGS[@]}\"")"
[ "$OUT" = "$(printf 'Extra sandbox binds: %s:/home/ben/awm:ro,%s:/home/ben/data\n--bind\n%s:/home/ben/awm:ro\n--bind\n%s:/home/ben/data' "$T/src1" "$T/src2" "$T/src1" "$T/src2")" ] \
    || fail "extra binds parsed wrongly: $OUT"
OUT="$(bash -c "${BLOCK}"$'\n'"echo n=\${#EXTRA_BIND_ARGS[@]}")"
[ "$OUT" = "n=0" ] || fail "unset POST_TRAIN_BENCH_EXTRA_BINDS should add nothing: $OUT"
if POST_TRAIN_BENCH_EXTRA_BINDS="$T/missing:/home/ben/x" bash -c "$BLOCK" >/dev/null 2>&1; then fail "a missing bind source was accepted"; fi

echo "PASS: claude_vertex_high_awm/solve.sh and the run_task.sh extra-binds block"
