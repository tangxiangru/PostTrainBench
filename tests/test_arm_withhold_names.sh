#!/bin/bash
# Assert that every `--withhold-skills` name an autor arm passes is a skill that
# arm's own payload actually has.
#
# Worth a file because the flag FAILS OPEN by design. `read_withheld_skills`
# says so in its own docstring: "Unknown names are not an error here ... a name
# matching nothing withholds nothing". That is the right behaviour for the flag
# and a trap for an experiment. A control arm built by withholding a renamed or
# mistyped skill installs the same 49 skills as the reference arm, runs for ten
# hours on eight H100s, scores the same, and reports a clean null for a
# mechanism that was never removed. Nothing in the run log looks wrong -- the
# arm did exactly what it was told -- so the only place to catch it is here,
# before the campaign is submitted.
#
# Checks the value bash itself expands from `AUTOR_EXTRA_FLAGS`, not a regex
# over the source line: the array's closing paren is one character away from
# being read as part of the skill name, and a check that makes that mistake
# reports a false failure on a correct arm.
#
# `agents/*/payload/` is gitignored -- the payload is installed, not committed
# -- so an arm with no payload on disk is SKIPPED, not failed. On a fresh clone
# this file passes with everything skipped; on the submit host it checks.
#
# Usage: bash tests/test_arm_withhold_names.sh
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS_DIR="${PTB_AGENTS_DIR:-$REPO_ROOT/agents}"

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }
note() { echo "       $*"; }

echo "agents dir under test: $AGENTS_DIR"
[ -d "$AGENTS_DIR" ] || { echo "FAIL: no such directory: $AGENTS_DIR" >&2; exit 1; }

# The value the flag really carries, obtained by letting bash expand the array
# assignment in a subshell. Only that one line is evaluated; the rest of
# solve.sh (which launches an agent) is never run.
withheld_names_of() {  # withheld_names_of <solve.sh>
    local line
    line="$(grep -m1 '^AUTOR_EXTRA_FLAGS=' "$1")" || return 0
    [ -n "$line" ] || return 0
    (
        eval "$line" 2>/dev/null || exit 0
        local tok
        for tok in "${AUTOR_EXTRA_FLAGS[@]:-}"; do
            case "$tok" in
                --withhold-skills=*) printf '%s\n' "${tok#--withhold-skills=}" | tr ',' '\n' ;;
            esac
        done
    )
}

# The names that arm's installed payload answers to. Read through the pack
# reader rather than `ls src/skills`, because the pack is what `withheld` is
# matched against and a directory without a readable SKILL.md is not an entry.
pack_names_of() {  # pack_names_of <payload dir>
    ( cd "$1" && python3 -B -c '
import pathlib, sys
sys.path.insert(0, ".")
from src.run_skills import read_skill_pack
for e in read_skill_pack(pathlib.Path("src/skills")):
    print(e.name)
' 2>/dev/null )
}

echo "[1] every withheld name is a live entry in that arm's own pack"
checked=0
skipped=0
for solve in "$AGENTS_DIR"/*/solve.sh; do
    [ -f "$solve" ] || continue
    arm="$(basename "$(dirname "$solve")")"
    mapfile -t want < <(withheld_names_of "$solve")
    [ "${#want[@]}" -gt 0 ] || continue

    payload="$(dirname "$solve")/payload"
    if [ ! -d "$payload/src/skills" ]; then
        note "SKIP $arm -- no payload installed"
        skipped=$((skipped + 1))
        continue
    fi
    mapfile -t have < <(pack_names_of "$payload")
    if [ "${#have[@]}" -eq 0 ]; then
        echo "  FAIL $arm: pack reader returned nothing for $payload"
        fail=1
        continue
    fi

    for one in "${want[@]}"; do
        [ -n "$one" ] || continue
        checked=$((checked + 1))
        if printf '%s\n' "${have[@]}" | grep -qxF -- "$one"; then
            echo "  PASS $arm withholds a real skill: $one"
        else
            echo "  FAIL $arm withholds a name its pack does not have: $one"
            note "this arm is silently identical to the no-flag reference"
            fail=1
        fi
    done
done
note "checked $checked name(s), skipped $skipped arm(s) with no payload"

# A check that never fires is not a check. If every arm was skipped above, the
# section passed without comparing anything, and the negative control below is
# the only evidence the comparison works at all -- so it always runs.
echo "[2] the comparison can fail (negative control)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-withhold.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/arm"
cat > "$WORK/arm/solve.sh" <<'EOF'
#!/bin/bash
AUTOR_EXTRA_FLAGS=(--withhold-skills=no-such-skill-anywhere,price-every-full-read-in-the-steps-it-displaces)
echo "this line must never run"
EOF
mapfile -t got < <(withheld_names_of "$WORK/arm/solve.sh")
chk "extracts both names from one flag" '[ "${#got[@]}" = 2 ]'
chk "and strips the array's closing paren" \
    '[ "${got[1]}" = "price-every-full-read-in-the-steps-it-displaces" ]'
note "got: ${got[*]-<none>}"
# Evaluating the assignment must not execute the rest of solve.sh.
chk "does not run the arm's body" \
    '! withheld_names_of "$WORK/arm/solve.sh" | grep -q "must never run"'
# And the membership test the loop uses must reject the bogus name.
real="price-every-full-read-in-the-steps-it-displaces"
chk "rejects a name no pack has" \
    '! printf "%s\n" "$real" | grep -qxF -- "no-such-skill-anywhere"'
chk "accepts the name that is real" \
    'printf "%s\n" "$real" | grep -qxF -- "$real"'

# An arm with no --withhold-skills at all must contribute nothing, not an empty
# string that then fails the membership test.
echo "[3] an arm without the flag is not checked"
cat > "$WORK/arm/plain.sh" <<'EOF'
#!/bin/bash
AUTOR_EXTRA_FLAGS=(--dose-gate --deliverable-gate)
EOF
mapfile -t none < <(withheld_names_of "$WORK/arm/plain.sh")
chk "no names from a gate-only arm" '[ "${#none[@]}" = 0 ]'
cat > "$WORK/arm/empty.sh" <<'EOF'
#!/bin/bash
AUTOR_EXTRA_FLAGS=()
EOF
mapfile -t nada < <(withheld_names_of "$WORK/arm/empty.sh")
chk "no names from the reference arm"  '[ "${#nada[@]}" = 0 ]'
mapfile -t absent < <(withheld_names_of "/dev/null")
chk "no names when the line is absent" '[ "${#absent[@]}" = 0 ]'

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
