#!/bin/bash
# Exercise the arm -> GPU-seat assignment in ptb_ops/ptb_pack.sbatch.
#
# Worth having as a file rather than a code read, because the defect it guards
# against was invisible in every log the pack ever wrote. `arm="${ARMS[$gpu]}"`
# is correct code; it fails only across jobs, when the operator types the arms
# in the same order every time. Across the whole published ten-hour contrast
# `sn` drew g0/g2/g4/g6 and `r2` drew g1/g3/g5/g7, so the arm effect and the
# seat effect (NVLink neighbour, inlet temperature, `sleep $((gpu*STAGGER))`,
# NUMA node, scratch bandwidth) were the same number. Nothing in a single job
# log looks wrong, which is why this has to be asserted over MANY job ids.
#
# The functions are extracted from the launcher and sourced, so the test
# exercises the shipped code and not a copy of it. No job is launched and no
# GPU is touched.
#
# Usage: bash tests/test_pack_seat_map.sh
#        PTB_PACK_SBATCH=/path/to/other/ptb_pack.sbatch bash tests/...
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH="${PTB_PACK_SBATCH:-$REPO_ROOT/ptb_ops/ptb_pack.sbatch}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ptb-seattest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  PASS $1"; else echo "  FAIL $1"; fail=1; fi; }
note() { echo "       $*"; }

echo "sbatch under test: $SBATCH"
[ -f "$SBATCH" ] || { echo "FAIL: no such file: $SBATCH" >&2; exit 1; }

echo "[1] the launcher parses, and no longer maps arm -> card by argument order"
p_old='ARMS[$gpu]'
p_launch='arm="${PTB_SEATS[$gpu]}"'
p_reap='cell_exit gpu=${gpu} arm=${PTB_SEATS[$gpu]}'
p_seat='echo "seat gpu='
p_flat='echo "seat_map'
chk "bash -n" 'bash -n "$SBATCH"'
# Comment lines may quote the old idiom; live code may not.
chk "no live ARMS[\$gpu] indexing" \
    '! grep -vE "^[[:space:]]*#" "$SBATCH" | grep -qF -- "$p_old"'
chk "launch loop reads PTB_SEATS"  'grep -qF -- "$p_launch" "$SBATCH"'
chk "reap loop reads PTB_SEATS"    'grep -qF -- "$p_reap" "$SBATCH"'
# The realised map has to reach the job log in a shape a later script can parse.
chk "prints a per-seat line"       'grep -qF -- "$p_seat" "$SBATCH"'
chk "prints a flat seat_map line"  'grep -qF -- "$p_flat" "$SBATCH"'
chk "prints the seed"              'grep -qF -- "seat_seed=" "$SBATCH"'

# Extract the block between the sentinels and source it.
sed -n '/^# >>> ptb seat assignment >>>$/,/^# <<< ptb seat assignment <<<$/p' \
    "$SBATCH" > "$WORK/seat.sh"
for f in ptb_seat_shuffle ptb_seat_confound ptb_seat_plan; do
    if ! grep -q "^${f}() {" "$WORK/seat.sh"; then
        echo "  FAIL could not extract ${f} from $SBATCH" >&2
        echo; echo "SOME TESTS FAILED"; exit 1
    fi
done
# shellcheck disable=SC1090
source "$WORK/seat.sh"

# Helpers. plan() runs ptb_seat_plan with a clean environment for the two knobs
# and reports the outcome through globals, exactly as the launcher sees it.
plan() {  # plan <seed> <arm>...
    unset PTB_SEATS PTB_SEAT_POS
    ptb_seat_plan "$@"
}
seats_of() { printf '%s' "${PTB_SEATS[*]}"; }

echo "[2] the draw is deterministic and recoverable"
unset PTB_SEAT_MAP
plan "77001|sn r2 sn r2 sn r2 sn r2" sn r2 sn r2 sn r2 sn r2 ; m1="$(seats_of)"
plan "77001|sn r2 sn r2 sn r2 sn r2" sn r2 sn r2 sn r2 sn r2 ; m2="$(seats_of)"
plan "77002|sn r2 sn r2 sn r2 sn r2" sn r2 sn r2 sn r2 sn r2 ; m3="$(seats_of)"
plan "77001|sn r2 sn r2 sn r2 sn r2" sn r2 sn r2 sn r2 sn r2 ; m4="$(seats_of)"
chk "same job id -> same map (twice)"      '[ "$m1" = "$m2" ]'
chk "same job id -> same map (after m3)"   '[ "$m1" = "$m4" ]'
chk "another job id -> another map"        '[ "$m1" != "$m3" ]'
# Same map out of a fresh interpreter: the draw must not depend on shell state,
# or a later script cannot recompute what a log line says happened.
m5="$(bash -c 'set -u; source "$1"; ptb_seat_plan "77001|sn r2 sn r2 sn r2 sn r2" \
        sn r2 sn r2 sn r2 sn r2 >/dev/null; printf "%s" "${PTB_SEATS[*]}"' _ "$WORK/seat.sh")"
chk "same map from a fresh interpreter"    '[ "$m1" = "$m5" ]'
note "job 77001 -> [$m1]"
note "job 77002 -> [$m3]"

echo "[3] over many job ids every arm gets both parities and both halves"
declare -A seen_seat=() distinct=()
bad_bal=0; ncase=0
for (( j=90000; j<90200; j++ )); do
    plan "${j}|sn r2 sn r2 sn r2 sn r2" sn r2 sn r2 sn r2 sn r2 || { bad_bal=1; break; }
    ncase=$(( ncase + 1 ))
    ptb_seat_confound 8 "${PTB_SEATS[@]}" >/dev/null || bad_bal=1
    distinct["$(seats_of)"]=1
    for (( g=0; g<8; g++ )); do seen_seat["${PTB_SEATS[$g]}:${g}"]=1; done
done
chk "200 job ids all planned"           '[ "$ncase" -eq 200 ]'
chk "no realised map is confounded"     '[ "$bad_bal" -eq 0 ]'
miss=""
for a in sn r2; do
    for (( g=0; g<8; g++ )); do [ -n "${seen_seat[$a:$g]:-}" ] || miss="${miss} ${a}:g${g}"; done
done
chk "each arm reached all 8 seats"      '[ -z "$miss" ]'
[ -z "$miss" ] || note "never seen:${miss}"
chk "more than 20 distinct maps drawn"  '[ "${#distinct[@]}" -gt 20 ]'
note "${#distinct[@]} distinct maps over 200 job ids"
# The old behaviour is a single map repeated 200 times; this is the assertion
# that would have caught it even if everything else above were satisfied.
chk "not one map repeated forever"      '[ "${#distinct[@]}" -gt 1 ]'

echo "[4] the drawn map is balanced for every pack shape, not just 4+4"
for spec in "a a a a b b b b" "a a b b c c d d" "a a a b b b c c" "ctl x1 x2 x3 x4 x5 x6 x7" "a a b b"; do
    read -r -a _arms <<< "$spec"
    bad=0; n="${#_arms[@]}"
    for (( j=1000; j<1050; j++ )); do
        plan "${j}|${spec}" "${_arms[@]}" || { bad=1; break; }
        ptb_seat_confound "$n" "${PTB_SEATS[@]}" >/dev/null || bad=1
    done
    chk "[$spec] 50 job ids, all balanced" '[ "$bad" -eq 0 ]'
done
# Below four seats there is nothing to balance, and a one-seat arm has no
# spread; both must still plan rather than refuse.
plan "1|a b" a b; rc=$?
chk "2-seat pack plans"    '[ "$rc" -eq 0 ] && [ "${#PTB_SEATS[@]}" -eq 2 ]'
plan "1|a b c" a b c; rc=$?
chk "3-seat pack plans"    '[ "$rc" -eq 0 ] && [ "${#PTB_SEATS[@]}" -eq 3 ]'
plan "1|a" a; rc=$?
chk "1-seat pack plans"    '[ "$rc" -eq 0 ] && [ "${PTB_SEATS[0]}" = "a" ]'

# A refusal that can fire on a legitimate pack is worse than the confound: it
# throws away an eight-GPU allocation for nothing. Every pack shape with
# 1 <= n <= 8 -- all 66 of them -- must plan, on every job id tried.
shapes="$(python3 - <<'PY'
def parts(n, mx=None):
    mx = n if mx is None else mx
    if n == 0:
        yield []
        return
    for k in range(min(n, mx), 0, -1):
        for r in parts(n - k, k):
            yield [k] + r
for n in range(1, 9):
    for p in parts(n):
        arms = []
        for i, c in enumerate(p):
            arms += [chr(97 + i)] * c
        print(" ".join(arms))
PY
)"
nshape=0; bad_shape=""
while read -r spec; do
    [ -n "$spec" ] || continue
    read -r -a _arms <<< "$spec"
    n="${#_arms[@]}"; nshape=$(( nshape + 1 ))
    for (( j=500; j<503; j++ )); do
        plan "${j}|${spec}" "${_arms[@]}" || { bad_shape="${bad_shape} [${spec}]:refused"; break; }
        ptb_seat_confound "$n" "${PTB_SEATS[@]}" >/dev/null \
            || bad_shape="${bad_shape} [${spec}]:confounded"
    done
done <<< "$shapes"
chk "all 66 pack shapes enumerated" '[ "$nshape" -eq 66 ]'
chk "no legitimate shape is refused" '[ -z "$bad_shape" ]'
[ -z "$bad_shape" ] || note "offenders:${bad_shape}"

echo "[5] a pinned map is honoured when it is balanced"
export PTB_SEAT_MAP="sn r2 r2 sn sn r2 r2 sn"
plan "77001|x" sn sn sn sn r2 r2 r2 r2; rc=$?
chk "accepted"                 '[ "$rc" -eq 0 ]'
chk "realised exactly as pinned" '[ "$(seats_of)" = "sn r2 r2 sn sn r2 r2 sn" ]'
chk "source recorded as pinned"  '[ "$PTB_SEAT_SOURCE" = "pinned" ]'
chk "typed_pos is a permutation" '[ "$(printf "%s\n" "${PTB_SEAT_POS[@]}" | sort -n | tr "\n" " ")" = "0 1 2 3 4 5 6 7 " ]'
export PTB_SEAT_MAP="sn,r2,r2,sn,sn,r2,r2,sn"
plan "77001|x" sn sn sn sn r2 r2 r2 r2
chk "comma form parses the same"  '[ "$(seats_of)" = "sn r2 r2 sn sn r2 r2 sn" ]'

echo "[6] a pinned map that reproduces the confound is REFUSED"
# This is the exact published pattern: sn on the even cards, r2 on the odd ones.
export PTB_SEAT_MAP="sn r2 sn r2 sn r2 sn r2"
plan "77001|x" sn sn sn sn r2 r2 r2 r2; rc=$?
chk "refused"                    '[ "$rc" -ne 0 ]'
chk "refusal is the confound one" '[ "$PTB_SEAT_REFUSAL_KIND" = "confound" ]'
chk "names sn"                   '[[ "$PTB_SEAT_REFUSAL" == *"sn:every-seat-even"* ]]'
chk "names r2"                   '[[ "$PTB_SEAT_REFUSAL" == *"r2:every-seat-odd"* ]]'
note "refusal: $PTB_SEAT_REFUSAL"
export PTB_SEAT_MAP="sn sn sn sn r2 r2 r2 r2"
plan "77001|x" sn sn sn sn r2 r2 r2 r2; rc=$?
chk "one-half-of-the-chassis refused" '[ "$rc" -ne 0 ] && [ "$PTB_SEAT_REFUSAL_KIND" = "confound" ]'
chk "names the half-plane"       '[[ "$PTB_SEAT_REFUSAL" == *half* ]]'
export PTB_SEAT_MAP="a b a b"
plan "77001|x" a a b b; rc=$?
chk "4-seat parity pin refused"  '[ "$rc" -ne 0 ] && [ "$PTB_SEAT_REFUSAL_KIND" = "confound" ]'
export PTB_SEAT_MAP="a b b a"
plan "77001|x" a a b b; rc=$?
chk "4-seat balanced pin accepted" '[ "$rc" -eq 0 ]'

echo "[7] a malformed pin is refused, and not as a confound"
export PTB_SEAT_MAP="sn r2 sn"
plan "77001|x" sn sn r2 r2; rc=$?
chk "wrong length refused"       '[ "$rc" -ne 0 ] && [ "$PTB_SEAT_REFUSAL_KIND" = "malformed" ]'
export PTB_SEAT_MAP="sn r2 r2 typo"
plan "77001|x" sn sn r2 r2; rc=$?
chk "non-permutation refused"    '[ "$rc" -ne 0 ] && [ "$PTB_SEAT_REFUSAL_KIND" = "malformed" ]'
note "refusal: $PTB_SEAT_REFUSAL"
unset PTB_SEAT_MAP

echo "[8] the confound predicate itself"
chk "8/4+4 alternating is confounded" '! ptb_seat_confound 8 a b a b a b a b'
chk "8/4+4 split-half is confounded"  '! ptb_seat_confound 8 a a a a b b b b'
chk "8/4+4 mixed is clean"            'ptb_seat_confound 8 a b b a b a a b'
chk "8 singleton arms are clean"      'ptb_seat_confound 8 a b c d e f g h'
chk "below 4 seats is exempt"         'ptb_seat_confound 2 a b'
chk "3 seats is exempt"               'ptb_seat_confound 3 a a b'
chk "4 seats is not exempt"           '! ptb_seat_confound 4 a b a b'

echo "[9] the launcher's own stanza: refuses with a non-zero exit, logs the map"
# The functions being right is half of it; the other half is that the launcher
# calls them, exits before any cell is started, and prints something parseable.
# Run the caller stanza verbatim, with the arrays it expects and nothing else --
# no allocation, no apptainer, no GPU.
{ echo 'set -uo pipefail'
  cat "$WORK/seat.sh"
  echo 'ARMS=(sn sn sn sn r2 r2 r2 r2); N=8'
  sed -n '/^SEAT_SEED=/,/^echo "seat_source=/p' "$SBATCH"
} > "$WORK/caller.sh"
chk "caller stanza extracted" 'grep -q "^if ! ptb_seat_plan" "$WORK/caller.sh"'

out="$(env -u PTB_SEAT_MAP -u PTB_ALLOW_SEAT_CONFOUND SLURM_JOB_ID=77001 \
        bash "$WORK/caller.sh" 2>"$WORK/err")"; rc=$?
chk "clean pack exits 0"        '[ "$rc" -eq 0 ]'
chk "8 per-seat lines logged"   '[ "$(grep -c "^seat gpu=" <<< "$out")" -eq 8 ]'
chk "one flat seat_map line"    '[ "$(grep -c "^seat_map " <<< "$out")" -eq 1 ]'
chk "flat line names 8 seats"   '[ "$(grep "^seat_map " <<< "$out" | grep -o "g[0-7]=" | wc -l)" -eq 8 ]'
chk "logs source and seed"      'grep -q "^seat_source=drawn seat_draws=[0-9]* seat_confound=none seat_seed=77001|" <<< "$out"'
# The map in the log must be the map the code computes -- that is what makes the
# log line a record and not decoration.
logged="$(grep "^seat_map " <<< "$out" | sed "s/^seat_map //; s/g[0-7]=//g")"
unset PTB_SEAT_MAP
plan "77001|sn sn sn sn r2 r2 r2 r2" sn sn sn sn r2 r2 r2 r2
chk "logged map == recomputed map" '[ "$logged" = "$(seats_of)" ]'
note "seat_map $(grep "^seat_map " <<< "$out" | sed "s/^seat_map //")"

env -u PTB_ALLOW_SEAT_CONFOUND PTB_SEAT_MAP="sn r2 sn r2 sn r2 sn r2" SLURM_JOB_ID=77001 \
    bash "$WORK/caller.sh" >"$WORK/o2" 2>"$WORK/e2"; rc=$?
chk "confounded pin exits non-zero" '[ "$rc" -ne 0 ]'
chk "and says FATAL on stderr"      'grep -q "^FATAL: PTB_SEAT_MAP reproduces" "$WORK/e2"'
chk "and starts no cell"            '! grep -q "^launched " "$WORK/o2"'
PTB_SEAT_MAP="sn r2 sn r2 sn r2 sn r2" PTB_ALLOW_SEAT_CONFOUND=1 SLURM_JOB_ID=77001 \
    bash "$WORK/caller.sh" >"$WORK/o3" 2>"$WORK/e3"; rc=$?
chk "the named override continues"  '[ "$rc" -eq 0 ]'
chk "and marks the log confounded"  'grep -q "seat_confound=confound" "$WORK/o3"'
# The escape hatch is for a deliberate confound, not for a typo.
PTB_SEAT_MAP="sn r2 sn" PTB_ALLOW_SEAT_CONFOUND=1 SLURM_JOB_ID=77001 \
    bash "$WORK/caller.sh" >/dev/null 2>&1; rc=$?
chk "override does not rescue a malformed pin" '[ "$rc" -ne 0 ]'

echo "[10] one realised map, as evidence"
unset PTB_SEAT_MAP
plan "89999|claude_autor_ctl claude_autor_sn claude_autor_r2 claude_autor_ctl claude_autor_sn claude_autor_r2 claude_autor_sn claude_autor_r2" \
     claude_autor_ctl claude_autor_sn claude_autor_r2 claude_autor_ctl \
     claude_autor_sn claude_autor_r2 claude_autor_sn claude_autor_r2
for (( g=0; g<8; g++ )); do echo "       seat gpu=${g} arm=${PTB_SEATS[$g]} typed_pos=${PTB_SEAT_POS[$g]}"; done
echo "       seat_source=${PTB_SEAT_SOURCE} seat_draws=${PTB_SEAT_DRAWS}"

echo "[11] the superseded launcher is the same confound by a second door"
# ptb_pack6.sbatch still contains `arm="${ARMS[$gpu]}"` and always will -- it is not being
# rewritten, it is being closed. Fixing the map in one launcher while a copy of the old one
# stays submittable fixes nothing, and this is not hypothetical: job 82648 was launched
# from a copy of that file. Only its refusal is executed here; the rest of it allocates
# scratch, reads GPUs and starts cells, none of which belongs in a test.
PACK6="$REPO_ROOT/ptb_ops/ptb_pack6.sbatch"
if [ ! -f "$PACK6" ]; then
    note "no ptb_pack6.sbatch in this tree -- the second door is already gone"
else
    chk "bash -n" 'bash -n "$PACK6"'
    chk "it does still pin arm to typed position" \
        'grep -vE "^[[:space:]]*#" "$PACK6" | grep -qF -- "$p_old"'
    bash "$PACK6" a b c d e f >"$WORK/p6o" 2>"$WORK/p6e"; rc=$?
    chk "so submitting it refuses"        '[ "$rc" -ne 0 ]'
    chk "and says FATAL on stderr"        'grep -q "^FATAL:" "$WORK/p6e"'
    chk "and names its replacement"       'grep -q "ptb_pack.sbatch" "$WORK/p6e"'
    # A refusal that leaves no way through is a peer's blocked afternoon, so the hatch has
    # to exist, be named in the message, and be greppable from a job log afterwards.
    chk "and names an escape hatch"       'grep -q "PTB_ALLOW_PACK6=1" "$WORK/p6e"'
    chk "which the file honours"          'grep -q "PTB_ALLOW_PACK6:-0" "$PACK6"'
    chk "and which marks its own log"     'grep -q "seat_confound=arm_equals_seat" "$PACK6"'
    # It refuses before it touches the node: no scratch, no nvidia-smi, no cell. Asserted
    # on stdout because everything this file does after the refusal announces itself there.
    chk "refusing costs the node nothing"  '[ ! -s "$WORK/p6o" ]'
fi

echo
if [ "$fail" = 0 ]; then echo "ALL TESTS PASSED"; else echo "SOME TESTS FAILED"; fi
exit $fail
