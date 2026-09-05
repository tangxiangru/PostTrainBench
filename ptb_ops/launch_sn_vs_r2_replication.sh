#!/usr/bin/env bash
# Powered replication of the claude_autor_sn vs claude_autor_r2 ten-hour contrast.
#
# WHY THIS EXISTS
# ---------------
# The 2026-09-02 board (packs 89727/89809/89810, job names ptb-sn-vs-r2-p10..p12)
# is a pilot, not a verdict:
#
#   * After two r2 cells were voided on 2026-09-04 for reading the gsm8k answer
#     key it stands at 12 v 10 cells, sn - r2 = +2.39 pt (SE 1.75, p = 0.195,
#     blocked permutation p = 0.173 over 85,750 arrangements).
#   * Pooled sd is 3.91 pt, so its detection floor is
#     2.8 * 3.91 * sqrt(1/12 + 1/10) = 4.7 pt. The observed +2.39 sits *under*
#     that floor: the board cannot tell it from zero.
#   * Every submit line of p10..p12 typed the arms in the same order, so sn ran
#     on g0/g2/g4/g6 and r2 on g1/g3/g5/g7 in all three packs. Arm was 100%
#     aliased with seat parity. A null survives that reading; a positive one
#     does not.
#
# So this campaign is a clean replication, not an extension:
#
#   n per arm for 80% power at the observed 2.39 pt, pooled sd 3.91:
#       n = 2 * (2.8 * 3.91 / 2.39)^2 = 42
#   13 packs * 4 cells per arm = 52 per arm, floor
#       2.8 * 3.91 * sqrt(2/52) = 2.15 pt  <  2.39 pt.
#
#   These packs draw their seat map (ptb_seat_plan in ptb_pack.sbatch refuses
#   any map that reproduces the arm/seat confound), so they are readable on
#   their own. Analyse them as their own board; do NOT silently pool them with
#   p10..p12, whose arms are seat-aliased.
#
# FENCING
# -------
# Nothing here overrides the launcher's own placement directives. ptb_pack.sbatch
# carries `--reservation=robtang-a3` and `--exclude=slurm2-a3nodeset0-[0-2]`, plus
# a runtime abort if it ever wakes up on one of Froilan's three nodes. A command
# line --exclude REPLACES the file's, and an empty --reservation= is a PIN onto
# Froilan's nodes rather than a relaxation, so this script passes neither.
#
#   ./ptb_ops/launch_sn_vs_r2_replication.sh            # submit
#   ./ptb_ops/launch_sn_vs_r2_replication.sh --dry-run  # sbatch --test-only only

set -euo pipefail

REPO="/rmeng_data/robtang/PostTrainBench"
LAUNCHER="${REPO}/ptb_ops/ptb_pack.sbatch"
LEDGER="${REPO}/ptb_ops/sn_vs_r2_replication_jobs.tsv"

# Pack numbers continue the p10..p12 series and are never reused: p13/p17..p20 were
# eaten by the slurm2-a3nodesetondem-0 shredder on 2026-09-05 (see ptb_pack.sbatch) and
# were replaced by p26..p30 rather than resubmitted under the same names, so every name
# still maps to exactly one job id in sacct.
FIRST_PACK="${FIRST_PACK:-13}"
LAST_PACK="${LAST_PACK:-25}"    # 13 packs, 52 cells per arm

# Every knob stated, including the ones that match the launcher's defaults --
# a default that drifts must not silently re-shape the campaign.
export PTB_TASK="gsm8k"
export PTB_MODEL="Qwen/Qwen3-1.7B-Base"
export PTB_NUM_HOURS="10"                    # launcher default is 1
export PTB_AGENT_CONFIG="claude-opus-5"
export PTB_CONTROL_ARM="claude_autor_r2"     # launcher default is claude_autor_ctl
export PTB_STAGGER="60"

WALLCLOCK="16:00:00"    # what p10..p12 actually used for a 10 h contract
ARMS=(claude_autor_sn claude_autor_sn claude_autor_sn claude_autor_sn
      claude_autor_r2 claude_autor_r2 claude_autor_r2 claude_autor_r2)

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

for (( p=FIRST_PACK; p<=LAST_PACK; p++ )); do
    name="ptb-sn-vs-r2-p${p}"
    if [ "$DRY" = "1" ]; then
        sbatch --test-only --job-name="$name" --time="$WALLCLOCK" \
               "$LAUNCHER" "${ARMS[@]}"
        continue
    fi
    out="$(sbatch --parsable --job-name="$name" --time="$WALLCLOCK" \
                  "$LAUNCHER" "${ARMS[@]}")"
    jobid="${out%%;*}"
    printf '%s\t%s\t%s\t%s\n' "$jobid" "$name" "$PTB_NUM_HOURS" "$WALLCLOCK" >> "$LEDGER"
    echo "submitted ${jobid}  ${name}"
done

[ "$DRY" = "1" ] || {
    echo "--- ledger: ${LEDGER}"
    squeue -u "$USER" -n "$(seq -s, -f 'ptb-sn-vs-r2-p%g' "$FIRST_PACK" "$LAST_PACK")" \
           -o '%.10i %.22j %.10T %.20R' || true
}
