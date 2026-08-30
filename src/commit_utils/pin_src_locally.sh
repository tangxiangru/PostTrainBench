#!/bin/bash
#
# Copy the shell half of src/ to node-local disk and echo the path of the copy's
# run_task.sh. A launcher runs that copy instead of the one in the checkout.
#
# The hazard this closes. Bash does not read a script into memory; it reads a chunk,
# runs what it parsed, then seeks back to the next byte and reads again. A script whose
# body is one ten-hour command therefore holds an open handle on its own inode for ten
# hours. Replace that inode -- which `git commit`, `git checkout` and every editor do,
# because they write a new file and rename it over the old one -- and the next read
# fails. Over NFS it fails as ESTALE.
#
# It has already cost a full batch. Jobs 82165 and 82166 both started at 07:44:28 on
# 2026-08-30; commit 2775447 rewrote src/run_task.sh at 08:28:29, forty-four minutes in.
# Neither job noticed, because neither needed to read anything for hours. Each died at
# the instant its agent phase returned and bash reached for the next line -- 82165 after
# 10:01:47, 82166 after 08:27:27 -- with
#
#     src/run_task.sh: error reading input file: Stale file handle
#
# Both had a finished 3.3 GB final_model/ sitting on node-local disk. Neither was ever
# scored. That is nineteen H100-hours, and the trigger was a one-line commit to an
# unrelated part of the same file.
#
# What is and is not copied:
#
#   src/eval  is 2.0 GB of task data and is symlinked, not copied. run_task.sh reaches
#             it through cwd-relative paths (`python src/eval/general/get_prompt.py`)
#             and the scoring container binds it by REPO_ROOT, so it is read from the
#             checkout either way. Each of those reads is a single short open, not a
#             handle held across hours.
#   .env      is symlinked rather than copied. set_env_vars.sh finds it at
#             "$(dirname "${BASH_SOURCE[0]}")/../../.env", so the copy has to keep the
#             src/commit_utils/ shape for that to resolve -- hence a mirrored tree
#             rather than three loose files. A symlink also keeps configuration off
#             node-local disk, where it would outlive the job.
#   the rest  of src/ is ~600 KB: run_task.sh, commit_utils/, judges/, trace_parsing/,
#             utils/. Copying it is instant and closes the two `source` lines too --
#             run_task.sh resolves both against ${BASH_SOURCE[0]}, so sourcing follows
#             the copy. judges/judge_lib.sh sets JUDGES_DIR the same way, which is why
#             the whole judges/ directory comes along rather than just the library.
#
# The working directory is deliberately NOT changed by this. run_task.sh line 542 takes
# REPO_ROOT from `pwd` and the scoring container binds "${REPO_ROOT}:${REPO_ROOT}"; point
# that at node-local scratch and the bind stops covering the results directory.
#
# Usage:  RUN_TASK="$(bash src/commit_utils/pin_src_locally.sh "$REPO_ROOT" "$DEST")"
#         bash "$RUN_TASK" <task> <agent> ...
#
# On any failure it echoes the checkout's own run_task.sh and warns on stderr: an
# unpinned run is worse than a pinned one and better than no run at all.
set -uo pipefail

REPO="${1:?usage: pin_src_locally.sh <repo_root> <dest_dir>}"
DEST="${2:?usage: pin_src_locally.sh <repo_root> <dest_dir>}"
FALLBACK="${REPO}/src/run_task.sh"

fail() { echo "WARN: pin_src_locally: $1; running from the checkout instead" >&2; echo "$FALLBACK"; exit 0; }

[ -f "$FALLBACK" ] || { echo "FATAL: no ${FALLBACK}" >&2; exit 1; }
mkdir -p "${DEST}/src" || fail "cannot create ${DEST}/src"

for entry in "${REPO}"/src/*; do
    base="$(basename "$entry")"
    if [ "$base" = "eval" ]; then
        ln -sfn "$entry" "${DEST}/src/eval" || fail "cannot link src/eval"
    else
        cp -a "$entry" "${DEST}/src/" || fail "cannot copy src/${base}"
    fi
done
# scripts/ is 152 KB and holds the other long-running driver, rerun_eval_n_times.sh --
# an eight-hour job with the same exposure. Its own `source src/commit_utils/...` is
# resolved against the working directory, which stays on the checkout, so that one line
# is still a short read from the shared tree; the eight hours of incremental reads of
# the driver itself are what moves here.
cp -a "${REPO}/scripts" "${DEST}/scripts" 2>/dev/null || echo "WARN: pin_src_locally: no scripts/ to pin" >&2
ln -sfn "${REPO}/.env" "${DEST}/.env" || fail "cannot link .env"

[ -f "${DEST}/src/run_task.sh" ] || fail "copy produced no run_task.sh"
# Named so a post-mortem can tell a pinned run from an unpinned one, and tell which
# bytes ran: the checkout's HEAD moves, this sum does not.
echo "pinned run_task.sh -> ${DEST}/src/run_task.sh sha256=$(sha256sum "${DEST}/src/run_task.sh" | cut -c1-12)" >&2
echo "${DEST}/src/run_task.sh"
