#!/bin/bash
# Build a PostTrainBench container on a Slurm cluster that has no system
# apptainer and no root.
#
# Differences from containers/build_container.sh, all of them forced by this
# cluster rather than by choice:
#
#   1. apptainer is not installed node-locally on every node, so the binary is
#      taken from an unprivileged unpack on the shared filesystem
#      (APPTAINER_BIN, default /rmeng_data/robtang/tools/apt-root/usr/bin).
#      That build needs its own LD_LIBRARY_PATH.
#   2. $HOME lives on a 91%-full NFS mount, so the build scratch and layer
#      cache are pointed at node-local SSD (BUILD_SCRATCH).
#   3. flash-attn 2.8.3 compiles from source in %post. Left unbounded it forks
#      one nvcc per core -- 208 of them on an a3 node -- and is OOM-killed.
#      MAX_JOBS/NVCC_THREADS are injected at the top of %post, which is the
#      only reason this script rewrites the definition file at all. The
#      rewrite is a two-line insertion; the upstream .def is never edited.
#
# Usage: bash containers/build_container_slurm.sh <def-name> [output-name]
#   e.g. bash containers/build_container_slurm.sh vllm_debug
set -euo pipefail

DEF_NAME="${1:?usage: build_container_slurm.sh <def-name> [output-name]}"
OUT_NAME="${2:-$DEF_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"   # %files paths in the .def are relative to the build cwd

APPTAINER_BIN="${APPTAINER_BIN:-/rmeng_data/robtang/tools/apt-root/usr/bin}"
APPTAINER_LIB="${APPTAINER_LIB:-/rmeng_data/robtang/tools/apt-root/usr/lib/x86_64-linux-gnu}"
BUILD_SCRATCH="${BUILD_SCRATCH:-/mnt/localssd/$USER/ptb-build}"
# Read the output directory from .env when it is not already exported, the same
# way containers/build_container.sh does, so the image lands where run_task.sh
# will look for it. Do not derive it from $USER: on this cluster $USER is the
# full principal (robtang_google_com) while the data root is /rmeng_data/robtang.
if [ -z "${POST_TRAIN_BENCH_CONTAINERS_DIR:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
    POST_TRAIN_BENCH_CONTAINERS_DIR="$(grep -E '^POST_TRAIN_BENCH_CONTAINERS_DIR=' "$REPO_ROOT/.env" \
        | head -1 | cut -d= -f2- | tr -d '"')"
fi
OUT_DIR="${POST_TRAIN_BENCH_CONTAINERS_DIR:?set POST_TRAIN_BENCH_CONTAINERS_DIR or put it in .env}"
MAX_JOBS="${MAX_JOBS:-32}"
NVCC_THREADS="${NVCC_THREADS:-2}"

SRC_DEF="$REPO_ROOT/containers/${DEF_NAME}.def"
[ -f "$SRC_DEF" ] || { echo "no such definition: $SRC_DEF" >&2; exit 1; }

mkdir -p "$BUILD_SCRATCH/tmp" "$BUILD_SCRATCH/cache" "$OUT_DIR"

# Insert the build-parallelism bound as the first line of %post.
GEN_DEF="$BUILD_SCRATCH/${OUT_NAME}.generated.def"
awk -v mj="$MAX_JOBS" -v nt="$NVCC_THREADS" '
    { print }
    /^%post[[:space:]]*$/ && !done {
        print "    # injected by build_container_slurm.sh: bound the flash-attn compile"
        print "    export MAX_JOBS=" mj
        print "    export NVCC_THREADS=" nt
        done = 1
    }
' "$SRC_DEF" > "$GEN_DEF"

grep -q "export MAX_JOBS=" "$GEN_DEF" || {
    echo "injection failed: no %post section in $SRC_DEF" >&2; exit 1; }

export PATH="$APPTAINER_BIN:$PATH"
export LD_LIBRARY_PATH="$APPTAINER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export APPTAINER_TMPDIR="$BUILD_SCRATCH/tmp"
export APPTAINER_CACHEDIR="$BUILD_SCRATCH/cache"
export APPTAINER_BIND=""

echo "apptainer : $(command -v apptainer) ($(apptainer --version))"
echo "definition: $SRC_DEF -> $GEN_DEF (MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS)"
echo "output    : $OUT_DIR/${OUT_NAME}.sif"
echo "scratch   : $BUILD_SCRATCH"

apptainer build --force "$OUT_DIR/${OUT_NAME}.sif" "$GEN_DEF"

ls -lh "$OUT_DIR/${OUT_NAME}.sif"
