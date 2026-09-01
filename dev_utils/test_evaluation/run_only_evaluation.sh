#!/bin/bash
export EVALUATION_TASK="$1"
export EVAL_DIR="$2"
export HOME="$3"
export CLUSTER="$4"

RANDOM_UUID="${SLURM_JOB_ID:-$(uuidgen)}"
RECOVERY_SCRATCH_ROOT="${POST_TRAIN_BENCH_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
export TMP_SUBDIR="${RECOVERY_SCRATCH_ROOT%/}/posttrain_container_${EVALUATION_TASK}_${RANDOM_UUID}"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"
export JOB_TMP="${TMP_SUBDIR}/tmp"
mkdir -p "${TMP_SUBDIR}"
mkdir -p "${HF_MERGED}"
mkdir -p "${JOB_TMP}"

cleanup() {
    fusermount -uz "${TMP_SUBDIR}/merged_huggingface" 2>/dev/null || true
    rm -rf -- "${TMP_SUBDIR}"
}
trap cleanup EXIT

source src/commit_utils/set_env_vars.sh

if [ -n "${POST_TRAIN_BENCH_APPTAINER_BIN:-}" ]; then
    export PATH="$(dirname "$POST_TRAIN_BENCH_APPTAINER_BIN"):${PATH}"
fi
if [ -n "${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${POST_TRAIN_BENCH_APPTAINER_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec 1>${EVAL_DIR}/z_new_${CLUSTER}_output.log
exec 2>${EVAL_DIR}/z_new_${CLUSTER}_error.log

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi

with_huggingface_overlay() {
    mkdir -p "$TMP_SUBDIR/merged_huggingface"
    mkdir -p "$TMP_SUBDIR/upper_huggingface"
    mkdir -p "$TMP_SUBDIR/fuse_workdir"
    fuse-overlayfs -o "lowerdir=$HF_HOME,upperdir=$TMP_SUBDIR/upper_huggingface,workdir=$TMP_SUBDIR/fuse_workdir" "$TMP_SUBDIR/merged_huggingface"
    
    "$@"
    local exit_code=$?
    
    fusermount -u "$TMP_SUBDIR/merged_huggingface"
    rm -r "$TMP_SUBDIR/merged_huggingface"
    rm -r "$TMP_SUBDIR/upper_huggingface"
    rm -r "$TMP_SUBDIR/fuse_workdir"
    
    return $exit_code
}

with_huggingface_overlay apptainer exec \
    --nv \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${REPO_ROOT}" \
    ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python src/utils/check_cuda_writing.py > "$EVAL_DIR/cuda_check.txt"

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export REPO_ROOT="$(pwd)"

export TMP_HF_CACHE="${TMP_SUBDIR}/hf_cache"

export EVAL_COUNTER=0

run_evaluation() {
    local max_tokens_arg="$1"
    local eval_num="$2"
    local eval_home="${JOB_TMP}/eval-home"
    local eval_cache="${eval_home}/.cache"
    mkdir -p \
        "${eval_cache}/vllm" \
        "${eval_cache}/torchinductor" \
        "${eval_cache}/triton" \
        "${eval_home}/.config"
    with_huggingface_overlay apptainer exec \
        --nv \
        -c \
        --cleanenv \
        --pid \
        --no-init \
        --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
        --env "XDG_CACHE_HOME=${HOME}/.cache" \
        --env "XDG_CONFIG_HOME=${HOME}/.config" \
        --env "VLLM_CACHE_ROOT=${HOME}/.cache/vllm" \
        --env "TORCHINDUCTOR_CACHE_DIR=${HOME}/.cache/torchinductor" \
        --env "TRITON_CACHE_DIR=${HOME}/.cache/triton" \
        --env "TMPDIR=/tmp" \
        --env "HF_HOME=${TMP_HF_CACHE}" \
        --env OPENAI_API_KEY="${OPENAI_API_KEY}" \
        --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --env VLLM_LOGGING_LEVEL="DEBUG" \
        --writable-tmpfs \
        --home "${eval_home}:${HOME}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${EVAL_DIR}:${EVAL_DIR}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python "evaluate.py" \
            --model-path "$EVAL_DIR/final_model" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            ${max_tokens_arg} \
            --json-output-file "${EVAL_DIR}/metrics.json" > "$EVAL_DIR/z_new_${CLUSTER}_final_eval_${eval_num}.txt"
}

run_evaluation_with_retry() {
    local max_retries="$1"
    local max_tokens_arg="$2"

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        sleep 5
        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi

        EVAL_COUNTER=$((EVAL_COUNTER + 1))
        export EVAL_COUNTER
        echo "Evaluation attempt $EVAL_COUNTER (phase attempt $attempt of $max_retries)"

        timeout --signal=TERM --kill-after=60s 28800s bash -c "$(declare -f run_evaluation with_huggingface_overlay); run_evaluation \"$max_tokens_arg\" \"$EVAL_COUNTER\""

        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

# First evaluation: up to 4 attempts
run_evaluation_with_retry 4 ""

# Second evaluation with adjusted max tokens: up to 3 attempts
case "${EVALUATION_TASK}" in
    aime2025)
        MAX_TOKENS_ARG="--max-tokens 12000"
        ;;
    arenahardwriting)
        MAX_TOKENS_ARG="--max-new-tokens 12288"
        ;;
    bfcl)
        MAX_TOKENS_ARG="--max-tokens 12000"
        ;;
    gpqamain)
        MAX_TOKENS_ARG="--max-tokens 12000"
        ;;
    gsm8k)
        MAX_TOKENS_ARG="--max-tokens 3000"
        ;;
    healthbench)
        MAX_TOKENS_ARG="--max-new-tokens 12288"
        ;;
    humaneval)
        MAX_TOKENS_ARG="--max-tokens 3000"
        ;;
    *)
        MAX_TOKENS_ARG=""
        ;;
esac

run_evaluation_with_retry 3 "$MAX_TOKENS_ARG"

# Third evaluation with further adjusted max tokens: up to 2 attempts
case "${EVALUATION_TASK}" in
    aime2025)
        MAX_TOKENS_ARG="--max-tokens 8000"
        ;;
    arenahardwriting)
        MAX_TOKENS_ARG="--max-new-tokens 8192"
        ;;
    bfcl)
        MAX_TOKENS_ARG="--max-tokens 8000"
        ;;
    gpqamain)
        MAX_TOKENS_ARG="--max-tokens 8000"
        ;;
    gsm8k)
        MAX_TOKENS_ARG="--max-tokens 2000"
        ;;
    healthbench)
        MAX_TOKENS_ARG="--max-new-tokens 8192"
        ;;
    humaneval)
        MAX_TOKENS_ARG="--max-tokens 2000"
        ;;
    *)
        MAX_TOKENS_ARG=""
        ;;
esac

run_evaluation_with_retry 2 "$MAX_TOKENS_ARG"

if [ ! -s "${EVAL_DIR}/metrics.json" ]; then
    echo "ERROR: evaluation recovery exhausted all attempts without metrics.json" >&2
    exit 1
fi

echo $(cat "$EVAL_DIR/z_new_${CLUSTER}_final_eval_${EVAL_COUNTER}.txt")

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"
