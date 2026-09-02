#!/bin/bash

export EVALUATION_TASK="$1"
AGENT="$2"
MODEL_TO_TRAIN="$3"
CLUSTER_ID="$4"
NUM_HOURS="$5"
AGENT_CONFIG="$6"
NUM_GPUS="${7:-1}"

source src/commit_utils/set_env_vars.sh

PTB_AGENT_PROVIDER="unknown"
PTB_AGENT_EFFORT="unknown"
PTB_AGENT_REQUESTED_CONTEXT_TOKENS="unknown"
if [ -f "agents/${AGENT}/profile.env" ]; then
    # Agent profiles are versioned harness metadata, not site secrets.
    source "agents/${AGENT}/profile.env"
fi
export PTB_AGENT_PROVIDER PTB_AGENT_EFFORT PTB_AGENT_REQUESTED_CONTEXT_TOKENS

# Select the judge backend for grader-based benchmarks (arenahardwriting / healthbench):
# default to the OpenAI-backed evaluate.py, but fall back to the OpenRouter variant when
# .env provides OPENROUTER_API_KEY but no OPENAI_API_KEY.
JUDGE_BACKEND="openai"
if { [ "$EVALUATION_TASK" = "arenahardwriting" ] || [ "$EVALUATION_TASK" = "healthbench" ]; } \
   && [ -z "${OPENAI_API_KEY}" ] && [ -n "${OPENROUTER_API_KEY}" ]; then
    JUDGE_BACKEND="openrouter"
fi

if [ "$JUDGE_BACKEND" = "openrouter" ]; then
    export EVAL_SCRIPT="evaluate_openrouter.py"
else
    export EVAL_SCRIPT="evaluate.py"
fi

RESULT_PREFIX_SAFE=$(echo "$MODEL_TO_TRAIN" | tr '/:[]' '____')

AGENT_CONFIG_SAFE=$(echo "$AGENT_CONFIG" | tr '/:[]' '____')

RANDOM_UUID=$(uuidgen)

GPU_SUFFIX=""
if [ "$NUM_GPUS" -gt 1 ] 2>/dev/null; then
    GPU_SUFFIX="_${NUM_GPUS}gpu"
fi

export EVAL_DIR="${POST_TRAIN_BENCH_RESULTS_DIR}/${AGENT}_${AGENT_CONFIG_SAFE}_${NUM_HOURS}h${GPU_SUFFIX}${POST_TRAIN_BENCH_EXPERIMENT_NAME}/${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${CLUSTER_ID}"

mkdir -p ${EVAL_DIR}

exec 1>${EVAL_DIR}/output.log
exec 2>${EVAL_DIR}/error.log

echo "$@"
echo "Judge backend: ${JUDGE_BACKEND} (eval script: ${EVAL_SCRIPT})"

python3 src/utils/record_runtime_provenance.py init \
    --output "${EVAL_DIR}/runtime_provenance.json" \
    --task "$EVALUATION_TASK" \
    --agent "$AGENT" \
    --agent-config "$AGENT_CONFIG" \
    --base-model "$MODEL_TO_TRAIN" \
    --hours "$NUM_HOURS" \
    --num-gpus "$NUM_GPUS"

PTB_SCRATCH_ROOT="${POST_TRAIN_BENCH_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
mkdir -p "${PTB_SCRATCH_ROOT}"
export TMP_SUBDIR="${PTB_SCRATCH_ROOT%/}/posttrain_container_${EVALUATION_TASK}_${RESULT_PREFIX_SAFE}_${RANDOM_UUID}"

JOB_DIR="${TMP_SUBDIR}/job_dir"
export JOB_TMP="${TMP_SUBDIR}/tmp"
export HF_MERGED="${TMP_SUBDIR}/merged_huggingface"

mkdir -p "${JOB_DIR}"
mkdir -p "${JOB_TMP}"

echo "Preparing job directory..." 
mkdir -p "${JOB_DIR}"

mkdir "${JOB_DIR}/task"

cp "src/eval/tasks/${EVALUATION_TASK}/${EVAL_SCRIPT}" "${JOB_DIR}/task/evaluate.py"
# hv-patches (upstream PR #45): aime2025 now ships a local scorer next to
# evaluate.py; carry those modules into the agent's task dir too.
for _extra_module in score.py task.py; do
    if [ -f "src/eval/tasks/${EVALUATION_TASK}/${_extra_module}" ]; then
        cp "src/eval/tasks/${EVALUATION_TASK}/${_extra_module}" "${JOB_DIR}/task"
    fi
done
if [ -d "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" ]; then
    cp -r "src/eval/tasks/${EVALUATION_TASK}/evaluation_code" "${JOB_DIR}/task"
fi
cp -r src/eval/templates "${JOB_DIR}/task/"

if [ -d "src/eval/tasks/${EVALUATION_TASK}/task_context" ]; then
    cp -r src/eval/tasks/${EVALUATION_TASK}/task_context/* "${JOB_DIR}/task"
fi
cp -r "containers/other_home_data/.codex" "${JOB_DIR}/"

BENCHMARK=$(cat src/eval/tasks/${EVALUATION_TASK}/benchmark.txt)
BASE_MODEL_REVISION="${POST_TRAIN_BENCH_BASE_MODEL_REVISION:-}"
BASE_MODEL_CACHE_KEY="models--${MODEL_TO_TRAIN//\//--}"
BASE_MODEL_SNAPSHOT_CONTAINER="${HF_HOME_NEW}/hub/${BASE_MODEL_CACHE_KEY}/snapshots/${BASE_MODEL_REVISION}"
PROMPT_ARGS=(
    --model-to-train "$MODEL_TO_TRAIN"
    --benchmark-id "$EVALUATION_TASK"
    --num-hours "$NUM_HOURS"
    --num-gpus "$NUM_GPUS"
    --agent "${AGENT}"
)
if [ -n "$BASE_MODEL_REVISION" ]; then
    PROMPT_ARGS+=(
        --model-revision "$BASE_MODEL_REVISION"
        --model-snapshot "$BASE_MODEL_SNAPSHOT_CONTAINER"
    )
fi
if ! PROMPT=$(python3 src/eval/general/get_prompt.py "${PROMPT_ARGS[@]}"); then
    echo "ERROR: failed to generate the agent prompt" >&2
    exit 1
fi
if [ -z "${PROMPT//[[:space:]]/}" ]; then
    echo "ERROR: generated agent prompt is empty" >&2
    exit 1
fi
echo "$PROMPT" > "${EVAL_DIR}/prompt.txt"

bash src/utils/create_timer.sh $NUM_HOURS $JOB_DIR/task/timer.sh

# set openai api keys appropriately
export CODEX_API_KEY="${OPENAI_API_KEY}"
unset OPENAI_API_KEY
if [ "$EVALUATION_TASK" == "arenahardwriting" ] || [ "$EVALUATION_TASK" == "healthbench" ]; then
    export OPENAI_API_KEY="${CODEX_API_KEY}"
fi

# --- API key allowlist ----------------------------------------------------
# Each agent declares the third-party API keys it may receive in
# agents/<agent>/api_keys.json; each benchmark declares the keys its grading
# needs in src/eval/tasks/<task>/info.json ("required_api_keys", default none).
# The agent sandbox is launched with -c --cleanenv, so it inherits NOTHING from
# the host: it receives ONLY the union of those two sets via --env (built into
# API_KEY_ENV_ARGS below). Every other provider key is never passed in, hence
# unset inside the sandbox. OPENAI_API_KEY thus reaches the agent only for the
# arenahardwriting/healthbench benchmarks (which list it in required_api_keys).
if ! ALLOWED_API_KEYS_RAW="$(python3 -c '
import json, sys
agent_keys = json.load(open(sys.argv[1]))["allowed_api_keys"]
bench = json.load(open(sys.argv[2]))
seen, out = set(), []
for k in list(agent_keys) + bench.get("required_api_keys", []):
    if k not in seen:
        seen.add(k); out.append(k)
print("\n".join(out))
' "agents/${AGENT}/api_keys.json" "src/eval/tasks/${EVALUATION_TASK}/info.json")"; then
    echo "ERROR: failed to compute API key allowlist for agent=${AGENT} task=${EVALUATION_TASK}" >&2
    exit 1
fi

ALLOWED_API_KEYS=()
[ -n "$ALLOWED_API_KEYS_RAW" ] && mapfile -t ALLOWED_API_KEYS <<< "$ALLOWED_API_KEYS_RAW"

if [ "$JUDGE_BACKEND" = "openrouter" ]; then
    ALLOWED_API_KEYS+=("OPENROUTER_API_KEY")
fi

API_KEY_ENV_ARGS=()
for _k in "${ALLOWED_API_KEYS[@]}"; do
    API_KEY_ENV_ARGS+=(--env "${_k}=${!_k}")
done
echo "API keys provisioned for agent=${AGENT} task=${EVALUATION_TASK}: ${ALLOWED_API_KEYS[*]:-<none>}"

# Non-secret provider routing may be declared by an agent scaffold. Only
# explicitly named, syntactically valid environment variables are forwarded;
# cleanenv continues to block every other ambient setting.
AGENT_ENV_ARGS=()
if [ -f "agents/${AGENT}/env_passthrough.txt" ]; then
    while IFS= read -r _env_name || [ -n "$_env_name" ]; do
        [[ -z "$_env_name" || "$_env_name" =~ ^[[:space:]]*# ]] && continue
        if ! [[ "$_env_name" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
            echo "ERROR: invalid variable name in agents/${AGENT}/env_passthrough.txt: $_env_name" >&2
            exit 1
        fi
        if [ -n "${!_env_name+x}" ]; then
            AGENT_ENV_ARGS+=(--env "${_env_name}=${!_env_name}")
        fi
    done < "agents/${AGENT}/env_passthrough.txt"
fi
echo "Provider environment provisioned for agent=${AGENT}: ${#AGENT_ENV_ARGS[@]} argument(s)"

# Copy scripts needed inside the container
cp src/utils/check_cuda.py "${JOB_DIR}/check_cuda.py"
cp src/utils/check_cuda_writing.py "${JOB_DIR}/check_cuda_writing.py"
cp src/utils/system_monitor.sh "${JOB_DIR}/system_monitor.sh"
cp src/utils/timestamp_lines.py "${JOB_DIR}/timestamp_lines.py"
cp src/utils/update_agent_cli.sh "${JOB_DIR}/update_agent_cli.sh"
cp "agents/${AGENT}/solve.sh" "${JOB_DIR}/agent_solve.sh"

# Self-decontamination tooling for the agent: the same n-gram checker and
# test-set copy the contamination judge gets, at the same paths (the judge
# phase re-copies both via prepare_judge_sandbox, so the judges never run
# agent-modified versions). The matching usage instructions are added to the
# agent prompt by get_prompt.py.
if [ ! -f "src/eval/tasks/${EVALUATION_TASK}/test_data.json" ]; then
    echo "ERROR: src/eval/tasks/${EVALUATION_TASK}/test_data.json not found — required for the agent's decontamination tooling" >&2
    exit 1
fi
cp src/judges/judge_tools/contamination_check.py "${JOB_DIR}/contamination_check.py"
cp "src/eval/tasks/${EVALUATION_TASK}/test_data.json" "${JOB_DIR}/test_data.json"

# Agent-specific auth: auth.json is bind-mounted at apptainer exec time so the
# codex CLI can write the rotated refresh token back to the shared source file
# (single-use refresh tokens otherwise burn after one job).
AGENT_AUTH_SRC=""
if [ -f "agents/${AGENT}/auth.json" ]; then
    AGENT_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/auth.json")" && pwd)/auth.json"
    # Placeholder file inside the sandbox .codex dir for the bind mount to overlay.
    mkdir -p "${JOB_DIR}/.codex"
    : > "${JOB_DIR}/.codex/auth.json"
fi
if [ -f "agents/${AGENT}/oauth_token" ]; then
    cp "agents/${AGENT}/oauth_token" "${JOB_DIR}/oauth_token"
fi

# Cursor CLI persists its OAuth tokens at ~/.config/cursor/auth.json. Bind-mount
# the agent's copy so the CLI inside the sandbox reads and rotates against the
# shared credential file. Distinct filename in the agent dir avoids collision
# with the codex auth.json check above.
CURSOR_AUTH_SRC=""
if [ -f "agents/${AGENT}/cursor_auth.json" ]; then
    CURSOR_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/cursor_auth.json")" && pwd)/cursor_auth.json"
    mkdir -p "${JOB_DIR}/.config/cursor"
    : > "${JOB_DIR}/.config/cursor/auth.json"
fi

# xAI Grok Build CLI persists its OAuth session at ~/.grok/auth.json. Bind-mount
# the agent's copy so the CLI inside the sandbox reads and rotates against the
# shared credential file. Distinct filename in the agent dir (grok_auth.json)
# avoids collision with the codex auth.json check above.
GROK_AUTH_SRC=""
if [ -f "agents/${AGENT}/grok_auth.json" ]; then
    GROK_AUTH_SRC="$(cd "$(dirname "agents/${AGENT}/grok_auth.json")" && pwd)/grok_auth.json"
    mkdir -p "${JOB_DIR}/.grok"
    : > "${JOB_DIR}/.grok/auth.json"
fi

# Utils
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

with_record_the_time() {
    local begin=$(date --iso-8601=seconds)
    "$@"
    local exit_code=$?
    local end=$(date --iso-8601=seconds)
    
    local time_taken=$(( $(date --date="$end" +%s) - $(date --date="$begin" +%s) ))
    printf '%02d:%02d:%02d\n' \
        $(( time_taken / 3600 )) \
        $(( (time_taken % 3600) / 60 )) \
        $(( time_taken % 60 )) > "${EVAL_DIR}/time_taken.txt"
    
    return $exit_code
}

SOLVE_OUT="${EVAL_DIR}/solve_out.txt"

solve_task() {
    AGENT_AUTH_BIND=()
    [ -n "$AGENT_AUTH_SRC" ] && AGENT_AUTH_BIND=(--bind "${AGENT_AUTH_SRC}:/home/ben/.codex/auth.json")
    CURSOR_AUTH_BIND=()
    [ -n "$CURSOR_AUTH_SRC" ] && CURSOR_AUTH_BIND=(--bind "${CURSOR_AUTH_SRC}:/home/ben/.config/cursor/auth.json")
    GROK_AUTH_BIND=()
    [ -n "$GROK_AUTH_SRC" ] && GROK_AUTH_BIND=(--bind "${GROK_AUTH_SRC}:/home/ben/.grok/auth.json")
    # Forward the CLI-auto-update opt-out into the sandbox so update_agent_cli.sh
    # can honor it. Only set when the user opts in via .env.
    CLI_UPDATE_ENV=()
    [ -n "${POST_TRAIN_BENCH_SKIP_CLI_UPDATE:-}" ] && CLI_UPDATE_ENV+=(--env "POST_TRAIN_BENCH_SKIP_CLI_UPDATE=${POST_TRAIN_BENCH_SKIP_CLI_UPDATE}")
    VISIBLE_GPUS_ENV=()
    [ -n "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ] && \
        VISIBLE_GPUS_ENV+=(--env "CUDA_VISIBLE_DEVICES=${POST_TRAIN_BENCH_VISIBLE_GPUS}")
    # CUDA_VISIBLE_DEVICES limits CUDA clients but does not hide device files.
    # On an exclusive Slurm node without working GPU GRES, nvidia-container-cli
    # restores the one-GPU device boundary that HTCondor normally provides.
    # This stays opt-in because --nvccli requires nvidia-container-cli on the
    # host and is unnecessary when the scheduler already applies a GPU cgroup.
    NVCCLI_ARGS=()
    if [ "${POST_TRAIN_BENCH_ISOLATE_GPUS:-}" = "1" ] && [ -n "${POST_TRAIN_BENCH_VISIBLE_GPUS:-}" ]; then
        export NVIDIA_VISIBLE_DEVICES="${POST_TRAIN_BENCH_VISIBLE_GPUS}"
        NVCCLI_ARGS=(--nvccli)
    fi
    # --- extra binds (begin) ---
    # POST_TRAIN_BENCH_EXTRA_BINDS="src:dst[:ro],src2:dst2" adds one bind per
    # comma-separated entry to the agent sandbox only (not to evaluation or the
    # judges). It is read on the host; nothing about it reaches the sandbox
    # environment. Unset or empty, the default, changes nothing. A study uses it
    # to mount its own read-only code or data next to the task.
    EXTRA_BIND_ARGS=()
    if [ -n "${POST_TRAIN_BENCH_EXTRA_BINDS:-}" ]; then
        IFS=',' read -r -a _extra_binds <<< "${POST_TRAIN_BENCH_EXTRA_BINDS}"
        for _bind in "${_extra_binds[@]}"; do
            [ -n "$_bind" ] || continue
            _bind_src="${_bind%%:*}"
            if [ ! -e "$_bind_src" ]; then
                echo "ERROR: POST_TRAIN_BENCH_EXTRA_BINDS source does not exist: ${_bind_src}" >&2
                exit 1
            fi
            EXTRA_BIND_ARGS+=(--bind "$_bind")
        done
        echo "Extra sandbox binds: ${POST_TRAIN_BENCH_EXTRA_BINDS}"
    fi
    # --- extra binds (end) ---
    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \
    apptainer exec \
        --nv \
        "${NVCCLI_ARGS[@]}" \
        -c \
        --cleanenv \
        --pid \
        --no-init \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env HF_HOME="${HF_HOME_NEW}" \
        "${API_KEY_ENV_ARGS[@]}" \
        --env VLLM_API_KEY="inspectai" \
        --env PYTHONNOUSERSITE="1" \
        --env NUM_GPUS="${NUM_GPUS}" \
        --env PROMPT="${PROMPT}" \
        --env AGENT_CONFIG="${AGENT_CONFIG}" \
        --env PTB_BASE_MODEL_ID="${MODEL_TO_TRAIN}" \
        --env PTB_BASE_MODEL_REVISION="${BASE_MODEL_REVISION}" \
        --env PTB_BASE_MODEL_SNAPSHOT="${BASE_MODEL_SNAPSHOT_CONTAINER}" \
        "${AGENT_ENV_ARGS[@]}" \
        "${VISIBLE_GPUS_ENV[@]}" \
        "${CLI_UPDATE_ENV[@]}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${HF_MERGED}:${HF_HOME_NEW}" \
        "${EXTRA_BIND_ARGS[@]}" \
        "${AGENT_AUTH_BIND[@]}" \
        "${CURSOR_AUTH_BIND[@]}" \
        "${GROK_AUTH_BIND[@]}" \
        --home "${JOB_DIR}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" \
        bash -c "set -o pipefail; { python /home/ben/check_cuda.py && python /home/ben/check_cuda_writing.py || exit 1; bash /home/ben/system_monitor.sh & MONITOR_PID=\$!; bash /home/ben/agent_solve.sh; AGENT_EXIT=\$?; kill \$MONITOR_PID 2>/dev/null || true; exit \$AGENT_EXIT; } 2>&1 | python /home/ben/timestamp_lines.py" > "${SOLVE_OUT}" 2>&1
}

# ---------- judge auth/profile precheck ----------
# Both official and research profiles use an isolated Claude Opus 5 judge.
# Official outputs keep the canonical ids; the research profile keeps separate
# ids. Neither reads the tested agent's Claude home or token.

echo "================================"
echo "======= JUDGE AUTH CHECK ======="
echo "================================"
source src/judges/judge_lib.sh
configure_judge_profile "${POST_TRAIN_BENCH_JUDGE_PROFILE:-official}" || exit 1
resolve_judge_auth_mode || exit 1
JUDGE_AUTH="${POST_TRAIN_BENCH_CODEX_JUDGE_AUTH_FILE:-agents/codex_non_api/auth.json}"
case "$JUDGE_AUTH_MODE" in
skip)
    echo "WARNING: judge auth precheck skipped (POST_TRAIN_BENCH_JUDGE_AUTH_MODE=skip)." >&2
    echo "WARNING: reward-hacking judges will not produce verdicts for this run." >&2
    ;;
apikey)
    echo "ERROR: POST_TRAIN_BENCH_JUDGE_AUTH_MODE=apikey is not implemented by judge_lib.sh" >&2
    exit 1
    ;;
chatgpt)
    if [ "$PTB_JUDGE_BACKEND" != "codex" ]; then
        echo "ERROR: chatgpt auth requires the Codex judge backend" >&2
        exit 1
    fi
    if [ ! -f "$JUDGE_AUTH" ]; then
        echo "ERROR: judge auth file missing at $JUDGE_AUTH" >&2
        exit 1
    fi
    JUDGE_ACCESS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"]["access_token"])' "$JUDGE_AUTH") \
        || { echo "ERROR: could not extract tokens.access_token from $JUDGE_AUTH" >&2; exit 1; }
    JUDGE_HTTP=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 \
        -H "Authorization: Bearer $JUDGE_ACCESS" \
        "https://chatgpt.com/backend-api/codex/models?client_version=0.124.0")
    if [ "$JUDGE_HTTP" != "200" ]; then
        echo "ERROR: judge OAuth precheck failed (HTTP ${JUDGE_HTTP})." >&2
        echo "The ChatGPT session in $JUDGE_AUTH may be invalidated." >&2
        echo "Re-login on the head node:" >&2
        echo "  codex logout && codex login && cp ~/.codex/auth.json $JUDGE_AUTH && chmod 600 $JUDGE_AUTH" >&2
        exit 1
    fi
    echo "Judge OAuth OK (HTTP 200)"
    ;;
claude_oauth|vertex)
    if [ "$PTB_JUDGE_BACKEND" != "claude" ]; then
        echo "ERROR: $JUDGE_AUTH_MODE requires the Claude judge backend" >&2
        exit 1
    fi
    if [ "$JUDGE_AUTH_MODE" = "claude_oauth" ]; then
        CLAUDE_JUDGE_AUTH="${POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE:-}"
        if [ -z "$CLAUDE_JUDGE_AUTH" ] || [ ! -r "$CLAUDE_JUDGE_AUTH" ] || [ ! -s "$CLAUDE_JUDGE_AUTH" ]; then
            echo "ERROR: POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE must name a readable, non-empty, judge-only token file" >&2
            exit 1
        fi
    else
        setup_judge_vertex_auth "$JOB_DIR" || exit 1
    fi
    CLAUDE_JUDGE_IMAGE="${POST_TRAIN_BENCH_CONTAINERS_DIR}/${JUDGE_CONTAINER}"
    if [ ! -r "$CLAUDE_JUDGE_IMAGE" ]; then
        echo "ERROR: Claude judge container is missing or unreadable: $CLAUDE_JUDGE_IMAGE" >&2
        exit 1
    fi
    if ! CLAUDE_JUDGE_HELP="$(apptainer exec --containall "$CLAUDE_JUDGE_IMAGE" claude --help 2>&1)"; then
        echo "ERROR: Claude Code CLI is not runnable in $CLAUDE_JUDGE_IMAGE" >&2
        exit 1
    fi
    if ! grep -q -- '--effort' <<< "$CLAUDE_JUDGE_HELP" \
        || ! grep -q -- '--setting-sources' <<< "$CLAUDE_JUDGE_HELP" \
        || ! grep -q -- '--safe-mode' <<< "$CLAUDE_JUDGE_HELP"; then
        echo "ERROR: Claude Code CLI in $CLAUDE_JUDGE_IMAGE lacks --effort, --setting-sources, or --safe-mode" >&2
        exit 1
    fi
    CLAUDE_JUDGE_VERSION="$(apptainer exec --containall "$CLAUDE_JUDGE_IMAGE" claude --version 2>&1 | head -n 1)"
    echo "Claude judge ready; profile=${JUDGE_PROFILE} auth=${JUDGE_AUTH_MODE} model=${JUDGE_DEFAULT_MODEL} effort=${JUDGE_DEFAULT_REASONING_EFFORT} cli=${CLAUDE_JUDGE_VERSION:-unknown}"
    ;;
*)
    echo "ERROR: unknown POST_TRAIN_BENCH_JUDGE_AUTH_MODE='${JUDGE_AUTH_MODE}' (want chatgpt|claude_oauth|vertex|skip)" >&2
    exit 1
    ;;
esac

echo "================================"
echo "========= RUNNING TASK ========="
echo "================================"

with_huggingface_overlay with_record_the_time solve_task
SOLVE_EXIT=$?

echo "--- SOLVE DIAGNOSTICS ---"
echo "exit_code: $SOLVE_EXIT"
if [ $SOLVE_EXIT -eq 0 ]; then
    echo "status: exited normally"
elif [ $SOLVE_EXIT -eq 124 ]; then
    echo "status: killed by timeout (reached ${NUM_HOURS}h limit)"
elif [ $SOLVE_EXIT -gt 128 ]; then
    echo "status: killed by signal $((SOLVE_EXIT - 128)) ($(kill -l $((SOLVE_EXIT - 128)) 2>/dev/null || echo unknown))"
else
    echo "status: exited with error code $SOLVE_EXIT"
fi
echo "final_model_files: $(ls "${JOB_DIR}/task/final_model/" 2>/dev/null | wc -l)"
echo "hostname: $(hostname)"
echo "fuse_overlayfs_alive: $(ps aux 2>/dev/null | grep fuse-overlay | grep -v grep | wc -l)"
echo "disk_job_dir: $(du -sh "${JOB_DIR}" 2>/dev/null | cut -f1)"
echo "disk_tmp: $(du -sh "${JOB_TMP}" 2>/dev/null | cut -f1)"
echo "memory: $(free -m 2>/dev/null | grep Mem | awk '{print "total=" $2 "MB used=" $3 "MB free=" $4 "MB"}')"
echo "--- END SOLVE DIAGNOSTICS ---"

# Record the (auto-updated) agent CLI version captured by update_agent_cli.sh.
if [ -f "${JOB_DIR}/cli_version.txt" ]; then
    cp "${JOB_DIR}/cli_version.txt" "${EVAL_DIR}/cli_version.txt"
    echo "--- AGENT CLI VERSION ---"
    cat "${EVAL_DIR}/cli_version.txt"
    echo "--- END AGENT CLI VERSION ---"
else
    echo "WARNING: ${JOB_DIR}/cli_version.txt not found (agent CLI version not recorded)" >&2
fi

python3 src/utils/record_runtime_provenance.py finalize \
    --output "${EVAL_DIR}/runtime_provenance.json" \
    --trace "$SOLVE_OUT" \
    --cli-version "${EVAL_DIR}/cli_version.txt"

echo "============================================"
echo "=== TASK COMPLETE, PARSING AGENT TRACE ==="
echo "============================================"

# Parse agent trace into human-readable format
python3 src/trace_parsing/parse_trace.py --agent "${AGENT}" "${SOLVE_OUT}" -o "${EVAL_DIR}/solve_parsed.txt"
cp "${EVAL_DIR}/solve_parsed.txt" "${JOB_DIR}/solve_parsed.txt"

echo "============================="
echo "======== CLEANING UP ========"
echo "============================="

echo "Task directory contents:"
if command -v tree >/dev/null 2>&1; then
    tree "${JOB_DIR}/task"
else
    find "${JOB_DIR}/task" -maxdepth 3 -print
fi
echo "================================"

if [ -d "${JOB_DIR}/task/final_model" ]; then
    cp -r "${JOB_DIR}/task/final_model" "$EVAL_DIR/final_model"
fi

if [ -f "${JOB_DIR}/task/system_monitor.log" ]; then
    cp "${JOB_DIR}/task/system_monitor.log" "$EVAL_DIR/system_monitor.log"
fi

python3 containers/delete_hf_models.py "${JOB_DIR}/task"

cp -r "${JOB_DIR}/task" "$EVAL_DIR/task"

echo "========================================="
echo "=== RUNNING REWARD-HACKING JUDGES ==="
echo "========================================="

# When the precheck above ran in `skip` mode there is no judge login. Skip the
# whole judge phase; the deterministic score still gets produced.
if [ "$JUDGE_AUTH_MODE" = "skip" ]; then
    echo "SKIPPED: POST_TRAIN_BENCH_JUDGE_AUTH_MODE=skip -> no judge verdicts for this run." >&2
    echo "SKIPPED: scripts/collect.py will refuse to aggregate this run until verdicts exist." >&2
else

# Make judge helper tooling and benchmark metadata available inside the judge
# sandbox. The final_model config comes from EVAL_DIR because delete_hf_models
# has already run on JOB_DIR/task during cleanup.
prepare_judge_sandbox "${JOB_DIR}" "${EVALUATION_TASK}" "${EVAL_DIR}/final_model/config.json"

# Set up profile-specific isolated config and auth. The Claude profile uses a
# separate CLAUDE_CONFIG_DIR and disables user/project/local setting sources.
setup_judge_auth "${JOB_DIR}" || exit 1

# ChatGPT subscription auth rotates a single-use refresh token. Serialize only
# that legacy auth path; Vertex-backed official Claude judges can run in parallel.
JUDGE_LOCK_HELD=0
if [ "$JUDGE_AUTH_MODE" = "chatgpt" ]; then
    JUDGE_LOCK_FILE="${POST_TRAIN_BENCH_JUDGE_LOCK_FILE:-}"
    if [ -z "$JUDGE_LOCK_FILE" ]; then
        echo "ERROR: POST_TRAIN_BENCH_JUDGE_LOCK_FILE is required for official judges" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$JUDGE_LOCK_FILE")"
    exec 9>"$JUDGE_LOCK_FILE"
    echo "Waiting for official judge auth lock: $JUDGE_LOCK_FILE"
    flock -x 9
    JUDGE_LOCK_HELD=1
    echo "Official judge auth lock acquired."
fi

JUDGE_EXTRA_APPTAINER_ARGS=(
    --nv
    --env HF_HOME="${HF_HOME_NEW}"
    --env VLLM_API_KEY="inspectai"
    --bind "${HF_MERGED}:${HF_HOME_NEW}"
)

for JUDGE_NAME in "${ALL_JUDGES[@]}"; do
    load_judge_conf "${JUDGE_NAME}" || exit 1

    echo "=== Judge: ${JUDGE_LABEL} ==="

    # Clean the file before every judge, including the first, so an
    # agent-created or stale judgement.json can never be collected as a fresh
    # verdict when a judge CLI fails.
    rm -f "${JOB_DIR}/task/judgement.json"

    JUDGE_PROMPT=$(build_judge_prompt "${JUDGE_NAME}" "${EVALUATION_TASK}" "${MODEL_TO_TRAIN}" "${AGENT}" "${AGENT_CONFIG}")

    with_huggingface_overlay run_judge_exec "${JOB_DIR}" "${JOB_TMP}" "${EVAL_DIR}/judge_output_${JUDGE_OUTPUT_ID}.json" "${JUDGE_PROMPT}"

    # missing_fatal=0: a judge that produces no verdict warns and moves on. The
    # agent's 10h of work is already done, so it must still be evaluated; the
    # rerun pipeline can supply the missing verdict afterwards.
    collect_judge_output "${JOB_DIR}" "${EVAL_DIR}" "" 0
done

if [ "$JUDGE_LOCK_HELD" = "1" ]; then
    flock -u 9
    exec 9>&-
    echo "Official judge auth lock released."
fi

fi  # hv-patches: end of JUDGE_AUTH_MODE != skip

echo "================================"
echo "========= EVALUATING ==========="
echo "================================"

export REPO_ROOT="$(pwd)"

export TMP_HF_CACHE="${PTB_SCRATCH_ROOT%/}/eval_hf_cache_${RANDOM_UUID}"

export EVAL_COUNTER=0
source src/utils/gpu_reap.sh

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
    ptb_reap_allocated_gpu_processes
    sleep 5
    with_huggingface_overlay apptainer exec \
        --nv \
        -c \
        --cleanenv \
        --pid \
        --no-init \
        --env "CUDA_VISIBLE_DEVICES=${POST_TRAIN_BENCH_VISIBLE_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" \
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
        --writable-tmpfs \
        --home "${eval_home}:${HOME}" \
        --bind "${JOB_TMP}:/tmp" \
        --bind "${REPO_ROOT}:${REPO_ROOT}" \
        --bind "${EVAL_DIR}:${EVAL_DIR}" \
        --bind "${HF_MERGED}:${TMP_HF_CACHE}" \
        --pwd "$(pwd)/src/eval/tasks/${EVALUATION_TASK}" \
        ${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif python "${EVAL_SCRIPT}" \
            --model-path "$EVAL_DIR/final_model" \
            --templates-dir ../../../../src/eval/templates \
            --limit -1 \
            ${max_tokens_arg} \
            --json-output-file "${EVAL_DIR}/metrics.json" > "$EVAL_DIR/final_eval_${eval_num}.txt"
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

        timeout --signal=TERM --kill-after=60s 28800s bash -c "$(declare -f run_evaluation with_huggingface_overlay ptb_reap_allocated_gpu_processes); run_evaluation \"$max_tokens_arg\" \"$EVAL_COUNTER\""

        if [ -f "${EVAL_DIR}/metrics.json" ]; then
            return 0
        fi
    done

    return 1
}

# First evaluation: up to 4 attempts
run_evaluation_with_retry 4 ""

# Second evaluation with adjusted max tokens: up to 2 attempts
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

echo $(cat "$EVAL_DIR/final_eval_${EVAL_COUNTER}.txt")

echo "================================"
echo "======= EVALUATION DONE ========"
echo "================================"

if [ "${POST_TRAIN_BENCH_REQUIRE_COMPLETE:-0}" = "1" ]; then
    python3 src/utils/validate_completed_run.py \
        "$EVAL_DIR" --judge-profile "$JUDGE_PROFILE"
fi
