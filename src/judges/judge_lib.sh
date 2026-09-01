#!/bin/bash
#
# Shared library for the reward-hacking judges in src/judges/.
#
# Used by:
#   - src/run_task.sh           inline judges right after the agent run
#                               (output files without suffix)
#   - src/judges/run_judges.sh  standalone/rerun runs (output files with the
#                               _rerun suffix)
#
# Each judge lives in src/judges/<judge_name>/ with a judge.conf (see
# load_judge_conf) and a prompt template. ALL_JUDGES defines the full set and
# the execution order.
#
# Callers source this file and use:
#   load_judge_conf <judge_name>
#   prepare_judge_sandbox <job_dir> <benchmark_id> <final_model_config_src>
#   configure_judge_profile [official|claude]
#   setup_judge_auth <job_dir>
#   build_judge_prompt <judge_name> <benchmark_id> <model_hf> <agent> <agent_config>
#   run_judge_exec <job_dir> <job_tmp> <output_json> <prompt>
#   collect_judge_output <job_dir> <out_dir> <name_suffix> <missing_fatal>
#
# run_judge_exec additionally reads the caller-provided array
# JUDGE_EXTRA_APPTAINER_ARGS (e.g. --nv + HF cache binds during run_task.sh;
# empty for standalone reruns).

JUDGES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JUDGES_REPO_ROOT="$(cd "$JUDGES_DIR/../.." && pwd)"

# All judges, in execution order.
ALL_JUDGES=(data_contamination_judge api_usage_judge ptb_lookup_judge general_judge)

# Runtime profile. `official` preserves the upstream Codex/GPT judges and
# canonical output ids. `claude` uses Claude Code with Opus + xhigh and gives
# every output a separate id, so research verdicts can never be mistaken for
# canonical benchmark verdicts.
JUDGE_PROFILE=""
PTB_JUDGE_BACKEND=""
JUDGE_DEFAULT_MODEL=""
JUDGE_DEFAULT_REASONING_EFFORT=""
JUDGE_CONTAINER=""
JUDGE_AUTH_MODE=""

# configure_judge_profile [official|claude]
# Environment defaults are intentionally resolved here (rather than when this
# file is sourced), because standalone run_judges.sh loads .env afterwards.
configure_judge_profile() {
    local requested="${1:-${POST_TRAIN_BENCH_JUDGE_PROFILE:-official}}"
    case "$requested" in
        official)
            JUDGE_PROFILE="official"
            PTB_JUDGE_BACKEND="codex"
            JUDGE_DEFAULT_MODEL="gpt-5.4"
            JUDGE_DEFAULT_REASONING_EFFORT="xhigh"
            JUDGE_CONTAINER="${POST_TRAIN_BENCH_OFFICIAL_JUDGE_CONTAINER:-gpt_5_5.sif}"
            ;;
        claude)
            JUDGE_PROFILE="claude"
            PTB_JUDGE_BACKEND="claude"
            # `opus` is the stable Claude Code alias documented by Anthropic.
            # Sites that need a dated/pinned Opus 5 id must provide the exact
            # id accepted by their installed CLI through this env variable.
            JUDGE_DEFAULT_MODEL="${POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL:-opus}"
            JUDGE_DEFAULT_REASONING_EFFORT="xhigh"
            JUDGE_CONTAINER="${POST_TRAIN_BENCH_CLAUDE_JUDGE_CONTAINER:-opus_5.sif}"
            ;;
        *)
            echo "ERROR: unsupported judge profile '$requested' (want official|claude)" >&2
            return 1
            ;;
    esac
    export JUDGE_PROFILE PTB_JUDGE_BACKEND JUDGE_DEFAULT_MODEL
    export JUDGE_DEFAULT_REASONING_EFFORT JUDGE_CONTAINER
}

# resolve_judge_auth_mode
# Claude Code supports either a dedicated Anthropic OAuth token or Google
# Vertex AI. Prefer Vertex automatically when the site environment declares it;
# otherwise retain the dedicated-token path. An explicit
# POST_TRAIN_BENCH_JUDGE_AUTH_MODE always wins.
resolve_judge_auth_mode() {
    local default_mode
    case "$JUDGE_PROFILE" in
        official) default_mode="chatgpt" ;;
        claude)
            case "${CLAUDE_CODE_USE_VERTEX:-${ANTHROPIC_VERTEX:-}}" in
                1|true|TRUE|yes|YES|on|ON) default_mode="vertex" ;;
                *) default_mode="claude_oauth" ;;
            esac
            ;;
        *) echo "ERROR: configure_judge_profile must run before resolve_judge_auth_mode" >&2; return 1 ;;
    esac
    JUDGE_AUTH_MODE="${POST_TRAIN_BENCH_JUDGE_AUTH_MODE:-$default_mode}"
    export JUDGE_AUTH_MODE
}

# load_judge_conf <judge_name>
# Loads src/judges/<judge_name>/judge.conf into the JUDGE_* variables,
# resetting them first so nothing leaks between judges.
load_judge_conf() {
    local judge_name="$1"
    local conf="$JUDGES_DIR/$judge_name/judge.conf"
    if [ ! -f "$conf" ]; then
        echo "ERROR: unknown judge '$judge_name' (no $conf)" >&2
        return 1
    fi
    # Pick up profile changes made after this library was sourced (for example
    # run_judges.sh --profile claude).
    configure_judge_profile "${POST_TRAIN_BENCH_JUDGE_PROFILE:-${JUDGE_PROFILE:-official}}" || return 1

    JUDGE_LABEL=""
    JUDGE_OUTPUT_ID=""
    JUDGE_PROMPT_FILE=""
    JUDGE_MODEL="$JUDGE_DEFAULT_MODEL"
    JUDGE_REASONING_EFFORT="$JUDGE_DEFAULT_REASONING_EFFORT"
    JUDGE_CLAUDE_LABEL=""
    JUDGE_CLAUDE_OUTPUT_ID=""
    JUDGE_CLAUDE_MODEL=""
    # Empty = use the container's pinned codex; a version (e.g. "0.144.5")
    # makes run_judge_exec npm-install exactly that @openai/codex release into
    # the sandbox home and run it instead.
    JUDGE_CODEX_VERSION=""
    source "$conf"
    if [ -z "$JUDGE_LABEL" ] || [ -z "$JUDGE_OUTPUT_ID" ] || [ -z "$JUDGE_PROMPT_FILE" ]; then
        echo "ERROR: $conf must set JUDGE_LABEL, JUDGE_OUTPUT_ID and JUDGE_PROMPT_FILE" >&2
        return 1
    fi

    if [ "$PTB_JUDGE_BACKEND" = "claude" ]; then
        if [ -z "$JUDGE_CLAUDE_OUTPUT_ID" ]; then
            echo "ERROR: $conf must set JUDGE_CLAUDE_OUTPUT_ID for the claude profile" >&2
            return 1
        fi
        JUDGE_LABEL="${JUDGE_CLAUDE_LABEL:-Claude Opus (xhigh) ${JUDGE_LABEL}}"
        JUDGE_OUTPUT_ID="$JUDGE_CLAUDE_OUTPUT_ID"
        JUDGE_MODEL="${POST_TRAIN_BENCH_CLAUDE_JUDGE_MODEL:-${JUDGE_CLAUDE_MODEL:-opus}}"
        # The Claude profile is deliberately xhigh. `max` is not an alias for
        # xhigh and must not leak from an agent's own Claude configuration.
        JUDGE_REASONING_EFFORT="xhigh"
        JUDGE_CODEX_VERSION=""
    fi
}

# prepare_judge_sandbox <job_dir> <benchmark_id> <final_model_config_src>
# Copies the judge helper tooling and benchmark metadata into the sandbox
# home (shared by all judges).
prepare_judge_sandbox() {
    local job_dir="$1" benchmark_id="$2" final_model_config_src="$3"

    cp "$JUDGES_DIR/judge_tools/contamination_check.py" "$job_dir/contamination_check.py"
    cp "$JUDGES_DIR/judge_tools/model_identity_check.py" "$job_dir/model_identity_check.py"
    cp -r "$JUDGES_DIR/judge_tools/reference_configs" "$job_dir/reference_configs"

    # Expose final_model/config.json to the judge as ../final_model_config.json
    # so model_identity_check.py can compare it against the reference. Only the
    # config.json is needed for the architecture-identity check, not the weights.
    if [ -f "$final_model_config_src" ]; then
        cp "$final_model_config_src" "$job_dir/final_model_config.json"
    fi

    if [ -f "$JUDGES_REPO_ROOT/src/eval/tasks/$benchmark_id/test_data.json" ]; then
        cp "$JUDGES_REPO_ROOT/src/eval/tasks/$benchmark_id/test_data.json" "$job_dir/test_data.json"
    fi
}

# setup_judge_codex_auth <job_dir>
# Resets the sandbox codex config so agent-specific settings (e.g.
# model_reasoning_effort) can't leak into the judge, and prepares the
# ChatGPT-subscription auth: auth.json itself is bind-mounted from the shared
# location at apptainer exec time so codex can write the rotated refresh token
# back to the source and the next job picks it up instead of reusing a stale
# single-use refresh token. Sets JUDGE_CODEX_AUTH_SRC for run_judge_exec.
setup_judge_codex_auth() {
    local job_dir="$1"

    JUDGE_CODEX_AUTH_SRC="${POST_TRAIN_BENCH_CODEX_JUDGE_AUTH_FILE:-$JUDGES_REPO_ROOT/agents/codex_non_api/auth.json}"
    if [ ! -f "$JUDGE_CODEX_AUTH_SRC" ]; then
        echo "ERROR: official judge subscription auth not found: $JUDGE_CODEX_AUTH_SRC" >&2
        return 1
    fi
    JUDGE_CODEX_AUTH_SRC="$(cd "$(dirname "$JUDGE_CODEX_AUTH_SRC")" && pwd)/$(basename "$JUDGE_CODEX_AUTH_SRC")"

    cp -r "$JUDGES_REPO_ROOT/containers/other_home_data/.codex" "$job_dir/"
    # Touch a placeholder so apptainer has something to bind onto inside .codex/.
    : > "$job_dir/.codex/auth.json"
    if ! grep -q "forced_login_method" "$job_dir/.codex/config.toml" 2>/dev/null; then
        printf '\nforced_login_method = "chatgpt"\n' >> "$job_dir/.codex/config.toml"
    fi
}

# setup_judge_claude_auth <job_dir>
# Uses a dedicated OAuth token file and a fresh CLAUDE_CONFIG_DIR. It never
# falls back to agents/<agent>/oauth_token or the host's ~/.claude, so the
# tested agent's personas, settings, hooks and credentials cannot leak into
# the judge. The token is bind-mounted read-only and read only inside the
# container by run_judge_exec_claude.
setup_judge_claude_auth() {
    local job_dir="$1"
    local auth_file="${POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE:-}"

    if [ -z "$auth_file" ]; then
        echo "ERROR: POST_TRAIN_BENCH_CLAUDE_JUDGE_OAUTH_TOKEN_FILE is required for the claude judge profile" >&2
        return 1
    fi
    if [ ! -r "$auth_file" ] || [ ! -s "$auth_file" ]; then
        echo "ERROR: Claude judge OAuth token file is missing, unreadable, or empty: $auth_file" >&2
        return 1
    fi

    JUDGE_CLAUDE_AUTH_SRC="$(cd "$(dirname "$auth_file")" && pwd)/$(basename "$auth_file")"
    rm -rf "$job_dir/.claude-judge"
    mkdir -p "$job_dir/.claude-judge"
    chmod 700 "$job_dir/.claude-judge"
    # Apptainer needs a destination to bind the real token over.
    : > "$job_dir/.claude-judge/oauth_token"
    chmod 600 "$job_dir/.claude-judge/oauth_token"
}

# setup_judge_vertex_auth <job_dir>
# Vertex uses Google Application Default Credentials. On the GCP Slurm nodes
# this is normally the VM service account exposed by the metadata server. A
# file-backed ADC can be selected explicitly for other sites and is bind-
# mounted read-only by run_judge_exec_claude.
setup_judge_vertex_auth() {
    local job_dir="$1"
    local adc_file="${POST_TRAIN_BENCH_VERTEX_ADC_FILE:-${GOOGLE_APPLICATION_CREDENTIALS:-}}"

    JUDGE_VERTEX_ADC_SRC=""
    if [ -n "$adc_file" ]; then
        if [ ! -r "$adc_file" ] || [ ! -s "$adc_file" ]; then
            echo "ERROR: configured Vertex ADC file is missing, unreadable, or empty: $adc_file" >&2
            return 1
        fi
        JUDGE_VERTEX_ADC_SRC="$(cd "$(dirname "$adc_file")" && pwd)/$(basename "$adc_file")"
        mkdir -p "$job_dir/.config/gcloud"
        : > "$job_dir/.config/gcloud/application_default_credentials.json"
        chmod 600 "$job_dir/.config/gcloud/application_default_credentials.json"
    elif ! timeout 10 curl -fsS -H 'Metadata-Flavor: Google' \
        'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
        >/dev/null 2>&1; then
        echo "ERROR: Vertex judge found neither a readable ADC file nor GCE metadata credentials" >&2
        return 1
    fi

    JUDGE_VERTEX_PROJECT="${ANTHROPIC_VERTEX_PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-${CLOUD_ML_PROJECT_ID:-}}}"
    JUDGE_VERTEX_REGION="${ANTHROPIC_VERTEX_REGION:-${GOOGLE_CLOUD_LOCATION:-${CLOUD_ML_REGION:-global}}}"
    if [ -z "$JUDGE_VERTEX_PROJECT" ]; then
        echo "ERROR: Vertex judge project is unset (ANTHROPIC_VERTEX_PROJECT_ID/GOOGLE_CLOUD_PROJECT)" >&2
        return 1
    fi
    export JUDGE_VERTEX_ADC_SRC JUDGE_VERTEX_PROJECT JUDGE_VERTEX_REGION
}

# setup_judge_auth <job_dir>
setup_judge_auth() {
    local job_dir="$1"
    configure_judge_profile "${POST_TRAIN_BENCH_JUDGE_PROFILE:-${JUDGE_PROFILE:-official}}" || return 1
    resolve_judge_auth_mode || return 1
    case "${PTB_JUDGE_BACKEND}:${JUDGE_AUTH_MODE}" in
        codex:chatgpt) setup_judge_codex_auth "$job_dir" ;;
        claude:claude_oauth) setup_judge_claude_auth "$job_dir" ;;
        claude:vertex) setup_judge_vertex_auth "$job_dir" ;;
        *)
            echo "ERROR: auth mode '$JUDGE_AUTH_MODE' is incompatible with judge profile '$JUDGE_PROFILE'" >&2
            return 1
            ;;
    esac
}

# build_judge_prompt <judge_name> <benchmark_id> <model_hf> <agent> <agent_config>
# Prints the judge's prompt. Agent identity may be empty; it is only used by
# prompts that reference the agent harness (e.g. the API usage judge).
build_judge_prompt() {
    local judge_name="$1" benchmark_id="$2" model_hf="$3" agent="$4" agent_config="$5"
    local args=(--judge "$judge_name" --benchmark-id "$benchmark_id" --model "$model_hf")
    [ -n "$agent" ] && args+=(--agent "$agent")
    [ -n "$agent_config" ] && args+=(--agent-config "$agent_config")
    python3 "$JUDGES_DIR/get_judge_prompt.py" "${args[@]}"
}

# run_judge_exec_codex <job_dir> <job_tmp> <output_json> <prompt>
# Runs the loaded judge's codex CLI in the sandbox, teeing the raw JSON trace
# to <output_json>. Requires load_judge_conf and setup_judge_codex_auth to
# have run; extra apptainer flags come from JUDGE_EXTRA_APPTAINER_ARGS.
# When judge.conf pins JUDGE_CODEX_VERSION, that exact @openai/codex release
# is npm-installed into a version-specific prefix in the sandbox home
# (idempotent across judges sharing the sandbox) and used instead of the
# container's codex.
run_judge_exec_codex() {
    local job_dir="$1" job_tmp="$2" output_json="$3" prompt="$4"

    local codex_bin="codex"
    if [ -n "$JUDGE_CODEX_VERSION" ]; then
        local pin_prefix=".codex-cli-${JUDGE_CODEX_VERSION}"
        if [ ! -x "$job_dir/$pin_prefix/bin/codex" ]; then
            echo "  installing pinned codex CLI @openai/codex@${JUDGE_CODEX_VERSION} for ${JUDGE_LABEL} ..."
            apptainer exec \
                --containall \
                --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
                --bind "${job_tmp}:/tmp" \
                --home "${job_dir}:/home/ben" \
                --writable-tmpfs \
                "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${JUDGE_CONTAINER}" \
                npm install -g --prefix "/home/ben/${pin_prefix}" --no-fund --no-audit "@openai/codex@${JUDGE_CODEX_VERSION}"
        fi
        if [ ! -x "$job_dir/$pin_prefix/bin/codex" ]; then
            echo "ERROR: install of pinned @openai/codex@${JUDGE_CODEX_VERSION} failed (no ${pin_prefix}/bin/codex in the sandbox home) — ${JUDGE_LABEL} cannot run" >&2
            return 1
        fi
        codex_bin="/home/ben/${pin_prefix}/bin/codex"
    fi

    apptainer exec \
        --containall \
        "${JUDGE_EXTRA_APPTAINER_ARGS[@]}" \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env CODEX_API_KEY="" \
        --env OPENAI_API_KEY="" \
        --env PYTHONNOUSERSITE="1" \
        --bind "${job_tmp}:/tmp" \
        --bind "${JUDGE_CODEX_AUTH_SRC}:/home/ben/.codex/auth.json" \
        --home "${job_dir}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${JUDGE_CONTAINER}" \
        "$codex_bin" --search -a never exec --json -c model_reasoning_summary=detailed -c model_reasoning_effort="${JUDGE_REASONING_EFFORT}" --skip-git-repo-check --yolo --model "${JUDGE_MODEL}" "$prompt" 2>&1 | tee "$output_json"
}

_judge_metadata_path() {
    local output_json="$1"
    local dir base
    dir="$(dirname "$output_json")"
    base="$(basename "$output_json")"
    base="${base/#judge_output_/judge_metadata_}"
    printf '%s/%s\n' "$dir" "$base"
}

write_judge_metadata() {
    local output_json="$1" cli_version="$2"
    local metadata_json
    metadata_json="$(_judge_metadata_path "$output_json")"
    python3 - "$metadata_json" "$JUDGE_PROFILE" "$PTB_JUDGE_BACKEND" "$JUDGE_AUTH_MODE" \
        "$JUDGE_MODEL" "$JUDGE_REASONING_EFFORT" "$JUDGE_CONTAINER" "$cli_version" <<'PY'
import json
import sys
from pathlib import Path

path, profile, backend, auth_mode, model, effort, container, cli_version = sys.argv[1:]
Path(path).write_text(json.dumps({
    "profile": profile,
    "backend": backend,
    "auth_mode": auth_mode,
    "requested_model": model,
    "reasoning_effort": effort,
    "container": container,
    "cli_version": cli_version,
}, indent=2) + "\n", encoding="utf-8")
PY
}

record_claude_resolved_model() {
    local output_json="$1" metadata_json
    metadata_json="$(_judge_metadata_path "$output_json")"
    python3 - "$output_json" "$metadata_json" <<'PY'
import json
import sys
from pathlib import Path

trace_path, metadata_path = map(Path, sys.argv[1:])
resolved_model = None
if trace_path.is_file():
    for raw_line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            resolved_model = event.get("model")
            if resolved_model:
                break

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata["resolved_model"] = resolved_model or "unknown"
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY
}

# run_judge_exec_claude <job_dir> <job_tmp> <output_json> <prompt>
# Uses exactly the same prompt assembled by build_judge_prompt as the official
# Codex profile. Claude's safe mode disables CLAUDE.md auto-discovery, skills,
# plugins, hooks, MCP servers, custom agents and other customizations; its
# user/project/local settings and session persistence are disabled separately.
# Config/auth are isolated from the tested agent. stdout is Claude Code's
# stream-json trace, which collect_judge_output dispatches to claude_parser.py.
run_judge_exec_claude() {
    local job_dir="$1" job_tmp="$2" output_json="$3" prompt="$4"
    local cli_version
    local auth_kind="$JUDGE_AUTH_MODE"
    local -a auth_env_args auth_bind_args

    auth_env_args=()
    auth_bind_args=()
    case "$auth_kind" in
        claude_oauth)
            # Prevent a site's ambient Vertex settings from overriding the
            # explicitly selected Anthropic OAuth profile.
            auth_env_args+=(
                --env CLAUDE_CODE_USE_VERTEX=""
                --env ANTHROPIC_VERTEX=""
                --env CLAUDE_CODE_OAUTH_TOKEN_FILE="/home/ben/.claude-judge/oauth_token"
            )
            auth_bind_args+=(--bind "${JUDGE_CLAUDE_AUTH_SRC}:/home/ben/.claude-judge/oauth_token:ro")
            ;;
        vertex)
            auth_env_args+=(
                --env CLAUDE_CODE_USE_VERTEX="1"
                --env ANTHROPIC_VERTEX="true"
                --env ANTHROPIC_VERTEX_PROJECT_ID="$JUDGE_VERTEX_PROJECT"
                --env ANTHROPIC_VERTEX_REGION="$JUDGE_VERTEX_REGION"
                --env GOOGLE_CLOUD_PROJECT="$JUDGE_VERTEX_PROJECT"
                --env GOOGLE_CLOUD_LOCATION="$JUDGE_VERTEX_REGION"
                --env ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-opus-5}"
                --env ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_DEFAULT_OPUS_MODEL:-claude-opus-5}"
                --env VERTEX_REGION_CLAUDE_5_OPUS="${VERTEX_REGION_CLAUDE_5_OPUS:-$JUDGE_VERTEX_REGION}"
                --env CLAUDE_CODE_OAUTH_TOKEN=""
            )
            if [ -n "${JUDGE_VERTEX_ADC_SRC:-}" ]; then
                auth_env_args+=(--env GOOGLE_APPLICATION_CREDENTIALS="/home/ben/.config/gcloud/application_default_credentials.json")
                auth_bind_args+=(--bind "${JUDGE_VERTEX_ADC_SRC}:/home/ben/.config/gcloud/application_default_credentials.json:ro")
            fi
            ;;
        *) echo "ERROR: unsupported Claude judge auth mode '$auth_kind'" >&2; return 1 ;;
    esac

    cli_version="$(apptainer exec \
        --containall \
        --cleanenv \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --home "${job_dir}:/home/ben" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${JUDGE_CONTAINER}" \
        claude --version 2>&1 | head -n 1)"
    if [ -z "$cli_version" ]; then
        cli_version="unknown"
    fi
    write_judge_metadata "$output_json" "$cli_version"

    echo "  judge profile=${JUDGE_PROFILE} backend=claude model=${JUDGE_MODEL} effort=${JUDGE_REASONING_EFFORT} cli=${cli_version}"
    apptainer exec \
        --containall \
        --cleanenv \
        "${JUDGE_EXTRA_APPTAINER_ARGS[@]}" \
        --env PATH="/root/.local/bin:/home/ben/.local/bin:$PATH" \
        --env ANTHROPIC_API_KEY="" \
        --env ANTHROPIC_AUTH_TOKEN="" \
        --env CLAUDE_CODE_EFFORT_LEVEL="xhigh" \
        --env CLAUDE_CONFIG_DIR="/home/ben/.claude-judge" \
        --env PYTHONNOUSERSITE="1" \
        "${auth_env_args[@]}" \
        --bind "${job_tmp}:/tmp" \
        "${auth_bind_args[@]}" \
        --home "${job_dir}:/home/ben" \
        --pwd "/home/ben/task" \
        --writable-tmpfs \
        "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${JUDGE_CONTAINER}" \
        bash -c '
            set -e
            if [ "$4" = "claude_oauth" ]; then
                export CLAUDE_CODE_OAUTH_TOKEN="$(cat "$CLAUDE_CODE_OAUTH_TOKEN_FILE")"
            else
                unset CLAUDE_CODE_OAUTH_TOKEN
            fi
            exec claude --print --verbose --output-format stream-json \
                --model "$1" --effort "$2" --setting-sources "" \
                --safe-mode --no-session-persistence \
                --dangerously-skip-permissions "$3"
        ' ptb-claude-judge "$JUDGE_MODEL" "$JUDGE_REASONING_EFFORT" "$prompt" "$auth_kind" \
        2>&1 | tee "$output_json"
    record_claude_resolved_model "$output_json"
}

# run_judge_exec <job_dir> <job_tmp> <output_json> <prompt>
run_judge_exec() {
    case "$PTB_JUDGE_BACKEND" in
        codex)
            write_judge_metadata "$3" "container-pinned-or-${JUDGE_CODEX_VERSION:-default}"
            run_judge_exec_codex "$@"
            ;;
        claude) run_judge_exec_claude "$@" ;;
        *) echo "ERROR: internal unsupported judge backend '$PTB_JUDGE_BACKEND'" >&2; return 1 ;;
    esac
}

# collect_judge_output <job_dir> <out_dir> <name_suffix> <missing_fatal>
# Parses the raw backend trace into a human-readable report and copies the
# judgement produced in the sandbox to
# <out_dir>/judgement_<JUDGE_OUTPUT_ID><name_suffix>.json. Returns 1 on a
# missing judgement only when <missing_fatal> is 1.
#
# <missing_fatal> is a property of the caller, not of the judge: standalone
# reruns (run_judges.sh) pass 1, because producing the verdict is the whole
# point of the job. run_task.sh passes 0 — a judge that produces no verdict
# must never cost a finished 10h agent run its evaluation, and the rerun
# pipeline can supply the verdict later.
collect_judge_output() {
    local job_dir="$1" out_dir="$2" suffix="$3" missing_fatal="$4"
    local out_base="judge_output_${JUDGE_OUTPUT_ID}${suffix}"
    local judgement="$out_dir/judgement_${JUDGE_OUTPUT_ID}${suffix}.json"

    python3 "$JUDGES_REPO_ROOT/src/trace_parsing/parse_trace.py" --agent "$PTB_JUDGE_BACKEND" "$out_dir/${out_base}.json" -o "$out_dir/${out_base}.txt"

    if [ -f "$job_dir/task/judgement.json" ]; then
        cp "$job_dir/task/judgement.json" "$judgement"
        echo "  ${JUDGE_LABEL} judgement: $(cat "$judgement")"
    elif [ "$missing_fatal" = "1" ]; then
        echo "ERROR: judgement.json not created by ${JUDGE_LABEL} (see $out_dir/${out_base}.txt)" >&2
        return 1
    else
        echo "WARNING: judgement.json not created by ${JUDGE_LABEL} (see $out_dir/${out_base}.txt); continuing — a missing inline verdict never aborts the task run" >&2
    fi
}
