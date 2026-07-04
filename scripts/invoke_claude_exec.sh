#!/usr/bin/env bash
# Non-interactive Claude Code (Fable) launcher for one autoresearch trial — Linux/cloud-box twin
# of scripts/invoke_claude_exec.ps1. Reads the rendered prompt, pipes it into `claude -p`, runs
# unattended, and writes stdout/stderr + the extracted final assistant message to disk. The outer
# autoresearch supervisor (run_autoresearch.py) is the real safety boundary — it audits the
# resulting diff against the editable surface and rolls back violations.
#
# Usage (matches run_research_trial.py's AUTORESEARCH_AGENT_COMMAND template):
#   bash scripts/invoke_claude_exec.sh \
#     --prompt-file "{prompt_file}" --repo-root "{repo_root}" \
#     --final-message-file "{final_message_file}" \
#     --stdout-file "{agent_stdout_file}" --stderr-file "{agent_stderr_file}" \
#     [--model claude-fable-5] [--claude-binary claude]
set -uo pipefail

MODEL="claude-fable-5"
CLAUDE_BINARY="claude"
PROMPT_FILE="" REPO_ROOT="" FINAL_MESSAGE_FILE="" STDOUT_FILE="" STDERR_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --prompt-file)        PROMPT_FILE="$2"; shift 2 ;;
        --repo-root)          REPO_ROOT="$2"; shift 2 ;;
        --final-message-file) FINAL_MESSAGE_FILE="$2"; shift 2 ;;
        --stdout-file)        STDOUT_FILE="$2"; shift 2 ;;
        --stderr-file)        STDERR_FILE="$2"; shift 2 ;;
        --model)              MODEL="$2"; shift 2 ;;
        --claude-binary)      CLAUDE_BINARY="$2"; shift 2 ;;
        *) echo "[invoke_claude_exec] unknown arg: $1" >&2; exit 2 ;;
    esac
done

for required in PROMPT_FILE REPO_ROOT FINAL_MESSAGE_FILE STDOUT_FILE STDERR_FILE; do
    [ -n "${!required}" ] || { echo "[invoke_claude_exec] --$(echo "$required" | tr '_A-Z' '-a-z') is required" >&2; exit 2; }
done
[ -f "$PROMPT_FILE" ] || { echo "[invoke_claude_exec] prompt file not found: $PROMPT_FILE" >&2; exit 2; }

mkdir -p "$(dirname "$FINAL_MESSAGE_FILE")" "$(dirname "$STDOUT_FILE")" "$(dirname "$STDERR_FILE")"

# Vast boxes put node/npm on PATH only in interactive login shells (nvm). Source it so the
# npm-installed `claude` binary resolves in detached/supervisor shells too.
if ! command -v "$CLAUDE_BINARY" >/dev/null 2>&1 && [ -s /opt/nvm/nvm.sh ]; then
    # shellcheck disable=SC1091
    . /opt/nvm/nvm.sh
fi
command -v "$CLAUDE_BINARY" >/dev/null 2>&1 || { echo "[invoke_claude_exec] cannot resolve '$CLAUDE_BINARY' on PATH" >&2; exit 127; }

# Claude Code refuses --dangerously-skip-permissions as root unless it is told it's in a
# sandbox. Cloud containers run as root and are throwaway, which is the intended habitat.
export IS_SANDBOX=1

# Without this guard (set -e is absent) a bad REPO_ROOT would run claude with
# --dangerously-skip-permissions in whatever directory we happened to start in.
cd "$REPO_ROOT" || { echo "[invoke_claude_exec] cannot cd to repo root: $REPO_ROOT" >&2; exit 1; }
"$CLAUDE_BINARY" -p \
    --model "$MODEL" \
    --output-format json \
    --dangerously-skip-permissions \
    < "$PROMPT_FILE" > "$STDOUT_FILE" 2> "$STDERR_FILE"
EXIT_CODE=$?

# Parse the machine-readable result and extract the final assistant message so downstream
# tooling reads it the same way it reads the Codex/PowerShell wrapper's output. Fall back to
# raw stdout if the JSON is unparseable; surface an agent-reported error as a non-zero exit.
if command -v jq >/dev/null 2>&1 && jq -e . "$STDOUT_FILE" >/dev/null 2>&1; then
    if jq -e '.result != null' "$STDOUT_FILE" >/dev/null 2>&1; then
        jq -r '.result' "$STDOUT_FILE" > "$FINAL_MESSAGE_FILE"
    else
        # Parity with the .ps1 twin: a parseable payload with a null/absent .result keeps
        # the raw stdout as the final message so error diagnostics aren't blanked.
        cp "$STDOUT_FILE" "$FINAL_MESSAGE_FILE"
    fi
    if [ "$(jq -r '.is_error // false' "$STDOUT_FILE")" = "true" ] && [ "$EXIT_CODE" -eq 0 ]; then
        EXIT_CODE=1
    fi
else
    cp "$STDOUT_FILE" "$FINAL_MESSAGE_FILE"
fi

exit "$EXIT_CODE"
