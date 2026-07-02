#!/usr/bin/env bash
# Upload the local GitHub token to a cloud box on start, so `git push` / auto send-back work.
#
# Policy (docs/multi_agent/0001): the token lives ONLY in the gitignored .secrets/gh_token on the
# local machine. It is never committed and never baked into an image — we push it to each box when
# the box comes up. This script is that "on start" step; run it once after bootstrap.
#
# Usage:
#   HOST=209.33.172.149 PORT=12338 bash scripts/cloud/upload_secrets.sh
# Env:
#   HOST=...              box IP/host                       (required)
#   PORT=22              ssh port                           (default 22)
#   SSH_USER=root        remote user                        (default root)
#   TOKEN_FILE=...       path to the token                  (default .secrets/gh_token)
set -euo pipefail

HOST="${HOST:?HOST required (from the Vast dashboard Connect button)}"
PORT="${PORT:-22}"
SSH_USER="${SSH_USER:-root}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
TOKEN_FILE="${TOKEN_FILE:-$REPO_ROOT/.secrets/gh_token}"
[ -s "$TOKEN_FILE" ] || { echo "[secrets] no token at $TOKEN_FILE — create it first (one line: the PAT)"; exit 1; }

SSHO="-p ${PORT} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=40"

# Send the token over stdin (never on the command line / process list). The remote reads one line
# and writes a git credential store entry that works in any shell, incl. detached sweeps.
{ tr -d '\r\n' < "$TOKEN_FILE"; printf '\n'; } | ssh $SSHO "${SSH_USER}@${HOST}" \
  'read -r t && git config --global credential.helper store \
     && printf "https://x-access-token:%s@github.com\n" "$t" > "$HOME/.git-credentials" \
     && chmod 600 "$HOME/.git-credentials" \
     && echo "[secrets] git credentials installed on box; auto send-back armed."' \
  2>&1 | grep -vE "Welcome to vast|Have fun|AI agents"
