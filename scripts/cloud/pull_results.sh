#!/usr/bin/env bash
# Pull sweep/run results off a cloud box BEFORE teardown, storage-consciously.
#
# The heavy stuff in runs/ lives entirely under each run's episode_*/ folders:
#   - episode_NN/.../inner_checkpoints/*.pt   ~54 MB each, one per episode  (~12 GB/sweep)
#   - episode_NN/.../epNN_env0_data.json      ~20 MB each, step-level dump  (~5 GB/sweep)
# The analysis-critical data (summary/runs CSV+JSON, sweep.out, per-cell logs, brain_trends
# charts, the 274 KB brain_model.pt, config) is a few MB. So by default we grab everything
# EXCEPT episode_*/  -> a full sweep drops from ~19 GB to ~10 MB.
#
# Usage:
#   HOST=209.33.172.149 PORT=12338 bash scripts/cloud/pull_results.sh [dest_dir]
# Examples:
#   # light bundle (default) -> sweeps/cloud_pull_<UTC>/
#   HOST=1.2.3.4 PORT=12338 bash scripts/cloud/pull_results.sh
#   # also pull the FINAL inner checkpoint + brain_model.pt for two cells you intend to extend
#   HOST=1.2.3.4 PORT=12338 FINAL_CKPT_CELLS="*trainable_seed1* *trainable_seed3*" \
#       bash scripts/cloud/pull_results.sh
#
# Env:
#   HOST=...            remote host/IP                              (required)
#   PORT=22             ssh port                                    (default 22)
#   SSH_USER=root       remote user                                 (default root)
#   REMOTE=/workspace/Lifelong-Learning   remote repo path         (default shown)
#   FINAL_CKPT_CELLS="" space-separated run-dir globs whose FINAL inner checkpoint to also
#                       fetch (opt-in; each ~54 MB). Empty = light bundle only.
set -euo pipefail

HOST="${HOST:?HOST required (get the current IP from the Vast dashboard Connect button)}"
PORT="${PORT:-22}"
SSH_USER="${SSH_USER:-root}"
REMOTE="${REMOTE:-/workspace/Lifelong-Learning}"
FINAL_CKPT_CELLS="${FINAL_CKPT_CELLS:-}"
DEST="${1:-sweeps/cloud_pull_$(date -u +%Y%m%d-%H%M%S)}"

SSHO="-p ${PORT} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=40 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
# Vast prints a 3-line login banner on stdout; strip it from control-command output.
NOBANNER='grep -vE "Welcome to vast|Have fun|AI agents"'
TGT="${SSH_USER}@${HOST}"

mkdir -p "$DEST"
echo "[pull] $TGT:$REMOTE  ->  $DEST"

# 1. Build the light bundle remotely (tar to a file, then scp — a piped tar stream would be
#    corrupted by the login banner; scp uses its own protocol and is banner-safe).
echo "[pull] building light bundle (excluding episode_*/) ..."
eval ssh $SSHO "$TGT" \
  "'cd $REMOTE && tar czf /tmp/pull_light.tgz --exclude=\"episode_*\" sweeps runs && du -h /tmp/pull_light.tgz'" \
  2>&1 | eval "$NOBANNER"

scp $SSHO "$TGT:/tmp/pull_light.tgz" "$DEST/pull_light.tgz"
tar xzf "$DEST/pull_light.tgz" -C "$DEST"
rm -f "$DEST/pull_light.tgz"
eval ssh $SSHO "$TGT" "'rm -f /tmp/pull_light.tgz'" 2>&1 | eval "$NOBANNER" || true
echo "[pull] light bundle extracted -> $DEST ($(du -sh "$DEST" | cut -f1))"

# 2. Opt-in: final inner checkpoint (+ brain_model.pt) for named cells, for resume/extension.
if [ -n "$FINAL_CKPT_CELLS" ]; then
  echo "[pull] fetching FINAL checkpoints for: $FINAL_CKPT_CELLS"
  for glob in $FINAL_CKPT_CELLS; do
    # newest run dir matching the glob; its highest-numbered episode's inner checkpoint.
    # shellcheck disable=SC2086
    ck=$(eval ssh $SSHO "$TGT" "'ls -d $REMOTE/runs/$glob 2>/dev/null | sort | tail -1'" 2>/dev/null | eval "$NOBANNER" | tr -d "\r")
    [ -z "$ck" ] && { echo "[pull]   no run dir matches $glob — skipping"; continue; }
    last=$(eval ssh $SSHO "$TGT" "'ls -d $ck/episode_*/ 2>/dev/null | sed s#/\$## | sort -t_ -k2 -n | tail -1'" 2>/dev/null | eval "$NOBANNER" | tr -d "\r")
    rel="${ck#"$REMOTE"/}"
    mkdir -p "$DEST/$rel"
    scp $SSHO "$TGT:$ck/brain_model.pt" "$DEST/$rel/" 2>/dev/null || true
    if [ -n "$last" ]; then
      lrel="${last#"$REMOTE"/}"
      mkdir -p "$DEST/$lrel"
      scp -r $SSHO "$TGT:$last/*/inner_checkpoints" "$DEST/$lrel/" 2>/dev/null || true
      echo "[pull]   $glob -> brain_model.pt + $(basename "$last")/inner_checkpoints"
    fi
  done
fi

echo "[pull] done. Remember to destroy the Vast instance to stop billing."
