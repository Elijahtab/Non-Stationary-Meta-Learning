#!/usr/bin/env bash
# Auto send-back (plan docs/multi_agent/0001, option A): on sweep completion, push the LIGHT
# results bundle to a durable place so results survive the ephemeral/flaky cloud box.
#
# It commits the sweep's analysis artifacts (summary/runs/sweep.out/logs + each run's brain_trends
# data.json, brain_model.pt, config.txt — a few MB, NO episode_*/ bulk) onto an orphan-style
# `results` branch on the git remote, under results/<sweep_name>/. The working branch/index is
# never touched (uses a throwaway worktree).
#
# Usage (normally called by run_sweep.sh, but standalone works):
#   GH_TOKEN=ghp_xxx bash scripts/cloud/push_results.sh <sweep_out_dir>
# Env:
#   GH_TOKEN / GITHUB_TOKEN   PAT with repo write. If unset, the push is SKIPPED (non-fatal) and
#                             the bundle is left tarred at <sweep_out_dir>/results_bundle.tgz so it
#                             can still be pulled manually. This is the box's current state.
#   RESULTS_BRANCH=results    branch to append to (default "results")
#   REMOTE_URL=...            override the push URL (default: origin's URL)
set -euo pipefail

SWEEP_DIR="${1:?sweep out dir required, e.g. sweeps/calib8x8_20260701-...}"
[ -d "$SWEEP_DIR" ] || { echo "[push] no such dir: $SWEEP_DIR"; exit 1; }
SWEEP_NAME="$(basename "$SWEEP_DIR")"
RESULTS_BRANCH="${RESULTS_BRANCH:-results}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"

cd "$REPO_ROOT"

# 1. Stage the light bundle into a temp dir mirroring results/<sweep>/ .
STAGE="$(mktemp -d)"
DST="$STAGE/results/$SWEEP_NAME"
mkdir -p "$DST"
# sweep-level artifacts (already light: KB of csv/json/logs)
cp -r "$SWEEP_DIR"/. "$DST/" 2>/dev/null || true
rm -f "$DST/results_bundle.tgz" 2>/dev/null || true
# per-run light files referenced by this sweep's runs.csv (brain_trends + brain_model.pt + config).
# run_dir is column 20 and holds an absolute path; resolve by basename so this works whether the
# CSV was written on the box (/workspace/...) or locally.
if [ -f "$SWEEP_DIR/runs.csv" ]; then
  while IFS= read -r rd; do
    [ -n "$rd" ] || continue
    src="$REPO_ROOT/runs/$(basename "$rd")"
    [ -d "$src" ] || continue
    rel="runs/$(basename "$rd")"; mkdir -p "$DST/$rel"
    cp -r "$src/brain_trends" "$DST/$rel/" 2>/dev/null || true
    cp "$src/brain_model.pt" "$src/config.txt" "$DST/$rel/" 2>/dev/null || true
  done < <(awk -F, 'NR>1{print $20}' "$SWEEP_DIR/runs.csv")
fi
# always leave a local tarball as the no-token fallback
tar czf "$SWEEP_DIR/results_bundle.tgz" -C "$STAGE" "results/$SWEEP_NAME" 2>/dev/null || true
echo "[push] staged light bundle ($(du -sh "$DST" | cut -f1)); fallback tar: $SWEEP_DIR/results_bundle.tgz"

# 2. Decide whether a push is even possible: an explicit token, or a git credential helper
#    (bootstrap.sh configures one from GH_TOKEN). Otherwise skip non-fatally — the tarball above
#    is left for a manual pull. Never allow an interactive prompt to hang a detached sweep.
export GIT_TERMINAL_PROMPT=0
if [ -z "$TOKEN" ] && ! git config --get credential.helper >/dev/null 2>&1; then
  echo "[push] no GH_TOKEN and no git credential helper — skipping git push."
  echo "[push] Pull manually with scripts/cloud/pull_results.sh, or set GH_TOKEN (docs/multi_agent/0001)."
  rm -rf "$STAGE"; exit 0
fi

# 3. Commit onto the results branch via a throwaway worktree (never touches the current
#    checkout). Multiple cells finishing in the same window race on the branch tip; a losing
#    non-fast-forward push used to be swallowed as "non-fatal" upstream and silently dropped
#    the dir (LOOP-0006/0007 lost ~20 per-seed dirs this way). Now every attempt re-fetches
#    the fresh tip, re-commits the bundle on top, and retries with backoff; exhausting the
#    retries is a LOUD failure (nonzero exit + greppable marker in the sweep log).
REMOTE_URL="${REMOTE_URL:-$(git remote get-url origin)}"
PUSH_URL="$REMOTE_URL"
# Embed the token only when we actually have one; otherwise rely on the credential helper.
if [ -n "$TOKEN" ]; then
  case "$REMOTE_URL" in
    https://github.com/*) PUSH_URL="https://x-access-token:${TOKEN}@github.com/${REMOTE_URL#https://github.com/}" ;;
  esac
fi
WT="$(mktemp -d)"
cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; rm -rf "$STAGE" "$WT"; }
trap cleanup EXIT
git worktree add -f "$WT" HEAD >/dev/null

MAX_PUSH_ATTEMPTS="${MAX_PUSH_ATTEMPTS:-6}"
pushed=0
for attempt in $(seq 1 "$MAX_PUSH_ATTEMPTS"); do
  git fetch --quiet "$PUSH_URL" "+refs/heads/${RESULTS_BRANCH}:refs/remotes/results_push/${RESULTS_BRANCH}" 2>/dev/null || true
  if git rev-parse -q --verify "refs/remotes/results_push/${RESULTS_BRANCH}" >/dev/null; then
    ( cd "$WT" && git checkout -qB "$RESULTS_BRANCH" "refs/remotes/results_push/${RESULTS_BRANCH}" )
  else
    # First-ever push: build the branch as an orphan (retries reuse the local branch).
    ( cd "$WT" && { git checkout -q --orphan "$RESULTS_BRANCH" 2>/dev/null \
                      && { git rm -rqf . >/dev/null 2>&1 || true; } \
                    || git checkout -q "$RESULTS_BRANCH"; } )
  fi
  rm -rf "$WT/results/$SWEEP_NAME"
  mkdir -p "$WT/results"
  cp -r "$STAGE/results/$SWEEP_NAME" "$WT/results/"
  if (
    cd "$WT"
    git add -A "results/$SWEEP_NAME"
    git -c user.email="cloud@runpod" -c user.name="cloud-sweep" \
        commit -q -m "results: $SWEEP_NAME ($(date -u +%Y-%m-%dT%H:%M:%SZ))" \
      || { echo "[push] nothing to commit — branch already has identical content"; exit 0; }
    git push -q "$PUSH_URL" "HEAD:refs/heads/${RESULTS_BRANCH}"
  ); then
    pushed=1
    break
  fi
  echo "[push] attempt $attempt/$MAX_PUSH_ATTEMPTS lost the branch race (or push failed); retrying on the fresh tip..."
  sleep $(( (RANDOM % 5) + attempt * 3 ))
done

if [ "$pushed" != "1" ]; then
  echo "[push] RESULTS_PUSH_FAILED: results/$SWEEP_NAME after $MAX_PUSH_ATTEMPTS attempts — bundle kept at $SWEEP_DIR/results_bundle.tgz"
  exit 1
fi
echo "[push] pushed results/$SWEEP_NAME to branch '${RESULTS_BRANCH}'."
