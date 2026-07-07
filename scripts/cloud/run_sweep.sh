#!/usr/bin/env bash
# Launch a seed sweep on a cloud box, auto-sizing concurrency to the core count and
# running detached so it survives SSH disconnects.
#
# Usage:
#   bash scripts/cloud/run_sweep.sh <preset> "<conditions>" "<seeds>" [out_name]
# Examples:
#   bash scripts/cloud/run_sweep.sh calib5x5 \
#       "brain_neuromod brain_no_neuromod brain_random_code brain_oracle_code" "0 1 2 3 4"
#   bash scripts/cloud/run_sweep.sh scout5x5 "brain_neuromod brain_no_neuromod" "0 1 2"
#
# Env toggles:
#   CORES_PER_CELL=8   cores budgeted per concurrent cell (default 8) -> max_parallel=nproc/CORES_PER_CELL
#   MAX_PARALLEL=...   override the computed concurrency
#   DEVICE=cuda        device passed to training (default cuda)
set -euo pipefail

PRESET="${1:?preset required (pilot|calib5x5|scout5x5)}"
CONDITIONS="${2:?conditions required, space-separated}"
SEEDS="${3:?seeds required, space-separated}"
OUT_NAME="${4:-${PRESET}_$(date -u +%Y%m%d-%H%M%S)}"

CORES_PER_CELL="${CORES_PER_CELL:-8}"
CORES="$(nproc)"
MAX_PARALLEL="${MAX_PARALLEL:-$(( CORES / CORES_PER_CELL ))}"
[ "$MAX_PARALLEL" -lt 1 ] && MAX_PARALLEL=1
DEVICE="${DEVICE:-cuda}"

# Resume is ON by default (spot-instance safe): forces periodic checkpoints and continues
# interrupted cells on re-launch. Set RESUME=0 to force all-fresh runs.
RESUME="${RESUME:-1}"
RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"

OUT_DIR="sweeps/${OUT_NAME}"
mkdir -p "$OUT_DIR"
LOG="${OUT_DIR}/sweep.out"

echo "[run_sweep] preset=$PRESET cores=$CORES cores_per_cell=$CORES_PER_CELL max_parallel=$MAX_PARALLEL"
echo "[run_sweep] conditions=[$CONDITIONS] seeds=[$SEEDS] -> $OUT_DIR"
echo "[run_sweep] streaming to $LOG (tail -f to watch); detached via nohup."

# Auto send-back (docs/multi_agent/0001): after the sweep finishes, push the light results bundle
# to the `results` branch (needs GH_TOKEN on the box; otherwise skips non-fatally). Disable with
# AUTO_PUSH=0.
AUTO_PUSH="${AUTO_PUSH:-1}"

# Run the sweep and the post-run push as one detached group so the push fires on completion and
# still survives an SSH disconnect. Braces keep $CONDITIONS/$SEEDS word-splitting intact.
# shellcheck disable=SC2086
{
    PYTHONPATH=src PYTHONUNBUFFERED=1 python scripts/run_seed_sweep.py \
        --preset "$PRESET" \
        --conditions $CONDITIONS \
        --seeds $SEEDS \
        --max-parallel "$MAX_PARALLEL" \
        --device "$DEVICE" \
        $RESUME_FLAG \
        --out "$OUT_DIR"
    sweep_ec=$?
    echo "[run_sweep] sweep exited ec=$sweep_ec"
    if [ "$AUTO_PUSH" = "1" ]; then
        # push_results.sh retries branch races internally; a final failure prints the
        # greppable RESULTS_PUSH_FAILED marker so watchers can alert instead of the
        # dir silently vanishing from origin/results.
        bash scripts/cloud/push_results.sh "$OUT_DIR" \
            || echo "[run_sweep] RESULTS_PUSH_FAILED for $OUT_DIR — recover ${OUT_DIR}/results_bundle.tgz manually"
    fi
} > "$LOG" 2>&1 &
disown 2>/dev/null || true

echo "[run_sweep] PID $! — results land in ${OUT_DIR}/runs.csv + summary.csv"
echo "[run_sweep] auto-push=$AUTO_PUSH (needs GH_TOKEN to actually push); watch: tail -f $LOG"
