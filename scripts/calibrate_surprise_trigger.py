"""Desk-calibrate the head-bank surprise trigger from archived eval traces ($0).

Replays the EMA change-point detector (same math as train.py's cand-4 spike: fire when
value_loss > EMA * (1 + threshold), EMA decay 0.9, cooldown 8 updates) over the `loss/value`
traces of archived control evals (default: the LOOP-0009 seed-1 trained arm, n=16), scoring
each candidate threshold against the ground-truth switch schedule (every 100k steps).

A fired trigger is a TRUE POSITIVE if it lands within `--window` updates after a true switch;
otherwise a false positive. Recall = detected switches / true switches. This picks the
threshold the A-R1 rung pre-registers (research-log 0009) — calibration data is the *control*
arm, disjoint from any head-bank result.

Usage:
    PYTHONPATH=src myenv\\Scripts\\python.exe scripts/calibrate_surprise_trigger.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

EMA_DECAY = 0.9
COOLDOWN_UPDATES = 8


def replay(trace: list[list[float]], threshold: float) -> list[int]:
    """Return global_steps at which the detector fires (EMA + cooldown, train.py semantics)."""
    ema = None
    cooldown = 0
    fired = []
    for step, v in trace:
        if ema is None:
            ema = v
            continue
        if cooldown > 0:
            cooldown -= 1
        elif v > ema * (1.0 + threshold):
            fired.append(int(step))
            cooldown = COOLDOWN_UPDATES
        ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * v
    return fired


def score(fired: list[int], switches: list[int], window_steps: int) -> tuple[int, int, int]:
    """(true_pos, false_pos, detected_switches)."""
    tp = fp = 0
    detected = set()
    for f in fired:
        hit = None
        for sw in switches:
            if 0 <= f - sw <= window_steps:
                hit = sw
                break
        if hit is not None:
            tp += 1
            detected.add(hit)
        else:
            fp += 1
    return tp, fp, len(detected)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="evals/loop9_s1_model_e*", help="eval dirs to calibrate on")
    p.add_argument("--steps-per-regime", type=int, default=100000)
    p.add_argument("--total-steps", type=int, default=800000)
    p.add_argument("--window", type=int, default=4, help="updates after a switch that count as detection")
    p.add_argument("--steps-per-update", type=int, default=2048)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    args = p.parse_args()

    switches = list(range(args.steps_per_regime, args.total_steps, args.steps_per_regime))
    window_steps = args.window * args.steps_per_update

    dirs = sorted(glob.glob(str(REPO_ROOT / args.glob)))
    traces = []
    for d in dirs:
        js = glob.glob(str(Path(d) / "**" / "*_data.json"), recursive=True)
        if js:
            traces.append(json.load(open(js[0]))["loss/value"])
    print(f"calibrating on {len(traces)} traces | switches/run: {len(switches)} | "
          f"window: {args.window} updates ({window_steps} steps)\n")
    print(f"{'threshold':>9} {'precision':>10} {'recall':>8} {'fires/run':>10} {'lag(upd)':>9}")

    best = None
    for thr in args.thresholds:
        tps = fps = dets = fires = 0
        lags = []
        for tr in traces:
            fired = replay(tr, thr)
            tp, fp, det = score(fired, switches, window_steps)
            tps += tp; fps += fp; dets += det; fires += len(fired)
            for f in fired:
                for sw in switches:
                    if 0 <= f - sw <= window_steps:
                        lags.append((f - sw) / args.steps_per_update)
                        break
        precision = tps / max(1, tps + fps)
        recall = dets / (len(switches) * len(traces))
        print(f"{thr:>9.2f} {precision:>10.2f} {recall:>8.2f} {fires/len(traces):>10.1f} "
              f"{np.mean(lags) if lags else float('nan'):>9.1f}")
        # pick: max recall subject to precision >= 0.60 (the A-R1 kill bar), tie-break precision
        if precision >= 0.60 and (best is None or recall > best[2] or
                                  (recall == best[2] and precision > best[1])):
            best = (thr, precision, recall)

    if best:
        print(f"\nchosen threshold: {best[0]} (precision {best[1]:.2f}, recall {best[2]:.2f})")
    else:
        print("\nNO threshold reaches precision >= 0.60 — A-R1's premise fails at the desk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
