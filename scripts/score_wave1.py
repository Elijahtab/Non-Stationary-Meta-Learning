"""Score + adjudicate the Wave-1 batches against the pre-registered gates (research-log 0008).

  ladder  G-DECOMP: arms heads/headsenc/wm/full (evals/wave1_decomp_*) vs the archived
          same-protocol control (evals/loop9_s1_model_e[1-8]_*).
          Computes H = mean(full) - mean(control), per-scope shares of H, Welch p per arm
          vs control, and evaluates P-W1a + P-W1b including the +/-10pp extension rule.

  k3      3-regime screen: arms init/model/o2 (evals/wave1_k3_*).
          Evaluates P-W1c against the ladder's H (pass --h2 with the K=2 headroom).

Scoring goes through the validated score_eval_dir staging + the frozen scorer. Results are
printed and written to evals/wave1_scores.json (local; the adjudicated numbers get committed
into the LOOP-0011 note).

Usage:
    PYTHONPATH=src myenv\\Scripts\\python.exe scripts/score_wave1.py ladder
    PYTHONPATH=src myenv\\Scripts\\python.exe scripts/score_wave1.py k3 --h2 <H from ladder>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from score_eval_dir import _stage_run_layout, welch  # noqa: E402
from lifelong_learning.research.benchmarking import score_brain_run  # noqa: E402

LADDER_ARMS = {
    "control": "evals/loop9_s1_model_e[1-8]_*",  # archived no-swap arm, same Brain/protocol/seeds
    "heads": "evals/wave1_decomp_heads_e*",
    "heads+encoder": "evals/wave1_decomp_headsenc_e*",
    "world_model": "evals/wave1_decomp_wm_e*",
    "full": "evals/wave1_decomp_full_e*",
}
K3_ARMS = {
    "init": "evals/wave1_k3_init_e*",
    "model": "evals/wave1_k3_model_e*",
    "o2": "evals/wave1_k3_o2_e*",
}


def score_arm_full(pattern: str, staging: Path) -> list[dict]:
    dirs = sorted(Path(p) for p in glob.glob(str(REPO_ROOT / pattern)) if Path(p).is_dir())
    if not dirs:
        raise FileNotFoundError(f"No eval dirs match {pattern}")
    rows = []
    for d in dirs:
        s = score_brain_run(_stage_run_layout(d, staging))
        rows.append({"eval_dir": d.name, "composite": s.composite_score,
                     "hit80": s.hit_rate_80, "hit95": s.hit_rate_95})
    return rows


def summarize(arms: dict[str, str]) -> dict[str, dict]:
    staging = Path(tempfile.mkdtemp(prefix="wave1_score_"))
    out = {}
    for label, pat in arms.items():
        rows = score_arm_full(pat, staging)
        comp = np.array([r["composite"] for r in rows])
        out[label] = {
            "n": len(rows),
            "mean": float(np.mean(comp)),
            "sd": float(np.std(comp, ddof=1)),
            "hit80": float(np.mean([r["hit80"] for r in rows])),
            "hit95": float(np.mean([r["hit95"] for r in rows])),
            "composites": comp,
        }
        print(f"  {label:<15} n={len(rows):>2}  mean {np.mean(comp):.4f}  sd {np.std(comp, ddof=1):.4f}  "
              f"hit80 {out[label]['hit80']:.3f}  hit95 {out[label]['hit95']:.3f}")
    return out


def adjudicate_ladder(a: dict[str, dict]) -> dict:
    c = a["control"]["composites"]
    H = float(np.mean(a["full"]["composites"]) - np.mean(c))
    wf = welch(a["full"]["composites"], c)
    print(f"\nH (full - control) = {H:+.4f}   Welch p = {wf['p']:.3g}")

    res = {"H": H, "full_vs_control_p": wf["p"], "shares": {}, "arm_vs_control": {}}
    for scope in ("heads", "heads+encoder", "world_model"):
        w = welch(a[scope]["composites"], c)
        share = (np.mean(a[scope]["composites"]) - np.mean(c)) / H if H != 0 else float("nan")
        res["shares"][scope] = float(share)
        res["arm_vs_control"][scope] = w
        print(f"  share({scope:<14}) = {share:+7.1%}   delta {w['delta']:+.4f}  p={w['p']:.3g}")

    p_w1a = (H > 0) and (wf["p"] < 0.05) and (H >= 0.05)
    sh, she, swm = (res["shares"][k] for k in ("heads", "heads+encoder", "world_model"))
    kill_weightspace = (sh < 0.20) and (she < 0.40)
    routing = []
    if sh >= 0.40:
        routing.append("W2A weight-space (G3) opens: heads >= 40%")
    if swm >= 0.20:
        routing.append("W2B MoWM (D) opens: world_model >= 20%")
    if kill_weightspace:
        routing.append("weight-space KILLED (heads<20% AND heads+enc<40%) -> route by larger of WM/scaling")
    if not routing:
        routing.append("no subtree threshold met -> mixed/inconclusive, apply extension rule")

    # Extension rule: any deciding share within +/-10pp of its threshold -> extend to n=16.
    near = []
    if abs(sh - 0.40) <= 0.10 or abs(sh - 0.20) <= 0.10:
        near.append("heads")
    if abs(she - 0.40) <= 0.10:
        near.append("heads+encoder")
    if abs(swm - 0.20) <= 0.10:
        near.append("world_model")

    res.update({"P_W1a_pass": bool(p_w1a), "routing": routing, "extend_arms_to_16": near})
    print(f"\nP-W1a (H>0, p<0.05, H>=+0.05): {'PASS' if p_w1a else 'FAIL'}")
    for r in routing:
        print(f"P-W1b routing: {r}")
    print(f"Extension rule (+/-10pp): {'extend ' + ', '.join(near) if near else 'no extension needed'}")
    return res


def adjudicate_k3(a: dict[str, dict], h2: float) -> dict:
    gap = float(np.mean(a["o2"]["composites"]) - np.mean(a["model"]["composites"]))
    w = welch(a["o2"]["composites"], a["model"]["composites"])
    ti = welch(a["model"]["composites"], a["init"]["composites"])
    p_w1c = gap > h2
    print(f"\n(o2 - model)@K3 = {gap:+.4f} (p={w['p']:.3g})  vs  H@K2 = {h2:+.4f}"
          f"  ->  P-W1c {'PASS: headroom grows with K' if p_w1c else 'FAIL: scaling axis dies'}")
    print(f"secondary trained-vs-init @K3: {ti['delta']:+.4f} (p={ti['p']:.3g})")
    return {"o2_minus_model_k3": gap, "p": w["p"], "h2": h2, "P_W1c_pass": bool(p_w1c),
            "trained_vs_init_k3": ti}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch", choices=["ladder", "k3"])
    p.add_argument("--h2", type=float, default=None, help="K=2 headroom H from the ladder (k3 only)")
    p.add_argument("--json-out", default="evals/wave1_scores.json")
    args = p.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    print(f"== Wave-1 {args.batch}: arm summaries ==")
    if args.batch == "ladder":
        arms = summarize(LADDER_ARMS)
        res = adjudicate_ladder(arms)
    else:
        if args.h2 is None:
            p.error("k3 needs --h2 (the ladder's H)")
        arms = summarize(K3_ARMS)
        res = adjudicate_k3(arms, args.h2)

    out_path = REPO_ROOT / args.json_out
    prior = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    prior[args.batch] = {
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "composites"} | {"composites": v["composites"].tolist()}
                 for k, v in arms.items()},
        "adjudication": res,
    }
    out_path.write_text(json.dumps(prior, indent=1), encoding="utf-8")
    print(f"\n[wave1] scores + adjudication -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
