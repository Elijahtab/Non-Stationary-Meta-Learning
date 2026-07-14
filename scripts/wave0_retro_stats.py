"""Wave-0 desk probes from the 2026-07-09 action tree — $0, CPU-only, read-only.

Three probes over archived artifacts (no new runs):

  power   W0a / branch-H kill-probe: bootstrap-subsample the archived n=32 eval arms
          (t1 trained-vs-init, the known +0.0292 effect) to n=8/n=16 cells and measure
          the power of an autoresearch-sized screen to detect it. Registered kill
          (action tree, branch H): power <60% at n=8 AND <80% at n=16 => H4 dies.
          Secondary: the same on the LOOP-0009 per-seed arms (n=16/arm, effect +0.0455).

  logstd  W0b / B-R0: read ``actor_log_std`` out of the trained Brain checkpoints
          (March ep130 + the four LOOP-0009 ep130 Brains) and compare the dead code
          dims (7-14) against the HP dims (0-6) and the init value (-0.5, sigma~0.61).
          Answers "is the 8-dim dead-code noise premise real?" before any B1 spend.

  iqm     W0d / H1+H2: IQM (25% trimmed mean) + bootstrap 95% CIs + the pre-authorized
          hit-0.95 reliability facet, retrospectively on the t1 and LOOP-0009 arms.
          The LOOP-0009 pooled contrast uses a seed-stratified bootstrap.

Scoring goes through the validated ``score_eval_dir`` staging + the frozen
``score_brain_run`` (adapter-fidelity gate: reproduces notes 0006/0007). Per-eval scores
are cached to a JSON so re-runs are instant. Committed per the provenance rule (note
0006's scratchpad-adapter lesson: analysis scripts do not live in ephemeral scratchpads).

Usage:
    set PYTHONPATH=src
    myenv\\Scripts\\python.exe scripts/wave0_retro_stats.py all
    myenv\\Scripts\\python.exe scripts/wave0_retro_stats.py power --boot 4000
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from score_eval_dir import _stage_run_layout, welch  # noqa: E402
from lifelong_learning.research.benchmarking import score_brain_run  # noqa: E402

# ---------------------------------------------------------------- arms & checkpoints

ARMS: dict[str, str] = {
    "t1_trained": "evals/t1_trained_ep130_seed*",
    "t1_init": "evals/t1c_initbrain_seed*",
}
for _s in (1, 2, 3, 4):
    ARMS[f"loop9_s{_s}_trained"] = f"evals/loop9_s{_s}_model_e*"
    ARMS[f"loop9_s{_s}_init"] = f"evals/loop9_s{_s}_init_e*"

MARCH_RUN = "runs/brain_2_regimes_8x8_neuromod_20260315-180839"
CHECKPOINTS: dict[str, tuple[str, str]] = {  # label -> (model ckpt, matched init ckpt)
    "march_s0": (f"{MARCH_RUN}/brain_model.pt", f"{MARCH_RUN}/brain_init_seed0.pt"),
}
for _s in (1, 2, 3, 4):
    _d = f"runs/loop9_final/runs/paper8x8_130_brain_neuromod_seed{_s}_*"
    CHECKPOINTS[f"loop9_s{_s}"] = (f"{_d}/brain_model.pt", f"{_d}/brain_init.pt")

LOGSTD_INIT = -0.5  # MLPActorCritic init: nn.Parameter(torch.ones(act_dim) * -0.5)
HP_DIMS = slice(0, 7)  # 7 scalar HP levers
CODE_DIMS = slice(7, 15)  # 8-d code -> mask pathway (proven inert, note 0006)

# ------------------------------------------------------------------------- scoring


def score_arms(cache_path: Path, arms: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """Score every eval dir in every arm (composite/hit80/hit95), with a JSON cache."""
    cache: dict[str, list[dict[str, Any]]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = {label: pat for label, pat in arms.items() if label not in cache}
    if missing:
        staging_root = Path(tempfile.mkdtemp(prefix="wave0_stats_"))
        for label, pattern in missing.items():
            eval_dirs = sorted(Path(p) for p in glob.glob(str(REPO_ROOT / pattern)) if Path(p).is_dir())
            if not eval_dirs:
                raise FileNotFoundError(f"No eval dirs match {pattern} (arm {label})")
            rows = []
            for d in eval_dirs:
                run_dir = _stage_run_layout(d, staging_root)
                s = score_brain_run(run_dir)
                rows.append(
                    {
                        "eval_dir": d.name,
                        "composite": s.composite_score,
                        "hit80": s.hit_rate_80,
                        "hit95": s.hit_rate_95,
                        "switches": s.switch_count,
                    }
                )
            cache[label] = rows
            print(f"[scores] {label}: n={len(rows)} scored", flush=True)
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return {label: cache[label] for label in arms}


def _arr(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def _sanity_anchors(scores: dict[str, list[dict[str, Any]]]) -> None:
    """Trust anchors: this scoring path must reproduce notes 0006 and 0007."""
    t1 = welch(_arr(scores["t1_trained"], "composite"), _arr(scores["t1_init"], "composite"))
    tr = np.concatenate([_arr(scores[f"loop9_s{s}_trained"], "composite") for s in (1, 2, 3, 4)])
    it = np.concatenate([_arr(scores[f"loop9_s{s}_init"], "composite") for s in (1, 2, 3, 4)])
    l9 = welch(tr, it)
    ok_t1 = abs(t1["delta"] - 0.0292) <= 5e-4
    ok_l9 = abs(l9["delta"] - 0.0455) <= 5e-4
    print(f"[anchor] t1 delta {t1['delta']:+.4f} vs note 0006 +0.0292: {'PASS' if ok_t1 else 'FAIL'}")
    print(f"[anchor] loop9 pooled delta {l9['delta']:+.4f} vs note 0007 +0.0455: {'PASS' if ok_l9 else 'FAIL'}")
    if not (ok_t1 and ok_l9):
        raise SystemExit("Scoring path does not reproduce the archived results — do not trust probes.")


# --------------------------------------------------------------------------- power


def _power_cell(a: np.ndarray, b: np.ndarray, n: int, boot: int, rng: np.random.Generator) -> dict[str, float]:
    """Power of an n-per-arm screen on subsamples of the archived arms (without replacement)."""
    hits_two = hits_one = hits_sign = 0
    for _ in range(boot):
        sa = rng.choice(a, size=n, replace=False)
        sb = rng.choice(b, size=n, replace=False)
        w = welch(sa, sb)
        pos = w["delta"] > 0
        hits_sign += pos
        hits_two += pos and w["p"] < 0.05
        hits_one += pos and (w["p"] / 2) < 0.05
    return {
        "n": n,
        "power_welch05_two_sided": hits_two / boot,
        "power_welch05_one_sided": hits_one / boot,
        "p_delta_positive": hits_sign / boot,
    }


def probe_power(scores: dict[str, list[dict[str, Any]]], boot: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    out: dict[str, Any] = {"boot": boot, "contrasts": {}}

    t1_a = _arr(scores["t1_trained"], "composite")
    t1_b = _arr(scores["t1_init"], "composite")
    out["contrasts"]["t1_march_effect"] = {
        "effect": float(np.mean(t1_a) - np.mean(t1_b)),
        "cells": [_power_cell(t1_a, t1_b, n, boot, rng) for n in (8, 16)],
    }
    for s in (1, 2, 3, 4):
        a = _arr(scores[f"loop9_s{s}_trained"], "composite")
        b = _arr(scores[f"loop9_s{s}_init"], "composite")
        full = welch(a, b)
        out["contrasts"][f"loop9_s{s}"] = {
            "effect": full["delta"],
            "observed_p_n16": full["p"],
            "cells": [_power_cell(a, b, 8, boot, rng)],
        }

    print(f"\n== W0a power probe (B={boot}, Welch, alpha=0.05) ==")
    print(f"{'contrast':<18}{'effect':>9}{'n':>4}{'2-sided':>9}{'1-sided':>9}{'P(d>0)':>8}")
    print("-" * 57)
    for name, c in out["contrasts"].items():
        for cell in c["cells"]:
            print(
                f"{name:<18}{c['effect']:>+9.4f}{cell['n']:>4}"
                f"{cell['power_welch05_two_sided']:>9.2f}{cell['power_welch05_one_sided']:>9.2f}"
                f"{cell['p_delta_positive']:>8.2f}"
            )
    t1_cells = {c["n"]: c for c in out["contrasts"]["t1_march_effect"]["cells"]}
    kill = t1_cells[8]["power_welch05_two_sided"] < 0.60 and t1_cells[16]["power_welch05_two_sided"] < 0.80
    out["h4_kill_rule"] = {
        "rule": "power(+0.029)@n8 < 0.60 AND @n16 < 0.80 (Welch 2-sided, alpha 0.05)",
        "power_n8": t1_cells[8]["power_welch05_two_sided"],
        "power_n16": t1_cells[16]["power_welch05_two_sided"],
        "h4_dies": bool(kill),
    }
    print(
        f"\nH4 kill rule: power@n8={out['h4_kill_rule']['power_n8']:.2f} (<0.60?) "
        f"AND power@n16={out['h4_kill_rule']['power_n16']:.2f} (<0.80?) -> "
        f"{'H4 DIES (screens cannot see +0.029-sized effects)' if kill else 'H4 SURVIVES'}"
    )
    return out


# -------------------------------------------------------------------------- logstd


def probe_logstd() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {"init_log_std": LOGSTD_INIT, "checkpoints": {}}
    print("\n== W0b dead-dim log_std probe (actor_log_std; init = -0.50 for every dim) ==")
    print(f"{'ckpt':<12}{'arm':<8}{'HP dims 0-6':>22}{'code dims 7-14':>22}")
    print("-" * 64)
    for label, (model_pat, init_pat) in CHECKPOINTS.items():
        row: dict[str, Any] = {}
        for arm, pat in (("model", model_pat), ("init", init_pat)):
            matches = sorted(glob.glob(str(REPO_ROOT / pat)))
            if not matches:
                raise FileNotFoundError(f"No checkpoint matches {pat}")
            payload = torch.load(matches[-1], map_location="cpu", weights_only=False)
            state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
            ls = state["actor_log_std"].detach().numpy().astype(float)
            row[arm] = ls.tolist()
            hp, code = ls[HP_DIMS], ls[CODE_DIMS]
            print(
                f"{label:<12}{arm:<8}"
                f"{np.mean(hp):>+8.3f} [{np.min(hp):+.3f},{np.max(hp):+.3f}]"
                f"{np.mean(code):>+8.3f} [{np.min(code):+.3f},{np.max(code):+.3f}]"
            )
        out["checkpoints"][label] = row
    models = np.array([out["checkpoints"][k]["model"] for k in out["checkpoints"]])
    hp_sigma = float(np.mean(np.exp(models[:, HP_DIMS])))
    code_sigma = float(np.mean(np.exp(models[:, CODE_DIMS])))
    out["summary"] = {"mean_trained_sigma_hp": hp_sigma, "mean_trained_sigma_code": code_sigma}
    print(f"\nmean trained sigma: HP dims {hp_sigma:.3f}, code dims {code_sigma:.3f} (init sigma {np.exp(LOGSTD_INIT):.3f})")
    return out


# ----------------------------------------------------------------------------- iqm


def _iqm(x: np.ndarray) -> float:
    from scipy import stats

    return float(stats.trim_mean(x, 0.25))


def _boot_ci(x: np.ndarray, stat, boot: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(boot)])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _stratified_delta_ci(
    pairs: list[tuple[np.ndarray, np.ndarray]], boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Bootstrap CI of pooled (trained - init) resampling within each stratum (seed)."""
    deltas = []
    for _ in range(boot):
        d_tr, d_it = [], []
        for a, b in pairs:
            d_tr.append(rng.choice(a, size=len(a), replace=True))
            d_it.append(rng.choice(b, size=len(b), replace=True))
        deltas.append(float(np.mean(np.concatenate(d_tr)) - np.mean(np.concatenate(d_it))))
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def probe_iqm(scores: dict[str, list[dict[str, Any]]], boot: int) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    out: dict[str, Any] = {"boot": boot, "arms": {}, "contrasts": {}}
    print(f"\n== W0d IQM + bootstrap 95% CIs + hit-0.95 facet (B={boot}) ==")
    print(f"{'arm':<18}{'n':>4}{'mean':>9}{'IQM':>9}{'mean 95% CI':>19}{'hit80':>8}{'hit95':>8}")
    print("-" * 75)
    for label, rows in scores.items():
        comp = _arr(rows, "composite")
        mean_ci = _boot_ci(comp, np.mean, boot, rng)
        iqm_ci = _boot_ci(comp, _iqm, boot, rng)
        arm = {
            "n": len(comp),
            "mean": float(np.mean(comp)),
            "mean_ci": mean_ci,
            "iqm": _iqm(comp),
            "iqm_ci": iqm_ci,
            "hit80": float(np.mean(_arr(rows, "hit80"))),
            "hit95": float(np.mean(_arr(rows, "hit95"))),
        }
        out["arms"][label] = arm
        print(
            f"{label:<18}{arm['n']:>4}{arm['mean']:>9.4f}{arm['iqm']:>9.4f}"
            f"  [{mean_ci[0]:.4f},{mean_ci[1]:.4f}]{arm['hit80']:>8.3f}{arm['hit95']:>8.3f}"
        )

    def contrast(name: str, a_lab: str, b_lab: str) -> None:
        a, b = _arr(scores[a_lab], "composite"), _arr(scores[b_lab], "composite")
        w = welch(a, b)
        out["contrasts"][name] = {
            **w,
            "delta_iqm": _iqm(a) - _iqm(b),
            "hit95_delta": float(
                np.mean(_arr(scores[a_lab], "hit95")) - np.mean(_arr(scores[b_lab], "hit95"))
            ),
        }

    contrast("t1_trained_vs_init", "t1_trained", "t1_init")
    for s in (1, 2, 3, 4):
        contrast(f"loop9_s{s}", f"loop9_s{s}_trained", f"loop9_s{s}_init")

    pairs = [
        (_arr(scores[f"loop9_s{s}_trained"], "composite"), _arr(scores[f"loop9_s{s}_init"], "composite"))
        for s in (1, 2, 3, 4)
    ]
    pooled_tr = np.concatenate([p[0] for p in pairs])
    pooled_it = np.concatenate([p[1] for p in pairs])
    w = welch(pooled_tr, pooled_it)
    strat_ci = _stratified_delta_ci(pairs, boot, rng)
    out["contrasts"]["loop9_pooled"] = {
        **w,
        "delta_iqm": _iqm(pooled_tr) - _iqm(pooled_it),
        "delta_stratified_ci": strat_ci,
        "hit95_trained": float(np.mean(np.concatenate([_arr(scores[f"loop9_s{s}_trained"], "hit95") for s in (1, 2, 3, 4)]))),
        "hit95_init": float(np.mean(np.concatenate([_arr(scores[f"loop9_s{s}_init"], "hit95") for s in (1, 2, 3, 4)]))),
    }
    print("\ncontrasts (delta = first - second):")
    for name, c in out["contrasts"].items():
        extra = (
            f"  strat-CI [{c['delta_stratified_ci'][0]:+.4f},{c['delta_stratified_ci'][1]:+.4f}]"
            if "delta_stratified_ci" in c
            else ""
        )
        print(
            f"  {name:<22} dMean {c['delta']:+.4f} (p={c['p']:.2g})  dIQM {c['delta_iqm']:+.4f}"
            f"  dHit95 {c.get('hit95_delta', c.get('hit95_trained', 0) - c.get('hit95_init', 0)):+.3f}{extra}"
        )
    return out


# ---------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe", choices=["scores", "power", "logstd", "iqm", "all"])
    parser.add_argument("--boot", type=int, default=4000, help="bootstrap iterations (power)")
    parser.add_argument("--boot-ci", type=int, default=10000, help="bootstrap iterations (iqm CIs)")
    parser.add_argument("--cache", default="evals/wave0_scores_cache.json")
    parser.add_argument("--json-out", default="evals/wave0_stats.json")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    results: dict[str, Any] = {}
    if args.probe in ("scores", "power", "iqm", "all"):
        scores = score_arms(REPO_ROOT / args.cache, ARMS)
        _sanity_anchors(scores)
    if args.probe in ("power", "all"):
        results["power"] = probe_power(scores, args.boot)
    if args.probe in ("logstd", "all"):
        results["logstd"] = probe_logstd()
    if args.probe in ("iqm", "all"):
        results["iqm"] = probe_iqm(scores, args.boot_ci)

    if results:
        out_path = REPO_ROOT / args.json_out
        out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"\n[wave0] probe results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
