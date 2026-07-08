"""Score trained-Brain *eval* directories with the frozen benchmark scorer.

Background
----------
The frozen scorer ``score_brain_run`` (``research/benchmarking.py``) expects a *run* layout::

    run_dir/
        config.txt                      # must carry `inner_steps_per_regime: N`
        episode_*/ep*_env*/*_data.json  # one JSON log per inner run

A trained-Brain *eval* produced by ``scripts/eval_brain.py`` instead lays out::

    eval_dir/
        config.txt                      # carries `steps_per_regime: N` (different key)
        <run_name>_<ts>/<run_name>_data.json   # ONE 800k-step inner run (16 envs aggregated)

This adapter stages each eval's single ``*_data.json`` into the run layout the frozen scorer
walks (a temp ``episode_00/ep0_env0/`` dir + a synthesized ``config.txt``) and then calls the
frozen scorer **unmodified**. It is the promoted, committed form of the ephemeral
``scratchpad/score_t1.py`` used for note 0006's T-series; see note 0006 Provenance and
docs/research-notes/0007-trained-brain-replication.md (assumption A3 — adapter fidelity).

Fidelity is enforced by the validation gate: re-scoring the archived ``evals/t1_*`` dirs must
reproduce note 0006's numbers exactly (trained ep130 composite 0.5577, init 0.5285,
Δ +0.0292, t≈2.97; hit80 0.982 both). Run ``--self-test`` to check.

Usage
-----
    set PYTHONPATH=src
    myenv\\Scripts\\python.exe scripts/score_eval_dir.py \\
        --arm "trained=evals/t1_trained_ep130_seed*" \\
        --arm "init=evals/t1c_initbrain_seed*"

    myenv\\Scripts\\python.exe scripts/score_eval_dir.py --self-test

The first two arms are contrasted with an unpaired Welch t-test (arm[0] - arm[1]).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from lifelong_learning.research.benchmarking import (
    parse_run_config,
    score_brain_run,
)

# Validation targets from research-note 0006 (the single-training-seed contrast being replicated).
NOTE_0006_TARGETS = {
    "trained": {"glob": "evals/t1_trained_ep130_seed*", "composite": 0.5577, "hit80": 0.982, "n": 32},
    "init": {"glob": "evals/t1c_initbrain_seed*", "composite": 0.5285, "hit80": 0.982, "n": 32},
    "delta": 0.0292,
    "t": 2.97,
}


def _nested_data_json(eval_dir: Path) -> Path:
    """Find the single ``*_data.json`` an eval produced (one inner run, 16 envs aggregated)."""
    candidates = sorted(eval_dir.rglob("*_data.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_data.json under {eval_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"Expected exactly one *_data.json under {eval_dir}, found {len(candidates)}: "
            f"{[str(c) for c in candidates]}"
        )
    return candidates[0]


def _steps_per_regime(eval_dir: Path) -> int:
    """Read the eval's `steps_per_regime` (the frozen scorer's `inner_steps_per_regime`)."""
    config = parse_run_config(eval_dir)  # eval config.txt is `key: value`, same parser
    if "steps_per_regime" in config:
        return int(config["steps_per_regime"])
    if "inner_steps_per_regime" in config:
        return int(config["inner_steps_per_regime"])
    raise KeyError(f"No steps_per_regime in {eval_dir/'config.txt'}")


def _stage_run_layout(eval_dir: Path, staging_root: Path) -> Path:
    """Materialize the frozen scorer's run layout for one eval; hardlink the data.json."""
    data_json = _nested_data_json(eval_dir)
    steps_per_regime = _steps_per_regime(eval_dir)

    run_dir = staging_root / eval_dir.name
    inner_dir = run_dir / "episode_00" / "ep0_env0"
    inner_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.txt").write_text(
        f"inner_steps_per_regime: {steps_per_regime}\n", encoding="utf-8"
    )

    dest = inner_dir / data_json.name
    if dest.exists():
        dest.unlink()
    try:
        os.link(data_json, dest)  # hardlink: no 6.7 MB copy, same volume
    except OSError:
        shutil.copy2(data_json, dest)  # cross-volume / permission fallback
    return run_dir


def score_one_eval(eval_dir: Path, staging_root: Path) -> dict[str, Any]:
    """Return {composite, hit80, switches} for a single eval dir via the frozen scorer."""
    run_dir = _stage_run_layout(eval_dir, staging_root)
    score = score_brain_run(run_dir)
    return {
        "eval_dir": eval_dir.name,
        "composite": score.composite_score,
        "hit80": score.hit_rate_80,
        "switches": score.switch_count,
    }


def score_arm(pattern: str, staging_root: Path) -> list[dict[str, Any]]:
    eval_dirs = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    if not eval_dirs:
        raise FileNotFoundError(f"No eval dirs match: {pattern}")
    return [score_one_eval(d, staging_root) for d in eval_dirs]


def summarize_arm(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    composites = np.array([r["composite"] for r in rows], dtype=float)
    hit80s = np.array([r["hit80"] for r in rows], dtype=float)
    return {
        "label": label,
        "n": len(rows),
        "composite_mean": float(np.mean(composites)),
        "composite_sd": float(np.std(composites, ddof=1)) if len(composites) > 1 else 0.0,
        "hit80_mean": float(np.mean(hit80s)),
        "composites": composites,
    }


def welch(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Unpaired Welch t-test (a - b), matching note 0006's trained-vs-init contrast."""
    from scipy import stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    )
    return {"delta": float(np.mean(a) - np.mean(b)), "t": float(t), "p": float(p), "df": float(df)}


def _run(arm_specs: list[tuple[str, str]], staging_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for label, pattern in arm_specs:
        rows = score_arm(pattern, staging_root)
        summaries.append(summarize_arm(label, rows))
    return summaries


def _print_report(summaries: list[dict[str, Any]]) -> None:
    print(f"\n{'arm':<14}{'n':>4}{'composite':>14}{'sd':>10}{'hit80':>10}")
    print("-" * 52)
    for s in summaries:
        print(
            f"{s['label']:<14}{s['n']:>4}{s['composite_mean']:>14.4f}"
            f"{s['composite_sd']:>10.4f}{s['hit80_mean']:>10.3f}"
        )
    if len(summaries) >= 2:
        a, b = summaries[0], summaries[1]
        w = welch(a["composites"], b["composites"])
        print(
            f"\ndelta ({a['label']} - {b['label']}): {w['delta']:+.4f}  "
            f"t={w['t']:.2f}  df={w['df']:.1f}  p={w['p']:.3g}"
        )


def self_test(staging_root: Path) -> int:
    """Re-score the archived T-series; assert reproduction of note 0006 (adapter-fidelity gate)."""
    print("Self-test: reproducing research-note 0006 from archived evals/t1_* ...")
    tgt = NOTE_0006_TARGETS
    summaries = _run([("trained", tgt["trained"]["glob"]), ("init", tgt["init"]["glob"])], staging_root)
    _print_report(summaries)

    ok = True
    tol = 5e-4  # numbers in note 0006 are quoted to 4 dp
    checks: list[tuple[str, float, float]] = []
    for s in summaries:
        exp = tgt[s["label"]]
        checks.append((f"{s['label']} n", s["n"], exp["n"]))
        checks.append((f"{s['label']} composite", s["composite_mean"], exp["composite"]))
        checks.append((f"{s['label']} hit80", s["hit80_mean"], exp["hit80"]))
    w = welch(summaries[0]["composites"], summaries[1]["composites"])
    checks.append(("delta", w["delta"], tgt["delta"]))
    checks.append(("t", w["t"], tgt["t"]))

    print("\nvalidation vs note 0006:")
    for name, got, exp in checks:
        if "n" == name.split()[-1]:
            passed = int(got) == int(exp)
            detail = f"{int(got)} vs {int(exp)}"
        elif name in ("t",):
            passed = abs(got - exp) <= 0.05
            detail = f"{got:.2f} vs {exp:.2f}"
        else:
            passed = abs(got - exp) <= tol
            detail = f"{got:.4f} vs {exp:.4f} (tol {tol})"
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<20} {detail}")

    print(f"\n{'ALL PASS — adapter reproduces note 0006.' if ok else 'FAIL — adapter does NOT match note 0006.'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--arm", action="append", default=[], metavar="LABEL=GLOB",
        help="An arm to score, e.g. --arm 'trained=evals/t1_trained_ep130_seed*'. Repeatable.",
    )
    parser.add_argument("--self-test", action="store_true", help="Reproduce note 0006 and assert (fidelity gate).")
    parser.add_argument("--staging-dir", default=None, help="Where to stage run layouts (default: a temp dir).")
    args = parser.parse_args(argv)

    # Windows consoles default to cp1252; keep ASCII-only output but harden anyway.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    tmp = args.staging_dir or tempfile.mkdtemp(prefix="score_eval_dir_")
    staging_root = Path(tmp)
    staging_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.self_test:
            return self_test(staging_root)
        if not args.arm:
            parser.error("provide at least one --arm LABEL=GLOB, or --self-test")
        arm_specs = []
        for spec in args.arm:
            if "=" not in spec:
                parser.error(f"--arm must be LABEL=GLOB, got: {spec}")
            label, pattern = spec.split("=", 1)
            arm_specs.append((label, pattern))
        summaries = _run(arm_specs, staging_root)
        _print_report(summaries)
        return 0
    finally:
        if args.staging_dir is None:
            shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
