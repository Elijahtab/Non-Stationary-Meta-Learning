#!/usr/bin/env python
"""
Regime-decoding probe — the headline analysis for the task-free neuromodulation study.

Question: does the Brain's learned 8-d context code carry information about the *hidden*
regime, even though it was trained with no task labels? We answer it by fitting a linear
probe (multinomial logistic regression) that maps the context code -> regime id, and
comparing its cross-validated accuracy to the majority-class baseline and uniform chance.

A code that was learned without labels yet linearly decodes the hidden regime is the
central evidence that neuromodulation acts as a regime-sensitive routing signal rather
than noise.

Inputs are the per-inner-run ``*_data.json`` logs written under a Brain run dir:
  - ``brain_context/context_0`` .. ``context_7``  -> the 8-d code at each Brain decision
  - ``charts/regime_id``                          -> the (otherwise hidden) active regime
Optionally also probes the decoded mask channels (``brain_neuromod/channel_mean_*``).

No sklearn dependency — the probe is a tiny torch logistic-regression with k-fold CV.

Examples
--------
python scripts/analyze_regime_decoding.py runs/brain_2_regimes_8x8_neuromod_20260315-180839
python scripts/analyze_regime_decoding.py runs/<run> --features mask --folds 5 --out probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lifelong_learning.research.benchmarking import (  # noqa: E402
    iter_inner_run_json_logs,
    load_json_log,
)

CONTEXT_KEYS = [f"brain_context/context_{i}" for i in range(8)]
REGIME_KEY = "charts/regime_id"


def _series_to_steps_values(series) -> tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.array([]), np.array([])
    arr = np.asarray(series, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def _regime_at_steps(regime_steps: np.ndarray, regime_vals: np.ndarray, query_steps: np.ndarray) -> np.ndarray:
    """For each query step, the most recent logged regime id at or before that step."""
    order = np.argsort(regime_steps)
    rs, rv = regime_steps[order], regime_vals[order]
    idx = np.searchsorted(rs, query_steps, side="right") - 1
    idx = np.clip(idx, 0, len(rs) - 1)
    return np.rint(rv[idx]).astype(int)


def _collect_feature_keys(data: dict, feature: str) -> list[str]:
    if feature == "code":
        return [k for k in CONTEXT_KEYS if k in data]
    # mask channels: variable count (brain_neuromod/channel_mean_<i>)
    keys = sorted(
        (k for k in data if k.startswith("brain_neuromod/channel_mean_")),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    return keys


def build_dataset(run_dir: str | Path, feature: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool aligned (feature_vector, regime) pairs across every inner run in the run dir."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n_runs = 0

    for json_path in iter_inner_run_json_logs(run_dir):
        data = load_json_log(json_path)
        feat_keys = _collect_feature_keys(data, feature)
        if not feat_keys or REGIME_KEY not in data:
            continue

        # Each feature key is logged at the same Brain-decision steps; use the first as the
        # step index and require all keys to share length.
        steps, _ = _series_to_steps_values(data[feat_keys[0]])
        if steps.size == 0:
            continue
        cols = []
        ok = True
        for key in feat_keys:
            s, v = _series_to_steps_values(data[key])
            if s.size != steps.size:
                ok = False
                break
            cols.append(v)
        if not ok:
            continue

        regime_steps, regime_vals = _series_to_steps_values(data[REGIME_KEY])
        if regime_steps.size == 0:
            continue
        labels = _regime_at_steps(regime_steps, regime_vals, steps)

        xs.append(np.stack(cols, axis=1))  # (T, F)
        ys.append(labels)
        n_runs += 1

    if not xs:
        return np.empty((0, 0)), np.empty((0,), dtype=int), 0
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(int)
    return X, y, n_runs


def _fit_logreg(X_tr, y_tr, X_te, n_classes, *, epochs=300, lr=0.05, weight_decay=1e-3, seed=0):
    torch.manual_seed(seed)
    device = "cpu"  # tiny problem; CPU is fine and deterministic
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    model = torch.nn.Linear(X_tr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = model(Xte).argmax(dim=1).cpu().numpy()
    return preds


def kfold_accuracy(X: np.ndarray, y: np.ndarray, *, folds: int, seed: int = 0) -> dict:
    n = len(y)
    classes = np.unique(y)
    n_classes = int(classes.max()) + 1
    if n < folds or len(classes) < 2:
        return {
            "n_samples": int(n),
            "n_classes": int(len(classes)),
            "decodable": False,
            "reason": "insufficient samples or single class",
        }

    # Standardize features (fit on full set; fine for a diagnostic probe).
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_ids = np.array_split(perm, folds)

    accs = []
    for k in range(folds):
        test_idx = fold_ids[k]
        train_idx = np.concatenate([fold_ids[j] for j in range(folds) if j != k])
        preds = _fit_logreg(Xn[train_idx], y[train_idx], Xn[test_idx], n_classes, seed=seed + k)
        accs.append(float((preds == y[test_idx]).mean()))

    counts = np.bincount(y, minlength=n_classes)
    majority = float(counts.max() / counts.sum())
    uniform = 1.0 / len(classes)
    mean_acc = float(np.mean(accs))
    return {
        "n_samples": int(n),
        "n_classes": int(len(classes)),
        "folds": folds,
        "cv_accuracy_mean": mean_acc,
        "cv_accuracy_std": float(np.std(accs)),
        "per_fold_accuracy": accs,
        "majority_baseline": majority,
        "uniform_chance": uniform,
        "accuracy_over_majority": mean_acc - majority,
        "decodable": mean_acc > majority + 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="Brain run directory (contains episode_*/ep*_env*/ inner logs)")
    parser.add_argument("--features", choices=["code", "mask"], default="code",
                        help="Probe the 8-d context code (default) or the decoded mask channel means")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    X, y, n_runs = build_dataset(args.run_dir, args.features)
    result = {
        "run_dir": str(args.run_dir),
        "features": args.features,
        "inner_runs_used": n_runs,
        "feature_dim": int(X.shape[1]) if X.size else 0,
    }
    result.update(kfold_accuracy(X, y, folds=args.folds, seed=args.seed))

    print(json.dumps(result, indent=2))
    if result.get("decodable"):
        print(
            f"\n[probe] {args.features}: {result['cv_accuracy_mean']:.3f} CV acc "
            f"vs {result['majority_baseline']:.3f} majority "
            f"(+{result['accuracy_over_majority']:.3f}) -> hidden regime IS decodable."
        )
    elif "cv_accuracy_mean" in result:
        print(
            f"\n[probe] {args.features}: {result['cv_accuracy_mean']:.3f} CV acc "
            f"vs {result['majority_baseline']:.3f} majority -> not clearly decodable."
        )
    else:
        print(f"\n[probe] {result.get('reason', 'no data')}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
