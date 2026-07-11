"""Run the LOOP-0009 eval ladder: trained-ep130 vs matched-init, per Brain, at the eval protocol.

For each (Brain seed) x (arm: model=ep130 selected / init=ep0 control) x (eval-seed), run
``scripts/eval_brain.py`` at the eval protocol (16 inner envs, decision_interval=1, 2 regimes x
100k / 800k total — the protocol note 0006/0007 require), spread across GPUs with bounded
concurrency and thread caps. Idempotent: an eval whose output dir already holds a ``*_data.json``
is skipped, so a re-run after an interruption only fills gaps.

Score the results afterward with the frozen adapter, e.g.:
    PYTHONPATH=src python scripts/score_eval_dir.py \
        --arm "s1_trained=evals/loop9_s1_model_e*" --arm "s1_init=evals/loop9_s1_init_e*"

Usage (on the box, GPUs 0-3):
    PYTHONPATH=src python scripts/run_loop9_evals.py --eval-seeds 1 2 3 4 5 6 7 8 \
        --gpus 0 1 2 3 --concurrency 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# The eval protocol — must match the T-series (note 0006) so the contrast is within-protocol.
EVAL_PROTOCOL = {
    "env_id": "MiniGrid-MultiGoal-8x8-v0",
    "total_timesteps": 800000,
    "steps_per_regime": 100000,
    "num_regimes": 2,
    "num_envs": 16,
    "decision_interval": 1,
}


def brain_run_dir(seed: int) -> Path:
    matches = sorted(glob.glob(str(REPO_ROOT / f"runs/paper8x8_130_brain_neuromod_seed{seed}_*")))
    if not matches:
        raise FileNotFoundError(f"No trained run dir for seed {seed} (runs/paper8x8_130_brain_neuromod_seed{seed}_*)")
    return Path(matches[-1])


def already_done(run_name: str, out_root: Path) -> bool:
    for d in glob.glob(str(out_root / f"{run_name}_*")):
        if glob.glob(os.path.join(d, "**", "*_data.json"), recursive=True):
            return True
    return False


def run_eval(job: dict, gpu: int, out_root: Path, log_dir: Path, threads: int) -> dict:
    run_name = job["run_name"]
    if already_done(run_name, out_root):
        return {**job, "status": "skipped_existing"}

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "eval_brain.py"),
        "--brain_checkpoint", job["ckpt"],
        "--seed", str(job["eval_seed"]),
        "--run_name", run_name,
        "--device", "cuda",
    ]
    for k, v in EVAL_PROTOCOL.items():
        cmd += [f"--{k}", str(v)]

    env = {
        **os.environ,
        "PYTHONPATH": str(SRC_DIR),
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
    }
    log_path = log_dir / f"{run_name}.log"
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=handle,
                              stderr=subprocess.STDOUT, text=True)
    return {
        **job, "gpu": gpu, "returncode": proc.returncode,
        "seconds": round(time.perf_counter() - started, 1),
        "status": "ok" if proc.returncode == 0 else "failed",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4], help="Brain training seeds")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--arms", nargs="+", default=["model", "init"], choices=["model", "init"],
                   help="model=trained ep130 (brain_model.pt); init=control (brain_init.pt)")
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent evals (e.g. 2 per GPU)")
    p.add_argument("--threads", type=int, default=4, help="Per-eval thread cap")
    p.add_argument("--out", default=str(REPO_ROOT / "evals"), help="Output root (eval dirs land here)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    out_root = Path(args.out)
    log_dir = out_root / "loop9_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    for seed in args.seeds:
        d = brain_run_dir(seed)
        for arm in args.arms:
            ckpt = d / f"brain_{'model' if arm == 'model' else 'init'}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing {ckpt}")
            for e in args.eval_seeds:
                jobs.append({
                    "seed": seed, "arm": arm, "eval_seed": e,
                    "ckpt": str(ckpt), "run_name": f"loop9_s{seed}_{arm}_e{e}",
                })

    print(f"[loop9-evals] {len(jobs)} evals | seeds={args.seeds} arms={args.arms} "
          f"eval_seeds={args.eval_seeds} gpus={args.gpus} concurrency={args.concurrency}")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run_name']}: {Path(j['ckpt']).name} seed={j['eval_seed']}")
        return 0

    results: list[dict] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for i, job in enumerate(jobs):
            gpu = args.gpus[i % len(args.gpus)]
            futures[pool.submit(run_eval, job, gpu, out_root, log_dir, args.threads)] = job
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            print(f"[loop9-evals] ({done}/{len(jobs)}) {r['run_name']} -> {r['status']}"
                  f"{' ' + str(r.get('seconds')) + 's' if 'seconds' in r else ''}", flush=True)

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped_existing"))
    failed = [r["run_name"] for r in results if r["status"] == "failed"]
    print(f"\n[loop9-evals] done: {ok}/{len(jobs)} ok/skipped; {len(failed)} failed")
    if failed:
        print("[loop9-evals] FAILED (re-run to retry): " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
