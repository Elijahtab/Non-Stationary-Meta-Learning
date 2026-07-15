"""Run the Wave-1 oracle-rung eval batches (action tree 2026-07-09; pre-reg: research-log 0008).

Two pre-registered batches, both frozen-Brain evals at the LOOP-0009 eval protocol on the
home GPU ($0):

  ladder  G-DECOMP swap-scope ladder: arms = swap_scope in {heads, heads+encoder,
          world_model, full}, all with --policy_swap_topline, Brain = LOOP-0009 seed-1 ep130.
          The no-swap control is the archived evals/loop9_s1_model_e* arm (same Brain, same
          protocol, same eval seeds) — no control re-runs needed.

  k3      3-regime difficulty screen (branch C, zero inner-code change): num_regimes=3
          (blocks A B C A B C A B at 100k/regime), arms = init (matched init Brain, no swap),
          model (trained Brain, no swap), o2 (trained Brain + full swap).

Run names are `wave1_<batch>_<arm>_e<seed>` so `score_eval_dir.py --arm` globs stay tidy.
Idempotent like run_loop9_evals.py: an eval whose dir already holds a *_data.json is skipped.

Usage (home 5070; sequential by default — the box concurrency lesson does not transfer to a
31 GB desktop):
    PYTHONPATH=src myenv\\Scripts\\python.exe scripts/run_wave1_evals.py ladder
    PYTHONPATH=src myenv\\Scripts\\python.exe scripts/run_wave1_evals.py k3
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

# The eval protocol — identical to run_loop9_evals.py so the archived loop9_s1 arms are
# valid same-protocol controls (num_regimes overridden by the k3 batch).
EVAL_PROTOCOL = {
    "env_id": "MiniGrid-MultiGoal-8x8-v0",
    "total_timesteps": 800000,
    "steps_per_regime": 100000,
    "num_regimes": 2,
    "num_envs": 16,
    "decision_interval": 1,
}

BRAIN_SEED = 1  # LOOP-0009 seed-1 Brain (largest replicated effect; n=16 control archived)
BRAIN_DIR_GLOB = f"runs/loop9_final/runs/paper8x8_130_brain_neuromod_seed{BRAIN_SEED}_*"

SCOPE_SLUGS = {  # '+' is unfriendly in dir names / globs
    "heads": "heads",
    "heads+encoder": "headsenc",
    "world_model": "wm",
    "full": "full",
}


def _brain_ckpt(name: str) -> str:
    matches = sorted(glob.glob(str(REPO_ROOT / BRAIN_DIR_GLOB)))
    if not matches:
        raise FileNotFoundError(f"No Brain run dir matches {BRAIN_DIR_GLOB}")
    ckpt = Path(matches[-1]) / name
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    return str(ckpt)


def build_jobs(batch: str, eval_seeds: list[int]) -> list[dict]:
    jobs: list[dict] = []
    if batch == "ladder":
        model = _brain_ckpt("brain_model.pt")
        for scope, slug in SCOPE_SLUGS.items():
            for e in eval_seeds:
                jobs.append({
                    "run_name": f"wave1_decomp_{slug}_e{e}",
                    "ckpt": model,
                    "eval_seed": e,
                    "extra": ["--policy_swap_topline", "--swap_scope", scope],
                    "num_regimes": 2,
                })
    elif batch == "k3":
        arms = [
            ("init", _brain_ckpt("brain_init.pt"), []),
            ("model", _brain_ckpt("brain_model.pt"), []),
            ("o2", _brain_ckpt("brain_model.pt"), ["--policy_swap_topline", "--swap_scope", "full"]),
        ]
        for arm, ckpt, extra in arms:
            for e in eval_seeds:
                jobs.append({
                    "run_name": f"wave1_k3_{arm}_e{e}",
                    "ckpt": ckpt,
                    "eval_seed": e,
                    "extra": extra,
                    "num_regimes": 3,
                })
    elif batch == "g3re":
        # LOOP-0013 selection rung (pre-reg: research-log 0010): learned trigger + content
        # addressing, WITH the spawn-until-full allocation fix. ar1re = the WM-reward-head
        # regime fingerprint (note 0013's recommendation); ar1ve2 = the value-error re-run
        # (LOOP-0012's P-G3c arm never exercised its selector — allocation was missing).
        model = _brain_ckpt("brain_model.pt")
        arms = [
            ("ar1re", "reward_error"),
            ("ar1ve2", "value_error"),
        ]
        for arm, select in arms:
            for e in eval_seeds:
                jobs.append({
                    "run_name": f"g3_{arm}_e{e}",
                    "ckpt": model,
                    "eval_seed": e,
                    "extra": ["--head_bank_slots", "2", "--head_bank_trigger", "surprise",
                              "--head_bank_select", select],
                    "num_regimes": 2,
                })
    elif batch == "ar1x":
        # n=16 extension of the LOOP-0012 A-R1 arm (paper-number precision; pass
        # --eval-seeds 9..16 — existing e1..8 are skipped as already done).
        model = _brain_ckpt("brain_model.pt")
        for e in eval_seeds:
            jobs.append({
                "run_name": f"g3_ar1_e{e}",
                "ckpt": model,
                "eval_seed": e,
                "extra": ["--head_bank_slots", "2", "--head_bank_trigger", "surprise",
                          "--head_bank_select", "other"],
                "num_regimes": 2,
            })
    elif batch == "g3":
        # LOOP-0012 de-oracling screen (pre-reg: research-log 0009). References are archived:
        # control = loop9_s1_model_e1..8, ceiling slice = wave1_decomp_heads_e1..8.
        model = _brain_ckpt("brain_model.pt")
        arms = [
            ("oracle", ["--head_bank_slots", "2"]),  # equivalence rung (oracle/oracle)
            ("ar1", ["--head_bank_slots", "2", "--head_bank_trigger", "surprise",
                     "--head_bank_select", "other"]),  # learned WHEN (A-R1)
            ("ar1ve", ["--head_bank_slots", "2", "--head_bank_trigger", "surprise",
                       "--head_bank_select", "value_error"]),  # learned WHEN + WHICH
        ]
        for arm, extra in arms:
            for e in eval_seeds:
                jobs.append({
                    "run_name": f"g3_{arm}_e{e}",
                    "ckpt": model,
                    "eval_seed": e,
                    "extra": extra,
                    "num_regimes": 2,
                })
    else:
        raise ValueError(batch)
    return jobs


def already_done(run_name: str, out_root: Path) -> bool:
    for d in glob.glob(str(out_root / f"{run_name}_*")):
        if glob.glob(os.path.join(d, "**", "*_data.json"), recursive=True):
            return True
    return False


def run_eval(job: dict, gpu: int, out_root: Path, log_dir: Path, threads: int) -> dict:
    if already_done(job["run_name"], out_root):
        return {**job, "status": "skipped_existing"}
    protocol = {**EVAL_PROTOCOL, "num_regimes": job["num_regimes"]}
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "eval_brain.py"),
        "--brain_checkpoint", job["ckpt"],
        "--seed", str(job["eval_seed"]),
        "--run_name", job["run_name"],
        "--device", "cuda",
    ]
    for k, v in protocol.items():
        cmd += [f"--{k}", str(v)]
    cmd += job["extra"]

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
    log_path = log_dir / f"{job['run_name']}.log"
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
    p.add_argument("batch", choices=["ladder", "k3", "g3", "g3re", "ar1x"])
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent evals (default 1: home desktop, ~10 GB free RAM)")
    p.add_argument("--threads", type=int, default=6, help="Per-eval thread cap")
    p.add_argument("--out", default=str(REPO_ROOT / "evals"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    out_root = Path(args.out)
    log_dir = out_root / "wave1_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.batch, args.eval_seeds)
    print(f"[wave1-{args.batch}] {len(jobs)} evals | seeds={args.eval_seeds} "
          f"gpus={args.gpus} concurrency={args.concurrency}", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['run_name']}: {Path(j['ckpt']).name} K={j['num_regimes']} {' '.join(j['extra'])}")
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
            print(f"[wave1-{args.batch}] ({done}/{len(jobs)}) {r['run_name']} -> {r['status']}"
                  f"{' ' + str(r.get('seconds')) + 's' if 'seconds' in r else ''}", flush=True)

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped_existing"))
    failed = [r["run_name"] for r in results if r["status"] == "failed"]
    print(f"\n[wave1-{args.batch}] done: {ok}/{len(jobs)} ok/skipped; {len(failed)} failed")
    if failed:
        print(f"[wave1-{args.batch}] FAILED (re-run to retry): " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
