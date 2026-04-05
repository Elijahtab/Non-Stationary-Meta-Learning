from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.train_brain as train_brain_script
from lifelong_learning.research.benchmarking import (
    FrozenBenchmarkSpec,
    get_frozen_benchmark,
    list_frozen_benchmarks,
    score_brain_run,
    write_score_report,
)


DEFAULT_RUN_ROOT = Path("runs")
DEFAULT_BENCHMARK_REPORT_ROOT = Path("benchmarks")


def build_frozen_train_args(
    spec: FrozenBenchmarkSpec,
    *,
    seed: int,
    run_name: str,
    device: str,
) -> SimpleNamespace:
    args = dict(spec.fixed_train_args)
    args.update(
        {
            "seed": seed,
            "device": device,
            "run_name": run_name,
            "resume_path": None,
            "new_run_dir": False,
            "save_every_episodes": 0,
            "plot_every_episodes": 0,
            "generate_high_scale_plots": False,
        }
    )
    return SimpleNamespace(**args)


def locate_new_run_dir(run_name: str, before_paths: set[Path], run_root: Path) -> Path:
    after_paths = {
        path.resolve()
        for path in run_root.glob(f"{run_name}_*")
        if path.is_dir()
    }
    new_paths = sorted(after_paths - before_paths, key=lambda path: path.stat().st_mtime)
    if len(new_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one new run directory for '{run_name}', found {len(new_paths)}"
        )
    return new_paths[0]


def run_benchmark(spec: FrozenBenchmarkSpec, *, device: str) -> dict:
    run_root = DEFAULT_RUN_ROOT
    run_root.mkdir(parents=True, exist_ok=True)

    benchmark_stamp = time.strftime("%Y%m%d-%H%M%S")
    report_dir = DEFAULT_BENCHMARK_REPORT_ROOT / f"{spec.name}_{benchmark_stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    seed_reports = []
    for seed in spec.seeds:
        run_name = f"{spec.name}_{benchmark_stamp}_seed{seed}"
        before_paths = {
            path.resolve()
            for path in run_root.glob(f"{run_name}_*")
            if path.is_dir()
        }

        args = build_frozen_train_args(spec, seed=seed, run_name=run_name, device=device)
        train_brain_script.train_brain(args)

        run_dir = locate_new_run_dir(run_name, before_paths, run_root)
        score = score_brain_run(
            run_dir,
            sustained_points_required=spec.sustained_points_required,
            post_switch_window_ratio=spec.post_switch_window_ratio,
        )
        score_payload = score.to_dict()
        score_payload["seed"] = seed
        score_payload["benchmark"] = spec.name
        seed_reports.append(score_payload)
        write_score_report(report_dir / f"seed_{seed}_score.json", score_payload)

    summary = {
        "benchmark": spec.name,
        "description": spec.description,
        "timestamp": benchmark_stamp,
        "device": device,
        "seeds": list(spec.seeds),
        "fixed_train_args": spec.fixed_train_args,
        "sustained_points_required": spec.sustained_points_required,
        "post_switch_window_ratio": spec.post_switch_window_ratio,
        "seed_reports": seed_reports,
    }
    write_score_report(report_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run a frozen benchmark or score an existing Brain run."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=sorted(spec["name"] for spec in list_frozen_benchmarks()),
        help="Frozen benchmark to execute.",
    )
    parser.add_argument(
        "--score-run-dir",
        type=str,
        help="Existing run directory to score without launching a benchmark.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen benchmark config without executing training.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available frozen benchmarks and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print(json.dumps(list_frozen_benchmarks(), indent=2))
        return

    if bool(args.benchmark) == bool(args.score_run_dir):
        parser.error("Provide exactly one of --benchmark or --score-run-dir.")

    if args.benchmark:
        spec = get_frozen_benchmark(args.benchmark)
        if args.dry_run:
            print(json.dumps(
                {
                    "benchmark": spec.name,
                    "description": spec.description,
                    "seeds": list(spec.seeds),
                    "fixed_train_args": spec.fixed_train_args,
                    "sustained_points_required": spec.sustained_points_required,
                    "post_switch_window_ratio": spec.post_switch_window_ratio,
                },
                indent=2,
            ))
            return

        summary = run_benchmark(spec, device=args.device)
        print(json.dumps(summary, indent=2))
        return

    score = score_brain_run(args.score_run_dir)
    print(json.dumps(score.to_dict(), indent=2))


if __name__ == "__main__":
    main()
