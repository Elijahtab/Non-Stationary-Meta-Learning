#!/usr/bin/env python
"""Rebuild the PNG charts for a run locally from its ``*_data.json`` files.

Cloud/sweep runs skip PNG rendering (LL_RENDER_CHARTS unset) to avoid writing ~1.4 GB of derived
images per sweep — the JSON logs are the source of truth. This script regenerates the figures on
demand from those JSONs, reusing the exact same DataLogger.plot() code path.

Usage:
    python scripts/render_charts.py <run_dir_or_data.json> [more ...]
    python scripts/render_charts.py sweeps/cloud_pull_20260701/runs/*   # globs work

For each ``*_data.json`` found (brain_trends and, if present, per-episode dumps), writes the
matching ``*_charts.png`` (+ ``*_success_rate.png``) next to it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["LL_RENDER_CHARTS"] = "1"  # force plot() to actually render

from lifelong_learning.utils.logger import DataLogger  # noqa: E402


def _iter_data_jsons(target: Path):
    if target.is_file() and target.name.endswith("_data.json"):
        yield target
    elif target.is_dir():
        yield from sorted(target.rglob("*_data.json"))


def render_one(data_json: Path) -> bool:
    run_name = data_json.name[: -len("_data.json")]
    save_dir = data_json.parent
    try:
        data = json.loads(data_json.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"[render] skip {data_json}: unreadable ({e})")
        return False
    # full_dir set to the existing dir so no new timestamped folder is created.
    logger = DataLogger(run_name=run_name, log_dir=str(save_dir), full_dir=str(save_dir))
    logger.data = data
    logger.plot(save_dir=str(save_dir), title=f"{run_name}")
    print(f"[render] {save_dir / (run_name + '_charts.png')}")
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    n = 0
    for arg in argv:
        for dj in _iter_data_jsons(Path(arg)):
            n += render_one(dj)
    print(f"[render] done — {n} chart set(s) rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
