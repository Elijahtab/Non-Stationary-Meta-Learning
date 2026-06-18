from __future__ import annotations

import ast
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrozenBenchmarkSpec:
    """Immutable benchmark definition for future autoresearch runs."""

    name: str
    description: str
    seeds: tuple[int, ...]
    fixed_train_args: dict[str, Any]
    sustained_points_required: int = 3
    post_switch_window_ratio: float = 0.5
    post_switch_buffer_steps: int = 500


FROZEN_BENCHMARKS: dict[str, FrozenBenchmarkSpec] = {
    "fast_switch_pilot_v1": FrozenBenchmarkSpec(
        name="fast_switch_pilot_v1",
        description=(
            "Very small pilot benchmark for validating the autoresearch loop end-to-end. "
            "Used for prompt/rendering/benchmark smoke tests before real scout runs."
        ),
        seeds=(0,),
        fixed_train_args={
            "env_id": "MiniGrid-MultiGoal-5x5-v0",
            "num_regimes": 2,
            "start_regime": 0,
            "randomize_start_regime": False,
            "inner_total_timesteps": 12_000,
            "inner_steps_per_regime": 3_000,
            "inner_num_envs": 2,
            "inner_num_steps": 32,
            "inner_mode": "dyna",
            "inner_intrinsic_coef": 0.015,
            "inner_imagined_horizon": 5,
            "inner_wm_lr": 1e-4,
            "brain_num_envs": 1,
            "brain_vectorization": "sync",
            "pretrain_episodes": 0,
            "pretrain_mode": "recovery",
            "brain_episodes": 1,
            "brain_lr": 1e-4,
            "brain_ent_coef": 0.0,
            "decision_interval": 2,
            "reward_alpha": 0.1,
            "reward_beta": 0.5,
            "reward_mode": "recovery",
            "disable_neuromodulation": False,
            "episodic_memory_capacity": 5_000,
            "max_inner_lr": 0.003,
            "min_inner_lr": 1e-4,
            "min_ent_coef": 0.001,
            "max_ent_coef": 0.1,
            "min_intrinsic_coef": 0.001,
            "max_intrinsic_coef": 0.5,
        },
        sustained_points_required=2,
        post_switch_window_ratio=0.5,
    ),
    "fast_switch_scout_v1": FrozenBenchmarkSpec(
        name="fast_switch_scout_v1",
        description=(
            "Primary fast-switch benchmark for neuromodulation research. "
            "Optimizes for recovery on a fixed 5x5 continual-learning setting."
        ),
        seeds=(0,),
        fixed_train_args={
            "env_id": "MiniGrid-MultiGoal-5x5-v0",
            "num_regimes": 2,
            "start_regime": 0,
            "randomize_start_regime": False,
            "inner_total_timesteps": 800_000,
            "inner_steps_per_regime": 100_000,
            "inner_num_envs": 8,
            "inner_num_steps": 128,
            "inner_mode": "dyna",
            "inner_intrinsic_coef": 0.015,
            "inner_imagined_horizon": 10,
            "inner_wm_lr": 1e-4,
            "brain_num_envs": 4,
            "brain_vectorization": "async",
            "pretrain_episodes": 0,
            "pretrain_mode": "recovery",
            "brain_episodes": 4,
            "brain_lr": 1e-4,
            "brain_ent_coef": 0.0,
            "decision_interval": 10,
            "reward_alpha": 0.1,
            "reward_beta": 0.5,
            "reward_mode": "recovery",
            "disable_neuromodulation": False,
            "episodic_memory_capacity": 50_000,
            "max_inner_lr": 0.003,
            "min_inner_lr": 1e-4,
            "min_ent_coef": 0.001,
            "max_ent_coef": 0.1,
            "min_intrinsic_coef": 0.001,
            "max_intrinsic_coef": 0.5,
        },
    ),
    "fast_switch_holdout_v1": FrozenBenchmarkSpec(
        name="fast_switch_holdout_v1",
        description=(
            "Harder holdout benchmark with a larger grid and multiple frozen seeds. "
            "Used to validate promising neuromodulation changes discovered on the scout benchmark."
        ),
        seeds=(11, 23, 37),
        fixed_train_args={
            "env_id": "MiniGrid-MultiGoal-8x8-v0",
            "num_regimes": 2,
            "start_regime": 0,
            "randomize_start_regime": False,
            "inner_total_timesteps": 800_000,
            "inner_steps_per_regime": 100_000,
            "inner_num_envs": 8,
            "inner_num_steps": 128,
            "inner_mode": "dyna",
            "inner_intrinsic_coef": 0.015,
            "inner_imagined_horizon": 10,
            "inner_wm_lr": 1e-4,
            "brain_num_envs": 4,
            "brain_vectorization": "async",
            "pretrain_episodes": 0,
            "pretrain_mode": "recovery",
            "brain_episodes": 4,
            "brain_lr": 1e-4,
            "brain_ent_coef": 0.0,
            "decision_interval": 10,
            "reward_alpha": 0.1,
            "reward_beta": 0.5,
            "reward_mode": "recovery",
            "disable_neuromodulation": False,
            "episodic_memory_capacity": 50_000,
            "max_inner_lr": 0.003,
            "min_inner_lr": 1e-4,
            "min_ent_coef": 0.001,
            "max_ent_coef": 0.1,
            "min_intrinsic_coef": 0.001,
            "max_intrinsic_coef": 0.5,
        },
    ),
}


@dataclass
class ThresholdRecoverySummary:
    threshold: float
    sustained_points_required: int
    switch_count: int
    hit_count: int
    hit_rate: float | None
    median_steps: float | None
    per_switch_steps: list[int] = field(default_factory=list)


@dataclass
class BrainRunScore:
    run_dir: str
    config: dict[str, Any]
    inner_run_count: int
    switch_count: int
    mean_episode_avg_success_rate: float | None
    mean_inner_time_avg_success_rate: float | None
    mean_post_switch_window_success_rate: float | None
    median_steps_to_80: float | None
    median_steps_to_95: float | None
    hit_rate_80: float | None
    hit_rate_95: float | None
    mean_post_switch_policy_kl: float | None
    mean_post_switch_value_delta_abs: float | None
    mean_post_switch_neuromod_activity: float | None
    composite_score: float | None
    threshold_80: ThresholdRecoverySummary | None = None
    threshold_95: ThresholdRecoverySummary | None = None
    per_inner_run: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def get_frozen_benchmark(name: str) -> FrozenBenchmarkSpec:
    try:
        return FROZEN_BENCHMARKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(FROZEN_BENCHMARKS))
        raise KeyError(f"Unknown benchmark '{name}'. Available benchmarks: {available}") from exc


def list_frozen_benchmarks() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in FROZEN_BENCHMARKS.values()]


def parse_run_config(run_dir: str | Path) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.txt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    config: dict[str, Any] = {}
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.endswith("Configuration:") or set(line) == {"-"}:
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_scalar(value.strip())
    return config


def load_json_log(log_path: str | Path) -> dict[str, list[list[float]]]:
    with open(log_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_regime_switch_steps(regime_points: list[list[float]] | list[tuple[float, float]]) -> list[int]:
    if len(regime_points) < 2:
        return []

    switches: list[int] = []
    prev_regime = int(round(float(regime_points[0][1])))
    for step, regime_value in regime_points[1:]:
        regime = int(round(float(regime_value)))
        if regime != prev_regime:
            switches.append(int(step))
            prev_regime = regime
    return switches


def find_first_sustained_threshold_step(
    success_points: list[list[float]] | list[tuple[float, float]],
    *,
    threshold: float,
    start_step: int,
    end_step: int,
    sustained_points_required: int,
) -> int | None:
    consecutive_hits = 0
    block_start_step: int | None = None

    for step_value, success_value in success_points:
        step = int(step_value)
        if step <= start_step or step >= end_step:
            continue

        if float(success_value) >= threshold:
            if consecutive_hits == 0:
                block_start_step = step
            consecutive_hits += 1
            if consecutive_hits >= sustained_points_required:
                return block_start_step
        else:
            consecutive_hits = 0
            block_start_step = None

    return None


def summarize_post_switch_success(
    success_points: list[list[float]] | list[tuple[float, float]],
    regime_points: list[list[float]] | list[tuple[float, float]],
    *,
    steps_per_regime: int,
    post_switch_window_ratio: float = 0.5,
    post_switch_buffer_steps: int = 500,
) -> dict[str, Any]:
    switches = detect_regime_switch_steps(regime_points)
    if not switches:
        return {
            "switch_count": 0,
            "window_count": 0,
            "mean_success_rate": None,
            "per_window_success_rate": [],
        }

    final_step = _series_final_step(success_points, regime_points)
    window_width = max(1, int(math.ceil(steps_per_regime * post_switch_window_ratio)))
    per_window_success_rate: list[float] = []

    for index, switch_step in enumerate(switches):
        next_switch_step = switches[index + 1] if index + 1 < len(switches) else final_step
        window_start = switch_step + post_switch_buffer_steps
        window_end = min(next_switch_step, window_start + window_width)
        if window_start >= window_end:
            continue

        window_success = [
            float(value)
            for step, value in success_points
            if int(step) > window_start and int(step) <= window_end
        ]
        if not window_success:
            continue
        per_window_success_rate.append(float(np.mean(window_success)))

    return {
        "switch_count": len(switches),
        "window_count": len(per_window_success_rate),
        "mean_success_rate": (
            float(np.mean(per_window_success_rate)) if per_window_success_rate else None
        ),
        "per_window_success_rate": per_window_success_rate,
    }


def summarize_threshold_recovery(
    success_points: list[list[float]] | list[tuple[float, float]],
    regime_points: list[list[float]] | list[tuple[float, float]],
    *,
    threshold: float,
    sustained_points_required: int,
    post_switch_buffer_steps: int = 500,
) -> ThresholdRecoverySummary:
    switches = detect_regime_switch_steps(regime_points)
    if not switches:
        return ThresholdRecoverySummary(
            threshold=threshold,
            sustained_points_required=sustained_points_required,
            switch_count=0,
            hit_count=0,
            hit_rate=None,
            median_steps=None,
        )

    final_step = _series_final_step(success_points, regime_points)

    per_switch_steps: list[int] = []
    for index, switch_step in enumerate(switches):
        next_switch_step = switches[index + 1] if index + 1 < len(switches) else final_step
        sustained_step = find_first_sustained_threshold_step(
            success_points,
            threshold=threshold,
            start_step=switch_step + post_switch_buffer_steps,
            end_step=next_switch_step,
            sustained_points_required=sustained_points_required,
        )
        if sustained_step is not None:
            per_switch_steps.append(int(sustained_step - switch_step))

    switch_count = len(switches)
    hit_count = len(per_switch_steps)
    hit_rate = (hit_count / switch_count) if switch_count else None
    median_steps = float(median(per_switch_steps)) if per_switch_steps else None
    return ThresholdRecoverySummary(
        threshold=threshold,
        sustained_points_required=sustained_points_required,
        switch_count=switch_count,
        hit_count=hit_count,
        hit_rate=hit_rate,
        median_steps=median_steps,
        per_switch_steps=per_switch_steps,
    )


def summarize_post_switch_neuromodulation(
    data: dict[str, list[list[float]]],
    *,
    steps_per_regime: int,
    post_switch_window_ratio: float = 0.5,
    post_switch_buffer_steps: int = 500,
) -> dict[str, Any]:
    regime_points = data.get("charts/regime_id", [])
    switches = detect_regime_switch_steps(regime_points)
    if not switches:
        return {
            "switch_count": 0,
            "window_count": 0,
            "mean_policy_kl": None,
            "mean_value_delta_abs": None,
            "mean_activity": None,
            "per_window_activity": [],
        }

    policy_points = data.get("brain_neuromod/policy_kl_vs_unmasked", [])
    value_points = data.get("brain_neuromod/value_delta_abs_vs_unmasked", [])
    if not policy_points and not value_points:
        return {
            "switch_count": len(switches),
            "window_count": 0,
            "mean_policy_kl": None,
            "mean_value_delta_abs": None,
            "mean_activity": None,
            "per_window_activity": [],
        }

    final_step = _series_final_step(regime_points, policy_points, value_points)
    window_width = max(1, int(math.ceil(steps_per_regime * post_switch_window_ratio)))

    policy_window_means: list[float] = []
    value_window_means: list[float] = []
    activity_window_means: list[float] = []

    for index, switch_step in enumerate(switches):
        next_switch_step = switches[index + 1] if index + 1 < len(switches) else final_step
        window_start = switch_step + post_switch_buffer_steps
        window_end = min(next_switch_step, window_start + window_width)
        if window_start >= window_end:
            continue

        window_policy = [
            float(value)
            for step, value in policy_points
            if int(step) > window_start and int(step) <= window_end
        ]
        window_value = [
            float(value)
            for step, value in value_points
            if int(step) > window_start and int(step) <= window_end
        ]

        if not window_policy and not window_value:
            continue

        policy_mean = float(np.mean(window_policy)) if window_policy else 0.0
        value_mean = float(np.mean(window_value)) if window_value else 0.0
        policy_window_means.append(policy_mean)
        value_window_means.append(value_mean)
        activity_window_means.append(value_mean + 0.1 * policy_mean)

    return {
        "switch_count": len(switches),
        "window_count": len(activity_window_means),
        "mean_policy_kl": float(np.mean(policy_window_means)) if policy_window_means else None,
        "mean_value_delta_abs": float(np.mean(value_window_means)) if value_window_means else None,
        "mean_activity": float(np.mean(activity_window_means)) if activity_window_means else None,
        "per_window_activity": activity_window_means,
    }


def score_brain_run(
    run_dir: str | Path,
    *,
    sustained_points_required: int = 3,
    post_switch_window_ratio: float = 0.5,
    post_switch_buffer_steps: int = 500,
) -> BrainRunScore:
    run_path = Path(run_dir)
    config = parse_run_config(run_path)
    steps_per_regime = int(config["inner_steps_per_regime"])

    brain_trends_json = _find_single_json(run_path / "brain_trends")
    brain_trends = load_json_log(brain_trends_json) if brain_trends_json else {}

    inner_run_scores: list[dict[str, Any]] = []
    all_steps_80: list[int] = []
    all_steps_95: list[int] = []
    total_switches = 0
    hit_count_80 = 0
    hit_count_95 = 0
    policy_kl_values: list[float] = []
    value_delta_values: list[float] = []
    activity_values: list[float] = []
    inner_time_avg_success_rates: list[float] = []
    post_switch_success_rates: list[float] = []

    for inner_json in iter_inner_run_json_logs(run_path):
        data = load_json_log(inner_json)
        success_points = data.get("charts/success_rate", [])
        post_switch_success = summarize_post_switch_success(
            success_points,
            data.get("charts/regime_id", []),
            steps_per_regime=steps_per_regime,
            post_switch_window_ratio=post_switch_window_ratio,
            post_switch_buffer_steps=post_switch_buffer_steps,
        )
        threshold_80 = summarize_threshold_recovery(
            success_points,
            data.get("charts/regime_id", []),
            threshold=0.80,
            sustained_points_required=sustained_points_required,
            post_switch_buffer_steps=post_switch_buffer_steps,
        )
        threshold_95 = summarize_threshold_recovery(
            success_points,
            data.get("charts/regime_id", []),
            threshold=0.95,
            sustained_points_required=sustained_points_required,
            post_switch_buffer_steps=post_switch_buffer_steps,
        )
        neuromod = summarize_post_switch_neuromodulation(
            data,
            steps_per_regime=steps_per_regime,
            post_switch_window_ratio=post_switch_window_ratio,
            post_switch_buffer_steps=post_switch_buffer_steps,
        )

        total_switches += threshold_80.switch_count
        hit_count_80 += threshold_80.hit_count
        hit_count_95 += threshold_95.hit_count
        all_steps_80.extend(threshold_80.per_switch_steps)
        all_steps_95.extend(threshold_95.per_switch_steps)
        if post_switch_success["mean_success_rate"] is not None:
            post_switch_success_rates.append(float(post_switch_success["mean_success_rate"]))
        if neuromod["mean_policy_kl"] is not None:
            policy_kl_values.append(float(neuromod["mean_policy_kl"]))
        if neuromod["mean_value_delta_abs"] is not None:
            value_delta_values.append(float(neuromod["mean_value_delta_abs"]))
        if neuromod["mean_activity"] is not None:
            activity_values.append(float(neuromod["mean_activity"]))
        time_avg_success_rate = (
            float(np.mean([float(point[1]) for point in success_points]))
            if success_points
            else None
        )
        if time_avg_success_rate is not None:
            inner_time_avg_success_rates.append(time_avg_success_rate)

        inner_run_scores.append(
            {
                "log_path": str(inner_json),
                "time_avg_success_rate": time_avg_success_rate,
                "post_switch_success": post_switch_success,
                "threshold_80": asdict(threshold_80),
                "threshold_95": asdict(threshold_95),
                "neuromodulation": neuromod,
            }
        )

    episode_success_points = brain_trends.get("brain/episode_avg_success_rate", [])
    if not episode_success_points:
        episode_success_points = brain_trends.get("brain/inner_final_success_rate", [])
    mean_episode_avg_success_rate = (
        float(np.mean([float(point[1]) for point in episode_success_points]))
        if episode_success_points
        else None
    )
    mean_inner_time_avg_success_rate = (
        float(np.mean(inner_time_avg_success_rates)) if inner_time_avg_success_rates else None
    )
    mean_post_switch_window_success_rate = (
        float(np.mean(post_switch_success_rates)) if post_switch_success_rates else None
    )

    threshold_80_summary = ThresholdRecoverySummary(
        threshold=0.80,
        sustained_points_required=sustained_points_required,
        switch_count=total_switches,
        hit_count=hit_count_80,
        hit_rate=(hit_count_80 / total_switches) if total_switches else None,
        median_steps=float(median(all_steps_80)) if all_steps_80 else None,
        per_switch_steps=all_steps_80,
    )
    threshold_95_summary = ThresholdRecoverySummary(
        threshold=0.95,
        sustained_points_required=sustained_points_required,
        switch_count=total_switches,
        hit_count=hit_count_95,
        hit_rate=(hit_count_95 / total_switches) if total_switches else None,
        median_steps=float(median(all_steps_95)) if all_steps_95 else None,
        per_switch_steps=all_steps_95,
    )

    composite_score = _compute_composite_score(
        mean_post_switch_window_success_rate=mean_post_switch_window_success_rate,
        hit_rate_80=threshold_80_summary.hit_rate,
        median_steps_to_80=threshold_80_summary.median_steps,
        steps_per_regime=steps_per_regime,
        post_switch_buffer_steps=post_switch_buffer_steps,
    )

    return BrainRunScore(
        run_dir=str(run_path),
        config=config,
        inner_run_count=len(inner_run_scores),
        switch_count=total_switches,
        mean_episode_avg_success_rate=mean_episode_avg_success_rate,
        mean_inner_time_avg_success_rate=mean_inner_time_avg_success_rate,
        mean_post_switch_window_success_rate=mean_post_switch_window_success_rate,
        median_steps_to_80=threshold_80_summary.median_steps,
        median_steps_to_95=threshold_95_summary.median_steps,
        hit_rate_80=threshold_80_summary.hit_rate,
        hit_rate_95=threshold_95_summary.hit_rate,
        mean_post_switch_policy_kl=float(np.mean(policy_kl_values)) if policy_kl_values else None,
        mean_post_switch_value_delta_abs=float(np.mean(value_delta_values)) if value_delta_values else None,
        mean_post_switch_neuromod_activity=float(np.mean(activity_values)) if activity_values else None,
        composite_score=composite_score,
        threshold_80=threshold_80_summary,
        threshold_95=threshold_95_summary,
        per_inner_run=inner_run_scores,
    )


def write_score_report(destination: str | Path, score: BrainRunScore | dict[str, Any]) -> None:
    payload = score.to_dict() if isinstance(score, BrainRunScore) else score
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def iter_inner_run_json_logs(run_dir: str | Path):
    run_path = Path(run_dir)
    for episode_dir in sorted(run_path.glob("episode_*")):
        if not episode_dir.is_dir():
            continue
        for inner_dir in sorted(episode_dir.glob("ep*_env*")):
            if not inner_dir.is_dir():
                continue
            json_path = _find_single_json(inner_dir)
            if json_path is not None:
                yield json_path


def _find_single_json(folder: str | Path) -> Path | None:
    folder = Path(folder)
    if not folder.exists():
        return None
    candidates = sorted(folder.glob("*_data.json"))
    if not candidates:
        return None
    return candidates[0]


def _compute_composite_score(
    *,
    mean_post_switch_window_success_rate: float | None,
    hit_rate_80: float | None,
    median_steps_to_80: float | None,
    steps_per_regime: int,
    post_switch_buffer_steps: int,
) -> float | None:
    if (
        mean_post_switch_window_success_rate is None
        and hit_rate_80 is None
        and median_steps_to_80 is None
    ):
        return None
    normalized_steps_to_80 = _normalize_recovery_steps(
        median_steps_to_80,
        steps_per_regime=steps_per_regime,
        post_switch_buffer_steps=post_switch_buffer_steps,
    )
    return float(
        0.5 * (mean_post_switch_window_success_rate or 0.0)
        + 0.25 * (hit_rate_80 or 0.0)
        + 0.25 * (normalized_steps_to_80 or 0.0)
    )


def _normalize_recovery_steps(
    median_steps: float | None,
    *,
    steps_per_regime: int,
    post_switch_buffer_steps: int,
) -> float | None:
    if median_steps is None:
        return None
    usable_window = max(1.0, float(steps_per_regime - post_switch_buffer_steps))
    elapsed_after_buffer = max(0.0, float(median_steps) - float(post_switch_buffer_steps))
    clipped_elapsed = min(elapsed_after_buffer, usable_window)
    return float(max(0.0, 1.0 - (clipped_elapsed / usable_window)))


def _series_final_step(
    *point_series: list[list[float]] | list[tuple[float, float]],
) -> int:
    max_step: float | None = None
    for series in point_series:
        for point in series:
            point_step = float(point[0])
            if max_step is None or point_step > max_step:
                max_step = point_step
    return int(max_step or 0) + 1


def _parse_scalar(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value
