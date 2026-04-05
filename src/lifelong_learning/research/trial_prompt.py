from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lifelong_learning.research.autoresearch import AutoresearchManifest


def load_baseline_summary(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    baseline_path = Path(path)
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def build_trial_context_payload(
    *,
    trial_index: int,
    repo_root: str | Path,
    manifest_path: str | Path,
    program_path: str | Path,
    trial_dir: str | Path,
    manifest: AutoresearchManifest,
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    primary = baseline_summary.get("primary_benchmark", {}) if baseline_summary else {}
    primary_aggregate = primary.get("aggregate", {}) if primary else {}
    holdouts = baseline_summary.get("holdout_benchmarks", []) if baseline_summary else []
    return {
        "trial_index": trial_index,
        "repo_root": str(Path(repo_root).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "program_path": str(Path(program_path).resolve()),
        "trial_dir": str(Path(trial_dir).resolve()),
        "editable_surface": list(manifest.editable_surface),
        "optional_editable_surface": list(manifest.optional_editable_surface),
        "immutable_surface": list(manifest.immutable_surface),
        "trial_limits": {
            "max_new_python_files": manifest.trial_limits.max_new_python_files,
            "max_net_new_lines": manifest.trial_limits.max_net_new_lines,
            "max_trials": manifest.trial_limits.max_trials,
        },
        "benchmark": {
            "primary": manifest.benchmark.primary,
            "holdout": list(manifest.benchmark.holdout),
            "device": manifest.benchmark.device,
            "target_composite_score": manifest.stopping.target_composite_score,
        },
        "baseline": {
            "primary_composite_score": primary_aggregate.get("composite_score"),
            "primary_success_rate": primary_aggregate.get("mean_episode_avg_success_rate"),
            "primary_median_steps_to_80": primary_aggregate.get("median_steps_to_80"),
            "primary_median_steps_to_95": primary_aggregate.get("median_steps_to_95"),
            "primary_hit_rate_80": primary_aggregate.get("hit_rate_80"),
            "primary_hit_rate_95": primary_aggregate.get("hit_rate_95"),
            "holdout_composite_scores": {
                item.get("name"): item.get("aggregate", {}).get("composite_score")
                for item in holdouts
            },
        },
    }


def render_research_trial_prompt(
    *,
    program_text: str,
    trial_index: int,
    repo_root: str | Path,
    manifest_path: str | Path,
    program_path: str | Path,
    trial_dir: str | Path,
    manifest: AutoresearchManifest,
    baseline_summary: dict[str, Any] | None,
) -> str:
    context = build_trial_context_payload(
        trial_index=trial_index,
        repo_root=repo_root,
        manifest_path=manifest_path,
        program_path=program_path,
        trial_dir=trial_dir,
        manifest=manifest,
        baseline_summary=baseline_summary,
    )
    baseline = context["baseline"]
    holdout_map = baseline["holdout_composite_scores"]
    holdout_lines = (
        "\n".join(f"- `{name}`: `{score}`" for name, score in holdout_map.items())
        if holdout_map
        else "- none"
    )

    return (
        f"# Trial {trial_index}: Neuromodulation Research Worker\n\n"
        f"You are the bounded researcher for a single autoresearch trial in this repo.\n"
        f"Work inside `{context['repo_root']}` and keep all permanent code changes inside the allowed surface.\n\n"
        "## Task\n\n"
        "Make one small, testable code change that could improve Meta-RL neuromodulation on the frozen benchmark.\n"
        "Prefer the smallest hypothesis that has a realistic chance to improve the composite score.\n\n"
        "## Current Architecture\n\n"
        "- Regime-switched MiniGrid envs hide the regime in observations and only change reward contingencies.\n"
        "- The inner learner is Dyna-PPO: real PPO rollouts plus replay, a world model, and dreamed updates.\n"
        "- The outer Brain is PPO over `MetaEnv`, which sees aggregate signals and sets inner hyperparameters.\n"
        "- Neuromodulation currently flows from the Brain context code into the inner CNN feature pipeline.\n"
        "- The main research question is whether better neuromodulation improves post-switch recovery.\n\n"
        "## Baseline Snapshot\n\n"
        f"- Primary benchmark: `{context['benchmark']['primary']}`\n"
        f"- Baseline composite score: `{baseline['primary_composite_score']}`\n"
        f"- Baseline mean success rate: `{baseline['primary_success_rate']}`\n"
        f"- Baseline median steps to 80%: `{baseline['primary_median_steps_to_80']}`\n"
        f"- Baseline median steps to 95%: `{baseline['primary_median_steps_to_95']}`\n"
        f"- Baseline hit rate 80%: `{baseline['primary_hit_rate_80']}`\n"
        f"- Baseline hit rate 95%: `{baseline['primary_hit_rate_95']}`\n"
        "- Holdout baseline scores:\n"
        f"{holdout_lines}\n\n"
        "## Editable Surface\n\n"
        + "\n".join(f"- `{path}`" for path in context["editable_surface"])
        + "\n\n## Optional Surface\n\n"
        + "\n".join(f"- `{path}`" for path in context["optional_editable_surface"])
        + "\n\n## Immutable Surface\n\n"
        + "\n".join(f"- `{path}`" for path in context["immutable_surface"])
        + "\n\n## Trial Constraints\n\n"
        f"- Max cumulative new Python files this session: `{context['trial_limits']['max_new_python_files']}`\n"
        f"- Max cumulative net new Python lines this session: `{context['trial_limits']['max_net_new_lines']}`\n"
        "- Do not touch benchmark, scoring, plotting, or environment code.\n"
        "- Keep the diff small and mechanically simple.\n"
        "- Do not widen the research scope beyond neuromodulation unless it is required for shape consistency.\n\n"
        "## Priority Order\n\n"
        "1. Prefer richer neuromodulation decoders or mask parameterizations before changing the Brain action shape.\n"
        "2. Prefer actor/critic-specific or channel/feature granularity experiments over broad refactors.\n"
        "3. Only change context dimensionality directly if the benefit is clear and the action/interface updates stay coherent.\n"
        "4. If you increase neuromodulation capacity, keep checkpoint and runtime shapes internally consistent.\n\n"
        "## Candidate Hypotheses\n\n"
        "- Increase decoder capacity while keeping the Brain action surface stable.\n"
        "- Try gain-based or affine modulation instead of purely suppressive masking.\n"
        "- Split actor and critic modulation paths.\n"
        "- Explore larger neuromodulation capacity such as `16`, `32`, or `64`, but only if the interface cost is justified.\n\n"
        "## Required Output\n\n"
        f"- Write a short note to `{Path(trial_dir).resolve() / 'agent_notes.md'}` containing the hypothesis, touched files, and expected effect.\n"
        "- Then make the code change directly in the repo.\n"
        "- Stop after one coherent experiment. Do not chain multiple unrelated ideas into the same trial.\n\n"
        "## Research Brief\n\n"
        f"{program_text.strip()}\n"
    )
