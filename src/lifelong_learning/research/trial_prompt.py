from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from lifelong_learning.research.autoresearch import AutoresearchManifest

# Rejections that constitute a *scientific* verdict on the hypothesis (vs. an infrastructure
# failure that says nothing about the idea). Accepted trials are science verdicts by definition.
_SCIENCE_REJECT_REASONS = {"primary_score_did_not_improve", "holdout_regressed"}


def load_hypothesis_queue(repo_root: str | Path) -> list[str]:
    """Read the human-curated hypothesis queue (ordered bullets/numbered items).

    The queue lives at config/hypothesis_queue.md — deliberately *outside* the editable
    surface so the diff audit prevents the trial agent from editing its own assignments.
    Missing file means an empty queue (agent free-picks).
    """
    path = Path(repo_root) / "config" / "hypothesis_queue.md"
    if not path.exists():
        return []
    items: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^(?:[-*]|\d+[.)])\s+(.*\S)", raw.strip())
        if match:
            items.append(match.group(1))
    return items


def _note_first_line(notes_path: Path, cap: int = 240) -> str | None:
    if not notes_path.exists():
        return None
    for line in notes_path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip().lstrip("#").strip()
        if text:
            return text[:cap]
    return None


def collect_trial_history(
    *,
    repo_root: str | Path,
    manifest: AutoresearchManifest,
    session_id: str,
    max_session_entries: int = 8,
    max_other_entries: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Summarize prior trial records from the ledger for prompt injection.

    Rejected trials are rolled back wholesale (including their research notes), so the
    ledger + per-trial agent_notes.md under the scratch dir are the only surviving record
    of what was already tried — without this, trial N can re-propose exactly what trial
    N-1 just failed at.
    """
    root = Path(repo_root)
    ledger_path = root / manifest.outputs.ledger_path
    scratch_dir = root / manifest.outputs.scratch_dir
    session: list[dict[str, Any]] = []
    previous: list[dict[str, Any]] = []
    if ledger_path.exists():
        for raw in ledger_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            if rec.get("record_type") != "trial":
                continue
            sid = rec.get("session_id")
            idx = rec.get("trial_index")
            hypothesis = None
            if sid and isinstance(idx, int):
                hypothesis = _note_first_line(
                    scratch_dir / str(sid) / f"trial_{idx:03d}" / "agent_notes.md"
                )
            aggregate = (rec.get("primary_benchmark") or {}).get("aggregate") or {}
            entry = {
                "session_id": sid,
                "trial_index": idx,
                "status": rec.get("status"),
                "reason": rec.get("reason"),
                "hypothesis": hypothesis,
                "changed_paths": list(rec.get("changed_paths") or [])[:6],
                # Verdicts are benchmark-scoped: a rejection on one primary benchmark does
                # not retire the hypothesis under a different one (scout v1 -> v2 switch).
                "benchmark": aggregate.get("benchmark"),
                "composite_score": aggregate.get("composite_score"),
                "science_verdict": (
                    rec.get("status") == "accepted"
                    or rec.get("reason") in _SCIENCE_REJECT_REASONS
                ),
            }
            (session if sid == session_id else previous).append(entry)
    return {
        "session": _cap_history(session, max_session_entries),
        "previous_sessions": _cap_history(previous, max_other_entries),
    }


def _cap_history(entries: list[dict[str, Any]], infra_cap: int) -> list[dict[str, Any]]:
    """Cap history without ever evicting science verdicts.

    A science verdict permanently retires (or validates) a hypothesis; if it scrolls out
    of the injected window the agent re-implements settled science (review 2026-07-03).
    Only infrastructure failures — retryable, low-information — age out.
    """
    infra_kept = {
        id(entry)
        for entry in [e for e in entries if not e.get("science_verdict")][-infra_cap:]
    }
    return [
        entry
        for entry in entries
        if entry.get("science_verdict") or id(entry) in infra_kept
    ]


def _format_history_lines(entries: list[dict[str, Any]]) -> str:
    lines = []
    for e in entries:
        score = e.get("composite_score")
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        if e.get("benchmark"):
            score_text += f" on {e['benchmark']}"
        verdict = "science verdict" if e.get("science_verdict") else "infrastructure failure; idea untested"
        hyp = f' — "{e["hypothesis"]}"' if e.get("hypothesis") else ""
        lines.append(
            f"- Trial {e.get('trial_index')} [{e.get('session_id')}]: "
            f"{e.get('status')} ({e.get('reason')}; {verdict}) — score {score_text}{hyp}"
        )
    return "\n".join(lines)


def next_research_note_number(repo_root: str | Path) -> int:
    """Next free NNNN under docs/research-notes/.

    Injected into the trial prompt so concurrent/sequential trials can't collide on note
    numbers (two same-day trials both minted 0003 before this existed; only a rollback
    avoided the on-disk collision — review 2026-07-03).
    """
    notes_dir = Path(repo_root) / "docs" / "research-notes"
    highest = 0
    if notes_dir.exists():
        for entry in notes_dir.glob("*.md"):
            match = re.match(r"(\d{4})-", entry.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


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
    hypothesis_queue: list[str] | None = None,
    trial_history: dict[str, list[dict[str, Any]]] | None = None,
    reserved_note_number: int | None = None,
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
        "hypothesis_queue": list(hypothesis_queue or []),
        "trial_history": trial_history or {"session": [], "previous_sessions": []},
        "reserved_note_number": reserved_note_number,
        "baseline": {
            "primary_composite_score": primary_aggregate.get("composite_score"),
            "primary_mean_post_switch_window_success_rate": primary_aggregate.get(
                "mean_post_switch_window_success_rate"
            ),
            "primary_inner_time_avg_success_rate": primary_aggregate.get(
                "mean_inner_time_avg_success_rate"
            ),
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
    hypothesis_queue: list[str] | None = None,
    trial_history: dict[str, list[dict[str, Any]]] | None = None,
    reserved_note_number: int | None = None,
) -> str:
    context = build_trial_context_payload(
        trial_index=trial_index,
        repo_root=repo_root,
        manifest_path=manifest_path,
        program_path=program_path,
        trial_dir=trial_dir,
        manifest=manifest,
        baseline_summary=baseline_summary,
        hypothesis_queue=hypothesis_queue,
        trial_history=trial_history,
        reserved_note_number=reserved_note_number,
    )

    reserved_note_line = ""
    if reserved_note_number is not None:
        reserved_note_line = (
            f"- If you create a research note under `docs/research-notes/`, number it "
            f"`{reserved_note_number:04d}` (reserved for this trial) and add it to the "
            "README index.\n"
        )

    queue_section = ""
    if context["hypothesis_queue"]:
        queue_lines = "\n".join(
            f"{i}. {item}" for i, item in enumerate(context["hypothesis_queue"], start=1)
        )
        primary_name = context["benchmark"]["primary"]
        queue_section = (
            "## Hypothesis Queue (human-curated)\n\n"
            "Ordered priorities from `config/hypothesis_queue.md` (read-only for you — it is\n"
            "outside the editable surface). Take the highest entry that does NOT already have a\n"
            f"science verdict **on the current primary benchmark (`{primary_name}`)** in the\n"
            "trial history below — verdicts recorded on a different benchmark are prior\n"
            "evidence, not retirement. If every entry is resolved, propose your own smallest\n"
            "next hypothesis instead.\n\n"
            f"{queue_lines}\n\n"
        )

    history = context["trial_history"]
    history_section = ""
    if history["session"] or history["previous_sessions"]:
        parts = ["## Trial History (same ledger)\n"]
        if history["session"]:
            parts.append("### Earlier trials in this session\n")
            parts.append(_format_history_lines(history["session"]) + "\n")
        if history["previous_sessions"]:
            parts.append("### Recent trials from previous sessions\n")
            parts.append(_format_history_lines(history["previous_sessions"]) + "\n")
        parts.append(
            "Rules: do not re-propose a hypothesis that already has a science verdict\n"
            "(accepted, or rejected with `primary_score_did_not_improve` / `holdout_regressed`)\n"
            f"**on the current primary benchmark (`{context['benchmark']['primary']}`)** — each\n"
            "entry above names the benchmark it was scored on, and verdicts from a different\n"
            "benchmark are prior evidence only. A trial that failed for infrastructure reasons\n"
            "left its idea untested — you may retry it if you avoid the recorded failure cause.\n\n"
        )
        history_section = "\n".join(parts)
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
        f"- Baseline mean post-switch window success rate: `{baseline['primary_mean_post_switch_window_success_rate']}`\n"
        f"- Baseline mean inner time-avg success rate: `{baseline['primary_inner_time_avg_success_rate']}`\n"
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
        f"{queue_section}"
        f"{history_section}"
        "## Required Output\n\n"
        f"- Write a short note to `{Path(trial_dir).resolve() / 'agent_notes.md'}` containing the hypothesis, touched files, and expected effect.\n"
        f"{reserved_note_line}"
        "- Then make the code change directly in the repo.\n"
        "- Stop after one coherent experiment. Do not chain multiple unrelated ideas into the same trial.\n\n"
        "## Research Brief\n\n"
        f"{program_text.strip()}\n"
    )
