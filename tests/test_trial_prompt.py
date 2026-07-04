import json
import os
import subprocess
import sys
from pathlib import Path

from lifelong_learning.research.autoresearch import load_research_manifest
from lifelong_learning.research.trial_prompt import (
    build_trial_context_payload,
    render_research_trial_prompt,
)


def _write_manifest(repo_root: Path) -> Path:
    manifest_path = repo_root / "research_manifest.toml"
    manifest_path.write_text(
        "\n".join(
            [
                "[trial_limits]",
                "max_trials = 3",
                "max_new_python_files = 2",
                "max_net_new_lines = 20",
                "",
                "[validation]",
                'tests_command = "fake-tests"',
                "",
                "[benchmark]",
                'primary = "fast_switch_scout_v1"',
                'holdout = ["fast_switch_holdout_v1"]',
                "",
                "[stopping]",
                "max_stale_trials = 2",
                "target_composite_score = 0.95",
                "",
                "[outputs]",
                'ledger_path = "autoresearch/trial_results.jsonl"',
                'scratch_dir = "autoresearch"',
                "",
                "[editable_surface]",
                "paths = [",
                '  "src/lifelong_learning/agents/brain/neuromod.py",',
                '  "src/lifelong_learning/agents/ppo/network.py",',
                "]",
                "",
                "[optional_editable_surface]",
                "paths = [",
                '  "src/lifelong_learning/agents/brain/signals.py",',
                '  "src/lifelong_learning/agents/brain/meta_agent.py",',
                "]",
                "",
                "[immutable_surface]",
                "paths = [",
                '  "scripts/run_frozen_benchmark.py",',
                '  "src/lifelong_learning/research/benchmarking.py",',
                '  "scripts/analyze_runs.py",',
                '  "scripts/plot_high_scale.py",',
                '  "src/lifelong_learning/envs",',
                "]",
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _touch_surface(repo_root: Path) -> None:
    for relative_path in (
        "src/lifelong_learning/agents/brain/neuromod.py",
        "src/lifelong_learning/agents/ppo/network.py",
        "src/lifelong_learning/agents/brain/signals.py",
        "src/lifelong_learning/agents/brain/meta_agent.py",
        "src/lifelong_learning/research/benchmarking.py",
        "src/lifelong_learning/envs/demo.py",
        "scripts/run_frozen_benchmark.py",
        "scripts/analyze_runs.py",
        "scripts/plot_high_scale.py",
    ):
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")


def test_render_research_trial_prompt_includes_baseline_and_constraints(tmp_path):
    _touch_surface(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    manifest = load_research_manifest(manifest_path)
    baseline = {
        "primary_benchmark": {
            "aggregate": {
                "composite_score": 0.42,
                "mean_post_switch_window_success_rate": 0.68,
                "mean_inner_time_avg_success_rate": 0.74,
                "median_steps_to_80": 120.0,
                "median_steps_to_95": 240.0,
                "hit_rate_80": 0.9,
                "hit_rate_95": 0.5,
            }
        },
        "holdout_benchmarks": [
            {"name": "fast_switch_holdout_v1", "aggregate": {"composite_score": 0.33}}
        ],
    }

    prompt = render_research_trial_prompt(
        program_text="# Brief\nTry a better decoder.\n",
        trial_index=2,
        repo_root=tmp_path,
        manifest_path=manifest_path,
        program_path=tmp_path / "program_neuromod.md",
        trial_dir=tmp_path / "autoresearch" / "trial_002",
        manifest=manifest,
        baseline_summary=baseline,
    )

    assert "Trial 2" in prompt
    assert "Baseline composite score: `0.42`" in prompt
    assert "Baseline mean post-switch window success rate: `0.68`" in prompt
    assert "Baseline mean inner time-avg success rate: `0.74`" in prompt
    assert "src/lifelong_learning/agents/brain/neuromod.py" in prompt
    assert "Try a better decoder." in prompt
    assert "16`, `32`, or `64`" in prompt


def test_run_research_trial_dry_run_writes_prompt_and_context(tmp_path):
    _touch_surface(tmp_path)
    manifest_path = _write_manifest(tmp_path)
    program_path = tmp_path / "program_neuromod.md"
    program_path.write_text("# Brief\nKeep the Brain action surface stable.\n", encoding="utf-8")
    baseline_path = tmp_path / "baseline_summary.json"
    baseline_path.write_text(
        json.dumps(
            {
                "primary_benchmark": {"aggregate": {"composite_score": 0.4}},
                "holdout_benchmarks": [],
            }
        ),
        encoding="utf-8",
    )
    trial_dir = tmp_path / "autoresearch" / "trial_001"
    runner_template = "echo prompt={prompt_file} notes={notes_file}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str((Path.cwd() / "src").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str((Path.cwd() / "scripts" / "run_research_trial.py").resolve()),
            "--program",
            str(program_path),
            "--manifest",
            str(manifest_path),
            "--trial",
            "1",
            "--trial-dir",
            str(trial_dir),
            "--repo-root",
            str(tmp_path),
            "--baseline-file",
            str(baseline_path),
            "--runner-template",
            runner_template,
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(completed.stdout)
    assert Path(payload["prompt_file"]).exists()
    assert Path(payload["context_file"]).exists()
    assert payload["final_message_file"].endswith("codex_final_message.md")
    assert payload["agent_stdout_file"].endswith("agent_runner.stdout.log")
    assert "prompt=" in payload["runner_command"]

    context = json.loads(Path(payload["context_file"]).read_text(encoding="utf-8"))
    assert context["baseline"]["primary_composite_score"] == 0.4


def test_cap_history_never_evicts_science_verdicts():
    from lifelong_learning.research.trial_prompt import _cap_history

    def entry(i, science):
        return {"trial_index": i, "science_verdict": science}

    entries = [entry(1, True)] + [entry(i, False) for i in range(2, 10)] + [entry(10, True)]
    capped = _cap_history(entries, infra_cap=3)

    kept_indices = [e["trial_index"] for e in capped]
    assert 1 in kept_indices and 10 in kept_indices, "science verdicts must never age out"
    assert kept_indices == [1, 7, 8, 9, 10], "only the last 3 infra failures are kept, order preserved"


def test_next_research_note_number_skips_existing(tmp_path):
    from lifelong_learning.research.trial_prompt import next_research_note_number

    notes = tmp_path / "docs" / "research-notes"
    notes.mkdir(parents=True)
    (notes / "0001-first.md").write_text("x", encoding="utf-8")
    (notes / "0007-later.md").write_text("x", encoding="utf-8")
    (notes / "README.md").write_text("x", encoding="utf-8")

    assert next_research_note_number(tmp_path) == 8
    assert next_research_note_number(tmp_path / "missing") == 1


def test_append_note_verdicts_touches_only_trial_created_notes(tmp_path):
    from lifelong_learning.research.autoresearch import append_note_verdicts

    notes = tmp_path / "docs" / "research-notes"
    notes.mkdir(parents=True)
    created = notes / "0004-new-idea.md"
    created.write_text("# 0004\n\n**Status:** OPEN\n", encoding="utf-8")
    readme = notes / "README.md"
    readme.write_text("index\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    code = tmp_path / "src" / "module.py"
    code.write_text("X = 1\n", encoding="utf-8")

    resolved = append_note_verdicts(
        tmp_path,
        ["docs/research-notes/0004-new-idea.md", "docs/research-notes/README.md", "src/module.py"],
        verdict_line="\n**Resolved:** accepted.\n",
    )

    assert resolved == ["docs/research-notes/0004-new-idea.md"]
    assert "**Resolved:** accepted." in created.read_text(encoding="utf-8")
    assert readme.read_text(encoding="utf-8") == "index\n"
    assert code.read_text(encoding="utf-8") == "X = 1\n"
