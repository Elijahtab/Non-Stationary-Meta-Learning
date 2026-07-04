import json
from pathlib import Path

from lifelong_learning.research.autoresearch import (
    AutoresearchSupervisor,
    BenchmarkExecution,
    CommandExecution,
    audit_repo_diff,
    compute_repo_diff,
    load_research_manifest,
    take_repo_snapshot,
)


def _write_manifest(repo_root: Path, *, include_holdout: bool = True) -> Path:
    holdout_line = 'holdout = ["fast_switch_holdout_v1"]\n' if include_holdout else ""
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
                "tests_timeout_seconds = 60",
                "",
                "[benchmark]",
                'primary = "fast_switch_scout_v1"',
                holdout_line.rstrip(),
                'device = "cpu"',
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


def _make_repo(repo_root: Path, *, include_holdout: bool = True) -> Path:
    files = {
        "src/lifelong_learning/agents/brain/neuromod.py": "VALUE = 1\n",
        "src/lifelong_learning/agents/ppo/network.py": "NETWORK = 1\n",
        "src/lifelong_learning/agents/brain/signals.py": "SIGNALS = 1\n",
        "src/lifelong_learning/agents/brain/meta_agent.py": "META = 1\n",
        "src/lifelong_learning/research/benchmarking.py": "BENCH = 1\n",
        "src/lifelong_learning/envs/demo_env.py": "ENV = 1\n",
        "scripts/run_frozen_benchmark.py": "print('frozen')\n",
        "scripts/analyze_runs.py": "print('analyze')\n",
        "scripts/plot_high_scale.py": "print('plot')\n",
    }
    for relative_path, content in files.items():
        target = repo_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return _write_manifest(repo_root, include_holdout=include_holdout)


def _fake_command_runner_factory(repo_root: Path, *, mutate_trial, fail_tests: bool = False):
    def _runner(command: str, cwd: Path, timeout, stdout_path: Path, stderr_path: Path):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(f"{command}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")

        if command.startswith("trial-"):
            mutate_trial(command, repo_root)
            return CommandExecution(
                command=command,
                returncode=0,
                duration_seconds=0.01,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )

        return CommandExecution(
            command=command,
            returncode=1 if fail_tests else 0,
            duration_seconds=0.01,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    return _runner


def _fake_benchmark_runner_factory(scores_by_name: dict[str, list[float]]):
    counts = {name: 0 for name in scores_by_name}

    def _runner(repo_root: Path, benchmark_name: str, device: str, timeout, stdout_path: Path, stderr_path: Path):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(f"{benchmark_name}:{device}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")

        index = counts[benchmark_name]
        counts[benchmark_name] += 1
        score = scores_by_name[benchmark_name][index]
        summary = {
            "benchmark": benchmark_name,
            "seed_reports": [
                {
                    "composite_score": score,
                    "mean_episode_avg_success_rate": score,
                    "mean_inner_time_avg_success_rate": score,
                    "mean_post_switch_window_success_rate": score,
                    "median_steps_to_80": 10.0,
                    "median_steps_to_95": 20.0,
                    "hit_rate_80": 1.0,
                    "hit_rate_95": 0.5,
                    "mean_post_switch_policy_kl": 0.01,
                    "mean_post_switch_value_delta_abs": 0.02,
                    "mean_post_switch_neuromod_activity": 0.03,
                }
            ],
        }
        return BenchmarkExecution(
            name=benchmark_name,
            duration_seconds=0.01,
            report_dir=str(repo_root / "benchmarks" / f"{benchmark_name}_fake"),
            summary=summary,
            aggregate={
                "benchmark": benchmark_name,
                "seed_count": 1,
                "composite_score": score,
                "mean_episode_avg_success_rate": score,
                "mean_inner_time_avg_success_rate": score,
                "mean_post_switch_window_success_rate": score,
                "median_steps_to_80": 10.0,
                "median_steps_to_95": 20.0,
                "hit_rate_80": 1.0,
                "hit_rate_95": 0.5,
                "mean_post_switch_policy_kl": 0.01,
                "mean_post_switch_value_delta_abs": 0.02,
                "mean_post_switch_neuromod_activity": 0.03,
                "report_dir": str(repo_root / "benchmarks" / f"{benchmark_name}_fake"),
            },
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    return _runner


def _write_cached_baseline(
    repo_root: Path,
    *,
    session_id: str,
    fingerprint: str,
    primary_score: float,
    holdout_scores: dict[str, float] | None = None,
) -> Path:
    holdout_scores = holdout_scores or {}
    session_dir = repo_root / "autoresearch" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    primary_report_dir = repo_root / "benchmarks" / f"cached_primary_{session_id}"
    primary_report_dir.mkdir(parents=True, exist_ok=True)
    holdout_payloads = []
    for benchmark_name, score in holdout_scores.items():
        report_dir = repo_root / "benchmarks" / f"{benchmark_name}_{session_id}"
        report_dir.mkdir(parents=True, exist_ok=True)
        holdout_payloads.append(
            {
                "name": benchmark_name,
                "duration_seconds": 1.0,
                "report_dir": str(report_dir),
                "stdout_path": str(session_dir / f"{benchmark_name}.stdout.log"),
                "stderr_path": str(session_dir / f"{benchmark_name}.stderr.log"),
                "aggregate": {
                    "benchmark": benchmark_name,
                    "composite_score": score,
                },
                "summary": {"benchmark": benchmark_name, "seed_reports": []},
            }
        )

    baseline_entry = {
        "session_id": session_id,
        "record_type": "baseline",
        "timestamp_utc": "2026-04-05T00:00:00+00:00",
        "status": "accepted",
        "reason": "baseline_snapshot",
        "primary_benchmark": {
            "name": "fast_switch_scout_v1",
            "duration_seconds": 1.0,
            "report_dir": str(primary_report_dir),
            "stdout_path": str(session_dir / "primary.stdout.log"),
            "stderr_path": str(session_dir / "primary.stderr.log"),
            "aggregate": {
                "benchmark": "fast_switch_scout_v1",
                "composite_score": primary_score,
            },
            "summary": {"benchmark": "fast_switch_scout_v1", "seed_reports": []},
        },
        "holdout_benchmarks": holdout_payloads,
        "best_primary_score": primary_score,
        "baseline_cache": {
            "enabled": True,
            "reused": False,
            "fingerprint": fingerprint,
            "fingerprint_paths": ["src/lifelong_learning/agents/brain/neuromod.py"],
            "source_session_id": None,
            "source_baseline_summary": None,
        },
    }
    summary_path = session_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(baseline_entry, indent=2), encoding="utf-8")
    return summary_path


def test_audit_repo_diff_allows_new_python_files_under_editable_parent(tmp_path):
    manifest_path = _make_repo(tmp_path)
    manifest = load_research_manifest(manifest_path)

    before = take_repo_snapshot(tmp_path, ignored_roots=manifest.ignored_roots())
    new_file = tmp_path / "src" / "lifelong_learning" / "agents" / "brain" / "new_mask.py"
    new_file.write_text("MASK = 2\n", encoding="utf-8")
    after = take_repo_snapshot(tmp_path, ignored_roots=manifest.ignored_roots())

    diff = compute_repo_diff(before, after)
    audit = audit_repo_diff(diff, manifest)

    assert audit.is_valid
    assert audit.new_python_paths == ("src/lifelong_learning/agents/brain/new_mask.py",)


def test_supervisor_accepts_improving_trial_and_keeps_changes(tmp_path):
    manifest_path = _make_repo(tmp_path, include_holdout=True)
    target_file = tmp_path / "src" / "lifelong_learning" / "agents" / "brain" / "neuromod.py"

    def mutate_trial(command: str, repo_root: Path):
        if command == "trial-1":
            with open(target_file, "a", encoding="utf-8") as handle:
                handle.write("IMPROVED = 2\n")

    supervisor = AutoresearchSupervisor(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        research_command="trial-{trial}",
        command_runner=_fake_command_runner_factory(tmp_path, mutate_trial=mutate_trial),
        benchmark_runner=_fake_benchmark_runner_factory(
            {
                "fast_switch_scout_v1": [0.40, 0.55],
                "fast_switch_holdout_v1": [0.35, 0.36],
            }
        ),
    )

    summary = supervisor.run(max_trials=1)

    ledger_lines = (tmp_path / "autoresearch" / "trial_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    baseline_record = json.loads(ledger_lines[0])
    trial_record = json.loads(ledger_lines[1])

    assert summary["accepted_trials"] == [1]
    assert summary["best_primary_score"] == 0.55
    assert baseline_record["record_type"] == "baseline"
    assert trial_record["status"] == "accepted"
    assert trial_record["reason"] == "primary_improved"
    assert "IMPROVED = 2" in target_file.read_text(encoding="utf-8")


def test_supervisor_rejects_immutable_change_and_restores_file(tmp_path):
    manifest_path = _make_repo(tmp_path, include_holdout=False)
    frozen_file = tmp_path / "scripts" / "run_frozen_benchmark.py"
    original_content = frozen_file.read_text(encoding="utf-8")

    def mutate_trial(command: str, repo_root: Path):
        if command == "trial-1":
            frozen_file.write_text("print('mutated')\n", encoding="utf-8")

    benchmark_runner = _fake_benchmark_runner_factory({"fast_switch_scout_v1": [0.40]})
    supervisor = AutoresearchSupervisor(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        research_command="trial-{trial}",
        command_runner=_fake_command_runner_factory(tmp_path, mutate_trial=mutate_trial),
        benchmark_runner=benchmark_runner,
    )

    summary = supervisor.run(max_trials=1)
    ledger_lines = (tmp_path / "autoresearch" / "trial_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    trial_record = json.loads(ledger_lines[1])

    assert summary["accepted_trials"] == []
    assert trial_record["status"] == "rejected"
    assert trial_record["reason"] == "immutable_surface_violation"
    assert frozen_file.read_text(encoding="utf-8") == original_content


def test_supervisor_dry_run_returns_command_preview(tmp_path):
    manifest_path = _make_repo(tmp_path, include_holdout=False)
    supervisor = AutoresearchSupervisor(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        research_command="trial-{trial} --program {program}",
        program_path=tmp_path / "notes.md",
    )
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")

    preview = supervisor.run(dry_run=True)

    assert preview["mode"] == "dry_run"
    assert "trial-1" in preview["trial_1_command_preview"]
    assert "notes.md" in preview["trial_1_command_preview"]


def test_supervisor_reuses_matching_cached_baseline(tmp_path):
    manifest_path = _make_repo(tmp_path, include_holdout=True)
    supervisor = AutoresearchSupervisor(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        research_command="trial-{trial}",
    )
    fingerprint = supervisor._build_baseline_fingerprint_payload()["fingerprint"]
    cached_summary = _write_cached_baseline(
        tmp_path,
        session_id="20260405-000000",
        fingerprint=fingerprint,
        primary_score=0.42,
        holdout_scores={"fast_switch_holdout_v1": 0.31},
    )

    benchmark_calls: list[str] = []

    def _unexpected_benchmark_runner(*args, **kwargs):
        benchmark_calls.append(args[1])
        raise AssertionError("benchmark runner should not execute when baseline cache matches")

    supervisor = AutoresearchSupervisor(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        research_command="trial-{trial}",
        benchmark_runner=_unexpected_benchmark_runner,
    )

    summary = supervisor.run(max_trials=0)
    ledger_lines = (tmp_path / "autoresearch" / "trial_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    baseline_record = json.loads(ledger_lines[0])

    assert benchmark_calls == []
    assert summary["trial_count"] == 0
    assert baseline_record["reason"] == "baseline_cache_hit"
    assert baseline_record["baseline_cache"]["reused"] is True
    assert baseline_record["baseline_cache"]["source_baseline_summary"] == str(cached_summary)


def test_supervisor_invalidates_cached_baseline_after_relevant_code_change(tmp_path):
    manifest_path = _make_repo(tmp_path, include_holdout=True)
    seed_supervisor = AutoresearchSupervisor(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        research_command="trial-{trial}",
    )
    fingerprint = seed_supervisor._build_baseline_fingerprint_payload()["fingerprint"]
    _write_cached_baseline(
        tmp_path,
        session_id="20260405-000000",
        fingerprint=fingerprint,
        primary_score=0.42,
        holdout_scores={"fast_switch_holdout_v1": 0.31},
    )

    target_file = tmp_path / "src" / "lifelong_learning" / "agents" / "brain" / "neuromod.py"
    target_file.write_text("VALUE = 2\n", encoding="utf-8")

    benchmark_calls: list[str] = []

    def _counting_benchmark_runner(repo_root: Path, benchmark_name: str, device: str, timeout, stdout_path: Path, stderr_path: Path):
        benchmark_calls.append(benchmark_name)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        score = 0.55 if benchmark_name == "fast_switch_scout_v1" else 0.44
        return BenchmarkExecution(
            name=benchmark_name,
            duration_seconds=0.01,
            report_dir=str(repo_root / "benchmarks" / f"{benchmark_name}_fresh"),
            summary={"benchmark": benchmark_name, "seed_reports": []},
            aggregate={
                "benchmark": benchmark_name,
                "composite_score": score,
            },
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    supervisor = AutoresearchSupervisor(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        research_command="trial-{trial}",
        benchmark_runner=_counting_benchmark_runner,
    )

    summary = supervisor.run(max_trials=0)
    ledger_lines = (tmp_path / "autoresearch" / "trial_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    baseline_record = json.loads(ledger_lines[0])

    assert summary["baseline_primary_score"] == 0.55
    assert benchmark_calls == ["fast_switch_scout_v1", "fast_switch_holdout_v1"]
    assert baseline_record["reason"] == "baseline_snapshot"
    assert baseline_record["baseline_cache"]["reused"] is False


def test_default_command_runner_kills_grandchild_on_timeout(tmp_path):
    """A research timeout must kill the whole process tree, not just the shell child.

    The grandchild here waits, then writes a sentinel; if only the direct shell is
    killed (the pre-review behavior), the orphan survives the timeout and the sentinel
    appears — exactly the post-rollback repo-mutation race from the 2026-07-03 review.
    """
    import sys
    import time

    from lifelong_learning.research.autoresearch import default_command_runner

    sentinel = tmp_path / "orphan_sentinel.txt"
    grandchild_script = tmp_path / "grandchild.py"
    grandchild_script.write_text(
        "import pathlib\n"
        "import time\n"
        "time.sleep(4)\n"
        f"pathlib.Path({str(sentinel)!r}).write_text('orphan lived')\n",
        encoding="utf-8",
    )
    spawner_script = tmp_path / "spawner.py"
    spawner_script.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        f"subprocess.Popen([sys.executable, {str(grandchild_script)!r}])\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    result = default_command_runner(
        f'"{sys.executable}" "{spawner_script}"',
        tmp_path,
        2,
        tmp_path / "out.log",
        tmp_path / "err.log",
    )

    assert result.timed_out is True
    time.sleep(5)
    assert not sentinel.exists(), "grandchild survived the timeout kill"


def test_repo_snapshot_tracks_and_restores_png_files(tmp_path):
    """Agent-written figures (.png) must be visible to the audit and undone by rollback."""
    from lifelong_learning.research.autoresearch import restore_repo_snapshot

    (tmp_path / "docs").mkdir()
    before = take_repo_snapshot(tmp_path, ignored_roots=())

    figure = tmp_path / "docs" / "0004-figure.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
    after = take_repo_snapshot(tmp_path, ignored_roots=())

    diff = compute_repo_diff(before, after)
    assert "docs/0004-figure.png" in diff.new_paths

    restore_repo_snapshot(tmp_path, before, after)
    assert not figure.exists(), "rejected-trial figure survived rollback"


def test_resolve_command_placeholders_substitutes_python():
    import sys

    from lifelong_learning.research.autoresearch import resolve_command_placeholders

    resolved = resolve_command_placeholders("{python} -m pytest -q")
    assert resolved == f'"{sys.executable}" -m pytest -q'
    assert resolve_command_placeholders("fake-tests") == "fake-tests"
