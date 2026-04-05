from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from lifelong_learning.research.benchmarking import get_frozen_benchmark


DEFAULT_IGNORED_ROOTS = (
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "benchmarks",
    "myenv",
    "runs",
)

MANAGED_FILE_SUFFIXES = {
    ".bat",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".pyi",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_PRIMARY_IMPROVEMENT_EPSILON = 1e-9
DEFAULT_HOLDOUT_REGRESSION_TOLERANCE = 0.0
DEFAULT_TESTS_TIMEOUT_SECONDS = 1_800


@dataclass(frozen=True)
class TrialLimits:
    max_trials: int
    max_new_python_files: int
    max_net_new_lines: int


@dataclass(frozen=True)
class ValidationConfig:
    tests_command: str
    tests_timeout_seconds: int = DEFAULT_TESTS_TIMEOUT_SECONDS
    research_timeout_seconds: int | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    primary: str
    holdout: tuple[str, ...] = ()
    device: str = "cuda"
    primary_improvement_epsilon: float = DEFAULT_PRIMARY_IMPROVEMENT_EPSILON
    holdout_regression_tolerance: float = DEFAULT_HOLDOUT_REGRESSION_TOLERANCE
    benchmark_timeout_seconds: int | None = None


@dataclass(frozen=True)
class StoppingConfig:
    max_stale_trials: int
    target_composite_score: float | None = None


@dataclass(frozen=True)
class OutputsConfig:
    ledger_path: str
    scratch_dir: str


@dataclass(frozen=True)
class AutoresearchManifest:
    trial_limits: TrialLimits
    validation: ValidationConfig
    benchmark: BenchmarkConfig
    stopping: StoppingConfig
    outputs: OutputsConfig
    editable_surface: tuple[str, ...]
    optional_editable_surface: tuple[str, ...]
    immutable_surface: tuple[str, ...]

    @property
    def allowed_surface(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.editable_surface, *self.optional_editable_surface)))

    @property
    def new_python_file_roots(self) -> tuple[str, ...]:
        roots: set[str] = set()
        for entry in self.allowed_surface:
            entry_path = PurePosixPath(entry)
            if entry.endswith("/"):
                roots.add(entry.rstrip("/"))
                continue
            if entry_path.suffix:
                parent = entry_path.parent.as_posix()
                roots.add("." if parent == "." else parent)
            else:
                roots.add(entry)
        return tuple(sorted(roots))

    def ignored_roots(self) -> tuple[str, ...]:
        output_roots = {
            _normalize_relative_path(self.outputs.scratch_dir),
            _normalize_relative_path(str(PurePosixPath(self.outputs.ledger_path).parent)),
        }
        output_roots.discard(".")
        return tuple(dict.fromkeys((*DEFAULT_IGNORED_ROOTS, *sorted(output_roots))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_limits": asdict(self.trial_limits),
            "validation": asdict(self.validation),
            "benchmark": asdict(self.benchmark),
            "stopping": asdict(self.stopping),
            "outputs": asdict(self.outputs),
            "editable_surface": list(self.editable_surface),
            "optional_editable_surface": list(self.optional_editable_surface),
            "immutable_surface": list(self.immutable_surface),
            "new_python_file_roots": list(self.new_python_file_roots),
            "ignored_roots": list(self.ignored_roots()),
        }


@dataclass(frozen=True)
class RepoSnapshot:
    root: Path
    files: dict[str, bytes]


@dataclass(frozen=True)
class RepoDiff:
    changed_paths: tuple[str, ...]
    modified_paths: tuple[str, ...]
    new_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.changed_paths)


@dataclass(frozen=True)
class PythonGrowthSummary:
    new_python_paths: tuple[str, ...]
    net_new_lines: int
    changed_python_paths: tuple[str, ...]


@dataclass(frozen=True)
class AuditResult:
    changed_paths: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    dynamic_allowed_paths: tuple[str, ...]
    new_python_paths: tuple[str, ...]
    immutable_violations: tuple[str, ...]
    unauthorized_paths: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.immutable_violations and not self.unauthorized_paths


@dataclass(frozen=True)
class CommandExecution:
    command: str
    returncode: int | None
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.returncode == 0


@dataclass(frozen=True)
class BenchmarkExecution:
    name: str
    duration_seconds: float
    report_dir: str | None
    summary: dict[str, Any]
    aggregate: dict[str, Any]
    stdout_path: str
    stderr_path: str


CommandRunner = Callable[[str, Path, int | None, Path, Path], CommandExecution]
BenchmarkRunner = Callable[[Path, str, str, int | None, Path, Path], BenchmarkExecution]


def load_research_manifest(path: str | Path) -> AutoresearchManifest:
    manifest_path = Path(path)
    with open(manifest_path, "rb") as handle:
        raw = tomllib.load(handle)

    trial_limits = raw["trial_limits"]
    validation = raw["validation"]
    benchmark = raw["benchmark"]
    stopping = raw["stopping"]
    outputs = raw["outputs"]
    editable_surface = _load_surface(raw, "editable_surface")
    optional_editable_surface = _load_surface(raw, "optional_editable_surface")
    immutable_surface = _load_surface(raw, "immutable_surface")

    manifest = AutoresearchManifest(
        trial_limits=TrialLimits(
            max_trials=int(trial_limits["max_trials"]),
            max_new_python_files=int(trial_limits["max_new_python_files"]),
            max_net_new_lines=int(trial_limits["max_net_new_lines"]),
        ),
        validation=ValidationConfig(
            tests_command=str(validation["tests_command"]),
            tests_timeout_seconds=int(
                validation.get("tests_timeout_seconds", DEFAULT_TESTS_TIMEOUT_SECONDS)
            ),
            research_timeout_seconds=_optional_int(validation.get("research_timeout_seconds")),
        ),
        benchmark=BenchmarkConfig(
            primary=str(benchmark["primary"]),
            holdout=tuple(str(item) for item in benchmark.get("holdout", [])),
            device=str(benchmark.get("device", "cuda")),
            primary_improvement_epsilon=float(
                benchmark.get(
                    "primary_improvement_epsilon", DEFAULT_PRIMARY_IMPROVEMENT_EPSILON
                )
            ),
            holdout_regression_tolerance=float(
                benchmark.get(
                    "holdout_regression_tolerance", DEFAULT_HOLDOUT_REGRESSION_TOLERANCE
                )
            ),
            benchmark_timeout_seconds=_optional_int(benchmark.get("benchmark_timeout_seconds")),
        ),
        stopping=StoppingConfig(
            max_stale_trials=int(stopping["max_stale_trials"]),
            target_composite_score=(
                None
                if stopping.get("target_composite_score") is None
                else float(stopping["target_composite_score"])
            ),
        ),
        outputs=OutputsConfig(
            ledger_path=_normalize_relative_path(outputs["ledger_path"]),
            scratch_dir=_normalize_relative_path(outputs["scratch_dir"]),
        ),
        editable_surface=editable_surface,
        optional_editable_surface=optional_editable_surface,
        immutable_surface=immutable_surface,
    )
    _validate_manifest(manifest)
    return manifest


def take_repo_snapshot(root: str | Path, *, ignored_roots: tuple[str, ...]) -> RepoSnapshot:
    root_path = Path(root)
    ignored = tuple(_normalize_relative_path(item) for item in ignored_roots if item)
    files: dict[str, bytes] = {}

    for current_root, dirnames, filenames in os.walk(root_path):
        current_dir = Path(current_root)
        rel_dir = (
            _normalize_relative_path(current_dir.relative_to(root_path).as_posix())
            if current_dir != root_path
            else ""
        )
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _path_matches_any(
                _join_relative(rel_dir, dirname),
                ignored,
            )
        ]
        for filename in filenames:
            rel_path = _join_relative(rel_dir, filename)
            if _path_matches_any(rel_path, ignored):
                continue
            if not _is_managed_file(rel_path):
                continue
            files[rel_path] = (root_path / rel_path).read_bytes()

    return RepoSnapshot(root=root_path, files=files)


def compute_repo_diff(before: RepoSnapshot, after: RepoSnapshot) -> RepoDiff:
    before_paths = set(before.files)
    after_paths = set(after.files)
    new_paths = tuple(sorted(after_paths - before_paths))
    deleted_paths = tuple(sorted(before_paths - after_paths))
    modified_paths = tuple(
        sorted(path for path in before_paths & after_paths if before.files[path] != after.files[path])
    )
    changed_paths = tuple(sorted(set(new_paths) | set(deleted_paths) | set(modified_paths)))
    return RepoDiff(
        changed_paths=changed_paths,
        modified_paths=modified_paths,
        new_paths=new_paths,
        deleted_paths=deleted_paths,
    )


def summarize_python_growth(base: RepoSnapshot, current: RepoSnapshot) -> PythonGrowthSummary:
    all_paths = sorted(set(base.files) | set(current.files))
    new_python_paths = tuple(
        sorted(path for path in current.files if path.endswith(".py") and path not in base.files)
    )
    changed_python_paths = []
    net_new_lines = 0

    for path in all_paths:
        if not path.endswith(".py"):
            continue
        base_bytes = base.files.get(path, b"")
        current_bytes = current.files.get(path, b"")
        if base_bytes != current_bytes:
            changed_python_paths.append(path)
            net_new_lines += _count_lines(current_bytes) - _count_lines(base_bytes)

    return PythonGrowthSummary(
        new_python_paths=new_python_paths,
        net_new_lines=net_new_lines,
        changed_python_paths=tuple(changed_python_paths),
    )


def audit_repo_diff(
    diff: RepoDiff,
    manifest: AutoresearchManifest,
    *,
    dynamic_allowed_paths: tuple[str, ...] = (),
) -> AuditResult:
    immutable_violations: list[str] = []
    unauthorized_paths: list[str] = []
    new_python_paths: list[str] = []
    allowed_paths: list[str] = []
    dynamic_allowed = tuple(sorted(dynamic_allowed_paths))

    for path in diff.changed_paths:
        if _path_matches_any(path, manifest.immutable_surface):
            immutable_violations.append(path)
            continue
        if _path_matches_any(path, manifest.allowed_surface) or path in dynamic_allowed:
            allowed_paths.append(path)
            continue
        if (
            path in diff.new_paths
            and path.endswith(".py")
            and _path_matches_any(path, manifest.new_python_file_roots)
        ):
            new_python_paths.append(path)
            allowed_paths.append(path)
            continue
        unauthorized_paths.append(path)

    return AuditResult(
        changed_paths=diff.changed_paths,
        allowed_paths=tuple(sorted(allowed_paths)),
        dynamic_allowed_paths=dynamic_allowed,
        new_python_paths=tuple(sorted(new_python_paths)),
        immutable_violations=tuple(sorted(immutable_violations)),
        unauthorized_paths=tuple(sorted(unauthorized_paths)),
    )


def restore_repo_snapshot(
    root: str | Path,
    target: RepoSnapshot,
    current: RepoSnapshot,
) -> list[str]:
    root_path = Path(root)
    restored_paths: list[str] = []
    for path in sorted(set(target.files) | set(current.files)):
        target_bytes = target.files.get(path)
        current_bytes = current.files.get(path)
        if target_bytes == current_bytes:
            continue

        abs_path = root_path / path
        if target_bytes is None:
            if abs_path.exists():
                abs_path.unlink()
                restored_paths.append(path)
            continue

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(target_bytes)
        restored_paths.append(path)

    return restored_paths


def aggregate_benchmark_summary(summary: dict[str, Any], *, report_dir: str | None) -> dict[str, Any]:
    seed_reports = summary.get("seed_reports", [])
    return {
        "benchmark": summary.get("benchmark"),
        "seed_count": len(seed_reports),
        "composite_score": _mean_field(seed_reports, "composite_score"),
        "mean_episode_avg_success_rate": _mean_field(seed_reports, "mean_episode_avg_success_rate"),
        "median_steps_to_80": _mean_field(seed_reports, "median_steps_to_80"),
        "median_steps_to_95": _mean_field(seed_reports, "median_steps_to_95"),
        "hit_rate_80": _mean_field(seed_reports, "hit_rate_80"),
        "hit_rate_95": _mean_field(seed_reports, "hit_rate_95"),
        "mean_post_switch_policy_kl": _mean_field(seed_reports, "mean_post_switch_policy_kl"),
        "mean_post_switch_value_delta_abs": _mean_field(
            seed_reports, "mean_post_switch_value_delta_abs"
        ),
        "mean_post_switch_neuromod_activity": _mean_field(
            seed_reports, "mean_post_switch_neuromod_activity"
        ),
        "report_dir": report_dir,
    }


def default_command_runner(
    command: str,
    cwd: Path,
    timeout_seconds: int | None,
    stdout_path: Path,
    stderr_path: Path,
) -> CommandExecution:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    try:
        with open(stdout_path, "w", encoding="utf-8") as stdout_handle, open(
            stderr_path, "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                shell=True,
                text=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=timeout_seconds,
            )
        return CommandExecution(
            command=command,
            returncode=completed.returncode,
            duration_seconds=time.perf_counter() - started_at,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return CommandExecution(
            command=command,
            returncode=None,
            duration_seconds=time.perf_counter() - started_at,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            timed_out=True,
        )


def default_benchmark_runner(
    repo_root: Path,
    benchmark_name: str,
    device: str,
    timeout_seconds: int | None,
    stdout_path: Path,
    stderr_path: Path,
) -> BenchmarkExecution:
    script_path = repo_root / "scripts" / "run_frozen_benchmark.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing benchmark runner: {script_path}")

    benchmarks_root = repo_root / "benchmarks"
    before_paths = {
        path.resolve()
        for path in benchmarks_root.glob(f"{benchmark_name}_*")
        if path.is_dir()
    }
    command = (
        f'"{sys.executable}" "{script_path}" --benchmark {benchmark_name} --device {device}'
    )
    execution = default_command_runner(
        command,
        repo_root,
        timeout_seconds,
        stdout_path,
        stderr_path,
    )
    if not execution.succeeded:
        raise RuntimeError(
            f"Frozen benchmark '{benchmark_name}' failed with return code {execution.returncode}"
        )

    after_paths = {
        path.resolve()
        for path in benchmarks_root.glob(f"{benchmark_name}_*")
        if path.is_dir()
    }
    new_paths = sorted(after_paths - before_paths, key=lambda path: path.stat().st_mtime)
    if len(new_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one new benchmark report for '{benchmark_name}', found {len(new_paths)}"
        )

    report_dir = new_paths[0]
    summary_path = report_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = aggregate_benchmark_summary(summary, report_dir=str(report_dir))
    return BenchmarkExecution(
        name=benchmark_name,
        duration_seconds=execution.duration_seconds,
        report_dir=str(report_dir),
        summary=summary,
        aggregate=aggregate,
        stdout_path=execution.stdout_path,
        stderr_path=execution.stderr_path,
    )


class AutoresearchSupervisor:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        manifest_path: str | Path,
        research_command: str,
        program_path: str | Path | None = None,
        device: str | None = None,
        command_runner: CommandRunner | None = None,
        benchmark_runner: BenchmarkRunner | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_research_manifest(self.manifest_path)
        self.research_command = research_command.strip()
        self.program_path = Path(program_path).resolve() if program_path else None
        self.device = device or self.manifest.benchmark.device
        self.command_runner = command_runner or default_command_runner
        self.benchmark_runner = benchmark_runner or default_benchmark_runner
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.repo_root / self.manifest.outputs.scratch_dir / self.session_id
        self.ledger_path = self.repo_root / self.manifest.outputs.ledger_path
        self.initial_snapshot: RepoSnapshot | None = None

    def validate(self) -> dict[str, Any]:
        for benchmark_name in (self.manifest.benchmark.primary, *self.manifest.benchmark.holdout):
            get_frozen_benchmark(benchmark_name)

        missing_surface = [
            path
            for path in (*self.manifest.allowed_surface, *self.manifest.immutable_surface)
            if not (self.repo_root / path).exists()
        ]
        if missing_surface:
            raise FileNotFoundError(
                f"Manifest references missing repo paths: {', '.join(sorted(missing_surface))}"
            )
        if self.program_path and not self.program_path.exists():
            raise FileNotFoundError(f"Missing program file: {self.program_path}")

        preview_command = (
            self._render_research_command(
                trial_index=1,
                trial_dir=self.session_dir / "trial_001",
            )
            if self.research_command
            else None
        )
        return {
            "repo_root": str(self.repo_root),
            "manifest_path": str(self.manifest_path),
            "program_path": str(self.program_path) if self.program_path else None,
            "device": self.device,
            "research_command_template": self.research_command or None,
            "trial_1_command_preview": preview_command,
            "manifest": self.manifest.to_dict(),
        }

    def run(self, *, max_trials: int | None = None, dry_run: bool = False) -> dict[str, Any]:
        validation = self.validate()
        if dry_run:
            return {
                "mode": "dry_run",
                "session_id": self.session_id,
                **validation,
            }
        if not self.research_command:
            raise ValueError("research_command must be provided unless dry_run is enabled.")

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.initial_snapshot = take_repo_snapshot(
            self.repo_root,
            ignored_roots=self.manifest.ignored_roots(),
        )

        baseline_entry, best_primary_score, baseline_holdout_scores = self._run_baseline()
        self._append_ledger_record(baseline_entry)

        target_score = self.manifest.stopping.target_composite_score
        if target_score is not None and _meets_score_target(best_primary_score, target_score):
            return self._write_session_summary(
                stop_reason="baseline_reached_target",
                baseline_entry=baseline_entry,
                best_primary_score=best_primary_score,
                accepted_trials=[],
                trial_count=0,
            )

        stale_trials = 0
        accepted_trials: list[int] = []
        trial_limit = self.manifest.trial_limits.max_trials if max_trials is None else max_trials
        stop_reason = "max_trials_reached"
        trials_executed = 0

        for trial_index in range(1, trial_limit + 1):
            trials_executed = trial_index
            trial_entry, accepted, best_primary_score = self._run_trial(
                trial_index=trial_index,
                best_primary_score=best_primary_score,
                baseline_holdout_scores=baseline_holdout_scores,
            )
            self._append_ledger_record(trial_entry)
            if accepted:
                accepted_trials.append(trial_index)
                stale_trials = 0
            else:
                stale_trials += 1

            if target_score is not None and _meets_score_target(best_primary_score, target_score):
                stop_reason = "target_score_reached"
                break
            if stale_trials >= self.manifest.stopping.max_stale_trials:
                stop_reason = "stale_trial_limit_reached"
                break

        return self._write_session_summary(
            stop_reason=stop_reason,
            baseline_entry=baseline_entry,
            best_primary_score=best_primary_score,
            accepted_trials=accepted_trials,
            trial_count=trials_executed,
        )

    def _run_baseline(self) -> tuple[dict[str, Any], float | None, dict[str, float | None]]:
        baseline_dir = self.session_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        primary_execution = self.benchmark_runner(
            self.repo_root,
            self.manifest.benchmark.primary,
            self.device,
            self.manifest.benchmark.benchmark_timeout_seconds,
            baseline_dir / "primary.stdout.log",
            baseline_dir / "primary.stderr.log",
        )
        holdout_executions = []
        baseline_holdout_scores: dict[str, float | None] = {}
        for benchmark_name in self.manifest.benchmark.holdout:
            execution = self.benchmark_runner(
                self.repo_root,
                benchmark_name,
                self.device,
                self.manifest.benchmark.benchmark_timeout_seconds,
                baseline_dir / f"{benchmark_name}.stdout.log",
                baseline_dir / f"{benchmark_name}.stderr.log",
            )
            holdout_executions.append(execution)
            baseline_holdout_scores[benchmark_name] = execution.aggregate.get("composite_score")

        baseline_entry = {
            "session_id": self.session_id,
            "record_type": "baseline",
            "timestamp_utc": _timestamp_utc(),
            "status": "accepted",
            "reason": "baseline_snapshot",
            "primary_benchmark": _benchmark_payload(primary_execution),
            "holdout_benchmarks": [_benchmark_payload(item) for item in holdout_executions],
            "best_primary_score": primary_execution.aggregate.get("composite_score"),
        }
        baseline_summary_path = self.session_dir / "baseline_summary.json"
        baseline_summary_path.write_text(json.dumps(baseline_entry, indent=2), encoding="utf-8")
        return (
            baseline_entry,
            primary_execution.aggregate.get("composite_score"),
            baseline_holdout_scores,
        )

    def _run_trial(
        self,
        *,
        trial_index: int,
        best_primary_score: float | None,
        baseline_holdout_scores: dict[str, float | None],
    ) -> tuple[dict[str, Any], bool, float | None]:
        assert self.initial_snapshot is not None

        trial_dir = self.session_dir / f"trial_{trial_index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        pre_trial_snapshot = take_repo_snapshot(
            self.repo_root,
            ignored_roots=self.manifest.ignored_roots(),
        )
        dynamic_allowed_paths = summarize_python_growth(
            self.initial_snapshot, pre_trial_snapshot
        ).new_python_paths

        research_command = self._render_research_command(trial_index=trial_index, trial_dir=trial_dir)
        research_result = self.command_runner(
            research_command,
            self.repo_root,
            self.manifest.validation.research_timeout_seconds,
            trial_dir / "research.stdout.log",
            trial_dir / "research.stderr.log",
        )

        post_research_snapshot = take_repo_snapshot(
            self.repo_root,
            ignored_roots=self.manifest.ignored_roots(),
        )
        diff = compute_repo_diff(pre_trial_snapshot, post_research_snapshot)
        audit = audit_repo_diff(
            diff,
            self.manifest,
            dynamic_allowed_paths=dynamic_allowed_paths,
        )
        trial_growth = summarize_python_growth(pre_trial_snapshot, post_research_snapshot)
        campaign_growth = summarize_python_growth(self.initial_snapshot, post_research_snapshot)

        trial_entry: dict[str, Any] = {
            "session_id": self.session_id,
            "record_type": "trial",
            "trial_index": trial_index,
            "timestamp_utc": _timestamp_utc(),
            "research_command": research_command,
            "research_execution": asdict(research_result),
            "changed_paths": list(diff.changed_paths),
            "modified_paths": list(diff.modified_paths),
            "new_paths": list(diff.new_paths),
            "deleted_paths": list(diff.deleted_paths),
            "audit": asdict(audit),
            "trial_python_growth": asdict(trial_growth),
            "campaign_python_growth": asdict(campaign_growth),
            "best_primary_score_before_trial": best_primary_score,
        }

        if not research_result.succeeded:
            restored = restore_repo_snapshot(self.repo_root, pre_trial_snapshot, post_research_snapshot)
            trial_entry.update(
                {
                    "status": "failed",
                    "reason": "research_command_failed",
                    "restored_paths": restored,
                }
            )
            return trial_entry, False, best_primary_score

        if not diff.has_changes:
            trial_entry.update(
                {
                    "status": "rejected",
                    "reason": "no_code_changes",
                    "restored_paths": [],
                }
            )
            return trial_entry, False, best_primary_score

        if not audit.is_valid:
            restored = restore_repo_snapshot(self.repo_root, pre_trial_snapshot, post_research_snapshot)
            reason = (
                "immutable_surface_violation"
                if audit.immutable_violations
                else "unauthorized_surface_violation"
            )
            trial_entry.update(
                {
                    "status": "rejected",
                    "reason": reason,
                    "restored_paths": restored,
                }
            )
            return trial_entry, False, best_primary_score

        if len(campaign_growth.new_python_paths) > self.manifest.trial_limits.max_new_python_files:
            restored = restore_repo_snapshot(self.repo_root, pre_trial_snapshot, post_research_snapshot)
            trial_entry.update(
                {
                    "status": "rejected",
                    "reason": "max_new_python_files_exceeded",
                    "restored_paths": restored,
                }
            )
            return trial_entry, False, best_primary_score

        if campaign_growth.net_new_lines > self.manifest.trial_limits.max_net_new_lines:
            restored = restore_repo_snapshot(self.repo_root, pre_trial_snapshot, post_research_snapshot)
            trial_entry.update(
                {
                    "status": "rejected",
                    "reason": "max_net_new_lines_exceeded",
                    "restored_paths": restored,
                }
            )
            return trial_entry, False, best_primary_score

        tests_result = self.command_runner(
            self.manifest.validation.tests_command,
            self.repo_root,
            self.manifest.validation.tests_timeout_seconds,
            trial_dir / "tests.stdout.log",
            trial_dir / "tests.stderr.log",
        )
        trial_entry["tests_execution"] = asdict(tests_result)
        if not tests_result.succeeded:
            restored = restore_repo_snapshot(self.repo_root, pre_trial_snapshot, post_research_snapshot)
            trial_entry.update(
                {
                    "status": "rejected",
                    "reason": "tests_failed",
                    "restored_paths": restored,
                }
            )
            return trial_entry, False, best_primary_score

        try:
            primary_execution = self.benchmark_runner(
                self.repo_root,
                self.manifest.benchmark.primary,
                self.device,
                self.manifest.benchmark.benchmark_timeout_seconds,
                trial_dir / "primary_benchmark.stdout.log",
                trial_dir / "primary_benchmark.stderr.log",
            )
        except Exception as exc:
            restored = restore_repo_snapshot(self.repo_root, pre_trial_snapshot, post_research_snapshot)
            trial_entry.update(
                {
                    "status": "failed",
                    "reason": "primary_benchmark_failed",
                    "error": str(exc),
                    "restored_paths": restored,
                }
            )
            return trial_entry, False, best_primary_score
        trial_entry["primary_benchmark"] = _benchmark_payload(primary_execution)
        primary_score = primary_execution.aggregate.get("composite_score")
        improved_primary = _is_primary_improvement(
            candidate_score=primary_score,
            best_score=best_primary_score,
            epsilon=self.manifest.benchmark.primary_improvement_epsilon,
        )

        holdout_results = []
        holdout_gate_passed = True
        if improved_primary:
            for benchmark_name in self.manifest.benchmark.holdout:
                try:
                    execution = self.benchmark_runner(
                        self.repo_root,
                        benchmark_name,
                        self.device,
                        self.manifest.benchmark.benchmark_timeout_seconds,
                        trial_dir / f"{benchmark_name}.stdout.log",
                        trial_dir / f"{benchmark_name}.stderr.log",
                    )
                except Exception as exc:
                    restored = restore_repo_snapshot(
                        self.repo_root,
                        pre_trial_snapshot,
                        post_research_snapshot,
                    )
                    trial_entry.update(
                        {
                            "status": "failed",
                            "reason": "holdout_benchmark_failed",
                            "error": str(exc),
                            "restored_paths": restored,
                            "holdout_benchmarks": [
                                _benchmark_payload(item) for item in holdout_results
                            ],
                        }
                    )
                    return trial_entry, False, best_primary_score
                holdout_results.append(execution)
                baseline_score = baseline_holdout_scores.get(benchmark_name)
                candidate_score = execution.aggregate.get("composite_score")
                if (
                    baseline_score is not None
                    and candidate_score is not None
                    and candidate_score
                    < baseline_score - self.manifest.benchmark.holdout_regression_tolerance
                ):
                    holdout_gate_passed = False
        trial_entry["holdout_benchmarks"] = [
            _benchmark_payload(item) for item in holdout_results
        ]

        if improved_primary and holdout_gate_passed:
            best_primary_score = primary_score
            trial_entry.update(
                {
                    "status": "accepted",
                    "reason": "primary_improved",
                    "restored_paths": [],
                    "best_primary_score_after_trial": best_primary_score,
                }
            )
            return trial_entry, True, best_primary_score

        restored = restore_repo_snapshot(self.repo_root, pre_trial_snapshot, post_research_snapshot)
        reason = "primary_score_did_not_improve"
        if improved_primary and not holdout_gate_passed:
            reason = "holdout_regressed"
        trial_entry.update(
            {
                "status": "rejected",
                "reason": reason,
                "restored_paths": restored,
                "best_primary_score_after_trial": best_primary_score,
            }
        )
        return trial_entry, False, best_primary_score

    def _render_research_command(self, *, trial_index: int, trial_dir: Path) -> str:
        return self.research_command.format(
            trial=trial_index,
            trial_dir=str(trial_dir),
            manifest=str(self.manifest_path),
            program=str(self.program_path) if self.program_path else "",
            repo_root=str(self.repo_root),
            session_id=self.session_id,
            session_dir=str(self.session_dir),
            baseline_file=str(self.session_dir / "baseline_summary.json"),
        )

    def _append_ledger_record(self, record: dict[str, Any]) -> None:
        with open(self.ledger_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    def _write_session_summary(
        self,
        *,
        stop_reason: str,
        baseline_entry: dict[str, Any],
        best_primary_score: float | None,
        accepted_trials: list[int],
        trial_count: int,
    ) -> dict[str, Any]:
        summary = {
            "session_id": self.session_id,
            "timestamp_utc": _timestamp_utc(),
            "stop_reason": stop_reason,
            "repo_root": str(self.repo_root),
            "manifest_path": str(self.manifest_path),
            "program_path": str(self.program_path) if self.program_path else None,
            "ledger_path": str(self.ledger_path),
            "session_dir": str(self.session_dir),
            "trial_count": trial_count,
            "accepted_trials": accepted_trials,
            "best_primary_score": best_primary_score,
            "baseline_primary_score": baseline_entry["primary_benchmark"]["aggregate"].get(
                "composite_score"
            ),
            "target_composite_score": self.manifest.stopping.target_composite_score,
        }
        summary_json_path = self.session_dir / "session_summary.json"
        summary_md_path = self.session_dir / "session_summary.md"
        summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_md_path.write_text(_format_session_summary_markdown(summary), encoding="utf-8")
        return summary


def _load_surface(raw_manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    table = raw_manifest.get(key, {})
    paths = table.get("paths", [])
    return tuple(_normalize_relative_path(path) for path in paths)


def _validate_manifest(manifest: AutoresearchManifest) -> None:
    for surface_name, surface in (
        ("editable_surface", manifest.editable_surface),
        ("optional_editable_surface", manifest.optional_editable_surface),
        ("immutable_surface", manifest.immutable_surface),
    ):
        for entry in surface:
            if PurePosixPath(entry).is_absolute():
                raise ValueError(f"{surface_name} entries must be repo-relative: {entry}")

    overlap = set(manifest.allowed_surface) & set(manifest.immutable_surface)
    if overlap:
        raise ValueError(
            f"Editable and immutable surfaces overlap: {', '.join(sorted(overlap))}"
        )
    get_frozen_benchmark(manifest.benchmark.primary)
    for benchmark_name in manifest.benchmark.holdout:
        get_frozen_benchmark(benchmark_name)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_relative_path(value: str) -> str:
    text = str(value).replace("\\", "/").strip()
    if text in ("", "."):
        return "."
    normalized = PurePosixPath(text).as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/") or "."


def _join_relative(base: str, child: str) -> str:
    if not base:
        return _normalize_relative_path(child)
    return _normalize_relative_path(f"{base}/{child}")


def _path_matches_any(path: str, roots: tuple[str, ...]) -> bool:
    normalized_path = _normalize_relative_path(path)
    for root in roots:
        normalized_root = _normalize_relative_path(root)
        if normalized_root == ".":
            if "/" not in normalized_path:
                return True
            continue
        if normalized_path == normalized_root or normalized_path.startswith(normalized_root + "/"):
            return True
    return False


def _is_managed_file(relative_path: str) -> bool:
    return PurePosixPath(relative_path).suffix.lower() in MANAGED_FILE_SUFFIXES


def _count_lines(payload: bytes) -> int:
    if not payload:
        return 0
    return len(payload.decode("utf-8", errors="ignore").splitlines())


def _mean_field(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _benchmark_payload(execution: BenchmarkExecution) -> dict[str, Any]:
    return {
        "name": execution.name,
        "duration_seconds": execution.duration_seconds,
        "report_dir": execution.report_dir,
        "stdout_path": execution.stdout_path,
        "stderr_path": execution.stderr_path,
        "aggregate": execution.aggregate,
        "summary": execution.summary,
    }


def _is_primary_improvement(
    *,
    candidate_score: float | None,
    best_score: float | None,
    epsilon: float,
) -> bool:
    if candidate_score is None:
        return False
    if best_score is None:
        return True
    return candidate_score > best_score + epsilon


def _meets_score_target(score: float | None, target: float) -> bool:
    return score is not None and score >= target


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_session_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Autoresearch Session Summary",
        "",
        f"- Session ID: `{summary['session_id']}`",
        f"- Stop reason: `{summary['stop_reason']}`",
        f"- Trials executed: `{summary['trial_count']}`",
        f"- Accepted trials: `{', '.join(map(str, summary['accepted_trials'])) or 'none'}`",
        f"- Baseline primary score: `{summary['baseline_primary_score']}`",
        f"- Best primary score: `{summary['best_primary_score']}`",
        f"- Target composite score: `{summary['target_composite_score']}`",
        f"- Ledger: `{summary['ledger_path']}`",
    ]
    return "\n".join(lines) + "\n"
