# 07 — Benchmarking & Autoresearch

This is the automation layer sitting above the two RL loops. It has two parts:

1. a **frozen benchmark + scorer** that turns "did the Brain recover faster?" into one
   reproducible `composite_score`, and
2. an **autoresearch supervisor** that drives an external coding agent through bounded,
   append-only research trials, keeping only the changes the frozen benchmark says helped.

The governing contract is [AUTORESEARCH.md](../../AUTORESEARCH.md); the per-trial research
brief is [program_neuromod.md](../../program_neuromod.md).

---

## Part 1 — The frozen benchmark

Source: [`research/benchmarking.py`](../../src/lifelong_learning/research/benchmarking.py),
runner [`scripts/run_frozen_benchmark.py`](../../scripts/run_frozen_benchmark.py).

A [`FrozenBenchmarkSpec`](../../src/lifelong_learning/research/benchmarking.py#L14) freezes the
whole experiment so trial-to-trial comparison is fair: `env_id`, `num_regimes`,
`inner_total_timesteps`, `inner_steps_per_regime`, inner mode/HP/bounds, brain config,
`decision_interval`, reward mode, `disable_neuromodulation`, episodic-memory capacity, and the
**seeds**. Three specs are registered
([`FROZEN_BENCHMARKS`](../../src/lifelong_learning/research/benchmarking.py#L27)):

| Spec | Grid | inner steps / per-regime | seeds | role |
| --- | --- | --- | --- | --- |
| `fast_switch_pilot_v1` | 5×5 | `12_000` / `3_000` | `(0,)` | end-to-end smoke test (sustained=2) |
| `fast_switch_scout_v1` | 5×5 | `800_000` / `100_000` | `(0,)` | **primary** research benchmark |
| `fast_switch_holdout_v1` | 8×8 | `800_000` / `100_000` | `(11, 23, 37)` | harder **holdout** validation |

All use `reward_mode="recovery"`, `num_regimes=2`, `pretrain_episodes=0`,
`brain_episodes=4`, `decision_interval=10`.

### What the runner does

[`run_benchmark`](../../scripts/run_frozen_benchmark.py#L118): for each frozen seed it calls
`train_brain()` with the frozen args (output → **`runs/<spec>_<ts>_seed<seed>_<ts>/`**), locates
that new run dir, scores it, generates high-scale plots, and writes
`benchmarks/<spec>_<ts>/seed_<seed>_score.json` and `summary.json`. So the benchmark **trains**
(into `runs/`) and **scores** (into `benchmarks/`).

```bash
python scripts/run_frozen_benchmark.py --benchmark fast_switch_scout_v1 --device cuda
python scripts/run_frozen_benchmark.py --list                 # list specs
python scripts/run_frozen_benchmark.py --benchmark X --dry-run # print frozen config only
python scripts/run_frozen_benchmark.py --score-run-dir runs/<dir>   # score an existing run
```

### The scorer — [`score_brain_run`](../../src/lifelong_learning/research/benchmarking.py#L441)

Inputs read from a run dir: `config.txt` (for `inner_steps_per_regime`), `brain_trends/*_data.json`
(Brain-level episode success), and every inner trace `episode_*/ep*_env*/*_data.json`. From each
inner trace it uses `charts/success_rate`, `charts/regime_id`, and the two neuromodulation
traces `brain_neuromod/policy_kl_vs_unmasked` and `brain_neuromod/value_delta_abs_vs_unmasked`.

**Recovery metric rule** (per regime switch), in
[`summarize_threshold_recovery`](../../src/lifelong_learning/research/benchmarking.py#L317):

1. Detect switches from the `charts/regime_id` trace
   ([`detect_regime_switch_steps`](../../src/lifelong_learning/research/benchmarking.py#L227)).
2. **Ignore the first `post_switch_buffer_steps` (=500)** steps after each switch (dead zone).
3. Within `[switch+500, next_switch)`, find the first step where success ≥ threshold and stays
   there for `sustained_points_required` (=3) consecutive logged points
   ([`find_first_sustained_threshold_step`](../../src/lifelong_learning/research/benchmarking.py#L241)).
4. Recovery steps = `first_sustained_step − switch_step`.
5. Aggregate across switches by **median**.

This is computed for thresholds **0.80** and **0.95**, yielding `median_steps_to_80/95`,
`hit_rate_80/95` (fraction of switches that recovered at all). The
**post-switch window success** ([`summarize_post_switch_success`](../../src/lifelong_learning/research/benchmarking.py#L270))
averages success over a window of `post_switch_window_ratio·steps_per_regime` after the same
dead zone.

### Reported metrics & composite score

`score_brain_run` returns a
[`BrainRunScore`](../../src/lifelong_learning/research/benchmarking.py#L167) with:
`mean_episode_avg_success_rate`, `mean_inner_time_avg_success_rate`,
`mean_post_switch_window_success_rate`, `median_steps_to_80/95`, `hit_rate_80/95`,
`mean_post_switch_policy_kl`, `mean_post_switch_value_delta_abs`,
`mean_post_switch_neuromod_activity`, and the **`composite_score`**
([`_compute_composite_score`](../../src/lifelong_learning/research/benchmarking.py#L624)):

```text
composite_score = 0.50 · mean_post_switch_window_success_rate
                + 0.25 · hit_rate_80
                + 0.25 · normalized_median_steps_to_80
```

where `normalized_median_steps_to_80 = max(0, 1 − clipped_elapsed / usable_window)` rewards
faster recovery ([`_normalize_recovery_steps`](../../src/lifelong_learning/research/benchmarking.py#L650)).
The supervisor compares trials on this single scalar (averaged across seeds by
[`aggregate_benchmark_summary`](../../src/lifelong_learning/research/autoresearch.py#L421)).

---

## Part 2 — The autoresearch supervisor

Source: [`research/autoresearch.py`](../../src/lifelong_learning/research/autoresearch.py),
entrypoint [`scripts/run_autoresearch.py`](../../scripts/run_autoresearch.py).

[`AutoresearchSupervisor`](../../src/lifelong_learning/research/autoresearch.py#L553) runs a
fixed number of **trials**. Each trial lets an external coding agent (Codex) make one bounded
code change, then gates that change behind tests + the frozen benchmark, keeping it only if it
strictly improves the primary score without regressing the holdout. See the flow diagram in
[01-research-flow.md](./01-research-flow.md#3c-the-autoresearch-supervisor-karpathy-style-bounded-loop).

### Trial lifecycle — [`_run_trial`](../../src/lifelong_learning/research/autoresearch.py#L767)

1. **Snapshot** the repo (managed text files only; `runs/`, `benchmarks/`, `myenv/`, etc. are
   ignored) — [`take_repo_snapshot`](../../src/lifelong_learning/research/autoresearch.py#L283).
2. **Run the research command**: render the per-trial command and invoke the external agent
   ([`_render_research_command`](../../src/lifelong_learning/research/autoresearch.py#L1003)).
3. **Diff + audit** against the surface
   ([`audit_repo_diff`](../../src/lifelong_learning/research/autoresearch.py#L355)): any change
   to the **immutable surface**, or to anything outside the **editable surface** (other than a
   permitted new `.py` under an editable root), → reject + rollback.
4. **Enforce growth caps** relative to the session baseline:
   `max_new_python_files` and `max_net_new_lines`
   ([autoresearch.py](../../src/lifelong_learning/research/autoresearch.py#L863-L883)).
5. **Run tests** (`validation.tests_command`); failure → reject + rollback.
6. **Run the primary (scout) benchmark**; compute `composite_score`.
7. If it **improves** over the best-so-far by `primary_improvement_epsilon`, **run the
   holdout** benchmark(s); reject if any holdout regresses beyond
   `holdout_regression_tolerance`.
8. **Accept** (keep the change, update best score) or **roll back** to the pre-trial snapshot
   ([`restore_repo_snapshot`](../../src/lifelong_learning/research/autoresearch.py#L394)).
9. **Append one ledger record** to `autoresearch/trial_results.jsonl`
   ([`_append_ledger_record`](../../src/lifelong_learning/research/autoresearch.py#L1015)).

A **baseline** benchmark runs once up front (fingerprint-**cached** across sessions so an
unchanged repo reuses prior scores —
[`_build_baseline_fingerprint_payload`](../../src/lifelong_learning/research/autoresearch.py#L1036)).
The session stops on `max_trials`, `target_composite_score`, or `max_stale_trials` consecutive
non-improving trials, then writes `session_summary.{json,md}`.

### The manifest — [`research_manifest.toml`](../../research_manifest.toml)

Loaded by [`load_research_manifest`](../../src/lifelong_learning/research/autoresearch.py#L217):

| Table | Keys (current values) |
| --- | --- |
| `[trial_limits]` | `max_trials=20`, `max_new_python_files=2`, `max_net_new_lines=600` |
| `[validation]` | `tests_command=…`, `tests_timeout_seconds=1800`, `research_timeout_seconds=1800` |
| `[benchmark]` | `primary="fast_switch_scout_v1"`, `holdout=["fast_switch_holdout_v1"]`, `device="cuda"`, `reuse_cached_baseline=true`, `primary_improvement_epsilon=1e-6`, `holdout_regression_tolerance=0.0` |
| `[stopping]` | `max_stale_trials=5`, `target_composite_score=0.75` |
| `[outputs]` | `ledger_path="autoresearch/trial_results.jsonl"`, `scratch_dir="autoresearch"` |
| `[editable_surface]` | `neuromod.py`, `network.py` |
| `[optional_editable_surface]` | `signals.py`, `meta_agent.py` |
| `[immutable_surface]` | `run_frozen_benchmark.py`, `benchmarking.py`, `analyze_runs.py`, `plot_high_scale.py`, `envs/` |

Variants: [`research_manifest_pilot.toml`](../../research_manifest_pilot.toml) (tiny smoke
test) and [`research_manifest_085.toml`](../../research_manifest_085.toml) (a higher target).

### The per-trial worker & agent launcher

[`scripts/run_research_trial.py`](../../scripts/run_research_trial.py) is the recommended
per-trial worker: it renders the trial prompt
([`research/trial_prompt.py`](../../src/lifelong_learning/research/trial_prompt.py)) from
[program_neuromod.md](../../program_neuromod.md), writes a JSON context payload, and invokes a
configurable external runner (via `AUTORESEARCH_AGENT_COMMAND` / `--runner-template`). For Codex
specifically, [`scripts/invoke_codex_exec.ps1`](../../scripts/invoke_codex_exec.ps1) pipes the
prompt into `codex exec` and captures stdout/stderr + the final message. This indirection keeps
the supervisor-facing `--research-command` stable even if the local launcher changes. Full
command recipes are in [AUTORESEARCH.md](../../AUTORESEARCH.md).

### The safety boundary in one sentence

The agent is free to redesign **neuromodulation** (`neuromod.py`/`network.py`), but it cannot
touch the benchmark, scorer, plotting, or environment — and every change must pay for itself on
a frozen benchmark or it is reverted and logged.
