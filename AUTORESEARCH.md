# Autoresearch Architecture

This document defines the immutable benchmark and trial contract for future
autoresearch runs focused on Meta-RL neuromodulation.

## Goal

The research agent should be free to explore neuromodulation architecture, but
it must not be able to mutate the benchmark, the scorer, or the plotting
pipeline that decides whether a change helped.

## Frozen Benchmark Surface

The benchmark runner is `scripts/run_frozen_benchmark.py`.
The benchmark/scoring implementation is `src/lifelong_learning/research/benchmarking.py`.

These files are intended to stay outside the future editable allowlist.

The first two frozen benchmark specs are:

- `fast_switch_scout_v1`
- `fast_switch_holdout_v1`

They freeze:

- `env_id`
- `num_regimes`
- `start_regime`
- `randomize_start_regime`
- `inner_total_timesteps`
- `inner_steps_per_regime`
- `inner_num_envs`
- `inner_num_steps`
- `inner_mode`
- `inner_intrinsic_coef`
- `inner_imagined_horizon`
- `inner_wm_lr`
- `brain_num_envs`
- `brain_vectorization`
- `pretrain_episodes`
- `pretrain_mode`
- `brain_episodes`
- `brain_lr`
- `brain_ent_coef`
- `decision_interval`
- `reward_alpha`
- `reward_beta`
- `reward_mode`
- `disable_neuromodulation`
- `episodic_memory_capacity`
- inner hyperparameter bounds
- fixed seeds

## Trial Model

The future supervisor should run a fixed number of trials, not a fixed number of
internal prompts. A trial means:

1. Apply a bounded code change.
2. Run tests.
3. Run the frozen benchmark.
4. Score the result.
5. Write one append-only trial record.

The benchmark runner and scorer should remain immutable across all trials.

The concrete supervisor entrypoint is `scripts/run_autoresearch.py`.
It should:

1. Snapshot the repo state before each trial.
2. Run one bounded research command.
3. Audit the resulting diff against the editable surface.
4. Enforce file-count and net-new-line caps relative to the session baseline.
5. Run tests.
6. Run the frozen scout benchmark, then holdout only on improvements.
7. Append one ledger record and either keep or roll back the trial.

The recommended per-trial worker entrypoint is `scripts/run_research_trial.py`.
This script renders a trial-specific prompt, writes a JSON context payload, and
then invokes a configurable external research-agent command.

The first recommended supervisor command is:

```powershell
$env:PYTHONPATH='src'
.\myenv\Scripts\python.exe scripts\run_autoresearch.py `
  --manifest research_manifest.toml `
  --program program_neuromod.md `
  --research-command ".\myenv\Scripts\python.exe scripts\run_research_trial.py --program {program} --manifest {manifest} --trial {trial} --trial-dir {trial_dir} --repo-root {repo_root} --baseline-file {baseline_file}"
```

`scripts/run_research_trial.py` expects the actual agent launcher to be provided
through `AUTORESEARCH_AGENT_COMMAND` or `--runner-template`. That runner command
receives:

- `{prompt_file}`
- `{context_file}`
- `{notes_file}`
- `{final_message_file}`
- `{agent_stdout_file}`
- `{agent_stderr_file}`
- `{trial}`
- `{trial_dir}`
- `{manifest}`
- `{program}`
- `{repo_root}`
- `{baseline_file}`

This keeps the supervisor-facing `--research-command` stable even if the local
Codex launcher syntax changes later.

For Codex CLI specifically, the recommended automation wrapper is
`scripts/invoke_codex_exec.ps1`. It reads the rendered prompt file, pipes it into
`codex exec`, sets the workspace root, uses explicit non-interactive permissions,
and writes separate stdout/stderr plus the final assistant message to disk.

Recommended setup:

```powershell
$env:AUTORESEARCH_AGENT_COMMAND = 'powershell -ExecutionPolicy Bypass -File scripts\invoke_codex_exec.ps1 -PromptFile "{prompt_file}" -RepoRoot "{repo_root}" -FinalMessageFile "{final_message_file}" -StdoutFile "{agent_stdout_file}" -StderrFile "{agent_stderr_file}"'
```

Then run:

```powershell
$env:PYTHONPATH='src'
.\myenv\Scripts\python.exe scripts\run_autoresearch.py `
  --manifest research_manifest.toml `
  --program program_neuromod.md `
  --research-command ".\myenv\Scripts\python.exe scripts\run_research_trial.py --program {program} --manifest {manifest} --trial {trial} --trial-dir {trial_dir} --repo-root {repo_root} --baseline-file {baseline_file}"
```

Why this wrapper uses these defaults:

- `codex exec` is the documented non-interactive mode.
- `PROMPT = -` lets us pipe the rendered prompt from stdin.
- `-C` sets the repo workspace explicitly.
- `-a never -s workspace-write` is a better unattended automation default than `--full-auto`, because the docs recommend `never` for non-interactive runs and `--full-auto` maps to `on-request` approvals.
- `--json` plus `-o` gives machine-readable event logs and a separate final message file.

## Editable Surface

The default editable surface for neuromodulation research should be narrow:

- `src/lifelong_learning/agents/brain/neuromod.py`
- `src/lifelong_learning/agents/ppo/network.py`

Optional second-stage expansion:

- `src/lifelong_learning/agents/brain/signals.py`
- `src/lifelong_learning/agents/brain/meta_agent.py`

Keep these immutable until explicitly opened:

- `scripts/run_frozen_benchmark.py`
- `src/lifelong_learning/research/benchmarking.py`
- `scripts/analyze_runs.py`
- `scripts/plot_high_scale.py`
- environment code
- score parsing and ledger files

New Python files are allowed only under the parent directories of the editable
surface, and only up to the manifest budget. Once a new Python file is accepted,
future trials may continue editing it.

## Metrics

The scorer reports:

- `mean_episode_avg_success_rate`
- `median_steps_to_80`
- `median_steps_to_95`
- `hit_rate_80`
- `hit_rate_95`
- `mean_post_switch_policy_kl`
- `mean_post_switch_value_delta_abs`
- `mean_post_switch_neuromod_activity`
- `composite_score`

### Recovery Metric Rule

`median_steps_to_80` and `median_steps_to_95` are computed from the inner-run
`charts/success_rate` trace and the `charts/regime_id` trace.

For each regime switch:

1. Look only between that switch and the next switch.
2. Find the first point where the threshold is sustained for `K` consecutive
   logged success-rate points.
3. Measure steps from the switch to the first point in that sustained block.
4. Aggregate across all switch windows using the median.

This avoids a single noisy point counting as recovery.

## Neuromodulation Freedom

The benchmark is frozen, but neuromodulation is intentionally not frozen beyond
the runner/scorer boundary. Future research is expected to explore:

- larger context dimensions such as `16`, `32`, or `64`
- different decoder widths and depths
- feature-wise vs channel-wise masking
- actor-only vs critic-only vs shared modulation
- suppressive vs gain-based modulation

The cleanup in Part 1 centralized the current implementation so these changes
can happen without giving the agent write access to the benchmark pipeline.
