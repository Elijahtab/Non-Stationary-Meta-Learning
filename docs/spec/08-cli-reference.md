# 08 — CLI Reference

Every entry point in [`scripts/`](../../scripts/): what it does, its key flags, and **where it
writes**. Defaults shown are the argparse defaults in the script (some differ from library
defaults). On Windows the repo is typically driven via `myenv\Scripts\python.exe` with
`PYTHONPATH=src`.

## Training & evaluation

### [`train_brain.py`](../../scripts/train_brain.py) → writes `runs/`
Trains the outer Brain meta-agent (this is a **run**). Optionally runs imitation pretraining,
then PPO over `brain_episodes`, writing `config.txt`, `brain_model.pt`, `brain_trends/`, and
per-episode inner logs.

Key flags (defaults): `--env_id` (`MiniGrid-MultiGoal-5x5-v0`), `--num_regimes` (`2`),
`--inner_total_timesteps` (`1150000`), `--inner_steps_per_regime` (`296000`),
`--inner_mode` (`dyna`), `--brain_episodes` (`50`), `--brain_num_envs` (`16`),
`--brain_vectorization` (`async`), `--decision_interval` (`10`), `--brain_lr` (`1e-4`),
`--reward_mode` (`auc`), `--pretrain_episodes` (`5`), `--pretrain_mode` (`recovery`),
`--disable_neuromodulation`, the inner HP bounds (`--max_inner_lr 0.003`, `--min_inner_lr 1e-4`,
`--min/max_ent_coef`, `--min/max_intrinsic_coef`), `--episodic_memory_capacity` (`50000`),
`--resume_path` / `--new_run_dir`, `--save_every_episodes` (`1`), `--plot_every_episodes` (`5`).

### [`eval_brain.py`](../../scripts/eval_brain.py) → writes `evals/`
Loads a trained Brain checkpoint and runs it **inference-only** over one long inner run (this is
an **eval**; no Brain training). Requires `--brain_checkpoint <runs/.../brain_model.pt>`.

Key flags: `--brain_checkpoint` (**required**), `--env_id`, `--total_timesteps`,
`--steps_per_regime`, `--num_regimes`, `--num_envs`, `--num_steps`, `--mode`,
`--decision_interval`, `--disable_neuromodulation`, `--run_name`, the inner HP bounds.

### [`eval_inner_hyperparams.py`](../../scripts/eval_inner_hyperparams.py) → writes `evals/`
Extracts the inner hyperparameters baked into a checkpoint (and the JSON log) and evaluates them
on a fresh inner run — i.e. "what do the Brain's learned static HPs do on their own?". Requires
`--checkpoint_path`. Defaults: `--total_timesteps 450000`, `--num_regimes 3`,
`--steps_per_regime 22500`.

### [`train_ppo.py`](../../scripts/train_ppo.py) → writes `runs/` (+ `checkpoints/`)
Trains the **inner** Dyna-PPO agent directly (no Brain) — the baseline learner. Useful for
inspecting forgetting curves without meta-control.

Key flags (defaults): `--env_id` (`MiniGrid-MultiGoal-8x8-v0`), `--total_timesteps` (`300_000`),
`--num_envs` (`8`), `--num_steps` (`128`), `--num_regimes` (`2`), `--steps_per_regime`,
`--mode` (`dyna`|`passive`), `--intrinsic_coef` (`0.015`), `--imagined_horizon` (`10`),
`--wm_lr` (`1e-4`), `--anchoring_weight` (`0.0`), `--replay_ratio` (`0.0`),
`--replay_prioritization` (`0.0`), `--anneal_lr`/`--no-anneal_lr`.

> Note: `train_ppo.py` logs inner runs into `runs/` too. The "**run = Brain training**"
> convention in [06-runs-and-evals.md](./06-runs-and-evals.md) is about the *Brain* workflow;
> a `runs/` dir with `brain_model.pt` + `episode_N/` is a Brain run, while a flat inner-PPO log
> dir under `runs/` is a direct `train_ppo.py` run.

## Benchmark & autoresearch

### [`run_frozen_benchmark.py`](../../scripts/run_frozen_benchmark.py) → writes `benchmarks/` (+ `runs/`)
Runs a frozen benchmark spec end-to-end (trains per seed into `runs/`, scores into
`benchmarks/`). Flags: `--benchmark <spec>` | `--score-run-dir <dir>` (exactly one),
`--device` (`cuda`), `--dry-run`, `--list`. See
[07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md).

### [`run_autoresearch.py`](../../scripts/run_autoresearch.py) → writes `autoresearch/`
Runs the bounded autoresearch supervisor over the immutable benchmark stack. Driven by
`--manifest`, `--program`, and `--research-command` (with `--max-trials`, `--dry-run`). See
[AUTORESEARCH.md](../../AUTORESEARCH.md) for full command recipes.

### [`run_research_trial.py`](../../scripts/run_research_trial.py)
The recommended per-trial worker: renders the trial prompt + JSON context and invokes the
external research agent (Codex via [`invoke_codex_exec.ps1`](../../scripts/invoke_codex_exec.ps1)).
Configured through `AUTORESEARCH_AGENT_COMMAND` / `--runner-template`.

## Analysis & plotting

### [`analyze_runs.py`](../../scripts/analyze_runs.py)
Summarizes a run/eval log folder into charts (success/return vs surprise, regime boundaries).
Usage: `python scripts/analyze_runs.py <run_folder>`. **Immutable** under autoresearch.

### [`plot_high_scale.py`](../../scripts/plot_high_scale.py)
Produces higher-resolution, smoothed plots for an inner-run log folder. Used internally by the
benchmark to render per-inner-run PNGs. **Immutable** under autoresearch.

## Validation & inspection utilities

| Script | Purpose |
| --- | --- |
| [`verify_rewards.py`](../../scripts/verify_rewards.py) | Sanity-check that the regime wrapper emits the correct +5 / −1 / −0.01 rewards |
| [`check_env.py`](../../scripts/check_env.py) | Smoke-test a single wrapped env (spaces, reset/step) |
| [`check_vec_env.py`](../../scripts/check_vec_env.py) | Smoke-test the vectorized env stack and shared regime counter |
| [`render_env.py`](../../scripts/render_env.py) | Render the environment for visual inspection |
| [`test_info.py`](../../scripts/test_info.py) | Inspect the `info` dict keys emitted by the env |

## Tests

`pytest tests/` covers environments, signal extraction, the network, `MetaEnv`, Dyna logic,
benchmarking/scoring, and the train/eval CLIs. The autoresearch manifest runs a focused subset
(see [`research_manifest.toml`](../../research_manifest.toml) `[validation].tests_command`).
