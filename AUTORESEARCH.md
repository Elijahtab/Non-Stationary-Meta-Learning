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

- `fast_switch_pilot_v1`
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
  --manifest config/research_manifest.toml `
  --program config/program_neuromod.md `
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
$env:AUTORESEARCH_AGENT_COMMAND = 'powershell -ExecutionPolicy Bypass -File scripts\invoke_codex_exec.ps1 -PromptFile "{prompt_file}" -RepoRoot "{repo_root}" -FinalMessageFile "{final_message_file}" -StdoutFile "{agent_stdout_file}" -StderrFile "{agent_stderr_file}" -Sandbox danger-full-access -Model gpt-5.4-mini'
```

Then run:

```powershell
$env:PYTHONPATH='src'
.\myenv\Scripts\python.exe scripts\run_autoresearch.py `
  --manifest config/research_manifest.toml `
  --program config/program_neuromod.md `
  --research-command ".\myenv\Scripts\python.exe scripts\run_research_trial.py --program {program} --manifest {manifest} --trial {trial} --trial-dir {trial_dir} --repo-root {repo_root} --baseline-file {baseline_file}"
```

Why this wrapper uses these defaults:

- `codex exec` is the documented non-interactive mode.
- `PROMPT = -` lets us pipe the rendered prompt from stdin.
- `-C` sets the repo workspace explicitly.
- `-a never` is the recommended unattended approval mode for non-interactive runs.
- On this Windows setup, `workspace-write` sandboxing fails once the agent tries to use tools, so the pilot wrapper uses `-s danger-full-access` and relies on the outer autoresearch supervisor for the real safety boundary.
- `-m gpt-5.4-mini` keeps pilot trials short enough to validate the loop before spending time on full runs.
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
- `mean_inner_time_avg_success_rate`
- `mean_post_switch_window_success_rate`
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

1. Ignore the first `500` steps after the switch.
2. Look only between that buffered start and the next switch.
3. Find the first point where the threshold is sustained for `K` consecutive
   logged success-rate points.
4. Measure steps from the switch to the first point in that sustained block.
5. Aggregate across all switch windows using the median.

This avoids a single noisy point counting as recovery.

### Composite Score

`composite_score` is now switch-centered instead of whole-run centered:

- `50%` `mean_post_switch_window_success_rate`
- `25%` `hit_rate_80`
- `25%` normalized `median_steps_to_80`

`mean_post_switch_window_success_rate` is computed over a fixed post-switch
window after the same `500`-step dead zone.

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

## Pilot Mode

Use `config/research_manifest_pilot.toml` for the first end-to-end smoke test.
It switches to the tiny `fast_switch_pilot_v1` benchmark, removes holdout runs,
cuts the test suite down, limits the session to one small trial, and uses a
shorter research timeout so stuck agent runs fail fast.

Recommended pilot command:

```powershell
$env:PYTHONPATH='src'
.\myenv\Scripts\python.exe scripts\run_autoresearch.py `
  --manifest config/research_manifest_pilot.toml `
  --program config/program_neuromod.md `
  --research-command ".\myenv\Scripts\python.exe scripts\run_research_trial.py --program {program} --manifest {manifest} --trial {trial} --trial-dir {trial_dir} --repo-root {repo_root} --baseline-file {baseline_file}" `
  --max-trials 1
```

## Cloud Execution & Long-Running Loops (Operational Notes, 2026-07-01)

How trials/sweeps actually run on a rented cloud GPU box — reference for when the autoresearch
supervisor runs unattended on cloud infra. Learned from the trainable-vs-frozen stabilizer sweeps;
full detail in [`docs/multi_agent/0001-reduce-gpu-artifacts.md`](docs/multi_agent/0001-reduce-gpu-artifacts.md)
and [`docs/handoffs/2026-07-01-trainable-vs-frozen-cloud-sweep.md`](docs/handoffs/2026-07-01-trainable-vs-frozen-cloud-sweep.md).

### Free compute — sweeps parallelize far cheaper than the old budget assumed (headline)

The earlier plan budgeted **16 cores per cell** (`MAX_PARALLEL = nproc/16`). Measured reality on a
128-vCPU / 4× RTX 3090 box: a `calib8x8` cell uses **~1.5 CPU cores** and ~0.5–1 GB VRAM.

- **8 cells:** load ~12/128, RAM 23/251 GB, GPUs ~1 GB/24 GB at 23–61% util — box ~90% idle.
- **Packed to 12 cells** (added a 4-seed condition into the spare capacity): load 15/128; existing
  cells slowed only **~7%** (1347→1249 SPS); one GPU reached 93% util at 3 cells/GPU.
- **The binding constraint is GPU utilization, not CPU or RAM.** Budget by GPU: ~3 cells/GPU stays
  healthy (~1200+ SPS); ~4–6/GPU is likely before ~1300 SPS degrades materially (inferred from the
  util headroom — validate before relying on the top end).
- **Because the box bills per hour regardless, idle GPUs are wasted money.** A parallel trial/sweep
  that fits inside the current window is effectively free. Long loops should pack trials up to the
  GPU-util ceiling rather than the `nproc/16` rule.
- **Still cap threads:** `OMP_NUM_THREADS=16` (+ `MKL`/`OPENBLAS`/`NUMEXPR`) is mandatory to stop
  torch's thread explosion (uncapped → ~131 threads/cell → load 321, GPUs idle, no progress). The
  16 is a *thread cap*, not a per-cell core budget — do not use it to size `MAX_PARALLEL`.

### Right-sizing the box (measured — provision leaner next time)

Measured with 12 cells running: **~14–16 CPU cores used of 128** (~1.2–1.5 cores/cell), **31 GB
RAM of 251**, **~1.6 GB VRAM of 24 GB per GPU**. The 128-vCPU / 4× RTX 3090 box used for the first
sweeps is **5–8× over-provisioned on CPU** and hugely over-provisioned on VRAM.

- **Sizing rule:** vCPU ≈ **2 × max concurrent cells** (each cell ≈ 1.5 cores), RAM ≈ 2 GB/cell.
  A full 4-GPU pack (12–16 cells) needs only **~32 vCPU + ~64 GB**.
- **Recommended next box:** **4× GPU, ~32 vCPU, ~64 GB RAM**, and scale the thread cap down with
  the core count (`OMP_NUM_THREADS=8`). Zero performance loss for this workload.
- **Vast pricing** is a bundled per-host offer dominated by the **GPUs**, not linear in vCPU — a
  leaner box isn't *guaranteed* cheaper, but leaner-vCPU 4×GPU offers usually run ~20–40% cheaper
  with no downside here. (This supersedes the "prioritize 48–64 vCPU, `max_parallel = vCPU/16`"
  guidance in research-log 0002, which was based on a wrong 16-cores/cell estimate.)
- **Bigger cost lever:** the GPU *tier* is over-provisioned too (~1.6 GB/24 GB, modest util). Keep
  the GPU **count** at 4 (parallelism = one condition per GPU) but a cheaper card class
  (RTX 3060 12 GB / A4000 / 2080-class) runs each cell fine and cuts price more than vCPU, since
  GPU class drives most of the Vast cost.

### Launch mechanics (proven shape)

- `run_seed_sweep.py` passes `--device cuda` (= `cuda:0`), so **all cells of one invocation land on
  the first visible GPU**. Spread by pinning one invocation per GPU with `CUDA_VISIBLE_DEVICES`:
  ```bash
  CUDA_VISIBLE_DEVICES=N MAX_PARALLEL=k RESUME=1 \
    bash scripts/cloud/run_sweep.sh <preset> "<condition>" "<seeds>" <outdir>
  ```
- **Healthy signal:** ~1300–1400 SPS/cell, load ≪ nproc, ~1 GB VRAM/cell.
- **Detachment:** `run_sweep.sh` backgrounds + disowns and survives SSH exit (verified from a fresh
  session). It calls `push_results.sh` on completion (`AUTO_PUSH=1`).
- **SSH gotchas:** filter the 3-line vast banner (`grep -vE "Welcome to vast|Have fun|AI agents"`);
  non-interactive shells do **not** auto-activate the venv (`source /venv/main/bin/activate`); `scp`
  uses `-P` (not `-p`); kill with the bracket trick — `pkill -9 -f "[t]rain_brain.py"` — so the
  pattern doesn't match your own SSH command and kill the shell.

### Box ephemerality & results durability (critical for unattended loops)

- **`/workspace` is NOT persistent** (overlay fs); recycle/destroy wipes everything. Check with
  `vast-capabilities | jq '.instance.workspace_is_volume'`. The SSH endpoint (ip:port) also changes
  per box — read it from the dashboard Connect button; don't hardcode (it went stale mid-session).
- **Auto send-back:** on completion `run_sweep.sh` → `scripts/cloud/push_results.sh` pushes a *light*
  bundle (summary/runs/logs + per-run `brain_trends`/`brain_model.pt`) to the **`results` branch**
  under `results/<sweep>/`. Survives box destroy. Pull for analysis with `scripts/cloud/pull_results.sh`
  (excludes the heavy `episode_*/`).
- **Token policy:** the GitHub PAT lives ONLY in gitignored `.secrets/gh_token`;
  `scripts/cloud/upload_secrets.sh` installs it on a box at start (or set `GH_TOKEN` as a Vast launch
  env var → `bootstrap.sh` arms git creds automatically). Never commit the token.
- **Disk:** a raw sweep leaves ~19 GB, almost all derived/step-level. Reductions are shipped
  (`docs/multi_agent/0001`): PNG rendering gated behind `LL_RENDER_CHARTS=1` (regenerate locally with
  `scripts/render_charts.py`), compact JSON, and the inner-checkpoint leak fixed (with
  `save_checkpoints=False`, the default, none are written). Run dir ~800 MB → a few MB.

### Scorer constraint (a trial must not break scoring)

`composite_score` reads per-episode `charts/success_rate` + `charts/regime_id` from each run's
`*_data.json`, and **scoring runs on the box**. Do not prune those series (only whitespace-compaction
is safe). `benchmarking.py` stays immutable per the manifest.

## Brainstorm: Long-Running Loop Architecture (2026-07-01, pre-design)

Ideas for the unattended loop that will eventually drive cloud experiments end-to-end. Not a
contract yet — a brainstorm grounded in what this session did manually. The manual flow it should
automate: *pick conditions → launch across GPUs → monitor → pull trajectories → interpret against
a registered prediction → write the research note → queue the follow-up*.

### Loop skeleton (observe → decide → launch → record)

1. **Observe:** poll box state (cells running, per-cell episode progress, GPU util) and the
   `results` branch (completed sweeps auto-push there).
2. **Decide:** compare finished results against the *registered prediction* written before launch
   (notes 0001/0002 pattern). Outcomes route to: extend (more seeds), escalate (longer horizon /
   next lever), or close (write the null, pivot to the next mechanism family).
3. **Launch:** new conditions are one-line entries in `run_seed_sweep.py::CONDITIONS`; spread via
   `CUDA_VISIBLE_DEVICES`, `RESUME=1` for spot-safety.
4. **Record:** append to the research note (never silently edit a prediction), push docs + data
   to git. The ledger/trial-record contract from the Trial Model section applies.

### REQUIRED: compute check after any run completion

**Every time a run/sweep completes (or the loop wakes), re-measure capacity and backfill.** The
box bills per hour whether GPUs work or idle — this session repeatedly found 60–90% of paid
capacity sitting idle after a wave finished, and each backfill (combo condition, frozen-long
control, extra seeds) was effectively free.

Concrete rule:
- On any completion event: read cells-per-GPU (`nvidia-smi` util/mem + `ps` cell count).
- If any GPU is below the packing density (**~3 cells/GPU** healthy; validate 4+ before relying
  on it), pop the next item off the experiment queue and launch it into the gap immediately.
- Queue priorities when backfilling: (1) missing controls for in-flight results (e.g. the
  frozen-long control), (2) extra seeds on the currently-leading hypothesis (tighter CIs beat new
  conditions), (3) next-lever conditions, (4) dose-response variants.
- If the queue is empty and all cells are done: pull/push everything, then **destroy the box**
  (idle box = pure waste; recreation is cheap and `bootstrap.sh` + launch env vars re-arm it).

### Experiment queue (what "next" means)

A small ordered file (e.g. `autoresearch/queue.toml`) of {condition, seeds, horizon, prediction,
priority} the loop can pop from — so backfilling never has to invent science, only schedule it.
Humans and the decide-step both append to it.

### Safety rails carried over from the Trial Model

- Immutable surface stays immutable (scorer/benchmark; see manifest) — the loop only adds
  CONDITIONS entries and launches.
- Budget caps: max cells in flight, max box-hours per day, hard stop if `results` pushes fail
  (never accumulate unpushed results on an ephemeral disk).
- Registered predictions are append-only; a surprised prediction is a *finding*, not an error.

### Open questions

- Where does the loop live — local machine (survives box death, needs SSH out) vs on-box
  (cheaper, dies with the box) vs both (on-box worker + local/cron supervisor)?
- Trigger mechanism: poll interval vs watching the `results` branch vs `run_sweep.sh` calling a
  webhook on completion.
- How much of the decide-step to trust to the agent unattended vs gate on human review (start:
  backfill/extend autonomous, escalate/close gated).

### Current experiment context (pointer)

Trainable neuromod decoder ≈ frozen on composite but **unstable** (late collapse; two co-adapting
learners = risk R1). Stabilizer sweep in flight — `slowbrain` (`brain_lr`), `declr` (separate
`--neuromod_decoder_lr`), `long` (60 ep), `slow_long`, plus the `declr_slowbrain` combo. New
conditions live in `run_seed_sweep.py::CONDITIONS`. See
[`docs/research-notes/0001-trainable-vs-frozen-decoder.md`](docs/research-notes/0001-trainable-vs-frozen-decoder.md).
