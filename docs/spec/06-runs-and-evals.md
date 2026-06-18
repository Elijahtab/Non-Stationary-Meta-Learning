# 06 — Runs vs. Evals (the Artifact Contract)

This is the doc to read when you are staring at a folder and asking "what is this?". It
defines the four top-level output directories and, above all, the distinction the rest of the
spec leans on:

> **A `run` is an outer-Brain *training* run. An `eval` is a *trained* outer-Brain
> *evaluation*.** A run **produces** `brain_model.pt`; an eval **consumes** one.

| Dir | What it is | Brain is… | Written by |
| --- | --- | --- | --- |
| [`runs/`](../../runs/) | **Brain training runs** | **being trained** | [`scripts/train_brain.py`](../../scripts/train_brain.py), and (per seed) [`run_frozen_benchmark.py`](../../scripts/run_frozen_benchmark.py) |
| [`evals/`](../../evals/) | **trained-Brain evaluations** | **frozen, inference-only** | [`scripts/eval_brain.py`](../../scripts/eval_brain.py), [`scripts/eval_inner_hyperparams.py`](../../scripts/eval_inner_hyperparams.py) |
| [`benchmarks/`](../../benchmarks/) | frozen-benchmark **scores** | trained, then scored | [`run_frozen_benchmark.py`](../../scripts/run_frozen_benchmark.py) |
| [`autoresearch/`](../../autoresearch/) | autoresearch **sessions** + ledger | trained+scored per trial | [`run_autoresearch.py`](../../scripts/run_autoresearch.py) |

`checkpoints/` and `graphs/` hold ad-hoc inner-PPO checkpoints and plots from direct
`train_ppo.py` usage.

---

## `runs/` — a Brain **training** run

Created by `train_brain.py` (directly, or via the benchmark which calls `train_brain()` per
seed). Naming: `runs/<run_name>_<timestamp>/`.

```text
runs/<run_name>_<timestamp>/
├── config.txt                          # full resolved config (parsed by the scorer)
├── brain_model.pt                      # ← THE TRAINED ARTIFACT (final Brain weights)
├── brain_trends/
│   ├── <run>_data.json                 # Brain-level trends (episode reward, success, …)
│   └── <run>_charts.png
├── episode_1/                          # one dir per Brain RL episode (and pretrain_<N>/)
│   ├── brain_ep1.pt                    # per-episode Brain checkpoint (save_every_episodes)
│   ├── ep1_env0_<ts>/                  # inner Dyna-PPO logs: Brain episode 1, vec-env worker 0
│   │   ├── ep1_env0_data.json          #   inner traces: charts/success_rate, charts/regime_id,
│   │   │                               #   brain_neuromod/*, world_model/*, …
│   │   ├── ep1_env0_charts.png
│   │   ├── ep1_env0_neuromodulation_dashboard.png
│   │   └── *_highres.png
│   ├── ep1_env1_<ts>/ … ep1_env7_<ts>/ # one per inner vec-env worker
│   └── inner_checkpoints/
├── episode_2/ …
└── …
```

The **defining marker of a run is `brain_model.pt`** — the Brain is being optimized here. The
nested `episode_N/ep_envI/*_data.json` files are the inner training traces the scorer reads.

> Tip: a benchmark-launched run (`runs/fast_switch_scout_v1_<ts>_seed0_<ts>/`) is still a normal
> training run; the benchmark just scores it afterward and stores the *report* in `benchmarks/`.

## `evals/` — a **trained-Brain** evaluation

Created by `eval_brain.py` from a `--brain_checkpoint` (a `brain_model.pt` produced by a run).
The Brain is **frozen**; it drives a single long inner run inference-only. There is **no
`brain_model.pt` being written here** — that is how you tell an eval from a run at a glance.

```text
evals/<eval_name>/
├── config.txt
└── eval_<...>_<timestamp>/
    ├── eval_<...>_data.json            # inner eval traces (success_rate, regime_id, …)
    ├── eval_<...>_charts.png
    └── eval_<...>_success_rate.png
```

Naming in practice often encodes the source checkpoint and episode, e.g.
`evals/eval_brain_3_regimes_ep54_success/`.

## `benchmarks/` — frozen-benchmark reports

Created by `run_frozen_benchmark.py`. Naming: `benchmarks/<spec>_<timestamp>/`.

```text
benchmarks/<spec>_<timestamp>/
├── summary.json        # spec + frozen args + seeds + per-seed scores (the aggregate report)
└── seed_<seed>_score.json   # one per frozen seed
```

`summary.json` keys: `benchmark`, `description`, `timestamp`, `device`, `seeds`,
`fixed_train_args`, `sustained_points_required`, `post_switch_window_ratio`,
`post_switch_buffer_steps`, and `seed_reports[]`. Each seed report includes
`run_dir` (pointing back into `runs/`), `composite_score`, the recovery metrics, and
`per_inner_run`. The scoring math is in
[07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md).

## `autoresearch/` — autoresearch sessions

Created by `run_autoresearch.py`. Naming: `autoresearch/<session_id>/` plus the append-only
ledger `autoresearch/trial_results.jsonl`.

```text
autoresearch/
├── trial_results.jsonl                 # append-only ledger (one JSON record per baseline/trial)
└── <session_id>/
    ├── baseline/                        # baseline benchmark stdout/stderr logs
    ├── baseline_summary.json
    ├── trial_001/
    │   ├── research_prompt.md           # rendered prompt handed to the agent
    │   ├── trial_context.json           # machine-readable trial context payload
    │   ├── agent_notes.md               # the agent's hypothesis/notes
    │   ├── codex_final_message.md       # the agent's final message
    │   ├── research.stdout/stderr.log
    │   ├── tests.stdout/stderr.log
    │   └── primary_benchmark.stdout/stderr.log
    ├── trial_002/ …
    ├── session_summary.json
    └── session_summary.md
```

## Quick identification cheatsheet

| If the folder has… | it is a… |
| --- | --- |
| `brain_model.pt` + `episode_N/` | **run** (`runs/`) |
| `config.txt` + `eval_*/` and **no** `brain_model.pt` | **eval** (`evals/`) |
| `summary.json` + `seed_*_score.json` | **benchmark report** (`benchmarks/`) |
| `trial_NNN/` + `session_summary.*` | **autoresearch session** (`autoresearch/`) |
