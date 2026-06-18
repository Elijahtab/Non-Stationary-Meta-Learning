# Lifelong-Learning — Specification

This directory is the engineering specification for the **Meta-Learning ("Brain") subsystem**
of the catastrophic-forgetting project. It describes what the code in this repository
actually does, module by module, grounded in the source — not in aspiration.

> These docs are a **non-canonical aid**. The source code is the single source of truth.
> Where a doc and the code disagree, the code wins; please fix the doc.
> Every claim here links to the file (and usually the line) it came from.

## What this repository is

This repo implements **one of the five strategies** studied in the paper
[Combating Catastrophic Forgetting](../../Combating%20Catastrophic%20Forgetting.pdf):
the **Meta-Learning / "double-RL" approach** (paper §3.7 and §4.3). The other four
strategies (standard PPO, Dyna-PPO baseline, Mixture-of-World-Models, and the
Context-Aware Transformer World Model) are described in the paper but are **not** the
focus of this codebase.

The core idea is a two-level reinforcement-learning system:

- an **inner** continual learner — a **Dyna-PPO** agent that learns a MiniGrid task whose
  reward regime flips periodically (catastrophic forgetting is the failure mode), and
- an **outer** meta-controller — the **"Brain"**, a PPO agent that treats *the inner
  agent's entire training run* as its environment and adjusts the inner learner's
  training dynamics online (7 scalar hyperparameter levers + an 8-dimensional
  neuromodulation context code).

On top of those two RL loops sits a **research-automation layer**: a frozen benchmark +
scorer, and an autoresearch supervisor that drives an external coding agent through
bounded, append-only research trials against that frozen benchmark.

## Vocabulary (read this first)

Two words are used precisely throughout this repo and these docs:

| Term     | Meaning                                                       | Written to    |
| -------- | ------------------------------------------------------------ | ------------- |
| **run**  | an **outer-Brain training run** — the Brain *is being trained* | [`runs/`](../../runs/) |
| **eval** | a **trained-outer-Brain evaluation** — a saved Brain checkpoint is *loaded and run inference-only* | [`evals/`](../../evals/) |

A **run** produces a `brain_model.pt`; an **eval** consumes one. See
[06-runs-and-evals.md](./06-runs-and-evals.md) for the full on-disk contract, including
how `benchmarks/` and `autoresearch/` relate to these two.

## How to read this spec

Start at [01-research-flow.md](./01-research-flow.md) — it has the diagrams that tie
everything together — then drill into whichever subsystem you need.

| Doc | Covers |
| --- | --- |
| [01-research-flow.md](./01-research-flow.md) | **Diagrams** + end-to-end control/data flow: the nested RL loops, the run→eval lifecycle, and the autoresearch loop |
| [02-environments.md](./02-environments.md) | Dual/Multi-Goal MiniGrid task, the wrapper stack, and regime switching |
| [03-inner-dyna-ppo.md](./03-inner-dyna-ppo.md) | The inner continual learner: PPO, world model, curiosity, dreaming, replay, anchoring, and the per-update cycle |
| [04-outer-brain-and-metaenv.md](./04-outer-brain-and-metaenv.md) | The Brain meta-agent, `MetaEnv`, the 19-dim observation, the 15-dim action, and the meta-reward modes |
| [05-neuromodulation.md](./05-neuromodulation.md) | The context decoder, the feature-wise mask, and the neuromodulation diagnostics |
| [06-runs-and-evals.md](./06-runs-and-evals.md) | The artifact contract: `runs/` vs `evals/` vs `benchmarks/` vs `autoresearch/` |
| [07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md) | The frozen benchmark, the scorer/composite score, and the autoresearch supervisor |
| [08-cli-reference.md](./08-cli-reference.md) | Every script in [`scripts/`](../../scripts/): purpose, key flags, and what it writes |

## Source map (one level deep)

```text
src/lifelong_learning/
├── agents/
│   ├── ppo/            # inner Dyna-PPO learner  → docs 03
│   │   ├── network.py        CNNActorCritic (+ neuromodulation hook)
│   │   ├── world_model.py    SimpleWorldModel (Dyna simulator)
│   │   ├── ppo.py            PPOConfig + ppo_update (clipped PPO + anchoring)
│   │   ├── buffers.py        RolloutBuffer
│   │   ├── episodic_memory.py EpisodicMemory (rehearsal store)
│   │   └── train.py          inner loop: init/step/close + run_inner_update
│   └── brain/          # outer meta-controller    → docs 04, 05
│       ├── meta_env.py       MetaEnv (inner run as a Gym env)
│       ├── meta_agent.py     Brain PPO actor-critic
│       ├── signals.py        19-dim observation extraction + normalization
│       └── neuromod.py       FeatureMaskNeuromodulator + action layout
├── envs/               # task + non-stationarity   → docs 02
│   ├── multi_goal.py / dual_goal.py
│   ├── regime_wrapper.py     RegimeGoalSwapWrapper
│   ├── make_env.py           wrapper stack factory
│   └── wrappers/             action reduction, one-hot encoding
├── research/           # automation layer          → docs 07
│   ├── benchmarking.py       frozen specs + scorer + composite score
│   ├── autoresearch.py       AutoresearchSupervisor (bounded trial loop)
│   └── trial_prompt.py       per-trial prompt rendering
└── utils/              # logging + seeding
```

## Background reading

- [Combating Catastrophic Forgetting](../../Combating%20Catastrophic%20Forgetting.pdf) — focus on §3.7 (Meta-Learning and Neuromodulation), §4.3 (Meta-Learning results), and §4.3.1 (the neuromodulation diagnostic dashboard).
- [PLAN.md](../../PLAN.md) — original project roadmap (note: predates some of the code; the spec reflects the current code).
- [AUTORESEARCH.md](../../AUTORESEARCH.md) — the immutable benchmark/trial contract for autoresearch.
- [program_neuromod.md](../../program_neuromod.md) — the research brief handed to the autoresearch agent each trial.
