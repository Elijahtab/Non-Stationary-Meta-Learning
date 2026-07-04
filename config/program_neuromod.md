# Neuromodulation Research Brief

You are running one bounded research trial inside this Meta-RL repository.

Your job is to improve the neuromodulation part of the system, not the benchmark.
This repo is the Meta-RL subsection of the larger catastrophic forgetting work:
the outer Brain controls the inner Dyna-PPO continual learner, and neuromodulation
changes how the inner policy network responds to regime shifts.

## Core Goal

Find one small architectural change that could improve post-switch recovery on the
frozen benchmark without expanding the codebase unnecessarily.

## Architecture To Respect

- Environment wrappers hide regime identity in observations and change only reward contingencies.
- The inner loop is Dyna-PPO with PPO, replay, a world model, and dreamed updates.
- `MetaEnv` turns a whole inner-training interval into a Brain action/observation step.
- The Brain outputs scalar control knobs plus a neuromodulation context code.
- The inner CNN consumes that neuromodulation signal to alter shared features before policy/value heads.

## Research Priorities

Start with the smallest changes most likely to help:

1. Improve the neuromodulation decoder or mask parameterization.
2. Improve where and how modulation is applied in the CNN.
3. Separate actor and critic modulation if the current shared path looks limiting.
4. Only after that, consider increasing neuromodulation capacity directly.

## Strong Guidance

- Prefer targeted edits in `neuromod.py` and `network.py`.
- Keep the Brain action surface stable unless there is a strong reason to change it.
- If you want more neuromodulation capacity, first consider expanding the decoded modulation space from the existing context code before changing the Brain output dimensionality itself.
- If you do decide to test larger context sizes like `16`, `32`, or `64`, keep all related shapes internally consistent and do not make unrelated refactors.
- Avoid changing benchmark code, scorer code, plotting code, or environment code.
- Avoid large abstractions unless they directly simplify the neuromodulation research surface.

## What A Good Trial Looks Like

- One hypothesis.
- One coherent code change.
- Small diff.
- Clear expected effect on post-switch recovery.
- Minimal collateral changes.

## What To Write Down

Before finishing the trial, write a short note with:

- the hypothesis,
- the exact files changed,
- why the change could help,
- the main risk or uncertainty.

Research notes under `docs/research-notes/` (per the repo convention) are allowed and
encouraged — create a new numbered note for your hypothesis and add it to the README index.
Two hard rules: **never rewrite or delete existing notes' predictions or results** (they are
append-only lab records; you may append a dated update section), and never touch notes you
did not create except to append.
