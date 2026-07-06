# CLAUDE.md — Lifelong-Learning

Project instructions for working in this repo. This is the **Meta-Learning ("Brain") subsystem**
of the catastrophic-forgetting project: an outer PPO "Brain" controls an inner Dyna-PPO continual
learner, with a frozen-benchmark + autoresearch layer on top.

**Start every non-trivial task with the `/orient` skill** — it maps the architecture from
[docs/spec/](docs/spec/) and grounds claims in source.

## Docs layout — what goes where

| Dir | Purpose |
| --- | --- |
| [docs/spec/](docs/spec/) | How the code works (non-canonical map; source is ground truth) |
| [docs/plans/](docs/plans/) | Roadmap, budget, experiment matrix |
| [docs/research-notes/](docs/research-notes/) | **Lab notebook for the paper** — see below |
| [docs/research-log/](docs/research-log/) | Dated, chronological trial/session records |
| [docs/autoresearch-loops/](docs/autoresearch-loops/) | **One note per autoresearch loop** — run info, obstacles, verdicts, pickup state |

## Research notes convention (important)

**[docs/research-notes/](docs/research-notes/) is the lab notebook for the neuromodulation
paper.** Use it to log **research assumptions, findings, and paper groundwork** — the scientific
argument (hypotheses → predictions → results → interpretation), accreting toward the write-up.

- One numbered note per idea/experiment (`NNNN-slug.md`); lead with the one-sentence hypothesis.
- State assumptions **explicitly** and flag their risks.
- When an experiment resolves, **update its note** with the outcome — don't leave stale predictions.
- Keep mechanics in `docs/spec/`, roadmap in `docs/plans/`, dated logs in `docs/research-log/`.
- When you make a research assumption, discover a finding, or take a design decision with
  rationale during a task here, **record it as (or into) a research note.**

## Autoresearch (important)

**Before doing ANY autoresearch work, read [AUTORESEARCH.md](AUTORESEARCH.md) (the checklist +
decisions register) and the relevant loop note(s) in
[docs/autoresearch-loops/](docs/autoresearch-loops/).** The register says what is currently
true; the latest loop note's *pickup state* says what to do next. Never edit snapshot-managed
files in the main tree while a local trial is in flight — use a worktree.

## Core vocabulary

- **run** = an outer-Brain *training* run (`runs/`, produces `brain_model.pt`).
- **eval** = a *trained*-Brain *evaluation* (`evals/`, consumes a checkpoint).

## Conventions

- Windows: drive via `myenv\Scripts\python.exe` with `PYTHONPATH=src`.
- The frozen benchmark / scorer / env code is an **immutable surface** for autoresearch — don't
  edit it without cause (see [config/research_manifest.toml](config/research_manifest.toml)).
- Neuromodulation research surface: `src/lifelong_learning/agents/brain/neuromod.py` +
  `src/lifelong_learning/agents/ppo/network.py`.
