# Research Notes

This folder is the **lab notebook** for the task-free neuromodulation paper. Each note records
a **research assumption, a finding, or a piece of experimental groundwork** — the reasoning
*around* the experiments, not the run logs themselves.

## What goes here vs. elsewhere

| Put it here (`docs/research-notes/`) | Put it elsewhere |
| --- | --- |
| Hypotheses, predictions, and *why* we expect them | Run outputs → `runs/`, `evals/`, `benchmarks/`, `sweeps/` |
| Interpretation of results and what they mean for the paper | How a subsystem works (mechanics) → `docs/spec/` |
| Assumptions we're relying on (and their risks) | The overall experiment plan / roadmap → `docs/plans/` |
| Dead ends, surprises, and design decisions with rationale | Dated, chronological trial records → `docs/research-log/` |

Rough division of labor:
- **`docs/spec/`** = how the code works (ground truth, non-canonical map).
- **`docs/plans/`** = what we intend to do (roadmap, budget, experiment matrix).
- **`docs/research-log/`** = dated, chronological "what happened in this session/trial".
- **`docs/research-notes/`** = the *scientific argument*: assumptions → hypotheses → predictions
  → results → interpretation, accreting into the paper's narrative.

## Conventions

- One note per idea/experiment, numbered: `NNNN-short-slug.md`.
- Lead with the **claim/hypothesis** in one sentence, then background, prediction, how-to-measure,
  possible outcomes + interpretations, risks, and the connection to the paper.
- State **assumptions explicitly** and mark them as assumptions (they are the things most likely
  to be wrong).
- When a note's experiment resolves, **update the note** with the result and what it changed —
  don't leave a stale prediction standing.
- Link to the relevant `docs/spec/` mechanics, `docs/plans/` line items, and any `sweeps/` output.

## Index

- [0001 — Trainable vs. Frozen Neuromodulation Decoder](./0001-trainable-vs-frozen-decoder.md)
- [0002 — Stabilizing the Trainable Decoder (slowbrain / declr / long / combo)](./0002-stabilizing-trainable-decoder.md)
