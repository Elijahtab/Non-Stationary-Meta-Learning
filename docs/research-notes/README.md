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
- [0003 — Code-Directed Plasticity Gating (gradient-side mask)](./0003-code-directed-plasticity-gating.md)
- [0004 — Code-free Plasticity & Stability Maintenance (LOOP-0007 family)](./0004-code-free-plasticity-stability.md)
- [0005 — The Untested Controller: adaptivity probe, bottom-rung control, and the oracle rung](./0005-untested-controller-bottom-rung-oracle.md)
- [0006 — The Controls Axis: thesis confirmed small, bottleneck relocated](./0006-controls-axis-thesis-relocated.md)
- [0007 — Multi-training-seed Replication of the Trained-Brain Effect](./0007-trained-brain-replication.md)
- [0008 — Paper Skeleton: meta-control helps a little, memory is the bottleneck](./0008-paper-skeleton.md)
- [0009 — Memory-facing Meta-control: can a learned trigger recover the O2 ceiling? (DRAFT for curation)](./0009-memory-levers-preregistration.md)
- [0010 — Related-work literature map: 62 verified papers vs. claims C1–C6](./0010-related-work-literature-map.md)
- [0011 — Wave-0 desk probes: screening power (H4 dies), dead-dim noise (B1 premise real), the 0.95 facet](./0011-wave0-desk-probes.md)
- [0012 — The ceiling decomposed: ~90% of the zero-forgetting headroom lives in the policy heads; MoWM rung null](./0012-ceiling-decomposition.md)
- [0013 — Learned WHEN works (54% of the slice, no oracle), learned WHICH fails via value fit — the selection frontier](./0013-learned-trigger-selection-gap.md)
