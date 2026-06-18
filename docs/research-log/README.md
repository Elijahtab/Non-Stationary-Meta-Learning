# Research Log

Dated records of **research decisions and their outcomes** for the Meta-Learning ("Brain")
subsystem. This is the lab notebook: *why* we changed something, *what we expected*, and
*what actually happened* — so we never re-run a sweep without knowing what the last one taught us.

This complements [`docs/spec/`](../spec/) (which describes *what the code does*) and
[`runs/`](../../runs/) / [`benchmarks/`](../../benchmarks/) / [`sweeps/`](../../sweeps/)
(which hold raw artifacts). The research log is the *interpretation* layer on top of those
artifacts.

## How to use this folder

- One file per decision, named `NNNN-YYYY-MM-DD-short-slug.md` (zero-padded sequence + date).
- Keep entries short and falsifiable. Link to the artifacts (`sweeps/...`, `runs/...`,
  `benchmarks/...`) and to the source lines (`file.py#Lnn`) that ground each claim.
- A decision starts as **Proposed**, becomes **Accepted/Rejected** when acted on, and gets an
  **Outcome** section filled in once the follow-up run reports back. Don't delete superseded
  records — mark them **Superseded by [NNNN]** so the reasoning trail survives.

## Entry template

```markdown
# NNNN — <title>

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Rejected | Superseded by [NNNN]
- **Touches:** <files / scorer / preset / env — and whether any immutable_surface file changes>

## Context
What prompted this? Link the run/sweep/benchmark that raised the question.

## Decision
The specific change, stated so someone could implement it from this paragraph alone.

## Rationale
Why this over the alternatives. Cite data and source lines.

## Outcome
(Filled in after the follow-up run.) What the evidence showed. Confirmed / refuted / surprised.

## Follow-ups
Concrete next actions or open questions.
```

## Index

| # | Date | Title | Status |
|---|------|-------|--------|
| [0001](./0001-2026-06-18-calibration-scoring.md) | 2026-06-18 | `calib_signal` null result + composite-score redesign | Proposed |
