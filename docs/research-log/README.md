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
| [0001](./0001-2026-06-18-calibration-scoring.md) | 2026-06-18 | `calib_signal` null result + composite-score redesign | Accepted |
| [0002](./0002-2026-06-18-calib8x8-cloud-sweep-plan.md) | 2026-06-18 | `calib8x8` cloud sweep: execution plan | Accepted (gated on pilot) |
| [0003](./0003-2026-06-18-recovery-reward-oscillation-fix.md) | 2026-06-18 | `recovery` reward farms oscillation; add `recovery_v2` | Accepted |
| [0004](./0004-2026-07-04-scout-v2-8x8.md) | 2026-07-04 | Scout v1 (5×5) saturates; primary → `fast_switch_scout_v2` (8×8) | Accepted |
| [0005](./0005-2026-07-05-confirmation-sweep-prediction.md) | 2026-07-05 | Registered prediction: actor-only & gain-α0.5 n=8 confirmation sweep | Accepted |
| [0006](./0006-2026-07-05-amortized-multiseed-baseline.md) | 2026-07-05 | Amortized 3-seed scout baseline (`baseline_primary`) | Accepted |
| [0007](./0007-2026-07-06-learning-dynamics-preregistration.md) | 2026-07-06 | Registered predictions + decision rules: LOOP-0006 learning-dynamics screening | Accepted |
| [0008](./0008-2026-07-14-wave1-oracle-rungs-preregistration.md) | 2026-07-14 | Registered gates P-W1a/b/c: Wave-1 G-DECOMP swap-scope ladder + 3-regime screen (LOOP-0011) | Resolved same-day (a PASS, b→W2A, c FAIL) |
| [0009](./0009-2026-07-14-g3-head-bank-preregistration.md) | 2026-07-14 | Registered gates P-G3a/b/c: G3 head-bank de-oracling screen — equivalence, A-R1 learned trigger (thr 1.0), value-error selection (LOOP-0012) | Resolved same-day (a PASS, b PASS +0.118, c FAIL) |
