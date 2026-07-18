# LOOP-0017 — conv1-targeted ReDo rung / branch F (2026-07-17, closed same-day)

**Goal:** Branch F's first live-targeted intervention: LOOP-0016 found first-conv-layer
dormancy accumulating (0.125→0.479) while everything downstream falls — does eliminating it
move the composite? User green-light 2026-07-17 ($0/home). Registered expectation: a null is
likely (the oracle encoder slice bounds the prize at ~+0.016, below the +0.05 MDE) and
informative — it becomes the "behaviorally cheap" bound in the paper's C4 paragraph.
**Verdict:** CLOSED 2026-07-17 (8/8 clean, ~61 min GPU) — **P-F1a FAIL at the mechanism
level: resets don't stick.** conv1 dormancy ends HIGHER with resets than without (f8 0.550
vs 0.479 untreated; rise +0.442) — re-initialised channels re-die within the 10-update
cadence. Composite unmoved (−0.0127, p=0.581; no harm flag). P-F1b not read per gate order.
**Reading: the accumulation is a converged input-sparsity attractor, not reset-repairable
damage — branch F closes with both edges measured** (heads never needed resets, LOOP-0007;
the input layer won't hold them, this rung). Paper plasticity-probe paragraph + limitations
updated. The cadence-1 re-registration contemplated by the pre-reg's risk clause is NOT
launched (attractor reading argues it buys churn, not retention) — user's call. Anatomy:
[note 0014 §Addendum](../research-notes/0014-encoder-dormancy-probe.md).

## Hardware

Home RTX 5070, $0, 8 sequential evals (~1.1 h). No box.

## Code state

`Auto-Research` @ the LOOP-0017/0018 pre-launch commit: `redo_conv1_interval` (default 0) —
`redo_reset_conv1` applies `redo_reset_heads` mechanics at conv1 channel granularity
(re-init incoming filter, zero bias, zero conv2 incoming, clear Adam), single disclosed
config (interval 10, τ=0.025), probe kept on for mechanism verification. 4 unit tests in
`tests/test_redo_conv1.py`; full suite 173 green. **No src/scripts edits while batches are
in flight.**

## Runs

`redoc1_e1..8` (batch `redoc1`). References archived: `loop9_s1_model_e*` (n=16),
`dorm_e*` (untreated trajectories). Logs `evals/wave1_logs/`; idempotent.

## Statistics & verdicts

Gates **P-F1a** (mechanism: conv1 f8 ≤ 0.15 AND rise ≤ +0.05 — read FIRST) and **P-F1b**
(composite ≥ +0.05, p<0.05; registered null reading = "first-layer plasticity loss is
behaviorally cheap") + harm flag (≤ −0.02), pre-registered in
[research-log 0015](../research-log/0015-2026-07-17-conv1-redo-preregistration.md)
(committed pre-results). Adjudicator: `scripts/score_wave1.py redoc1`.

## Cost

$0; ~1.1 h GPU wall-clock.

## Sibling loops

[LOOP-0016](./LOOP-0016-encoder-dormancy-probe.md) (the probe that armed this rung) ·
[LOOP-0018](./LOOP-0018-action-flip-calibration.md) (runs back-to-back on the same GPU) ·
[LOOP-0007](./LOOP-0007-brainstorm.md) (the heads-ReDo mirage this must not repeat).

## Pickup state

Loop CLOSED; records updated in the close-out commit (note 0014 addendum, paper ×2 sites,
register + indexes). Nothing launches from here without a user call: the only recorded
follow-on is the cadence-1 re-registration (one re-run allowed by log 0015's risk clause;
recommendation AGAINST — the attractor reading says re-death, not slow cadence, is the
mechanism). LOOP-0018 (ipcal batch) was in flight on the same GPU at close-out.
