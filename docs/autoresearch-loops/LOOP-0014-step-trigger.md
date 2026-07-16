# LOOP-0014 — Per-step trigger: attacking the lag wall (2026-07-15–16, closed)

**Goal:** Replace the per-update value-loss trigger with a per-step success-collapse
detector and test whether cutting detection lag (~2,048+ → ~400 steps) recovers a
substantial further slice of the oracle heads ceiling (+0.228), per LOOP-0013's
adjudication that lag — not precision — is the binding trigger cost.
**Verdict:** CLOSED 2026-07-16 (8/8 clean) — **P-S1a FAIL + P-S1b FAIL, the registered
both-fail reading applies: the collapse statistic does not survive live dynamics.** The lag
was cut exactly as designed (median 280 steps, recall 1.00) but the gain FELL to +0.0916
(40% of slice) vs ar1's +0.1188: live precision degraded to 0.65 at 15.5 fires/run because
**a false flip guarantees its own secondary collapse** (the wrongly-loaded head fails →
correction fire → ~2 cooldowns of churn per false fire) — shadow cluster-precision (~0.9)
cannot see this feedback loop. The t15 finding "precision is free" is cadence-bounded: free
at ~2 false fires/run, expensive at ~8. **The trigger family is complete: per-update
value-loss @ thr 1.0 (+0.1188, n=16) stands as the method**; the residual ~0.10 gap to the
oracle slice is structural to K=2-flip + any self-supervised trigger (detection economics +
first-exposure spawn dynamics), not to any single knob. Full anatomy:
[note 0013 §Per-step addendum](../research-notes/0013-learned-trigger-selection-gap.md).

## Hardware

Home RTX 5070, $0. 4 shadow-calibration evals (seeds 101/102, non-registered) + 8 scored
evals (~1.1 h).

## Code state

`Auto-Research` @ the LOOP-0014 pre-launch commit: `head_bank_trigger="step_surprise"` —
model-free fast/slow success EMAs at episode terminations, fires mid-rollout, K=2 flip,
warmup 50k / cooldown 2,048 / collapse ratio 0.25 (flag) / shape constants fixed
(`STEP_*` in train.py). Shadow mode (`--head_bank_step_shadow`) logs would-be fires
without switching. v1/v2 (WM reward-prediction-error variants) were built, calibrated,
and abandoned same-day — per-event WM prediction error is noise (precision 0.02–0.04);
see log 0012 §Calibration history. 27 unit tests in `tests/test_oracle_controls.py`;
full suite green pre-launch. **No src/scripts edits while the batch is in flight.**

## Runs

`stpcal_e101/102` (shadow ×3 iterations; final kept) → `g3_stp_e1..8` (batch `stp`).
Logs `evals/wave1_logs/`; idempotent.

## Statistics & verdicts

Gates **P-S1a / P-S1b** pre-registered in [log 0012](../research-log/0012-2026-07-16-step-trigger-preregistration.md)
(committed pre-results): precision ≥ 0.75 & recall ≥ 5/6 live; gain ≥ +0.139.
Adjudicator: `scripts/score_wave1.py stp`.

## Cost

$0; ~2.5 h GPU wall-clock including calibration.

## Sibling loops

[LOOP-0013](./LOOP-0013-selection-rung.md) (the lag adjudication this answers) ·
[LOOP-0012](./LOOP-0012-g3-head-bank.md) (the head-bank mechanism + ar1 reference arm).

## Pickup state

**CLOSED — verdict above; scores in `evals/wave1_scores.json` (stp key).** The method number
is unchanged: **+0.1188 (n=16, per-update trigger)**. Next (user direction "1 then 3"):
**paper consolidation** — fold notes 0011–0013 + LOOP-0011..0014 into `paper/main.tex`
(decomposition table/figure, the learned-trigger method result, the negative anatomy:
selection drift, precision null, per-step churn, K=3 flat premium). After the paper: the
remaining research options are drift-robust fingerprint selection (also the false-fire
suppressor — staying put when fingerprints match would attack the churn mechanism this loop
identified) and the deprioritized mem-Brain (box). Box remains unneeded.
