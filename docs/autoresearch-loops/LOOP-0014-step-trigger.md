# LOOP-0014 — Per-step trigger: attacking the lag wall (2026-07-15–OPEN)

**Goal:** Replace the per-update value-loss trigger with a per-step success-collapse
detector and test whether cutting detection lag (~2,048+ → ~400 steps) recovers a
substantial further slice of the oracle heads ceiling (+0.228), per LOOP-0013's
adjudication that lag — not precision — is the binding trigger cost.
**Verdict:** OPEN — scored arm launched 2026-07-16 after three-iteration shadow calibration
(fully disclosed in [log 0012](../research-log/0012-2026-07-16-step-trigger-preregistration.md)).

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

Scored arm in flight. On completion: `scripts/score_wave1.py stp` → adjudicate P-S1a/b →
update this note + note 0013 lineage (new results note if the method number moves) → then
**paper consolidation** (user direction: "do 1 then 3") — fold notes 0011–0013 + this loop
into `paper/main.tex` (decomposition table, learned-trigger method, negative anatomy).
Box remains unneeded; the mem-Brain question stays deprioritized per LOOP-0013.
