# LOOP-0015 — Drift-robust fingerprint selection (2026-07-16, closed same-day)

**Goal:** Fix content-addressed slot selection by scoring each slot's fingerprint in its own
frozen feature space (bank the full WM per slot, scoring-only), with the live WM scoring the
active slot — one mechanism that both unlocks the learned WHICH (K>2) and verifies fires
(the LOOP-0014 churn suppressor). User green-light 2026-07-16 ($0/home only).
**Verdict:** CLOSED 2026-07-16 (8/8 clean) — **P-FP1 FAIL (flip-when-should 0.16 / stay
0.64) + P-FP2 FAIL (+0.0367, p=0.27)**. The drift fix worked; a third, distinct mechanism
surfaced: the live reward head adapts to the new regime within the trigger's one-update lag,
so the live-scored "no switch" hypothesis wins even at true switches. **Selection family
closed (3 probes / 3 identified mechanisms: drift, churn, live-side adaptation) — the blind
K=2 flip (+0.1188, n=16) stands as the method.** Full anatomy:
[note 0013 §Drift-robust fingerprint addendum](../research-notes/0013-learned-trigger-selection-gap.md).

## Hardware

Home RTX 5070, $0, 8 sequential evals (~1.1 h). No box (none owned; none needed).

## Code state

`Auto-Research` @ the LOOP-0015 pre-launch commit: `head_bank_select="reward_fp"` —
`_head_bank_snapshot` banks `wm_full` (~1.5M params) in this mode; selection scores banked
slots through a scratch WM module (live WM never touched), active slot through the live WM;
restores load policy heads only. Trigger identical to the method arm (surprise thr 1.0).
26 unit tests in `tests/test_oracle_controls.py` (new: frozen-feature scoring, live-WM
isolation, heads-only restore); full suite green pre-launch. **No src/scripts edits while
the batch is in flight.**

## Runs

`g3_fp_e1..8` (batch `fp`). References archived (control/ar1 n=16, ar1re drift-bound null,
oracle heads slice). Logs `evals/wave1_logs/`; idempotent.

## Statistics & verdicts

Gates **P-FP1 / P-FP2** (+ secondary suppression dividend) pre-registered in
[research-log 0013](../research-log/0013-2026-07-16-fingerprint-selection-preregistration.md)
(committed pre-results). Adjudicator: `scripts/score_wave1.py fp` (gains + per-fire
flip/stay behavior reconstructed from fire + active traces).

## Cost

$0; ~1.1 h GPU wall-clock.

## Sibling loops

[LOOP-0013](./LOOP-0013-selection-rung.md) (the drift null this fixes) ·
[LOOP-0014](./LOOP-0014-step-trigger.md) (the churn mechanism this suppresses) ·
[LOOP-0012](./LOOP-0012-g3-head-bank.md) (the bank + method reference arm).

## Pickup state

Scored arm in flight. On completion: `scripts/score_wave1.py fp` → adjudicate P-FP1/FP2 →
update this note + note 0013 lineage → paper §6.3 closing paragraph (currently frames this
rung as future work) updates with the outcome either way → report to the user. If FP1+FP2
pass, natural follow-ons (NOT pre-approved): K=3 fingerprint arm (content addressing where
flip is undefined) and a re-visit of the per-step trigger with verification. Box unneeded.
