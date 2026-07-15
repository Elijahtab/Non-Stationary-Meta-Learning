# LOOP-0013 — Selection rung: reward fingerprint vs value fit, done right (2026-07-14–OPEN)

**Goal:** Adjudicate the learned-WHICH properly after LOOP-0012's selection arm was
invalidated (missing slot allocation): does content-addressable slot selection work, and is
the regime's identity signal in the reward function (WM reward head) rather than value fit?
Plus the A-R1 headline's n=16 extension.
**Verdict:** OPEN — launched 2026-07-14 night.

## Hardware

Home RTX 5070, $0, 24 sequential evals (~3.3 h).

## Code state

`Auto-Research` @ the LOOP-0013 pre-launch commit: spawn-until-full allocation in
`_head_bank_surprise_check` (the artifact fix, caught by the mode's smoke test);
`head_bank_select="reward_error"` (banks the 257-param WM reward head per slot as the regime
fingerprint; restore loads it with the policy heads in this mode); value-error selector
unchanged but now actually reachable. 24 unit tests in `tests/test_oracle_controls.py`
(new: spawn-then-address, reward-fingerprint discrimination); full suite green pre-launch;
smoke-verified end-to-end. **No src/scripts edits while the batch is in flight.**

## Runs

`g3_ar1re_e1..8`, `g3_ar1ve2_e1..8` (batch `g3re`) + `g3_ar1_e9..16` (batch `ar1x`).
Logs `evals/wave1_logs/`; idempotent.

## Statistics & verdicts

Gates **P-G3d / P-G3d-ve / P-G3e** pre-registered in
[research-log 0010](../research-log/0010-2026-07-14-selection-rung-preregistration.md)
(committed pre-results). Adjudicator: `scripts/score_wave1.py g3re`.

## Cost

$0; ~3.3 h GPU wall-clock.

## Sibling loops

[LOOP-0012](./LOOP-0012-g3-head-bank.md) (the trigger result + the invalidated selection arm
this corrects) · [LOOP-0010](./LOOP-0010-memory-levers.md) (selection failure here would make
the Brain's selection lever the remaining candidate — the box decision).

## Pickup state

Batches in flight. On completion: `scripts/score_wave1.py g3re` → adjudicate → update this
note + note 0013 (correction resolution) → ping the user. Routing: ar1re passes → the
learned-O2 method is complete (trigger + selection, no oracle) and the mem-Brain question
narrows to "can a Brain lever beat content addressing?"; both selectors fail → selection
defers to the mem-Brain lever (box ping) or the K=2 flip stands as the method.
