# LOOP-0018 — IP-1 action-flip instrument calibration (2026-07-17, open)

**Goal:** Decide, for ~2 GPU-hours, whether the IP-1 port (regimes differ in *dynamics* —
odd regimes mirror left/right — with reward fixed) buys the program a live learned-WHICH:
the recorded escape hatch from the selection family's three deaths (note 0013). User
approval 2026-07-17 of note 0015's three decision points (master env edit, IP-1, calibration).
**Verdict:** OPEN — launch queued behind LOOP-0017's batch (same GPU).

## Hardware

Home RTX 5070, $0, 16 sequential evals (~2.1 h). No box.

## Code state

`Auto-Research` @ the LOOP-0017/0018 pre-launch commit: `regime_effect={"goal_swap",
"action_flip"}` on `RegimeGoalSwapWrapper`/`make_env` (master-approved edit to the immutable
env surface — register row updated same commit), default `goal_swap` byte-identical
(regression-tested); plumbed through `train.py`/`eval_brain.py`. 4 unit tests in
`tests/test_regime_action_flip.py`; full suite 173 green.

## Runs

`ipflip_ctrl_e1..8` + `ipflip_o2_e1..8` (batch `ipcal`), Brain = loop9 s1 ep130
(normalizer caveat registered). Logs `evals/wave1_logs/`; idempotent.

## Statistics & verdicts

Gates **C-IP-a** (O2-analog ceiling ≥ +0.10, p<0.05) and **C-IP-b** (WM next-state-error
persistence at true switches: median ≥ 4 consecutive updates above baseline+2sd) pre-registered
in [research-log 0016](../research-log/0016-2026-07-17-action-flip-calibration-preregistration.md)
(committed pre-results). Adjudicator: `scripts/score_wave1.py ipcal`.

## Cost

$0; ~2.1 h GPU wall-clock.

## Sibling loops

[LOOP-0017](./LOOP-0017-conv1-redo-rung.md) (shares the pre-launch commit + GPU queue) ·
[LOOP-0015](./LOOP-0015-fingerprint-selection.md) (the live-side-adaptation death this
routes around) · [LOOP-0013](./LOOP-0013-selection-rung.md) (the selection frontier).

## Pickup state

Launch `run_wave1_evals.py ipcal` once the redoc1 batch releases the GPU. On completion:
`score_wave1.py ipcal` → C-IP-a and C-IP-b → note 0015 gains a §Calibration outcome →
close this note + register row → report. Both gates pass → next rungs (EACH a new
registration + user go): flip-instrument decomposition ladder, then the fingerprint
selector re-run scored by next-state error. C-IP-a fail → IP-2 is the fallback, user call.
C-IP-b fail → the WHICH frontier closes instrument-generally at this scale; paper sentence.
