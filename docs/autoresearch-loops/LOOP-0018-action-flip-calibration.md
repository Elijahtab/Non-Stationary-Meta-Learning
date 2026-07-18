# LOOP-0018 — IP-1 action-flip instrument calibration (2026-07-17, closed same-day)

**Goal:** Decide, for ~2 GPU-hours, whether the IP-1 port (regimes differ in *dynamics* —
odd regimes mirror left/right — with reward fixed) buys the program a live learned-WHICH:
the recorded escape hatch from the selection family's three deaths (note 0013). User
approval 2026-07-17 of note 0015's three decision points (master env edit, IP-1, calibration).
**Verdict:** CLOSED 2026-07-17 (16/16 clean, ~2 h GPU) — **C-IP-a FAIL + C-IP-b FAIL: IP-1
is a dead instrument.** Flip-control composite 0.9532 (hit80 1.000) — a mirrored policy is
equally competent, so the flip has ~no forgetting cost; the O2-analog restore HURTS
(−0.0373, p=0.034 — new member of the O1 "oracle interventions can hurt" family); WM
next-state-error persistence at true switches: median 0 updates (56 switches; the flip's
error dilutes into the full-grid mean). The port dies at the calibration rung, before any
ladder spend — the gates did their job. IP-2 (slippery-floor) remains the recorded
fallback, fresh user call required. Anatomy + honest scope notes:
[note 0015 §Calibration outcome](../research-notes/0015-instrument-port-dynamics-regimes.md).

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

Loop CLOSED; records updated in the close-out commit (note 0015 §Calibration outcome,
register + indexes). The `action_flip` mode stays in the tree as default-off
infrastructure. Nothing launches from here without a user call. Open decisions on the
user's desk after this loop: (1) IP-2 slippery-floor registration — the recorded fallback,
with note 0015's recommendation to gate any future C-IP-b on a turn-conditioned WM error
rather than the mean; (2) the paper/venue track (v2 + the day's C4/plasticity edits still
need their first Overleaf compile); (3) the Vast console destroy-confirmation check
(LOOP-0009 box). GPU idle at close-out; no box exists.
