# 0016 — 2026-07-17 — IP-1 action-flip instrument: calibration pre-registration (LOOP-0018)

**Registered BEFORE any results exist.** User approval 2026-07-17 ("1. and 2."): the
master-level env edit for the IP-1 instrument port (note 0015 decision points 1–3: edit
approved, IP-1 selected, calibration green-lit, $0/home). Context: the learned WHICH died on
the frozen benchmark because regimes differ only in reward and the live reward head adapts
within the trigger lag (note 0013, three mechanisms). IP-1 inverts the instrument: **regimes
differ in dynamics (odd regimes mirror left/right), reward mapping fixed** — the regime
signal moves to the WM's next-state head, a dense, high-dimensional surface that should
outlive the lag. This calibration decides whether the port is worth a ladder, **before**
any ladder is built.

- **Touches (master edit, register row updated in the same commit):**
  `envs/regime_wrapper.py` + `envs/make_env.py` gain `regime_effect={"goal_swap","action_flip"}`,
  default `goal_swap` **byte-identical to the frozen benchmark** (unit-tested regression);
  plumbed through `train.py`/`eval_brain.py` as default-off kwargs. The frozen benchmark
  presets, scorer, and existing env ids are untouched.

## Arms (16 evals, home 5070, ~2.1 h, $0)

`ipflip_ctrl_e1..8` (action_flip, no mechanism flags) and `ipflip_o2_e1..8` (action_flip +
full oracle swap — the ceiling analog; the snapshot dict is regime-keyed so O2 transfers
unchanged). Brain = LOOP-0009 seed-1 ep130 (scale-blind; **registered caveat:** its obs
normalizer was fit on goal_swap — all numbers are instrument-local and descriptive vs
future flip-trained Brains). Protocol otherwise identical to the standard eval (8×8, 800k,
100k/regime, K=2). Scored by `scripts/score_wave1.py ipcal`.

## Pre-registered gates

- **C-IP-a (headroom exists):** O2-analog ceiling vs flip-control ≥ **+0.10** composite,
  Welch p < 0.05. Below that, restoring a mirrored-competent policy buys too little room
  for any memory method to show — the port dies for ~2 GPU-hours.
- **C-IP-b (the premise — regime signal outlives adaptation):** on the control arm, the
  per-update WM next-state error (dense `debug/wm_raw_error_mean`, bucketed to update
  means) stays above baseline_mean + 2·baseline_sd (baseline = 20 pre-switch updates) for
  a **median ≥ 4 consecutive updates** after true switches — vs the <1-update reward-head
  contrast that killed fingerprints (note 0013). If next-state error also re-fits within
  the lag, live-side adaptation generalizes and the learned WHICH is dead here too.
- **Readings.** Both pass → the port opens: next rungs (each its own registration) are the
  flip-instrument decomposition ladder and the fingerprint selector re-run with state-error
  scoring. C-IP-a fail → instrument lacks headroom; try IP-2 (slippery-floor, tunable) only
  by a fresh user call. C-IP-b fail → the third selection-death mechanism is
  instrument-general; the WHICH frontier closes at this scale and the paper says so.
- **Extension rule:** ceiling within ±0.02 of +0.10 → extend to n=16 before adjudicating.

## Registered risks

The flip may be *too easy* to detect (an agent mid-corridor feels it instantly) — fine for
the WHICH question, weaker for WHEN claims; recorded, not gated. Success asymmetry: regime-1
blocks require mirrored controls the from-scratch learner must discover — first-exposure
windows may be slower than goal_swap's, so cross-instrument composite comparisons are
meaningless (only within-instrument contrasts are read; this is the note 0012 lesson,
registered here explicitly). The composite's post-switch windows and the scorer are used
unmodified — regime clocking is identical by construction.
