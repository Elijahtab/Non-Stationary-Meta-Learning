# 0012 — 2026-07-16 — Per-step success-collapse trigger: pre-registration (LOOP-0014)

**Registered BEFORE any scored results exist.** Context: LOOP-0013 closed with the oracle-vs-
learned gap attributed to detection **lag** (per-update detectors pay ≥2,048 steps of
wrong-head time per switch; precision is measured to be free). This rung attacks the lag
wall with a per-step trigger. User direction 2026-07-15: "do 1 then 3" (this rung, then
paper consolidation).

## Mechanism (committed pre-launch, `head_bank_trigger="step_surprise"`)

Model-free success-collapse detector at episode terminations: fast (decay 0.8) vs slow
(decay 0.995) success EMAs over terminal events (success = terminal reward > 0.05); fires
mid-rollout (oracle-swap precedent) when `fast < 0.25 × slow`, gated by slow ≥ 0.5 (only
fire *out of a learned regime*), ≥ 8 events since last fire, warmup 50,000 steps, cooldown
2,048 steps. On fire: K=2 flip (content addressing is drift-broken, note 0013), fast EMA
resets to slow (a correct restore stays quiet; a wrong flip re-collapses and self-corrects
after the cooldown). Constants fixed; only the collapse ratio is an exposed flag.

## Calibration history (instrument development, fully disclosed; shadow mode, seeds 101/102 only)

1. **v1 — WM reward-prediction error, absolute bar 0.25:** precision 0.04 (≈220 fires/run).
   The live reward head is chronically underconfident at goal events; per-event prediction
   error is noise.
2. **v2 — relative bar (err > 0.5 × max(r², r̂²)):** precision 0.02–0.03 AND recall fell to
   2–4/7 — same cause, worse trade. Model-based per-event triggering abandoned.
3. **v3 — success collapse (this design):** live-equivalent precision (first fire per
   collapse cluster, post-warmup) **6/6 and 6/7** on the two shadow runs; recall 6/6
   scoreable switches; **median lag 448 / 352 steps** (vs 2,048+ structural for per-update).
   All learning-phase false fires sat below ~45k steps → warmup set to 50,000.

The first switch (100k) is structurally exempt from recall: nothing is banked before it, so
a fire there can only spawn — identical behavior to every prior trigger arm.

## Arm (8 evals, home 5070, ~1.1 h, $0)

`g3_stp` × eval seeds 1–8: `head_bank_slots=2, trigger=step_surprise, select=other`,
defaults as above. References archived: control (n=16), `g3_ar1` (n=16, +0.1188),
`wave1_decomp_heads` (oracle ceiling slice +0.218–0.228 by seed subset). Scored by
`scripts/score_wave1.py stp`.

## Pre-registered gates

- **P-S1a (trigger quality transfers live):** in-run precision ≥ 0.75 (a fire is true iff
  within 4,096 steps after a switch) AND recall ≥ 5/6 on switches 2–7.
- **P-S1b (the lag hypothesis pays):** gain ≥ **+0.139** (= ar1 n=16 + 0.02 — same bar the
  threshold arm failed).
- **Prediction (stated):** gain +0.14–0.18. Lag shrinks ~2,000→~400 steps per switch and
  the late-detection tail (the 20–25% of switches the per-update detector caught late)
  should largely vanish; the residual gap to the +0.22 oracle slice is flip-side cost at
  first exposures and any remaining mid-regime false flips (measured ~1 per 2 runs in
  shadow — cheap, per the t15 result).
- **Readings:** P-S1a pass + P-S1b fail ⇒ lag was NOT the binding cost after all — the gap
  is in the K=2-flip dynamics themselves (e.g., first-exposure thrash), which caps the
  trigger family and makes drift-robust *selection* the only remaining $0 lever. Both
  fail ⇒ the collapse statistic doesn't survive live dynamics (the flip changes the very
  statistic being watched); revert to ar1 as the method and consolidate.
- **Extension rule:** deciding quantities within ±0.02 composite (or ±5 pp precision /
  1 switch of recall) of a bar → extend to n=16 before adjudicating.
