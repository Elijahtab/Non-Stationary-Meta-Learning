# 0011 — 2026-07-15 — Trigger hardening at threshold 1.5: pre-registration (LOOP-0013 addendum)

**Registered BEFORE any results exist.** Context: at n=16 the K=2-flip method holds +0.1188
(52% of the oracle heads slice) and the oracle-vs-learned gap is entirely trigger cost
(note 0013 §LOOP-0013 results): in-run precision at threshold 1.0 is 0.69 (desk 0.84), and
each false fire actively loads the wrong head. The desk calibration's precision-optimal point
with acceptable recall is threshold 1.5 (precision 0.95, recall 0.81, cooldown 8).

## Arm (8 evals, home 5070, ~1.1 h, $0)

`g3_ar1t15` × eval seeds 1–8: `head_bank_slots=2, trigger=surprise, select=other,
surprise_threshold=1.5`. References archived: control (n=16), `g3_ar1` @ thr 1.0 (n=16).
Zero code change — the threshold is already a flag. Scored by `scripts/score_wave1.py t15`.

## Pre-registered gates

- **P-T15a (precision transfers):** in-run trigger precision ≥ 0.80 (desk 0.95 minus the
  ~0.15 in-run degradation observed at thr 1.0).
- **P-T15b (it matters):** gain(thr 1.5) ≥ gain(thr 1.0, n=16) + 0.02, i.e. ≥ **+0.139**.
- **Readings:** both pass → trigger hardening works; the method number moves up and the
  remaining gap shrinks toward recall/lag. P-T15a passes but P-T15b fails → false fires
  were not the binding cost; the gap is misses/lag (recall), and threshold tuning is done —
  remaining routes are drift-robust selection (K>2) or the mem-Brain. Both fail → in-run
  dynamics defeat desk calibration; the detector needs a redesign (hysteresis /
  post-flip-verification), not a threshold.
- **Prediction (stated):** precision ~0.85–0.90 in-run; gain +0.13–0.15.
- **Extension rule:** deciding quantities within ±0.02 composite (or ±5 pp precision) of a
  bar → extend to n=16 before adjudicating.
