# LOOP-0012 — G3 head-bank: the de-oracling screen (2026-07-14, closed same-day)

**Goal:** Turn the Wave-1 routing verdict (heads carry ~90% of the +0.243 ceiling, note 0012)
into a *method*: a K-slot head bank whose trigger (WHEN) and selection (WHICH) are learned
instead of oracle — the candidate learned-O2 result.
**Verdict:** CLOSED 2026-07-14 (24/24 clean) — **P-G3a PASS** (bank ≡ heads swap, p=0.84);
**P-G3b PASS — the A-R1 learned trigger recovers +0.1180 = 54% of the ceiling slice**
(p=1.6e-4, in-run precision 0.70/recall 0.75, hit80 1.000); **P-G3c FAIL — corrected same day
(implementation artifact):** the value-error arm ran without slot allocation, so its selector
was never exercised (bank-without-restore; composite == control). Science claim withdrawn;
re-run with the spawn-until-full fix in [LOOP-0013](./LOOP-0013-selection-rung.md). Full
analysis + correction: [note 0013](../research-notes/0013-learned-trigger-selection-gap.md).

## Hardware

Home RTX 5070 only, $0 (24 sequential evals ≈ 3.2 h). No box (per the 2026-07-14 user call).

## Code state

`Auto-Research` @ the LOOP-0012 pre-launch commit: head bank in `train.py`
(`head_bank_slots/trigger/select/surprise_threshold`, flag-guarded default-off, validated
combos, own detector state decoupled from cand-4), `eval_brain.py` passthrough, calibrator
`scripts/calibrate_surprise_trigger.py` (threshold 1.0: precision .84 / recall .85 on
archived control traces), driver batch `run_wave1_evals.py g3`, adjudicator
`score_wave1.py g3` (incl. in-run trigger precision from logged fire steps). 156 tests green
(5 new head-bank tests); both trigger modes smoke-verified end-to-end via eval_brain.py.
**No src/scripts edits while the batch is in flight.**

## Runs

`g3_oracle` / `g3_ar1` / `g3_ar1ve` × eval seeds 1–8 (`evals/g3_*`); references archived
(`loop9_s1_model_e1..8` control, `wave1_decomp_heads_e1..8` ceiling slice). Logs
`evals/wave1_logs/`. Idempotent re-run fills gaps.

## Statistics & verdicts

Pre-registered gates **P-G3a/b/c** in
[research-log 0009](../research-log/0009-2026-07-14-g3-head-bank-preregistration.md)
(committed pre-results). P-G3b is the tree's A-R1 kill: gain < +0.05 OR trigger precision
< 0.60 ⇒ the mem-Brain trigger premise dies.

## Cost

$0; ~3.2 h GPU wall-clock.

## Sibling loops

[LOOP-0011](./LOOP-0011-wave1-oracle-rungs.md) (the routing verdict + reference arms) ·
[LOOP-0010](./LOOP-0010-memory-levers.md) (consumes P-G3b: the restore-gate lever premise).

## Pickup state

**CLOSED — verdict above; scores in `evals/wave1_scores.json` (g3 key).** The next rung is a
**strategy fork** (note 0013 §Consequences), deliberately left for a fresh session/human read:

- **(b) reward-head-classifier selection (recommended first, $0):** bank the WM reward head
  (257 params) per slot as a regime classifier; select by banked reward-prediction error on
  fresh transitions. New `head_bank_select="reward_error"` + pre-registration; unlocks K>2.
- **(a) trigger hardening:** hysteresis / flip-back-on-no-improvement to push in-run
  precision 0.70 → 0.85+; cheap but K=2-bound without (b).
- **(c) mem-Brain levers (LOOP-0010):** Brain-driven gate/selection — needs Brain fine-tunes
  = **the box ping** (~4×BR50 ≈ 1.5 box-nights, $20–35). Honest gate: run (b) first so the
  Brain is only asked to do what content-addressing provably cannot.

The A-R1 headline (+0.118, 54% of slice, fully learned WHEN) is paper-ready as-is; consider
extending `g3_ar1` to n=16 for the publication number when the next batch runs anyway.
