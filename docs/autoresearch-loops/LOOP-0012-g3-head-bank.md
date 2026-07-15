# LOOP-0012 — G3 head-bank: the de-oracling screen (2026-07-14–OPEN)

**Goal:** Turn the Wave-1 routing verdict (heads carry ~90% of the +0.243 ceiling, note 0012)
into a *method*: a K-slot head bank whose trigger (WHEN) and selection (WHICH) are learned
instead of oracle — the candidate learned-O2 result.
**Verdict:** OPEN — screen launched 2026-07-14 evening.

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

Screen in flight (launched 2026-07-14 ~17:45 local). On completion:
`PYTHONPATH=src myenv/Scripts/python.exe scripts/score_wave1.py g3` → adjudicate
P-G3a/b/c → update this note + a results research note → ping the user (gate resolution =
"interesting"). If P-G3b+c pass: next is the LOOP-0010 mem-Brain interface (B1 prune +
memory levers, Brain fine-tunes) — **that is the box ping** (~4×BR50 ≈ 1.5 box-nights,
$20–35). If P-G3b fails: the learned-trigger family dies; W2A falls back to self-inferred
selection at oracle timing + the paper documents the trigger negative.
