# LOOP-0016 — Encoder-dormancy probe / W0c (2026-07-17, open)

**Goal:** Close the last Wave-0 item and paper claim C4's blind spot: C4 ("plasticity loss
absent at this horizon") was measured on the actor/critic heads only; this rung measures
dormancy in the **encoder** (conv channels) with a probe-only, flag-guarded instrument riding
an otherwise-plain trained-Brain eval arm. User direction 2026-07-17 (new run from the
post-campaign state, $0/home, no box; GPU verified idle pre-launch).
**Verdict:** OPEN — scored arm pre-registered, launch pending pre-flight (tests green).

## Hardware

Home RTX 5070, $0, 8 sequential evals (~1.1 h). No box (none owned; none needed).

## Code state

`Auto-Research` @ the LOOP-0016 pre-launch commit: `dormancy_probe_interval` (default 0 = off)
in `train.py` — every N updates, a no-grad hooked forward on `s.obs_t` logs normalized
mean-abs dormant fractions at τ=0.025 (primary, = `REDO_TAU`) and τ=0.1 (descriptive) for
encoder conv1/2/3 (per-channel) and actor/critic head hidden units (per-unit, identical to
the `redo_reset_heads` statistic behind C4). No resets, no optimizer writes. Plumbed through
`eval_brain.py --dormancy_probe_interval`; batch `dorm` in `run_wave1_evals.py`; adjudicator
`score_wave1.py dorm`. **No src/scripts edits while the batch is in flight.**

## Runs

`dorm_e1..8` (batch `dorm`): standard eval protocol, LOOP-0009 seed-1 ep130 Brain, no
mechanism flags, probe interval 1. Reference archived: `loop9_s1_model_e*` (n=16) for the
composite-inertness check. Logs `evals/wave1_logs/`; idempotent.

## Statistics & verdicts

Gates **P-W0c2** (instrument integrity: heads fall replicates C4 AND composite unchanged vs
control — read FIRST) and **P-W0c1** (encoder no-accumulation: median rise from the
early-training trough ≤ +0.05 per conv layer) pre-registered in
[research-log 0014](../research-log/0014-2026-07-17-encoder-dormancy-preregistration.md)
(committed pre-results). Adjudicator: `scripts/score_wave1.py dorm`.

## Cost

$0; ~1.1 h GPU wall-clock.

## Sibling loops

[LOOP-0007](./LOOP-0007-brainstorm.md) (the heads-only probe + ReDo mirage this scopes) ·
[LOOP-0011](./LOOP-0011-wave1-oracle-rungs.md) (Wave-0/1 structure; note 0011 lists W0c as
the remaining item).

## Pickup state

Scored arm launching. On completion: `scripts/score_wave1.py dorm` → adjudicate P-W0c2 then
P-W0c1 → write note 0014 (results), update note 0011 §remaining + the action-tree status
block + paper C4 (scope correction if W0c1 fails; blind-spot-closed sentence if it passes) →
update this note + register loop-list row → report. If W0c1 FAILS: branch F gains a live
target — do NOT launch intervention runs; that is a new registration and a human call.
Strategy fork from the 2026-07-16 hand-off (paper/venue, new-instrument port, mem-Brain box
work) remains the user's call — a candidate note for the instrument port is being drafted
separately (no benchmark edits without a master decision).
