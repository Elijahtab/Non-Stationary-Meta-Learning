# LOOP-0016 — Encoder-dormancy probe / W0c (2026-07-17, closed same-day)

**Goal:** Close the last Wave-0 item and paper claim C4's blind spot: C4 ("plasticity loss
absent at this horizon") was measured on the actor/critic heads only; this rung measures
dormancy in the **encoder** (conv channels) with a probe-only, flag-guarded instrument riding
an otherwise-plain trained-Brain eval arm. User direction 2026-07-17 (new run from the
post-campaign state, $0/home, no box; GPU verified idle pre-launch).
**Verdict:** CLOSED 2026-07-17 (8/8 clean, ~61 min GPU) — **P-W0c2 PASS** (heads fall
replicates C4: actor 0.789→0.516, critic 0.788→0.605; composite inert +0.0115, p=0.586) /
**P-W0c1 FAIL, decisively:** first-conv-layer dormancy **accumulates 0.125→0.479** (median
rise vs early trough +0.318, 6× the bar; robust at τ=0.1) while conv2/3 stay flat. Extension
list empty — final at n=8. **Paper claim C4 scope-corrected** (heads-only; five sites in
`main.tex` updated); branch F's activation condition (rising dormancy) is now met —
intervention runs remain a new registration + human call. Full anatomy:
[note 0014](../research-notes/0014-encoder-dormancy-probe.md).

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

Loop CLOSED; Wave 0 is now fully done. All records updated in the close-out commit (note
0014, note 0011 §remaining, note 0008 C4 row, action-tree status block, paper `main.tex`
×5 sites, register + indexes). Open items, all human calls, none pre-approved:

1. **Branch F is live-targeted for the first time** — conv1 dormancy rises where everything
   else falls. A probe-guided conv1 intervention rung (ReDo-on-conv1 / shrink-and-perturb,
   $0 home evals) is the natural next loop IF the user wants it: new pre-registration
   required (log 0014's reading), and note 0014's caveat stands — no evidence yet that the
   accumulation costs composite.
2. **Instrument port** (strategy fork option 2): candidates + calibration gates drafted in
   [note 0015](../research-notes/0015-instrument-port-dynamics-regimes.md) — needs the
   master-level env-edit decision.
3. Paper v2 still needs its first Overleaf compile (now including the C4 scope-correction
   edits); venue call open.
4. Vast console: LOOP-0009 box destroy was never confirmed (storage may still bill) —
   human-only check.
