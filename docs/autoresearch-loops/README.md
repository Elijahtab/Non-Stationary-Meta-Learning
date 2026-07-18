# Autoresearch Loops

**One note per autoresearch loop — the granular points of truth for the autoresearch program.**
[AUTORESEARCH.md](../../AUTORESEARCH.md) is the checklist and decisions register; these notes
are where the detail lives: what ran, on what hardware, what broke, what the numbers were, and
what the next loop should do first. A new session picking up autoresearch reads AUTORESEARCH.md
plus the most recent note(s) here — nothing else is required to resume.

## What is a loop?

**One queue of work driven to its verdicts** — a screening campaign, a confirmation sweep, a
shakedown batch. Concurrent loops are normal (a home campaign and a box sweep often overlap and
feed each other); use the *Sibling loops* field to cross-link rather than pretending loops
serialize. Boundary judgment calls are fine — note them in the note.

The in-flight scratchpad is `autoresearch/live/RUN-*.md` (gitignored, safe to write mid-trial).
The loop note is its curated, committed afterlife, written at loop close-out.

## Note lifecycle

- Created OPEN when a loop starts (or at close-out for short loops); finalized when verdicts land.
- The **Pickup state** section replaces handoff docs for autoresearch work (docs/handoffs is
  retired for this purpose) — it must let a fresh session start the next loop without this
  conversation.
- Append-only after finalization (dated addenda only).

## Template

```markdown
# LOOP-NNNN — <slug> (<start date>–<end date | OPEN>)

**Goal:** <one sentence>
**Verdict:** <one sentence once closed>

## Hardware
<home: GPU model; box: provider, GPUs, vCPU, $/hr if known>

## Code state
<branch @ commits; conditions/flags in play; manifest used>

## Runs
<count, wall-clock, SPS, VRAM/cell — measured, with sources (ledger session ids, sweeps/ dirs)>

## Obstacles
<what broke, root cause, fix commit>

## Statistics & verdicts
<scores with CIs where multi-seed; link ledger/results-branch paths>

## Cost
<box-hours, agent-$, benchmark-hours>

## Sibling loops
<links to concurrent loops>

## Pickup state
<what the next loop should do first — specific enough to start cold>

## Links
<ledger sessions, sweeps dirs, research-log/notes entries, living doc>
```

## Index

| Loop | Dates | One-liner | Status |
| --- | --- | --- | --- |
| [LOOP-0001](./LOOP-0001-pilot-shakedown.md) | 2026-07-04 | Pilot shakedown: 3 runs hardened the harness (surface, interpreter, science-reject paths) | closed |
| [LOOP-0002](./LOOP-0002-v1-scout-campaign.md) | 2026-07-04 | v1 (5×5) campaign: 6 mechanism rejections on a saturated instrument → scout v2 | closed |
| [LOOP-0003](./LOOP-0003-v2-scout-campaign.md) | 2026-07-04–05 | v2 (8×8) campaign: 4/5 beat the n=1 anchor, 0 survived holdout → anchor flaw exposed | closed |
| [LOOP-0004](./LOOP-0004-confirmation-sweep.md) | 2026-07-05 | n=8 box sweep: falsifier fired, mechanism family closed, 3-seed baseline adopted | closed |
| [LOOP-0005](./LOOP-0005-brainstorm.md) | 2026-07-05–06 | Brainstorm: 5 learning-dynamics candidates (top pick: plasticity gating, note 0003); all 5 human-committed to queue | closed |
| [LOOP-0006](./LOOP-0006-learning-dynamics-campaign.md) | 2026-07-06–07 | Dual-box campaign: all 5 candidates + scaled variants closed NULL at n=8; two n=3 "reliability leads" exposed as sampling artifacts | closed |
| [LOOP-0007](./LOOP-0007-brainstorm.md) | 2026-07-07 | Code-free plasticity/stability family: 0/5 at n≥8; A1 (plasticity loss) adjudicated FALSE by the registered probe; redo = third small-n mirage | closed |
| [LOOP-0008](./LOOP-0008-controls-campaign.md) | 2026-07-07–08 | Controls campaign: Brain adds ≤0 (D0), A2 dead (O1), pathway dead (oracle_code), ceiling +0.249 (O2), trained-Brain +0.029 p<0.01 confirmed (T-series) | closed |
| [LOOP-0009](./LOOP-0009-trained-brain-replication.md) | 2026-07-08–11 | Trained-Brain replication: P-R1a PASSED — pooled Δ+0.0455 (p=1.3e-6, n=16/arm), 3/4 Brains positive, converged; 4 fresh ep130 Brains archived (`results/loop9_final/`); paper C1 upgraded to across-training-seeds | closed |
| [LOOP-0010](./LOOP-0010-memory-levers.md) | 2026-07-08– | Memory-facing meta-control: can a Brain with memory levers recover a slice of the +0.249 O2 ceiling? Note 0009 draft refined by the [2026-07-09 action tree](../plans/2026-07-09-research-action-tree.md) — Wave-1 oracle rungs (G-DECOMP first) decide which memory subtree opens | brainstorm |
| [LOOP-0011](./LOOP-0011-wave1-oracle-rungs.md) | 2026-07-14 | Wave-1 oracle rungs ($0, home): ceiling re-anchored +0.2434 at eval protocol; **heads carry ~90%** (heads+enc 96%, WM null → MoWM closed, G3 opens); K=3 premium flat → scaling axis dies ([note 0012](../research-notes/0012-ceiling-decomposition.md)) | closed |
| [LOOP-0012](./LOOP-0012-g3-head-bank.md) | 2026-07-14 | G3 de-oracling ($0, home): bank ≡ swap (P-G3a); **learned trigger recovers +0.118 = 54% of the ceiling slice** (P-G3b, hit80 1.000); P-G3c selection arm **invalidated same day** (missing slot allocation — selector never ran; corrected in note 0013, re-run in LOOP-0013) | closed |
| [LOOP-0013](./LOOP-0013-selection-rung.md) | 2026-07-14–15 | Selection + trigger rungs done right ($0): **flip method converges n=16 +0.1188 (52% of slice)**; content selectors = real nulls (trunk drift, 1/55 & 9/55 restores); precision 0.69→0.94 moves nothing → **the gap to oracle is detection lag** — per-step trigger & drift-robust fingerprints are the $0 frontier; mem-Brain lever premise weakened ([note 0013](../research-notes/0013-learned-trigger-selection-gap.md)) | closed |
| [LOOP-0014](./LOOP-0014-step-trigger.md) | 2026-07-15– | Per-step trigger vs the lag wall ($0): model-free success-collapse detector (fast/slow EMAs at episode terminations), 3-iteration shadow calibration disclosed (WM per-event error = noise; final: live-equiv precision ~0.9, median lag ~400 steps); P-S1a+b both FAIL: lag cut to 280 steps but gain FELL to +0.0916 — false flips self-generate collapses (churn > lag savings); **trigger family complete, method stands at +0.1188** ([note 0013 addendum](../research-notes/0013-learned-trigger-selection-gap.md)) | closed |
| [LOOP-0015](./LOOP-0015-fingerprint-selection.md) | 2026-07-16– | Drift-robust fingerprint selection ($0): bank the full WM per slot (scoring-only, frozen feature space), live WM scores the active slot — P-FP1+FP2 both FAIL (+0.037, flip-when-should 0.16): the drift fix worked but the live reward head adapts within the trigger lag — the 'no switch' hypothesis wins even at true switches. **Selection family closed (3 probes / 3 mechanisms); the blind flip +0.1188 stands** ([note 0013 addendum](../research-notes/0013-learned-trigger-selection-gap.md)) | closed |
| [LOOP-0016](./LOOP-0016-encoder-dormancy-probe.md) | 2026-07-17 | Encoder-dormancy probe / W0c ($0, home): probe-only instrument closes C4's heads-only blind spot — **P-W0c2 PASS (valid: heads fall, composite inert) / P-W0c1 FAIL: conv1 dormancy accumulates 0.125→0.479 (τ-robust) while conv2/3 + heads fall → C4 scope-corrected in the paper; branch F's activation condition met** ([note 0014](../research-notes/0014-encoder-dormancy-probe.md)) | closed |
| [LOOP-0017](./LOOP-0017-conv1-redo-rung.md) | 2026-07-17 | conv1-ReDo rung ($0, home): branch F's first live-targeted intervention — **P-F1a FAIL: resets don't stick (f8 0.550 with resets vs 0.479 without; re-init channels re-die in-cadence), composite unmoved (−0.013, p=0.58) → conv1 dormancy is a converged input-sparsity attractor; branch F closed with both edges measured** ([note 0014 addendum](../research-notes/0014-encoder-dormancy-probe.md)) | closed |
| [LOOP-0018](./LOOP-0018-action-flip-calibration.md) | 2026-07-17 | IP-1 action-flip calibration ($0, home): **C-IP-a + C-IP-b both FAIL — IP-1 dead: control 0.9532 (mirrored policy equally competent, ~no forgetting), O2 restore hurts (−0.037, p=0.034), WM-error persistence 0 updates; port killed pre-ladder for ~2 GPU-h; IP-2 fallback = user call** ([note 0015 §Calibration outcome](../research-notes/0015-instrument-port-dynamics-regimes.md)) | closed |
