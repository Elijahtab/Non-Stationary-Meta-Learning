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
