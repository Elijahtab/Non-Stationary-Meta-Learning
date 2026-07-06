# AUTORESEARCH.md Restructure: Checklist + Loop Notes

**Goal:** Replace the 439-line AUTORESEARCH.md with a short two-part operating document —
(1) a current-decisions register, (2) an indexed pointer list to per-loop notes — and introduce
`docs/autoresearch-loops/` as the granular points of truth, one note per autoresearch loop.
**Status:** proposed 2026-07-05 · **PARKED** in `autoresearch/parked/` (home campaign mid-trial;
installs to `docs/plans/` at execution time)

## Summary

AUTORESEARCH.md currently mixes contract, stale metrics text, cloud ops that duplicate
[cloud-setup.md](../../docs/plans/cloud-setup.md), a brainstorm superseded by the parked
`/autoresearch-run` skill, and two stale state-pointer sections. It becomes the **checklist for
how we run autoresearch** (per the user's framing, verbatim in its header): decisions in force +
loop-note index + the run procedure. Everything granular — box specs, wall-clocks, obstacles,
box statistics, verdicts, pickup state — moves to one note per loop in `docs/autoresearch-loops/`,
which Claude must read (along with AUTORESEARCH.md) before doing any autoresearch work. All
drafting can happen NOW against parked paths; only final placement waits for the campaign to end.

## Decisions & answers

| Question | Answer |
| --- | --- |
| Loop-note location | **New `docs/autoresearch-loops/`**, `LOOP-NNNN-<slug>.md` + README/template |
| Displaced content | **Redistribute + archive**: cloud ops → cloud-setup.md; trial-contract detail → spec/07; brainstorm + stale state sections → `docs/archive/AUTORESEARCH-v1.md` verbatim |
| Backfill | **All 4 loops**: pilot shakedown, v1 scout campaign, v2 campaign (finalize at close), box confirmation sweep |

Flagged assumptions (veto anytime):

- **A1 — Header language** states, near-verbatim from the user: *"This document is the checklist
  for how we run autoresearch. Loop notes are the granular points of truth we use to pick up new
  autoresearch loops. Before doing anything, read this document and the relevant loop notes."*
  A matching one-line pointer goes into CLAUDE.md's docs table and the /orient skill's fallback list.
- **A2 — "One loop" =** one campaign/sweep batch driven to a verdict set: (0001) pilot shakedown
  runs, (0002) v1 scout campaign incl. its 429-split sessions, (0003) v2 campaign, (0004) box
  confirmation sweep. The living doc (`autoresearch/live/RUN-*.md`) remains the *in-flight*
  scratch state; the loop note is its committed, curated afterlife.
- **A3 — Loop-note template fields:** loop id/dates/goal · hardware (home + box: GPU model,
  vCPU, provider, $/hr) · code state (branch/commits/conditions) · runs executed (count,
  wall-clock, SPS, VRAM) · obstacles hit & fixes (with commit refs) · statistics & verdicts
  (with CIs where multi-seed) · cost (box-hours, agent-$) · **sibling loops** (cross-links to
  concurrent loops, per A7) · pickup state (what the next loop should do first — this section
  REPLACES handoff docs for autoresearch work, per A8) · links (ledger session ids, sweeps
  dirs, research-log/notes entries).
- **A4 — The decisions register replaces the stale Metrics/Composite section** (currently
  [AUTORESEARCH.md:169–211](../../AUTORESEARCH.md) — still describes the 50/25/25 formula the
  scorer dropped 2026-06-18). Register rows link to source-of-truth (code + research-log) rather
  than restating formulas: composite = `mean_post_switch_window_success_rate`
  (`benchmarking.py::_compute_composite_score`, research-log 0001), scout→holdout gate setup
  (scout_v2 n=1 strict-improve → holdout_v1 n=3 zero-tolerance; research-log 0004), surfaces
  (manifest), trial agent + wrappers, `{python}` rule, STOP/journal kill-safety, budget rails.
  The pending two-criterion-gate question stays an explicit "open decision" row until
  research-log 0005 is adjudicated.
- **A5 — Same-window convergence:** execution bundles the other parked installs (skill →
  `.claude/skills/autoresearch-run/`, confirmation plan + handoff → `docs/`), research-log 0005
  Outcome (if sweep adjudicated), and the `confirmation-sweep-conditions` merge.
- **A6 — Register-update rule (anti-rot):** the new AUTORESEARCH.md checklist carries an
  explicit line: *"any commit that changes a register truth (scorer, gates, benchmark
  versions, surfaces, trial agent, budget rails) MUST update the corresponding register row
  in the same commit."* The register is a map, not a mirror — but a map that must be redrawn
  the moment the territory moves. (This is the fix-class for the stale-50/25/25 incident.)
- **A7 — Loop boundary definition:** a loop = **one queue of work driven to its verdicts**
  (a screening campaign, a confirmation sweep, a shakedown batch). Concurrent loops are
  normal (the v2 campaign and the confirmation sweep overlapped and fed each other); the
  loop-note template gets a **"Sibling loops"** field with cross-links instead of pretending
  loops serialize. Judgment at the edges is accepted and noted in the note itself.
- **A8 — Handoffs retired for autoresearch work:** the loop note's **pickup-state** section
  subsumes the handoff doc's job. `docs/handoffs/` stays for history (a README note marks it
  superseded for autoresearch; the 2026-07-05 dual-box handoff installs with a header line
  pointing at the loop notes as the living successor). research-notes (paper) and
  research-log (methodology decisions) remain distinct — different audiences.

## Stages

### Stage 0 — Draft everything now (parked paths; safe mid-campaign)

- Consolidate scratchpad-parked artifacts into `autoresearch/parked/` (durability — scratchpads
  aren't guaranteed to survive).
- Draft in `autoresearch/parked/restructure/`: the new `AUTORESEARCH.md`; `docs/autoresearch-loops/README.md`
  (charter + template); `LOOP-0001-pilot-shakedown.md`, `LOOP-0002-v1-scout-campaign.md`,
  `LOOP-0003-v2-scout-campaign.md` (open until campaign closes), `LOOP-0004-confirmation-sweep.md`
  (open until sweep adjudicated). Sources: `autoresearch/trial_results*.jsonl`,
  `autoresearch/live/RUN-20260705.md`, handoffs, research-log 0004/0005, this session's fixes
  (429s, fork/spawn crash `e756bad`, mojibake, cmd.exe interpreter — each becomes an "obstacles" entry).
- Draft the cloud-setup.md merge block (thread caps, packing densities incl. tonight's
  3060 numbers ~1.5 GB/cell @ 99% util, right-sizing) and the spec/07 contract fold.

**Verify:** drafts complete; every number traceable to a ledger/log path cited inline.

### Stage 1 — Install (gate: `autoresearch/v2_campaign1.exitcode` exists)

- Move drafts into place: new [AUTORESEARCH.md](../../AUTORESEARCH.md),
  `docs/autoresearch-loops/*`, `docs/archive/AUTORESEARCH-v1.md` (verbatim old file),
  [cloud-setup.md](../../docs/plans/cloud-setup.md) merge,
  [spec/07](../../docs/spec/07-benchmarking-and-autoresearch.md) fold.
- Fix the two section-level inbound refs:
  [research-notes/0002:83](../../docs/research-notes/0002-stabilizing-trainable-decoder.md)
  ("→ cloud notes") and [handoffs/2026-07-01:57](../../docs/handoffs/2026-07-01-trainable-vs-frozen-cloud-sweep.md)
  ("→ Right-sizing") → point at cloud-setup.md.
- CLAUDE.md: add `docs/autoresearch-loops/` row + the read-before-acting line (A1).
- `docs/handoffs/README.md` (new, 3 lines): handoffs superseded for autoresearch work by
  loop-note pickup state (A8); header pointer added to the installed 2026-07-05 handoff.
- New AUTORESEARCH.md checklist includes the A6 same-commit register rule as a literal
  checklist item.
- Finalize LOOP-0003 (trial-5 verdict + session close data) and LOOP-0004 (sweep CIs + 0005
  Outcome) if available.
- Bundle A5's converging installs. Merge `confirmation-sweep-conditions` — **watch for
  conflicts in `network.py`/`neuromod.py` if the campaign's trial 5 was ACCEPTED** (both the
  branch and an accepted trial touch those files; resolve manually, agent-diff wins on
  mechanism, branch wins on flags).

**Verify:** `grep -rn "AUTORESEARCH" docs/ scripts/` shows no dangling section refs; new
AUTORESEARCH.md ≤ ~120 lines; /orient dry-read makes sense; full test suite green post-merge.

### Stage 2 — Commit, push, and re-point the live loop

- Commit in two pieces: (a) restructure + loop notes + archive, (b) branch merge + parked
  installs. Push.
- Update `autoresearch/handoffs/2026-07-05-autoresearch-dual-box.md` closing note: future
  pickups start from AUTORESEARCH.md + loop notes (the handoff becomes historical).

**Verify:** fresh-session smoke: "read AUTORESEARCH.md and the latest loop note, state the next
action" produces the correct pickup (the LOOP-0004 pickup-state line).

## Files touched

| File | Change |
| --- | --- |
| AUTORESEARCH.md | rewritten: header charter + decisions register + checklist + loop index (~120 lines) |
| docs/autoresearch-loops/README.md + LOOP-0001..0004 | new: charter, template, 4 backfilled notes |
| docs/archive/AUTORESEARCH-v1.md | new: verbatim old document |
| docs/plans/cloud-setup.md | absorbs cloud-ops sections (+3060 measurements) |
| docs/spec/07-benchmarking-and-autoresearch.md | absorbs trial-contract detail; link updates |
| docs/research-notes/0002-*.md, docs/handoffs/2026-07-01-*.md | 2 one-line ref fixes |
| CLAUDE.md | docs-table row + read-before-acting pointer |
| (bundled) .claude/skills/autoresearch-run/SKILL.md, docs/plans/2026-07-05-confirmation-sweep-*.md, docs/handoffs/2026-07-05-*.md | parked installs |

## Risks & alternatives

- **Trial 5 acceptance conflict** (see Stage 1) — known, handled at merge time.
- **Losing operational nuggets in the archive**: mitigated by explicitly redistributing the
  load-bearing ones (thread caps, packing, durability rules) into cloud-setup.md and keeping the
  archive verbatim rather than deleting.
- **Two sources of truth drift (register vs code)**: register rows carry links, not formulas —
  the code/research-log stay canonical; the register is a map. (Rejected alternative: restating
  formulas in the register — exactly how the current staleness happened.)

## Open questions

- LOOP-0004's final content waits on the sweep verdicts (hours away).
- Whether the two-criterion gate (composite non-inferiority + hit80 superiority) gets adopted is
  research-log 0005's outcome, not this restructure — the register just tracks it as open.
