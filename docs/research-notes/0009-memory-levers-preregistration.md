# 0009 — Memory-facing meta-control: can a learned trigger recover the O2 ceiling?

**Status:** 📋 DRAFT FOR CURATION 2026-07-08 — the memory-phase brainstorm + pre-registration
sketch. **Not yet locked**: the user curates scope/predictions before any implementation
(LOOP-0010). Numbers/levers below are proposals, not commitments.
**Owner:** Elijah · **Relates to:** [note 0006](./0006-controls-axis-thesis-relocated.md) (the
+0.249 ceiling that motivates this), [note 0008](./0008-paper-skeleton.md) §Discussion,
[LOOP-0010](../autoresearch-loops/LOOP-0010-memory-levers.md) (ops note), hand-off
`docs/hand-offs/2026-07-08-loop-0008-findings-strategic-fork.md`.

## Hypothesis (one sentence, draft)

> Most of the **+0.249** composite the zero-forgetting oracle (O2) leaves on the table is
> recoverable by a Brain given **memory-facing levers** (restore trigger, snapshot gating,
> replay-source selection) — i.e. meta-control over *knowledge restoration* rather than
> plasticity — and the hard part is the **learned trigger/selection**, not the restore mechanism.

## Background: why memory, and why now

- The control ladder (note 0006) relocated the bottleneck: HP meta-control buys **+0.029**, while
  a zero-forgetting policy/world-model swap buys **+0.249** on the *same instrument*. The headroom
  is in knowledge restoration, and it is ~8× larger than the lever we have been optimizing.
- The modulation *pathway* is dead even with oracle information (oracle_code NULL; note 0006 C3)
  and plasticity loss is absent (C4). So "better plasticity meta-control" has a low ceiling by
  construction. The code half of the 15-lever interface is proven inert; the HP half is worth
  +0.029. **The interface should change.**
- O2 is an *oracle*: it snapshots at regime end and restores on regime return using ground-truth
  switch timing and identity. The scientific question is what happens when the Brain must
  **decide when to snapshot/restore and which memory to pull** — from signals, not an oracle.

## Design (pre-registered sketch — to curate)

**Arms on the existing controls axis** (same instrument, single-variable where possible):
- **static** — no controller, tuned HPs (D0 rung) — the floor.
- **HP-Brain** — the trained HP-only Brain (note 0007's replicated arm) — the +0.029 rung.
- **mem-Brain (learned)** — Brain with memory-facing levers, learned trigger + selection.
- **O2 (oracle swap)** — the +0.249 ceiling — the upper bound this arm chases.

**Candidate memory levers** (replace the dead 8-d code dims; keep the useful HP scalars):
1. **restore trigger** — a scalar/gate the Brain raises to request a restore of a stored
   snapshot (vs. oracle switch-timing in O2).
2. **snapshot gating** — when to *commit* the current policy/world-model to the memory bank
   (vs. O2's "always at regime end").
3. **replay-source selection** — which stored snapshot / regime memory to restore or replay from
   (vs. O2's ground-truth regime identity).

**Localization ablations** (which half is hard — trigger, or selection?):
- oracle-trigger + learned-selection.
- learned-trigger + oracle-selection.
- surprise-driven trigger (world-model error / value whiplash as the routing signal) vs. a pure
  Brain-lever trigger.

**Reuse:** the O2 harness (`brain_oracle_policy_swap`, per the LOOP-0008 hand-off — **verify it
merged into `Auto-Research` @ 742ba8c and locate it** before building) already implements oracle
snapshot/restore; the work is making trigger + selection *learned*, not re-implementing restore.

## Pre-registered predictions (draft — to curate)

- **P-R2a (primary, proposed bar):** mem-Brain (learned) captures **≥ 1/3 of the O2 gap over
  static** — i.e. Δ(mem-Brain − static) ≥ **+0.083** composite, p<0.01, with the
  convergence-vs-decay criterion at n≥8/arm. (⅓ is a proposal; the point is a *pre-committed*
  fraction-of-ceiling bar, not "beats HP-Brain by any margin.")
- **P-R2b (localization):** if oracle-trigger + learned-selection ≫ learned-trigger +
  oracle-selection, the trigger is the bottleneck (and vice versa) — reported regardless of P-R2a.
- **P-R2c (registered failure reading):** mem-Brain ≈ HP-Brain (≤ +0.03 over static, or estimate
  decays with n) → learned memory routing does **not** recover the ceiling on this instrument;
  the oracle gap is a genuine information/credit-assignment barrier, not a lever we were missing.
  This is a *publishable negative* that sharpens note 0008's discussion.

## Statistical honesty (up front)

- Same n≥8 + convergence-vs-decay discipline that killed three mirages and survived one real
  effect (note 0006). No arm graduates on n<8.
- A "fraction of ceiling" primary keeps the bar meaningful: beating HP-Brain by an epsilon is
  not the claim; recovering a pre-committed slice of +0.249 is.
- If mem-Brain adds levers, it adds capacity — the honest control is **HP-Brain with the same
  parameter budget / same training compute**, so a win is attributable to the *memory levers*,
  not to a bigger/longer-trained controller.

## Assumptions (explicit, with risk)

- **A1 — the ceiling is reachable without the oracle.** O2's +0.249 used ground-truth timing +
  identity. Risk: the gap is mostly *information* (knowing exactly when/what), not *mechanism* —
  in which case P-R2c fires. This is the core scientific bet.
- **A2 — snapshot/restore is cheap enough to meta-control online.** Storing policy+world-model
  snapshots per regime has memory/compute cost; the lever budget must stay within the inner
  run's footprint. Verify against the O2 harness's actual cost.
- **A3 — the interface redesign doesn't break the HP half.** Dropping the code dims and adding
  memory levers changes `neuromod.py` / the Brain action space (`BRAIN_ACTION_DIM`,
  `compose_brain_action`) — the +0.029 HP gain must survive the refactor (re-verify HP-Brain
  after the interface change).
- **A4 — trainability.** A memory-lever Brain is a harder credit-assignment problem (restore
  now, reward later). Risk: it simply won't train in 130 episodes; mitigate with the
  surprise-driven trigger as a shaped/eased variant before the pure-learned one.

## Decision matrix

- **Recovers the ceiling (P-R2a):** the positive result becomes "learned memory meta-control,"
  a much stronger paper than C1 alone — the discussion of note 0008 becomes the headline.
- **Localizes but doesn't win (P-R2b, ¬P-R2a):** we report *where* the oracle's advantage lives
  (trigger vs selection) — a mechanistic contribution even without a headline gain.
- **Fails (P-R2c):** publishable negative; the oracle gap is an information barrier; note 0008's
  memory-is-the-bottleneck framing is *confirmed and bounded* (you can't cheaply close it with
  a learned lever), which is itself a clean result.

## Open questions for curation (before this locks)

1. Is ⅓-of-ceiling the right primary bar, or should it be absolute (beat HP-Brain by ≥X) or
   ceiling-relative at a different fraction?
2. Scope: full interface redesign (drop code dims, add 3 memory levers) vs. minimal (add only a
   restore-trigger lever on top of the existing HP interface)?
3. Which trigger signal is the pre-registered primary — pure Brain-lever, or surprise-driven?
4. Compute: is this a box campaign (new training runs per arm) or does it reuse LOOP-0009's 4
   Brains as the HP-Brain control + fine-tune from them?

## Links

- Motivating ceiling: note [0006](./0006-controls-axis-thesis-relocated.md) (O2 = +0.249).
- Paper segue: note [0008](./0008-paper-skeleton.md) §Discussion / §Venue.
- Ops/campaign: [LOOP-0010](../autoresearch-loops/LOOP-0010-memory-levers.md) (draft).
- Research surface to change: `src/lifelong_learning/agents/brain/neuromod.py`,
  `src/lifelong_learning/agents/ppo/network.py` (per CLAUDE.md neuromodulation surface).
