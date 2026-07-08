# LOOP-0010 — memory-facing meta-control (MoWM × Brain / interface redesign)

**Goal:** Test whether a Brain given **memory levers** (restore trigger, snapshot gating,
replay-source selection) recovers a pre-committed slice of the **+0.249** zero-forgetting ceiling
(O2) that HP-only meta-control (+0.029) leaves on the table — meta-control over *knowledge
restoration* rather than plasticity.
**Verdict:** 🧠 BRAINSTORM / PRE-OPEN — pre-registration **drafted** (note 0009), **awaiting user
curation**; nothing designed-final, nothing launched. Opens only after the LOOP-0009 gate and a
curation pass.

## Why this loop exists

The LOOP-0008 control ladder relocated the catastrophic-forgetting bottleneck from learning
dynamics to knowledge restoration (note 0006): the modulation pathway is inert even with oracle
information, plasticity loss is absent, and a zero-forgetting policy/world-model swap buys ~8×
what HP meta-control buys. The natural next lever is explicit memory — this loop's subject.

## Relationship to the strategic fork

Per the 2026-07-08 user decision ("replicate + draft both"), this loop is the **draft half**: its
pre-registration is written now so it is ready if the LOOP-0009 replication gate (note 0007)
resolves toward the memory phase. The final write-up-vs-memory call is made at the note-0007 gate.
- **LOOP-0009 replicates (C1 strong):** the paper (note 0008) can ship; LOOP-0010 becomes the
  *next* paper / follow-on.
- **LOOP-0009 fails (C1 withdrawn):** LOOP-0010 becomes the primary route to a positive result.

## Design & statistics

All in the pre-registration draft [note 0009](../research-notes/0009-memory-levers-preregistration.md):
arms (static / HP-Brain / mem-Brain / O2 ceiling), candidate memory levers, localization
ablations, pre-registered predictions (P-R2a ⅓-of-ceiling primary, P-R2b localization, P-R2c
registered negative), and the equal-budget honesty control. Same n≥8 + convergence-vs-decay
discipline as every graduated result.

## Code state (nothing built)

Research surface to change: `src/lifelong_learning/agents/brain/neuromod.py` +
`src/lifelong_learning/agents/ppo/network.py` (drop the proven-dead 8-d code dims; add
memory-facing levers; `BRAIN_ACTION_DIM` / `compose_brain_action` change). **Reuse target:** the
O2 oracle snapshot/restore harness (`brain_oracle_policy_swap`) — *verify it merged into
`Auto-Research` @ 742ba8c and locate it* before any build. The refactor must preserve the +0.029
HP-Brain gain (note 0009 A3).

## Cost (rough, pre-scope)

If a box campaign: new training runs per arm (mem-Brain harder to train — note 0009 A4), likely a
multi-night sweep. Open question (note 0009 §Curation Q4): reuse LOOP-0009's 4 Brains as the
HP-Brain control / fine-tune source vs. train fresh. Sized after curation.

## Sibling loops

[LOOP-0008](./LOOP-0008-controls-campaign.md) (produced the +0.249 ceiling this chases) ·
[LOOP-0009](./LOOP-0009-trained-brain-replication.md) (the HP-Brain replication whose gate
decides whether this loop is next-paper or primary-result).

## Pickup state

1. **User curates note 0009** — the open questions (primary bar, scope, trigger signal, compute
   model) resolve before the pre-registration locks.
2. **Locate + verify the O2 harness** in the merged tree; measure its snapshot/restore cost
   (note 0009 A2).
3. **Design-final the interface change** (which levers, parameter/compute budget matched to
   HP-Brain) and lock note 0009.
4. **Then** implement + pre-register commit + launch — only after the LOOP-0009 gate.

## Links

Pre-registration draft: [note 0009](../research-notes/0009-memory-levers-preregistration.md) ·
motivating ceiling: [note 0006](../research-notes/0006-controls-axis-thesis-relocated.md) ·
paper segue: [note 0008](../research-notes/0008-paper-skeleton.md) · hand-off
`docs/hand-offs/2026-07-08-loop-0008-findings-strategic-fork.md` ·
[AUTORESEARCH.md](../../AUTORESEARCH.md) register.
