# LOOP-0005 — brainstorm (OPEN, created 2026-07-05)

**Goal:** Generate the next hypothesis family for the neuromodulation research program. The
feature-mask mechanism family is CLOSED (LOOP-0004, research-log 0005); the queue is empty;
compute is not the constraint — idea quality is.
**Verdict:** _(open)_

## What a brainstorm session should do

Produce **3–7 queue-ready hypotheses** (one line each, mechanism + why it could beat frozen,
per `config/hypothesis_queue.md` format), plus a short research-note draft arguing the chosen
direction. No code. Output = queue entries (human commits them) + a `docs/research-notes/`
entry registering assumptions before any trial runs.

## Evidence base to reason FROM (read these, in order)

1. [AUTORESEARCH.md](../../AUTORESEARCH.md) — decisions register (what the scorer/gates
   actually measure NOW).
2. [LOOP-0004](./LOOP-0004-confirmation-sweep.md) + research-log
   [0005](../research-log/0005-2026-07-05-confirmation-sweep-prediction.md) — what closed & why.
3. Research-notes [0001](../research-notes/0001-trainable-vs-frozen-decoder.md)/
   [0002](../research-notes/0002-stabilizing-trainable-decoder.md) — the trainable-decoder
   null + the R1 co-adaptation instability findings.

## Constraints & hard-won priors (do not re-derive)

- **Closed (don't requeue):** suppress-mask variants — actor-only scope, two-sided/affine gain,
  FiLM per-channel, split actor/critic masks, context-dim expansion, channel gates, trainable
  decoder (± every LR-family stabilizer), longer horizons. 20 variants, zero wins vs frozen.
- **What consistently DOES carry signal** (research-log 0001, notes 0001): the Brain's
  **scalar-HP control** drives adaptation; and the *diagnostic* channels separate conditions
  even when scores don't — `mean_post_switch_neuromod_activity` (oracle 0.43 < neuromod 0.60 <
  random 0.77) and `post_switch_policy_kl` (oracle 0.0034 < 0.0072 < 0.011). The code
  CONTAINS regime information; multiplying features by its decode just isn't a lever that
  helps. Directions that USE the information differently are open: e.g. code → per-module
  learning-rate/plasticity gating; code → replay/memory selection or anchoring weight; code →
  world-model/dreaming modulation; code as an auxiliary prediction target (regime inference);
  code → optimizer state (momentum/trust-region) modulation. Treat these as seeds, not a queue.
- **Every hypothesis must name its predicted effect on** composite (= post-switch window
  success, nothing else) and, optionally, hit_rate_80 as a registered secondary — and should
  survive the question "why would this beat frozen at n=8, not just beat a noisy n=1 anchor?"
- Trial budget shape: ~85 min + ~$3.60 per n=1 trial; editable surface is
  `neuromod.py`/`network.py` (+ optional `signals.py`/`meta_agent.py`) — hypotheses requiring
  edits beyond that surface need a human decision first.

## Sibling loops
None yet — the next screening campaign (LOOP-0006) starts from this loop's output.

## Pickup state

**Start here in a fresh session:** read this note top to bottom, then the three evidence-base
docs. Deliver: (1) 3–7 one-line queue entries with mechanisms outside the closed family,
(2) a numbered research-note draft (next free number) stating assumptions/risks for the top
pick, (3) explicit predicted-effect statements per hypothesis. Do not write code; do not
launch anything. Human reviews → commits queue → next campaign launches under the 3-seed
baseline (first run pays its one-time ~3.8 h anchor).

## Links
`config/hypothesis_queue.md` (closed-family guard + format); `config/program_neuromod.md`
(trial brief the eventual campaign will use); spec
[05-neuromodulation.md](../spec/05-neuromodulation.md) (current mechanism, ground truth).
