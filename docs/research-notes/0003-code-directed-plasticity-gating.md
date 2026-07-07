# 0003 — Code-Directed Plasticity Gating (gradient-side mask)

**Status:** ❌ CLOSED NULL 2026-07-07 (LOOP-0006, n=8). H1 FALSIFIED: gradgate n=3 composite
0.53658 sat 0.0168 *below* the control mean — the code-directed learning-dynamics family (gradgate
+ critic_code + auxcode + adamflush + scaled/stacked variants) does not beat the frozen control at
n=8 on the scout primary. The two apparent n=3 reliability leads (critic_code, auxcode_hi) were
lucky-triple artifacts that washed out at n=8. This closes the note as **outcome 2** ("scalar-HP
control is sufficient"), now standing on BOTH the forward-modulation and learning-dynamics
families. Trail: [research-log 0007 §Outcome](../research-log/0007-2026-07-06-learning-dynamics-preregistration.md),
[RUN-20260706](../../autoresearch/live/RUN-20260706.md). Pivot → code-free family,
[note 0004](./0004-code-free-plasticity-stability.md).
_(original proposal below, preserved for the record.)_

**Status (historical):** 📋 PROPOSED 2026-07-05 — registered before any trial exists; top pick of the
LOOP-0005 brainstorm, awaiting human queue curation.
**Owner:** Elijah · **Relates to:** [LOOP-0005](../autoresearch-loops/LOOP-0005-brainstorm.md)
(sibling hypotheses + predictions), notes [0001](./0001-trainable-vs-frozen-decoder.md)/
[0002](./0002-stabilizing-trainable-decoder.md) (the closed forward-modulation family),
[research-log 0005](../research-log/0005-2026-07-05-confirmation-sweep-prediction.md) (n=8
falsifier), [spec 05](../spec/05-neuromodulation.md) (current mechanism).

## Hypothesis (one sentence)

> Decoding the Brain's context code into a **per-feature gate on the backward pass** of the
> shared CNN features — so the code steers **where the encoder learns** rather than what the
> network computes — will improve post-switch recovery over the frozen forward-mask baseline,
> because it re-aims the code's demonstrated regime information at the one causal channel that
> has ever moved the composite: learning-rate control.

## Background: why this direction, given 20 failures

Two facts from the program so far, held together:

1. **Forward-path modulation is dead.** Twenty variants of "decode the code → transform the
   features in the forward pass" (suppress masks, gains, affine, FiLM, actor-only, split
   decoders, trainable decoder ± every stabilizer, longer horizons) produced zero wins against
   the frozen baseline, culminating in the pre-registered n=8 falsifier (research-log 0005).
   The family is closed.
2. **The code is informative and the scalar-HP channel is causal.** The Brain's scalar levers
   (foremost lever 0, the inner learning rate) demonstrably drive adaptation (research-log
   0001), and the code separates conditions on the diagnostics even when scores don't
   (`mean_post_switch_neuromod_activity`: oracle 0.43 < neuromod 0.60 < random 0.77;
   `post_switch_policy_kl`: 0.0034 < 0.0072 < 0.011). The code *contains* regime information;
   multiplying features by its decode just isn't a lever that helps.

The obvious synthesis: **route the code's information into the learning-rate channel.** The
Brain currently has exactly one plasticity knob — a single global LR scalar
(`optimizer.param_groups[0]["lr"]`, [meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py)
action[0]). It cannot say "protect the representation, re-fit the readout," which is the
canonical continual-learning move for reward-contingency switches (observations don't change
in this benchmark — only reward contingencies do — so post-switch the *percepts* stay valid
while the *mappings* must change; encoder churn induced by head gradients chasing the new
contingency is pure damage). A code-decoded plasticity mask gives the Brain that vocabulary.

Biological footnote for the paper: this is also what neuromodulators mostly *do* — dopamine
and acetylcholine gate plasticity and learning rates far more than they gate activity. The
mechanism story shifts from "neuromodulation as activity routing" (tested, dead) to
"neuromodulation as plasticity control" (untested here).

## The change (mechanism, no code yet)

A flag-guarded mode in `neuromod.py` + `network.py` (default off = current frozen behavior):

- Keep the existing frozen decoder and mask parameterization exactly as-is:
  `plasticity_mask = 1 − strength·template ∈ (0,1]`, `strength = ‖code‖₂/√8` clamped.
- **Remove the mask from the forward pass.** Features feed both heads unmodified.
- Insert a **gradient gate** at the shared-feature layer: identity in the forward direction,
  and on the backward pass the incoming gradient is multiplied element-wise by
  `plasticity_mask` before it flows into the encoder (a custom autograd op — the standard
  identity-forward/scaled-backward trick).
- **Heads keep full plasticity.** The gate sits between features and encoder, so actor/critic
  head weight gradients are computed from unmodified features and are untouched; only the
  encoder's effective per-feature learning rate is modulated.

Properties preserved from the current design:

- `code = 0` ⇒ `mask = 1` ⇒ **exact baseline in both directions** (forward identity, gradients
  unchanged) — the mechanism can never hurt a zero-code policy, same safety property as today.
- Suppress-only: the Brain can *protect* features (gate → 0) but not super-charge them; global
  boosts remain lever 0's job, so the two levers compose ("raise LR globally, shield the
  regime-general features selectively").
- Brain action surface, decoder weights, benchmark, scorer: all unchanged.

## Predictions (registered)

- **H1 (primary, composite = mean post-switch window success):** plasticity-gating beats the
  3-seed `baseline_primary` anchor on `fast_switch_scout_v2` and survives the zero-regression
  holdout. To be real rather than anchor noise it needs **Δ ≥ +0.005** (the n=8 control CI
  half-width is ~0.0055; anything smaller is indistinguishable and will/should die at the
  gates). Honest prior: the base rate of this program is 20/20 nulls, so a null is the most
  likely single outcome — but this is the highest-expected-value direction available because
  it is the first hypothesis that points the code at a channel with a proven causal path.
- **H2 (mechanism):** the Brain's learned gating is *time-structured*: mask strength
  (‖code‖-driven suppression) rises in stable stretches and drops right after switches
  (open plasticity to relearn), or the converse protect-after-switch pattern — either way,
  mask stats should correlate with switch phase rather than sit flat. Flat, phase-blind
  gating predicts H1 fails via A3/R2 below.
- **H3 (manipulation check, not a result):** forward diagnostics
  (`policy_kl_vs_unmasked`, `value_delta_abs_vs_unmasked`) drop to ~0 **by construction** —
  the mask is no longer in the forward pass. The trial must log gradient-side diagnostics
  instead (mask mean/std per Brain decision; gated-vs-ungated encoder grad-norm ratio) and
  must NOT misread the zeroed forward diagnostics as "modulation is off."

## How we'll measure

Standard autoresearch trial: `fast_switch_scout_v2` n=1 screen against the amortized 3-seed
`baseline_primary` anchor, then the `fast_switch_holdout_v1` gate (seeds 11/23/37, zero
regression) on improvement. Composite only decides; hit_rate_80 as registered secondary
(prediction: ↑ or flat — less encoder churn should make recoveries more reliable, the
opposite of the trainable decoder's reliability collapse in note 0001). Editable surface:
`neuromod.py` + `network.py` only — within the manifest, no human surface-extension needed.

## Assumptions (explicit — the things most likely to be wrong)

- **A1 — encoder churn is a real post-switch cost.** The readout-vs-representation story
  assumes head gradients chasing new contingencies damage shared features enough to matter in
  the scored windows. *Risk:* with reward-only switches the encoder may already be near
  regime-invariant, leaving little to protect — in which case gating encoder plasticity can
  only slow beneficial adaptation and H1 nulls (or worsens).
- **A2 — Adam does not absorb the gate.** Adam's per-parameter second-moment normalization
  largely cancels *constant* gradient rescaling. The effect here rides on (i) the gate varying
  per Brain decision, faster than the second-moment EMA adapts, and (ii) near-zero gating
  regions, which Adam cannot undo. *Risk:* if the Brain settles on a static mask, Adam washes
  much of it out and the lever goes inert — pair this reading with H2's flat-gating signature.
- **A3 — the Brain can do credit assignment through a plasticity lever.** The lever's effect is
  delayed (it changes future updates, not the current batch). *Support:* lever 0 (global LR)
  has exactly the same delay structure and the Brain demonstrably uses it. The ‖code‖ scaling
  also gives the Brain a scalar "how much" dimension to learn first, before shaping direction.
- **A4 — diagnostics side-effect is benign.** The scorer's neuromod-activity metric reads the
  forward diagnostics, which go to ~0 under this mechanism. The composite is success-rate-only
  so scoring is unaffected, but the activity diagnostic becomes uninformative for this arm —
  known and accepted, not a bug (and no scorer edit is permitted or needed).

## Possible outcomes and how we'd read each (all publishable)

1. **Wins and survives holdout.** First positive result of the program, and a clean paper
   arc: activity-gating neuromodulation is inert (notes 0001/0002, log 0005) but the *same
   code* becomes useful when re-aimed at plasticity — neuromodulation as plasticity control,
   matching the biology. Next: ablate vs a code-blind plasticity mask (fixed random template)
   to prove the *code content* matters, not just having a gate.
2. **Null.** Sharpens the central thesis to its strongest form: the Brain's *scalar* HP
   control is not just what works — structured, code-directed refinement of the same channel
   adds nothing. "Scalar sufficiency" becomes a claim defended on both the activity and the
   plasticity side. (Distinguish via H2/A2 whether the Brain never used the lever vs used it
   flat vs Adam absorbed it.)
3. **Worse / holdout regression.** Plasticity suppression at the wrong moments is actively
   harmful (A1 inverted) — evidence that in reward-only-switch regimes the correct policy is
   maximal uniform plasticity, which the global LR lever already provides. Also informative
   for the paper's "when does structured plasticity help" framing.

## Why this could beat frozen at n=8, not just a noisy n=1 anchor

The mechanism acts deterministically at every regime switch in every seed (it changes the
learning dynamics of each post-switch window, the exact quantity the composite averages), and
it operates through the one channel with an established causal effect on recovery. It does not
rely on a lucky baseline draw, a reliability-tail artifact, or n=1 selection — the failure
modes that killed the previous family. If the effect exists, it should be systematic; the
Δ ≥ +0.005 bar in H1 encodes that expectation.

## Connection to the paper

This is the qualitative pivot the negative results have been pointing at since note 0001:
"the code contains regime information, but feature-gating cannot cash it in." If plasticity
gating cashes it in, the paper gains a positive mechanism with a biological hook; if it
nulls, the paper's negative result graduates from "one mechanism family failed" to "the
information-bearing code fails to help through *either* activity or plasticity routing —
scalar HP control is sufficient." Either way the contribution sharpens.

## Sibling hypotheses

The full LOOP-0005 candidate list (critic code-conditioning, regime-inference auxiliary loss,
optimizer-state flush, code-weighted anchoring — with per-hypothesis predictions and surface
footnotes) lives in the loop note:
[LOOP-0005-brainstorm.md](../autoresearch-loops/LOOP-0005-brainstorm.md).
