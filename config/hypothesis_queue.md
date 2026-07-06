# Hypothesis Queue — Neuromodulation Mechanism Family

Human-curated, ordered steering for the autoresearch trial loop. The trial harness injects
this list into each trial prompt; the agent takes the highest entry without a science verdict
(per the injected trial history) and free-picks only when the queue is exhausted.

**Agents must not edit this file** — it is outside the manifest's editable surface, so the
diff audit rejects any trial that touches it. Humans append/reorder; keep entries one line.

## STATUS 2026-07-06: LIVE — LOOP-0005 learning-dynamics family (human-committed)

Five entries from the LOOP-0005 brainstorm (docs/autoresearch-loops/LOOP-0005-brainstorm.md,
research note 0003), ranked by confidence. **Routing for the current dual-box run:** entries
1–4 are assigned to the Vast box as master-implemented sweep conditions (branch
`autoresearch-run-20260706`) — home trial agents must NOT take them while that run is live;
**home trials start at entry 5.** Every entry routes the context code somewhere other than
forward-path feature modulation (that family is closed, see below).

1. Plasticity-gated features (gradient-side mask) — decode the code into a per-feature gate on the *backward* pass of the shared features (forward untouched; code=0 ⇒ identity; heads keep full plasticity): the Brain steers WHERE the encoder learns, re-aiming the code's proven regime info at the proven-causal LR channel. Predictions: research note 0003. [BOX, in flight]
2. Critic code-conditioning — feed the 8-D code to the critic head as an extra *input* (concatenation; policy path and shared mask unchanged): value re-fits per regime through a small fast pathway, targeting the measured post-switch critic whiplash (|ΔV|≈0.917 vs policy KL≈0.0014 → biased GAE in scored windows). [BOX, in flight]
3. Regime-inference auxiliary loss — small head predicts the Brain's code from inner features (code as teaching signal, not modulator), making representations regime-separable so post-switch readout re-mapping is faster; also tests note-0001's open H2 probe. Surface: needs ppo.py/train.py (opened 2026-07-06). [BOX, in flight]
4. Code-norm-triggered optimizer-state flush — reset Adam moments when ‖code‖ jumps; stale curvature mis-scales post-switch updates and no scalar lever touches optimizer state. Surface: needs train.py/meta_env.py (opened 2026-07-06). [BOX, in flight]
5. Code-weighted feature anchoring — per-feature anchor penalty weighted by the decoded code ("protect these features, release those") instead of the uniform scalar anchoring_weight; targeted consolidation the global lever cannot express (that lever is the control to beat). Surface: needs ppo.py/train.py (opened 2026-07-06). [HOME — take this one first]

## CLOSED 2026-07-05: forward-modulation family (research-log 0005/0006) — do not requeue

All five entries below received science verdicts on `fast_switch_scout_v2` (n=1 campaign,
session 20260704-203740) and the two strongest were refuted at n=8 (results/confirm_g0..g3):
actor-only and gain-α0.5 sit at/below the frozen control on BOTH composite and hit_rate_80
(control 0.5534 [.548,.559] / hit80 .829). The apparent n=1 "signature" was regression to the
mean around a low baseline draw — fixed going forward by the 3-seed `baseline_primary` anchor.
**Do not re-queue members of this family without a qualitatively new mechanism idea.**

1. ~~Actor-only modulation~~ — closed at n=8 (composite 0.5479, hit80 0.829 vs control 0.5534/0.829).
2. ~~Two-sided gain mask (α=0.5)~~ — closed at n=8 (composite 0.5460, hit80 0.820).
3. ~~FiLM-style input-conditioned modulation~~ — closed (v2 campaign science verdict).
4. ~~Separate actor/critic decoders~~ — closed (v2 campaign science verdict).
5. ~~Larger context dimensionality (16/32)~~ — closed (v2 campaign science verdict).
