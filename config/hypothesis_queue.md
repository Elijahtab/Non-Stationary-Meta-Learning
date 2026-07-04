# Hypothesis Queue — Neuromodulation Mechanism Family

Human-curated, ordered steering for the autoresearch trial loop. The trial harness injects
this list into each trial prompt; the agent takes the highest entry without a science verdict
(per the injected trial history) and free-picks only when the queue is exhausted.

**Agents must not edit this file** — it is outside the manifest's editable surface, so the
diff audit rejects any trial that touches it. Humans append/reorder; keep entries one line.

Priority order (from notes 0001/0002 "next levers" + pilot diagnostics):

1. Actor-only modulation — apply the mask to the actor pathway only; the critic reads raw encoder features. Motivation: baseline post-switch policy KL ≈ 0.0014 vs |value delta| ≈ 0.917 — today's shared mask barely steers the policy while whiplashing the critic (biased GAE exactly in the scored post-switch windows).
2. Two-sided gain mask — scale features in [1−α, 1+α] instead of suppress-only. A pilot trial (2026-07-04, session 20260704-010600) was accepted on the 12k-step smoke (+0.06 composite, n=1); needs a real scout/holdout verdict before it counts as evidence.
3. FiLM-style input-conditioned modulation — per-channel scale+shift conditioned on the context code (and optionally the observation encoding), rather than a static decoded mask.
4. Separate actor/critic decoders — two masks decoded from the same 8-D context code, so the Brain can steer policy and value plasticity independently.
5. Larger context dimensionality (16/32) — only if the Brain action-surface changes stay coherent end-to-end (checkpoint/interface cost must be justified; see the brief's Strong Guidance).
