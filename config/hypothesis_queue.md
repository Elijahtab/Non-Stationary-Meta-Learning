# Hypothesis Queue — Neuromodulation Mechanism Family

Human-curated, ordered steering for the autoresearch trial loop. The trial harness injects
this list into each trial prompt; the agent takes the highest entry without a science verdict
(per the injected trial history) and free-picks only when the queue is exhausted.

**Agents must not edit this file** — it is outside the manifest's editable surface, so the
diff audit rejects any trial that touches it. Humans append/reorder; keep entries one line.

## STATUS 2026-07-05: FAMILY CLOSED (research-log 0005/0006)

All five entries below received science verdicts on `fast_switch_scout_v2` (n=1 campaign,
session 20260704-203740) and the two strongest were refuted at n=8 (results/confirm_g0..g3):
actor-only and gain-α0.5 sit at/below the frozen control on BOTH composite and hit_rate_80
(control 0.5534 [.548,.559] / hit80 .829). The apparent n=1 "signature" was regression to the
mean around a low baseline draw — fixed going forward by the 3-seed `baseline_primary` anchor.
**Do not re-queue members of this family without a qualitatively new mechanism idea.**
The queue is EMPTY of live priorities; new entries need human curation.

Priority order (from notes 0001/0002 "next levers" + pilot diagnostics):

1. Actor-only modulation — apply the mask to the actor pathway only; the critic reads raw encoder features. Motivation: baseline post-switch policy KL ≈ 0.0014 vs |value delta| ≈ 0.917 — today's shared mask barely steers the policy while whiplashing the critic (biased GAE exactly in the scored post-switch windows).
2. Two-sided gain mask — scale features in [1−α, 1+α] instead of suppress-only. Untested on the real benchmark: a 12k-step n=1 pilot smoke (2026-07-04, session 20260704-010600, pilot ledger) once accepted a tanh variant, but that trial's prompt was corrupted by a since-fixed encoding bug and the diff was reverted — treat it as a weak prior only.
3. FiLM-style input-conditioned modulation — per-channel scale+shift conditioned on the context code (and optionally the observation encoding), rather than a static decoded mask.
4. Separate actor/critic decoders — two masks decoded from the same 8-D context code, so the Brain can steer policy and value plasticity independently.
5. Larger context dimensionality (16/32) — only if the Brain action-surface changes stay coherent end-to-end (checkpoint/interface cost must be justified; see the brief's Strong Guidance).
