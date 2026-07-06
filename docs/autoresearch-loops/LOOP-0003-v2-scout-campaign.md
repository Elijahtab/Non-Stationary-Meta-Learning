# LOOP-0003 — v2-scout-campaign (2026-07-04 → 2026-07-05)

**Goal:** Retest the mechanism family on `fast_switch_scout_v2` (8×8) with real headroom.
**Verdict:** 4 of 5 trials beat the n=1 scout anchor; all died at the 3-seed holdout on
noise-scale deltas — the pattern that exposed the n=1-anchor flaw (fixed by research-log 0006).

## Hardware
Home RTX 5070 (~85%+ util through ~28 h incl. three ~3.3 h holdout runs).

## Code state
`trainable-vs-frozen-sweep` @ `32c8676` (scout v2 + benchmark-scoped history);
manifest primary = scout_v2; ledger session `20260704-203740`.

## Runs
Baseline: composite 0.5499 / hit80 0.786 (n=1, ~76 min — later shown a low draw; true control
mean 0.5534/0.829 at n=8). 5 trials, each ~85 min + holdout when scout was beaten.

## Obstacles
None mechanical — the loop ran clean end-to-end (post-remediation harness). The scientific
obstacle WAS the loop's economics: 4 scout-passes ≈ coin flips against a low anchor bought
~13 h of holdout compute adjudicating noise.

## Statistics & verdicts
| trial | mechanism | scout (vs 0.5499) | holdout (vs 0.5477) | verdict |
| --- | --- | --- | --- | --- |
| 1 | actor-only (retest) | 0.5505 ✓ | 0.5475 | holdout_regressed |
| 2 | gain α=0.5 (retest) | 0.5491 | — | primary_score_did_not_improve |
| 3 | FiLM (retest) | 0.5620 ✓ | 0.5459 | holdout_regressed |
| 4 | split masks (retest) | 0.5552 ✓ | 0.5427 | holdout_regressed |
| 5 | context-16 (retest) | 0.5525 ✓ | 0.5425 | holdout_regressed |

Queue discipline + benchmark-scoped verdicts worked exactly as designed (each trial named its
retest status and took the next unresolved item).

## Cost
~$18 agent fees; ~28 h home GPU.

## Sibling loops
[LOOP-0004](./LOOP-0004-confirmation-sweep.md) ran concurrently on the box, adjudicating this
loop's trials 1–2 while its trials 3–5 were still running.

## Pickup state
(historical) Family closed by LOOP-0004; see its pickup state.

## Links
Ledger session `20260704-203740`; `docs/research-log/0004-*.md` (outcome section);
`autoresearch/live/RUN-20260705.md` (home gate log).
