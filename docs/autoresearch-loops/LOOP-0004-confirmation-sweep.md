# LOOP-0004 — confirmation-sweep (2026-07-05)

**Goal:** Adjudicate the two n=1 "hit80 signature" leads (actor-only, gain α=0.5) at n=8
against a pre-registered prediction (research-log 0005).
**Verdict:** Falsifier fired — no arm separates from control on either endpoint; the
mechanism family is CLOSED; the n=1 signature was regression to the mean around a low
baseline draw. 3-seed amortized baseline adopted (research-log 0006).

## Hardware
Vast.ai box `171.101.231.45:54340` — 4× RTX 3060 12 GB, 256 vCPU (offered as 85), 251 GB RAM,
no volume. ($/hr not recorded — add for future loops.)

## Code state
Branch `confirmation-sweep-conditions` @ `9991565` (+ `e756bad` spawn fix, `0bd6d84` 0005
prediction, `ff55349` 3-seed baseline) — later fast-forwarded into `trainable-vs-frozen-sweep`.
Conditions: `brain_neuromod` (control), `brain_neuromod_actor_only`, `brain_neuromod_gain05`;
preset `scoutv2` (programmatic copy of the frozen spec).

## Runs
24 cells (3 arms × seeds 1–8), 6 sequential cells per GPU (1 cell/GPU — compute-bound at 99%
util, ~1.45 GB VRAM, SPS ≈ 544/stream). Relaunch 10:14Z → last group pushed 20:11Z ≈ **9.9 h**.
Auto-pushed per group: `results` branch `confirm_g{0..3}` (`81cc267` et al.).

## Obstacles
- **First launch killed all 24 cells in ~7 s**: `Cannot re-initialize CUDA in forked
  subprocess` — Linux forks the async MetaEnv workers after the parent touches CUDA; async
  Brain configs had only ever run on Windows (spawn default). Fixed with `context="spawn"`
  (`e756bad`). Load-bearing for ALL future Linux runs of benchmark-shaped configs.
- Two monitor false-positives from stale first-launch logs (mtime-window filtering fixed it);
  a between-cell VRAM dip was mistaken for idling once — cells hand off with ~30–60 s gaps.

## Statistics & verdicts (n=8 per arm, bootstrap 95% CIs)
| arm | composite | hit_rate_80 |
| --- | --- | --- |
| frozen control | **0.5534** [0.5478, 0.5588] | 0.829 [0.815, 0.846] |
| actor-only | 0.5479 [0.5437, 0.5528] | 0.829 [0.810, 0.850] |
| gain α=0.5 | 0.5460 [0.5416, 0.5514] | 0.820 [0.795, 0.845] |

No separation anywhere; per the pre-committed rule both arms closed. Control's true mean
sits ABOVE the n=1 scout anchor (0.5499) — the anchor was the artifact.

## Cost
~10 box-hours + ~0.5 h bootstrap/diagnostics; $0 agent fees (no trial agent on box).

## Sibling loops
[LOOP-0003](./LOOP-0003-v2-scout-campaign.md) (home campaign) ran concurrently; its trials
3–5 produced the same anchor-beat/holdout-reject pattern this sweep explains.

## Pickup state
Mechanism family CLOSED — do not requeue its members (queue carries the guard). The next
loop is **idea generation** (LOOP-0005): the constraint is hypothesis quality, not compute.
Box destroyed/destroyable — recreation is ~20 min via `bootstrap.sh` (Ampere needs no torch
surgery; Pascal gotcha documented in cloud-setup.md). First campaign under the new 3-seed
baseline pays a one-time ~3.8 h anchor run, then amortizes via the fingerprint cache.

## Links
`docs/research-log/0005-*.md` (prediction + outcome), `0006-*.md` (baseline fix);
`results` branch `confirm_g{0..3}`; living doc `autoresearch/live/RUN-20260705.md`;
plan `docs/plans/2026-07-05-confirmation-sweep-actor-only-gain.md`.
