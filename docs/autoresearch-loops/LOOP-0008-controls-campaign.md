# LOOP-0008 — bottom-rung & oracle controls campaign (2026-07-07–OPEN)

**Goal:** After three mechanism families closed null, screen the pre-registered CONTROL
conditions that adjudicate the strategic fork (write up the null vs pivot the instrument):
D0 `brain_constant_action` (does the Brain add anything over tuned static HPs?), O1
`brain_critic_lr_oracle` (upper bound on any Brain-learned critic damp), O2
`brain_oracle_policy_swap` (the instrument's zero-forgetting ceiling), and optionally
`brain_oracle_code` (decouples "code was noise" from "pathway is dead").
**Verdict:** _(open)_

## Hardware
Box: Vast.ai `76.67.137.57:26061`, 4× RTX 3060 12 GB — 1 cell/GPU, 4 cells/wave, ~85 min/wave.
Home 5070 idle (held).

## Code state
Run branch `autoresearch-run-20260706` @ `cfa20d5` (+`b8c3e23` notes): conditions
`brain_constant_action` / `brain_critic_lr_oracle` / `brain_oracle_policy_swap`, all
flag-guarded default-off; results-push race fix (retry on fresh tip, `RESULTS_PUSH_FAILED`
marker); probe durability (`probes/<run>_probes.json` in pushed sweep dirs). Box on `cfa20d5`.
Pre-registration: [note 0005](../research-notes/0005-untested-controller-bottom-rung-oracle.md)
(committed BEFORE any screening results).

## Runs
Wave plan (out dirs `confirm_d0_s*` / `confirm_o1_s*` / `confirm_o2_s*`):
W11 D0 s1–4 [launched 19:25Z] → W12 D0 s5–8 → **GATE D0 n=8** → W13/W14 O1 s1–8 →
**GATE O1 n=8** → W15 O2 s1–4 (ceiling read, no fork) → W16/W17 `brain_oracle_code` s1–8
(pending user confirm). Gate log: `autoresearch/live/RUN-20260706.md`.

## Statistics & verdicts
Comparison anchor: LOOP-0004 frozen control n=8 — composite 0.5534 [0.5478, 0.5588],
hit80 0.829 [0.815, 0.846]. D0/O1 gated by the reliability fork at n≥8 ONLY; O2 is a
diagnostic ceiling (report topline − control gap; no fork). Decision matrix in note 0005.

## Sibling loops
[LOOP-0007](./LOOP-0007-brainstorm.md) (closed null; its retrospective finding — the scout
Brain is effectively untrained — motivates this whole campaign).

## Pickup state
Read note 0005 (predictions + decision matrix) and the living doc's newest Master Decisions
entries. If this session's watcher is dead: check `sweeps/confirm_d0_*` / `confirm_o1_*` /
`confirm_o2_*` summaries on the box (or `origin/results`), gate per note 0005, launch the
next wave in the sequence above, re-arm ONE watcher. Never idle the box; box destroy is
human-only. After all gates: assemble the note-0005 decision matrix and surface the fork
(write-up vs instrument pivot) to the user — do NOT auto-start a new mechanism family.

## Links
Note 0005 · session plan `docs/plans/2026-07-07-session-plan-redo-endgame-oracle-rung.md` ·
living doc `autoresearch/live/RUN-20260706.md` · results branch `confirm_d0_*`/`confirm_o1_*`/
`confirm_o2_*` · paper §4.3 Table 2 (the founding claim D0 re-tests).
