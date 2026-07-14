# Hand-off: LOOP-0008 controls campaign — all findings + the write-up-vs-holistic-research fork (2026-07-08)

## Goal
LOOP-0008 (bottom-rung + oracle controls campaign) is **complete and closed** — one overnight
session answered every pre-registered question the three-family null era left open. This
hand-off transfers the findings and frames the strategic decision the next session must drive:
**(a) write up and submit now, or (b) run one more, more-holistic research phase first**
(MoWM/memory × Brain combination, or a redesign of the 15-lever Brain interface).

**Orient first:** run `/orient`, then read the research notes **0001–0006 in order**
(`docs/research-notes/`) — they are the scientific narrative of the whole program and are
deliberately written to be read as a sequence. ⚠️ Notes 0005/0006 currently live on the run
branch `autoresearch-run-20260706` (worktree `..\LL-run-20260706`), not yet merged to
`Auto-Research` — read them there (`git show autoresearch-run-20260706:docs/research-notes/0006-controls-axis-thesis-relocated.md`).
Then read `AUTORESEARCH.md` + `docs/autoresearch-loops/README.md` (index of all 8 loops) and
the LOOP-0008 note's Pickup state.

## State
- **Branches:** `Auto-Research` @ `ec806bb` (LOOP-0008 closed in register/index); run branch
  `autoresearch-run-20260706` @ `871bdc1` (notes 0005/0006, all conditions). Both pushed.
  Remote moved to `Elijahtab/Non-Stationary-Meta-Learning` (old URL redirects). **Run worktree
  merge to main is still pending** (close-out step 9).
- **Data:** all box results on `origin/results` (`confirm_d0_s*`, `confirm_o1_s*`,
  `confirm_o2_s*`, `confirm_oc_s*`, `confirm_redo_*`, plus LOOP-0006/0007 dirs). T-series
  trained-Brain evals (96 runs) in local `evals/t1_*` on the home machine (gitignored — local
  only). Living doc / full gate log: `autoresearch/live/RUN-20260706.md` (gitignored).
- **Hardware: both idle.** Vast box `ssh -p 26061 root@76.67.137.57` (4× RTX 3060) — IDLE,
  billing, destroy is human-only; everything on it is archived off-disk, so destroy is now
  safe if chosen. Home RTX 5070 idle.
- **Uncommitted:** `docs/hand-offs/` (this file + prior hand-offs; untracked by convention).

## Major findings (each n≥8, pre-registered; full detail in note 0006)
Scout control reference: composite 0.5534 [0.5478, 0.5588], hit80 0.829 [0.815, 0.846].
1. **The scout instrument never contained a trained controller.** Scored cells train the Brain
   from scratch for only 4 episodes (~160 gradient steps) — every LOOP-0002→0007 screen tested
   mechanisms under an effectively untrained, noise-emitting Brain (adaptivity probe: note 0005).
2. **D0 (no controller, tuned static HPs): composite 0.5608, hit80 0.786** — the untrained
   Brain adds ≤0 composite; its action jitter was buying ~4 pts of hit80 reliability.
3. **O1 (oracle-timed critic-LR damp): 0.5451/0.806 — REJECT.** Perfect timing upper-bounds any
   learned damp, so A2 is dead: the post-switch critic "whiplash" is functional, not a pathology.
4. **oracle_code (ground-truth regime code → mask): 0.5481/0.826 — NULL.** The modulation
   pathway is inert even with perfect information → the forward-modulation family null is about
   the pathway and stands at full strength despite finding (1).
5. **O2 (zero-forgetting policy-swap ceiling): composite 0.8022 (sd 0.009) — +0.249 headroom.**
   The metric was never blind and there was never a ceiling; the bottleneck is knowledge
   restoration (policy/world-model memory), not learning dynamics.
6. **The thesis is TRUE but small: trained Brain (March ep130 checkpoint,
   `runs/brain_2_regimes_8x8_neuromod_20260315-180839`) vs random-init Brain at matched eval
   protocol: +0.0292 composite, t≈3.0, p<0.01, n=32/arm** — estimate converged across
   n=8/16/32 (mirages decay; this didn't). ~12% of the O2 ceiling. hit80 identical between arms.
7. **Cross-protocol comparisons are invalid** (measured: eval protocol shifts composite −0.045,
   hit80 +0.18 vs box protocol at identical HPs — 16-env smoothing inflates hit80 hugely).
8. **Paper-era Table 2 audit:** static-baseline row (0.4638, "never reaches 80%") contradicted
   (weak-baseline artifact — D0 hits 80% on 79% of switches); trained-Brain direction confirmed
   at ~10× smaller magnitude; the Brain+neuromod increment was noise.
9. **A1 (plasticity loss) is FALSE** at this horizon — dormant fraction *falls* 0.9→0.35
   (recovered probe, note 0005 addendum); LOOP-0007 closed 0/5 with redo as the third
   reproduced small-n mirage (n=6 pass → died at n=10 + inert probe + failed holdout).

## Gotchas
- **n≥8 discipline + convergence-vs-decay test** — three mirages died to it; the one real
  effect survived it. Non-negotiable for anything new.
- The T-series scoring adapter lived in the (ephemeral) session scratchpad — its method is
  documented in note 0006 Provenance; recreate as `scripts/score_eval_dir.py` if eval-protocol
  scoring is needed again (stages eval logs into run layout, then frozen `score_brain_run`).
- Trained-Brain evals must run at the checkpoint's protocol (16 inner envs,
  `decision_interval=1`) and be compared only within-protocol (finding 7).
- Business-hours commit blocker: rescinded 2026-07-07. Never-idle-box directive: deliberately
  paused at queue exhaustion — do not auto-launch waves without a strategy decision.

## Next step (the decision this session must drive)
**Decide: (a) write up now, or (b) one more holistic research phase first.** Frame:
1. **(a) Write-up path:** note 0006 contains the paper skeleton (positive claim + ceiling bound
   + mechanism results + methods contribution). Venue calibration from the closing session:
   workshop-ready today; CoLLAs-tier needs the one mandatory fix — the +0.029 rests on ONE
   Brain training seed → train ≥4 fresh Brains (40–50 episodes, ~15 h each, box does 4 in
   parallel ≈ 2 nights) and replicate the matched contrast across training seeds.
2. **(b) Holistic phase** — go coarser than the LOOP-0005/0007 mechanism granularity, aimed at
   the +0.249 the ceiling exposed:
   - **MoWM × Brain:** give the system explicit memory restoration (the paper's §4.2 MoWM
     lineage). The O2 harness (`brain_oracle_policy_swap` in `scripts/run_seed_sweep.py`, run
     branch) already implements oracle snapshot/restore — the research question is making the
     *trigger and selection* learned (surprise-based routing, or a Brain lever that requests a
     restore) instead of oracle.
   - **15-lever interface redesign:** the Brain's action space (7 scalar HPs + 8-d code,
     `neuromod.py`) was fixed through the whole program; the code half is now proven dead
     (finding 4) and the HP half is worth +0.029. Candidate redesign: drop the code dims,
     add memory-facing levers (restore trigger, snapshot gating, replay source selection) —
     i.e., meta-control over MEMORY rather than plasticity.
   - These two converge: the natural experiment is "Brain with memory levers vs oracle swap
     (ceiling) vs static" on the existing controls axis.
3. Either path: decide box destroy-vs-keep (idle billing); if (b), a brainstorm note
   (LOOP-0009) with pre-registered predictions comes before any implementation.

## References
- **Consolidated results:** note `0006-controls-axis-thesis-relocated.md` · pre-registration +
  probe/adaptivity findings: note `0005-untested-controller-bottom-rung-oracle.md` (both on
  run branch).
- Loop index: `docs/autoresearch-loops/README.md` (LOOP-0001→0008 one-liners) · LOOP-0008 note
  Pickup state · register: `AUTORESEARCH.md`.
- Gate-by-gate log: `autoresearch/live/RUN-20260706.md` (2026-07-07 18:55Z → 2026-07-08).
- Paper-era claims being audited: `docs/references/Combating Catastrophic Forgetting.pdf`
  §4.2–4.3 + Table 2.
- Prior hand-offs: `2026-07-07-loop-0007-screening-status.md`,
  `2026-07-07-three-family-null-strategic-inflection.md`.
