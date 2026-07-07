
## Outcome addendum (2026-07-07)

All five candidates resolved. Cand 1 critic_lr_lo REJECT n=8 (composite 0.5490/hit80 0.8248);
cand 5 encoder_lr_lo REJECT n=8 (0.5531/0.7589 — reliability −7 pts, two-timescale retired);
cand 3 plasticity_norm KILLED n=2 (degenerate LayerNorm(4096) operationalization — A1 untested
by a fair norm at kill time); cand 4 surprise_spike REJECT n=8 (0.5427/0.8103 — the guard
fired: over-exploration); cand 2 redo: n=8 Path-B pass (0.5522/0.8527) that FAILED at n=10
(0.5480/0.8393) → holdout (W10) decisive. **A1 subsequently adjudicated FALSE by the
registered dormant-fraction probe** — dormancy *falls* ~0.9→0.35 over the run, never rises
across switches (see note 0005 addendum) — retiring the plasticity-loss premise for cands 2/3
regardless of the holdout outcome. Family verdict pending only the redo holdout formality;
interpretation continues in [note 0005](./0005-untested-controller-bottom-rung-oracle.md).
