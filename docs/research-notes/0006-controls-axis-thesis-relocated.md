# 0006 — The controls axis: the thesis confirmed small, the bottleneck relocated

**Finding (one sentence):** With every rung of the control ladder finally measured, trained
hyperparameter meta-control **does** improve post-switch recovery — **+0.029 composite,
p<0.01, n=32/arm, single-variable matched contrast** — but that gain is ~12% of what
zero-forgetting memory restoration achieves on the same instrument (**+0.25**), locating the
catastrophic-forgetting bottleneck in **policy/world-model knowledge**, not learning dynamics.

All experiments pre-registered in [note 0005](./0005-untested-controller-bottom-rung-oracle.md)
(D0/O1/O2) and the RUN-20260706 living doc (T1 series, oracle_code) before their results
existed. Every number below is n=8 unless stated; scout control reference: composite
0.5534 [0.5478, 0.5588], hit80 0.829 [0.815, 0.846].

## The axis (box protocol, comparable to the frozen control)

| condition | composite | hit80 | verdict |
| --- | --- | --- | --- |
| control — untrained noisy Brain | 0.5534 | 0.829 | reference |
| D0 — no controller (mid-bound HPs) | 0.5608 | 0.7857 | Brain adds ≤0 composite; its jitter bought ~4 pts hit80 |
| O1 — oracle-timed critic damp | 0.5451 | 0.8058 | REJECT: mildly hurts both → **A2 dead; critic whiplash is the fix, not the bug** |
| oracle_code — ground-truth code → mask | 0.5481 | 0.8259 | NULL: **pathway dead, not just the code** — forward-mod family null reinstated |
| **O2 — zero-forgetting ceiling** | **0.8022** (n=8, sd 0.009) | 0.873 | **+0.249 headroom — reading ① (ceiling) and ④ (blind metric) both dead** |

## The matched trained-Brain contrast (eval protocol; single variable = Brain weights)

Frozen evals of the March-era 130-episode Brain (`brain_2_regimes_8x8_neuromod_20260315-180839`)
vs a random-init Brain, identical protocol (16 inner envs, decision_interval 1, same 800k/100k
switch schedule), n=32 seeds per arm:

| arm | composite | hit80 |
| --- | --- | --- |
| trained ep130 | **0.5577** (sd 0.051) | 0.982 |
| init control | **0.5285** (sd 0.022) | 0.982 |

**Δ = +0.0292, t≈2.97, p<0.01.** The estimate *converged* with sample size
(+0.046 n=8 → +0.0275 n=16 → +0.0292 n=32) — the signature of a real effect, in contrast to
the program's three small-n mirages, which all decayed. Training-dose ordering held
throughout (init < ep91 < ep130). hit80 is identical between arms: the effect is purely
mean-recovery. Checkpoint selection: ep130 chosen by best 10-episode-average training reward;
the ep91 single-episode peak was evaluated too (n=8: 0.5415) and lost — selection documented.

## Cross-protocol warning (measured, load-bearing)

The eval protocol shifts BOTH metrics at identical hyperparameters (vs box protocol:
composite −0.045, hit80 +0.18 — 16-env smoothing inflates threshold-crossing enormously).
**No cross-protocol comparison is valid.** The early "reliability gradient" reading died on
exactly this: an untrained Brain at eval protocol scores hit80 0.964, so T1's 64/64 was
p≈0.10 against the right control. Every claim above is within-protocol.

## What died tonight, with its killer

- **A2 (critic whiplash causal/pathological)** — killed by O1: perfectly-timed damping,
  which upper-bounds any learned damping, mildly *hurts* both metrics.
- **A1 (plasticity loss)** — killed earlier today by the recovered dormant-fraction probe
  (falls 0.9→0.35, never accumulates; note 0005 addendum).
- **The code/modulation channel** — killed by oracle_code: perfect regime information
  through the mask pathway does nothing; LOOP-0002/0004's null is about the pathway and is
  reinstated at full strength despite the untrained-controller confound.
- **Reading ① (no headroom) and ④ (blind metric)** — killed by O2: +0.25 available, sd 0.008.
- **The paper's Table-2 static baseline** — killed by D0: a tuned static agent scores ≥ the
  Brain control and hits 80% on ~79% of switches ("never reaches 80%" was a weak-baseline
  artifact). The Table-2 *trained-Brain direction* survives, at honest magnitude (+0.029,
  not +0.27).

## Paper skeleton this yields

1. **Positive claim:** trained HP meta-control improves fast-switch recovery (+0.029
   composite, ~5% relative, p<0.01, n=32/arm, matched contrast) — a modern, controlled
   replication of the paper's meta-learning direction.
2. **Bound:** that gain is ~12% of the measured zero-forgetting ceiling (+0.25) — the
   bottleneck is knowledge restoration, consistent with the paper's own MoWM results; the
   next-strategy implication is explicit memory, not better meta-control.
3. **Mechanism results:** critic re-fit is functional (O1); plasticity loss absent at this
   horizon (probe); modulation pathway inert even with oracle information (oracle_code);
   HP jitter trades mean for reliability (D0 vs control).
4. **Methods contribution:** three reproduced small-n mirages (critic_code, auxcode_hi,
   redo — the last surviving to a mechanical holdout pass before dying at n=10 + probe),
   the convergence-vs-decay criterion that separated the real effect from them, the
   protocol-confound trap (T1c), and the instrument audit (an untrained controller sat in
   the scored loop for three families of screening — validate the controller before
   screening its enhancements).

## Provenance

Conditions `brain_constant_action` / `brain_critic_lr_oracle` / `brain_oracle_policy_swap`
(run branch cfa20d5, note 0005) + `brain_oracle_code` (pre-existing); results on
`origin/results` (`confirm_d0_s*`, `confirm_o1_s*`, `confirm_o2_s*`, `confirm_oc_s*`); T-series
evals in `evals/t1_*` (home 5070, 96 evals total); gate log in RUN-20260706 living doc
(2026-07-07 18:55Z → 2026-07-08). Scoring: frozen scorer throughout (eval logs staged into
run layout — `scratchpad/score_t1.py`, to be promoted to `scripts/score_eval_dir.py`).
