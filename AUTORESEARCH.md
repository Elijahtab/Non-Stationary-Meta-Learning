# AUTORESEARCH.md — the autoresearch checklist

> **This document is the checklist for how we run autoresearch.** The loop notes in
> [docs/autoresearch-loops/](docs/autoresearch-loops/) are the granular points of truth we use
> to pick up new autoresearch loops. **Before doing anything, Claude must read this document
> and the relevant loop note(s)** — the register below says what is currently true; the latest
> loop note's *pickup state* says what to do next. History and rationale live behind the links,
> not here. Full v1 of this document: [docs/archive/AUTORESEARCH-v1.md](docs/archive/AUTORESEARCH-v1.md).

## 1. Decisions in force (register)

Rows link to the source of truth — code and dated records — rather than restating them.
**Register rule: any commit that changes one of these truths MUST update its row in the same
commit** (this is how the register stays alive; the archived v1 doc rotted by restating).

| Decision | Currently | Source of truth |
| --- | --- | --- |
| Composite score | `mean_post_switch_window_success_rate` — nothing else | [`_compute_composite_score`](src/lifelong_learning/research/benchmarking.py) · [log 0001](docs/research-log/0001-2026-06-18-calibration-scoring.md) |
| Primary scout | `fast_switch_scout_v2` (8×8, n=1, ~85 min/trial) | [log 0004](docs/research-log/0004-2026-07-04-scout-v2-8x8.md) |
| Baseline anchor | `fast_switch_scout_v2_baseline` (3-seed mean; amortized via fingerprint cache) | [log 0006](docs/research-log/0006-2026-07-05-amortized-multiseed-baseline.md) |
| Holdout gate | `fast_switch_holdout_v1` (seeds 11/23/37, zero regression tolerance), runs only on scout improvement | [manifest](config/research_manifest.toml) |
| Gate refinement (hit80 as second criterion — the "reliability fork") | **ADOPTED (fork), applied at n≥8 ONLY.** Accept on EITHER path: (A) composite improves, OR (B) composite non-degraded (≥ control CI-lower) AND hit_80 > control hit80 CI-upper. **The n≥8 guard is load-bearing:** at n=3 the fork over-selected lucky seed triples — LOOP-0006's two apparent reliability leads (critic_code, auxcode_hi) both washed out at n=8 → campaign closed NULL. | [log 0007 §Fork addendum + Outcome](docs/research-log/0007-2026-07-06-learning-dynamics-preregistration.md) · [RUN-20260706 fork re-adjudication](autoresearch/live/RUN-20260706.md) |
| Editable / immutable surfaces | see manifest (`docs/research-notes` is editable; scorer/benchmark/envs immutable; `ppo.py`/`train.py` opened as optional surface 2026-07-06 for the learning-dynamics family, flag-guarded default-off only; `train_brain.py`/`meta_env.py` opened 2026-07-07 for the bottom-rung/oracle CONTROL conditions D0/O1/O2 — [note 0005](docs/research-notes/0005-untested-controller-bottom-rung-oracle.md), user-approved, flag-guarded default-off only) | [config/research_manifest.toml](config/research_manifest.toml) |
| Trial agent | local `claude -p` (claude-fable-5) via [invoke_claude_exec.ps1](scripts/invoke_claude_exec.ps1)/[.sh](scripts/invoke_claude_exec.sh); ~$3.60 + ~85 min per trial | wrappers + [run_local_loop.ps1](scripts/run_local_loop.ps1) |
| Interpreter rule | manifests use `{python}` (resolved to `sys.executable`) — never hardcode venv paths | [log 0006](docs/research-log/0006-2026-07-05-amortized-multiseed-baseline.md) commit trail |
| Kill safety | `autoresearch/STOP` sentinel (graceful, trial-boundary) + write-ahead journal with startup recovery — the supervisor is safe to kill any time | [autoresearch.py](src/lifelong_learning/research/autoresearch.py) |
| Verdict semantics | benchmark-scoped; science verdicts (`accepted`/`primary_score_did_not_improve`/`holdout_regressed`) retire a hypothesis on that benchmark, infra failures are retryable; science verdicts never age out of injected history | [trial_prompt.py](src/lifelong_learning/research/trial_prompt.py) |
| Hypothesis queue | [config/hypothesis_queue.md](config/hypothesis_queue.md) — human-curated, agent-read-only. **Currently: EXHAUSTED** (all 5 LOOP-0005 entries science-rejected; LOOP-0007/0008/0009 run as master-implemented conditions, not queue entries) | queue file + [LOOP-0005](docs/autoresearch-loops/LOOP-0005-brainstorm.md) |
| Dual-box orchestration | `/autoresearch-run` skill (home = n=1 screening; box = n=8 confirmation + idle exploration; gates via living doc) | [.claude/skills/autoresearch-run/SKILL.md](.claude/skills/autoresearch-run/SKILL.md) |
| Linux async runs | require `context="spawn"` (in code); Pascal boxes require the cu126 torch downgrade | [cloud-setup.md](docs/plans/cloud-setup.md) |

## 2. The loop checklist

**Pre-flight**
1. Read this file + the latest loop note(s) ([index](docs/autoresearch-loops/README.md)).
2. Master-side repo edits go in a **fresh worktree** (`git worktree add ../LL-run-<stamp> -b
   autoresearch-run-<stamp>`) — the main tree belongs to the running loop; a mid-trial edit
   there poisons the trial's audit. Worktree tests need `PYTHONPATH=<worktree>/src`.
3. Confirm: queue has live entries; no stale `autoresearch/STOP`; machine won't sleep; box
   (if any) bootstrapped from the right branch with GH token armed.

**Run**
4. Home screening: `scripts/run_local_loop.ps1 -RunName <name>` (detached; watch
   `autoresearch/<name>.exitcode` + the manifest ledger). Box work: per the skill.
5. Keep the living doc (`autoresearch/live/RUN-*.md`) current at every gate — it is the only
   safe write surface while trials run.
6. Register any pre-registered prediction in `docs/research-log/` (via worktree) BEFORE its
   results exist; commit the register row in the same commit as any truth change.

**Close-out**
7. Verdicts → research-log/notes per conventions; queue annotated; results pushed
   (`results` branch — box disks are ephemeral).
8. Write/finalize the loop note (curated from the living doc), including **pickup state** and
   sibling links — this replaces handoff docs for autoresearch work.
9. Destroy idle boxes; merge the run worktree; push everything.

## 3. Loop notes (the granular truth)

Index with one-liners: [docs/autoresearch-loops/README.md](docs/autoresearch-loops/README.md)

- [LOOP-0001 — pilot shakedown](docs/autoresearch-loops/LOOP-0001-pilot-shakedown.md): harness hardened (3 pilots; surface/interpreter/mojibake fixes)
- [LOOP-0002 — v1 scout campaign](docs/autoresearch-loops/LOOP-0002-v1-scout-campaign.md): 6 rejections on a saturated 5×5 instrument → scout v2
- [LOOP-0003 — v2 scout campaign](docs/autoresearch-loops/LOOP-0003-v2-scout-campaign.md): 4/5 beat the n=1 anchor, 0 survived holdout → anchor flaw exposed
- [LOOP-0004 — confirmation sweep](docs/autoresearch-loops/LOOP-0004-confirmation-sweep.md): falsifier at n=8; mechanism family closed; 3-seed baseline adopted
- [LOOP-0005 — brainstorm](docs/autoresearch-loops/LOOP-0005-brainstorm.md): 5 learning-dynamics candidates ([note 0003](docs/research-notes/0003-code-directed-plasticity-gating.md)); all queued 2026-07-06
- [LOOP-0006 — learning-dynamics campaign](docs/autoresearch-loops/LOOP-0006-learning-dynamics-campaign.md): closed NULL at n=8 (two n=3 "reliability leads" were sampling artifacts)
- [LOOP-0007 — code-free plasticity/stability](docs/autoresearch-loops/LOOP-0007-brainstorm.md): closed NULL 0/5; A1 (plasticity loss) FALSE by registered probe
- [LOOP-0008 — bottom-rung & oracle controls](docs/autoresearch-loops/LOOP-0008-controls-campaign.md): closed — Brain adds ≤0 (D0); A2 + code pathway dead (O1/oracle_code); ceiling +0.249 (O2); trained-Brain +0.029 p<0.01 ([note 0006](docs/research-notes/0006-controls-axis-thesis-relocated.md))
- [LOOP-0009 — trained-Brain replication](docs/autoresearch-loops/LOOP-0009-trained-brain-replication.md): closed ✅ CONFIRMED — P-R1a passed: pooled Δ+0.0455 composite (p=1.3e-6, n=16/arm), 3/4 Brains positive, converged ([note 0007 §Results](docs/research-notes/0007-trained-brain-replication.md)); 4 fresh ep130 Brains archived (`results/loop9_final/` on the results branch); paper C1 upgraded to across-training-seeds (note 0008 → `paper/main.tex` v1)
- [LOOP-0010 — memory-facing meta-control](docs/autoresearch-loops/LOOP-0010-memory-levers.md): BRAINSTORM/pre-open — memory levers vs the +0.249 O2 ceiling; note 0009 draft refined by the [2026-07-09 action tree](docs/plans/2026-07-09-research-action-tree.md): Wave-1 oracle rungs (★G-DECOMP swap-scope ladder first) decide which memory subtree (weight-space / MoWM / mem-Brain) opens
- [LOOP-0011 — Wave-1 oracle rungs](docs/autoresearch-loops/LOOP-0011-wave1-oracle-rungs.md): closed 2026-07-14 same-day ($0, 56/56 evals) — **P-W1a: ceiling protocol-robust (H=+0.2434, p=2.8e-8); P-W1b: heads carry +89.6% (heads+enc +96.3%, WM −8.2% null) → G3 weight-space memory OPENS, MoWM subtree CLOSED; P-W1c: K=3 premium flat (+0.2351) → scaling axis dies** ([note 0012](docs/research-notes/0012-ceiling-decomposition.md); pre-reg [log 0008](docs/research-log/0008-2026-07-14-wave1-oracle-rungs-preregistration.md)). Next: G3 build (home, $0); box ping at Brain fine-tune stage
- [LOOP-0012 — G3 head-bank de-oracling](docs/autoresearch-loops/LOOP-0012-g3-head-bank.md): closed 2026-07-14 same-day ($0, 24/24) — **bank ≡ heads swap (P-G3a); A-R1 learned trigger recovers +0.1180 = 54% of the ceiling slice with zero oracle bits (P-G3b PASS, precision 0.70, hit80 1.000)**; P-G3c selection arm **invalidated same day** — missing slot allocation meant the selector never ran (artifact caught by LOOP-0013's smoke; correction in [note 0013](docs/research-notes/0013-learned-trigger-selection-gap.md))
- [LOOP-0013 — selection rung done right](docs/autoresearch-loops/LOOP-0013-selection-rung.md): selection RESOLVED 2026-07-15 ($0, 24/24) — **P-G3e PASS: the K=2-flip learned-trigger method converges at n=16, +0.1188 (p=4.7e-7, 52% of slice); P-G3d + P-G3d-ve both FAIL as real nulls with an identified mechanism: banked heads scored through the live drifting trunk almost never restore (1/55, 9/55 fires)** — the selection frontier is drift-robust fingerprints (bank the scoring path) or the mem-Brain lever. **Addendum resolved + LOOP CLOSED 2026-07-15: precision 0.69→0.94 (thr 1.5) moved the composite by nothing (P-T15a PASS / P-T15b FAIL) — the oracle gap is detection LAG, structural to per-update detectors including a Brain gate at the same cadence** ([log 0011](docs/research-log/0011-2026-07-15-trigger-hardening-preregistration.md), [note 0013](docs/research-notes/0013-learned-trigger-selection-gap.md)); $0 frontier: per-step WM-surprise trigger, drift-robust fingerprints; mem-Brain premise weakened
- [LOOP-0014 — per-step trigger vs the lag wall](docs/autoresearch-loops/LOOP-0014-step-trigger.md): OPEN (scored arm launched 2026-07-16, home, $0) — model-free success-collapse detector fires mid-rollout (median lag ~400 steps vs 2,048+ per-update); shadow-calibrated over 3 disclosed iterations (per-event WM prediction error abandoned at precision 0.02-0.04); **CLOSED 2026-07-16: P-S1a+b both FAIL** — lag cut to 280 steps (recall 1.00) but gain FELL to +0.0916: a false flip guarantees its own secondary collapse (live precision 0.65 @ 15.5 fires/run; churn > lag savings). **Trigger family complete — the method is the per-update trigger at +0.1188 (n=16)**; residual ~0.10 gap structural; next: paper consolidation ([note 0013 addendum](docs/research-notes/0013-learned-trigger-selection-gap.md))
- [LOOP-0015 — drift-robust fingerprint selection](docs/autoresearch-loops/LOOP-0015-fingerprint-selection.md): OPEN (launched 2026-07-16, home, $0, user green-light) — fixes the LOOP-0013 drift null by banking the full WM per slot (scoring-only) so fingerprints read their own frozen features; live WM scores the active slot → the selector doubles as a fire verifier (the LOOP-0014 churn suppressor); **CLOSED same-day: P-FP1+FP2 both FAIL** (+0.0367 p=0.27; flip-when-should 0.16) — the drift fix worked, but the live reward head adapts within the trigger's one-update lag, so the live-scored 'no switch' hypothesis wins even at true switches. **Selection family adjudicated closed on this instrument (drift → churn → live-side adaptation); the blind K=2 flip +0.1188 (n=16) stands as the method** ([log 0013](docs/research-log/0013-2026-07-16-fingerprint-selection-preregistration.md), [note 0013 addendum](docs/research-notes/0013-learned-trigger-selection-gap.md))

- [LOOP-0016 — encoder-dormancy probe (W0c)](docs/autoresearch-loops/LOOP-0016-encoder-dormancy-probe.md): OPEN (launched 2026-07-17, home, $0, user direction) — closes paper claim C4's blind spot (the "plasticity loss absent" verdict was heads-only); probe-only instrument on a plain trained-Brain arm (`dorm_e1..8`), gates pre-registered ([log 0014](docs/research-log/0014-2026-07-17-encoder-dormancy-preregistration.md)). **CLOSED same-day: P-W0c2 PASS (instrument valid: heads fall replicates C4, composite inert +0.0115 p=0.586) / P-W0c1 FAIL — first-conv-layer dormancy ACCUMULATES 0.125→0.479 (rise +0.318, 6× the bar, τ-robust) while conv2/3 and heads fall. C4 scope-corrected in the paper (5 sites) + note 0008; Wave 0 complete; branch F's activation condition met — interventions need a new registration + human call** ([note 0014](docs/research-notes/0014-encoder-dormancy-probe.md))

## Deeper references

- Benchmark/trial contract mechanics: [docs/spec/07](docs/spec/07-benchmarking-and-autoresearch.md)
  · CLI recipes: [docs/spec/08](docs/spec/08-cli-reference.md)
- Cloud ops (sizing, packing, thread caps, durability, gotchas): [docs/plans/cloud-setup.md](docs/plans/cloud-setup.md)
- Research narrative: [docs/research-notes/](docs/research-notes/README.md) · dated decisions:
  [docs/research-log/](docs/research-log/README.md)
