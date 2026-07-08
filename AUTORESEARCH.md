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
- [LOOP-0009 — trained-Brain replication](docs/autoresearch-loops/LOOP-0009-trained-brain-replication.md): OPEN — ≥4 fresh Brain training seeds vs matched inits at eval protocol, pre-registered in [note 0007](docs/research-notes/0007-trained-brain-replication.md); pre-launch verified (A3 adapter reproduces note 0006, A5 config, seed/init fix); strategic fork decided at its gate (box destroyed 2026-07-08; new box to provision)
- [LOOP-0010 — memory-facing meta-control](docs/autoresearch-loops/LOOP-0010-memory-levers.md): BRAINSTORM/pre-open — memory levers (restore trigger / snapshot gating / replay selection) vs the +0.249 O2 ceiling; pre-registration drafted ([note 0009](docs/research-notes/0009-memory-levers-preregistration.md)), awaiting curation

## Deeper references

- Benchmark/trial contract mechanics: [docs/spec/07](docs/spec/07-benchmarking-and-autoresearch.md)
  · CLI recipes: [docs/spec/08](docs/spec/08-cli-reference.md)
- Cloud ops (sizing, packing, thread caps, durability, gotchas): [docs/plans/cloud-setup.md](docs/plans/cloud-setup.md)
- Research narrative: [docs/research-notes/](docs/research-notes/README.md) · dated decisions:
  [docs/research-log/](docs/research-log/README.md)
