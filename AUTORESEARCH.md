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
| Gate refinement (ε margins, hit80 as second criterion) | **OPEN — deliberately not adopted**: the 0005 falsifier removed the evidence of need | [log 0005](docs/research-log/0005-2026-07-05-confirmation-sweep-prediction.md) |
| Editable / immutable surfaces | see manifest (`docs/research-notes` is editable; scorer/benchmark/envs immutable; `ppo.py`/`train.py` opened as optional surface 2026-07-06 for the learning-dynamics family, flag-guarded default-off only) | [config/research_manifest.toml](config/research_manifest.toml) |
| Trial agent | local `claude -p` (claude-fable-5) via [invoke_claude_exec.ps1](scripts/invoke_claude_exec.ps1)/[.sh](scripts/invoke_claude_exec.sh); ~$3.60 + ~85 min per trial | wrappers + [run_local_loop.ps1](scripts/run_local_loop.ps1) |
| Interpreter rule | manifests use `{python}` (resolved to `sys.executable`) — never hardcode venv paths | [log 0006](docs/research-log/0006-2026-07-05-amortized-multiseed-baseline.md) commit trail |
| Kill safety | `autoresearch/STOP` sentinel (graceful, trial-boundary) + write-ahead journal with startup recovery — the supervisor is safe to kill any time | [autoresearch.py](src/lifelong_learning/research/autoresearch.py) |
| Verdict semantics | benchmark-scoped; science verdicts (`accepted`/`primary_score_did_not_improve`/`holdout_regressed`) retire a hypothesis on that benchmark, infra failures are retryable; science verdicts never age out of injected history | [trial_prompt.py](src/lifelong_learning/research/trial_prompt.py) |
| Hypothesis queue | [config/hypothesis_queue.md](config/hypothesis_queue.md) — human-curated, agent-read-only. **Currently: 5 live LOOP-0005 entries (learning-dynamics family); 1–4 box-assigned, home starts at 5** | queue file + [LOOP-0005](docs/autoresearch-loops/LOOP-0005-brainstorm.md) |
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
- [LOOP-0006 — learning-dynamics campaign](docs/autoresearch-loops/LOOP-0006-learning-dynamics-campaign.md): **OPEN** — dual-box screen of the 5 candidates (box 1–4, home 5)

## Deeper references

- Benchmark/trial contract mechanics: [docs/spec/07](docs/spec/07-benchmarking-and-autoresearch.md)
  · CLI recipes: [docs/spec/08](docs/spec/08-cli-reference.md)
- Cloud ops (sizing, packing, thread caps, durability, gotchas): [docs/plans/cloud-setup.md](docs/plans/cloud-setup.md)
- Research narrative: [docs/research-notes/](docs/research-notes/README.md) · dated decisions:
  [docs/research-log/](docs/research-log/README.md)
