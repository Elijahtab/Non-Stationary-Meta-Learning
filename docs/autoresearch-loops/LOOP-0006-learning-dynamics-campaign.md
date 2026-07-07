# LOOP-0006 — learning-dynamics campaign (CLOSED NULL, 2026-07-06 → 07-07)

**Goal:** Screen the 5 LOOP-0005 hypotheses (context code → learning dynamics instead of
forward-path modulation): box battle-tests entries 1–4 as master-implemented conditions,
home screens entry 5 via the audited trial loop. Non-rejected candidates go to holdout;
if everything closes, spin a new brainstorm loop.
**Verdict:** ❌ **NULL at n=8.** No condition in the code-directed learning-dynamics family
(gradgate / critic_code / auxcode / adamflush + scaled/stacked/dose variants) beats the frozen
control on the scout primary. The two apparent n=3 *reliability* leads (critic_code, auxcode_hi)
were lucky-triple sampling artifacts — both washed out at n=8 (hit80 0.860→0.835, 0.851→0.819);
the one composite lead (gradgate_gain) failed holdout. Home entry 5 (code-weighted anchoring)
rejected on both fork paths. **Through-line across two closed families: injecting the regime code
into the inner agent — any pathway — does not robustly help.** → pivot to LOOP-0007 (code-free).

## Hardware
Home: RTX 5070 12 GB (local screening loop). Box: Vast.ai `76.67.137.57:26061`,
4× RTX 3060 12 GB, 80 vCPU → `BOX_CAPACITY = 4` parallel cells (1 cell/GPU at ~6.5 GB).

## Code state
Main tree: `trainable-vs-frozen-sweep` + queue/surface commits (5 live queue entries,
`ppo.py`/`train.py` opened as optional surface). Run worktree branch:
`autoresearch-run-20260706` — master-implemented flag-guarded conditions
`brain_neuromod_gradgate`, `brain_neuromod_critic_code`, `brain_neuromod_auxcode`,
`brain_neuromod_adamflush` (preset `scoutv2`). Box clones this branch.

## Runs
Box (scoutv2, seeds from 1..8 + holdout 11/23/37), all pushed to `origin/results` as
`confirm_ld_*`/`explore_ld_*`: 4 candidates × waves to n=3, then n=8 extensions for the
apparent leads, holdout batteries, the explore/scaled variants (gradgate_gain,
critic_code_gradgate, auxcode_hi, adamflush_lo), the reliability-fork follow-ups
(critic_code_auxhi, auxcode_010/025), and the auxcode_010 dose-response n=5. Home: one
audited trial (entry 5, code-weighted anchoring). Full gate-by-gate trail: living doc
`autoresearch/live/RUN-20260706.md`.

## Obstacles
- **The n=3 trap (central lesson).** n=3 over-selected lucky seed triples → two false
  reliability leads that evaporated at n=8. Fixed forward: screen at **n≥8** (carried to
  LOOP-0007) and the reliability fork is register-adopted *with an n≥8 guard*.
- **~7 h idle-box incident** (box agent's watcher died with its session; master looked in `~`
  not `/workspace`). Lesson: OS-detach long work AND own/watch it from the master session;
  verify paths before declaring box state.
- Non-interactive ssh misses the venv → prefix `PATH=/venv/main/bin`. SSH login banner pollutes
  stdout → parse by grepping a marker line, never by position.

## Statistics & verdicts
Box comparison anchor: LOOP-0004 frozen control at n=8, same preset/GPU class —
composite **0.5534 [0.5478, 0.5588]**, hit_rate_80 0.829 [0.815, 0.846]. Home anchor:
3-seed `baseline_primary`. Final n=8 reads: critic_code composite 0.54618 / hit80 0.83482;
auxcode_hi 0.54694 / 0.81920 — both inside/below the control band. Pre-registered predictions,
decision rules, fork thresholds, and the NULL Outcome: research-log 0007 (this branch).

## Cost
Box rented ~06:54Z 07-06 onward; ran ~13 h+ over the campaign (over the 12 h soft rail, which
the user explicitly lifted for continuous compute — results were promising until the n=8
walk-back). Home: one ~8 h cache-miss trial. Box-destroy is human-only and off the table.

## Sibling loops
[LOOP-0005](./LOOP-0005-brainstorm.md) (source of the queue); home and box halves of this
campaign share this note.

## Pickup state
**CLOSED NULL.** Successor is **LOOP-0007** (code-free plasticity/stability,
[note 0004](../research-notes/0004-code-free-plasticity-stability.md)): 5 candidates
implemented on this same run branch (`brain_neuromod_{critic_lr_lo, encoder_lr_lo,
plasticity_norm, surprise_spike, redo}`), screening at n≥8 on the box — see the LOOP-0007
loop note + living doc for current state. Remaining LOOP-0006 close-out chore: **merge the
`autoresearch-run-20260706` branch** once LOOP-0007 also resolves (both campaigns share this
branch, so merge once at the end). Results are durable on `origin/results`.

## Links
Queue: `config/hypothesis_queue.md` (LIVE section) · predictions: research-log 0007 ·
note [0003](../research-notes/0003-code-directed-plasticity-gating.md) · living doc
`autoresearch/live/RUN-20260706.md` · results branch dirs `confirm_ld_*`/`explore_ld_*`.
