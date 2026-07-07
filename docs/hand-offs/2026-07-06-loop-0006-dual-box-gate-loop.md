# Hand-off: LOOP-0006 dual-box campaign — gate loop takeover (2026-07-06)

## Goal
Drive the LOOP-0006 learning-dynamics campaign to verdicts: 5 hypotheses (context code →
learning dynamics, not forward masks) are being screened — 4 on a Vast box, 1 on the home
GPU. Master duties: watch the two live workloads below, adjudicate gates by the
pre-registered rules, route survivors to holdout, keep the box at zero idle, then close out
the loop. **Primary state doc: `autoresearch/live/RUN-20260706.md`** (gitignored, main tree)
— read it in full before acting; it has constants, verdicts so far, incidents, and the exact
decision rules. Checklist: `AUTORESEARCH.md`.

## Two LIVE workloads (both survive session restarts; their WATCHERS do not — re-arm yours)

**1. HOME trial `ld_home_t2`** (RTX 5070, main tree `C:\Users\elija\Desktop\Projects\Lifelong-Learning`)
- One audited trial (MaxTrials=1) on queue entry 5 (code-weighted feature anchoring),
  launched OS-detached via `Start-Process` at 08:52 local. In the 3-seed `baseline_primary`
  anchor phase first (~3.8 h), then the ~85-min trial → expect completion ~13:00–15:00 local.
- Watch: `autoresearch/ld_home_t2.exitcode` (appears at end) + a NEW line in
  `autoresearch/trial_results.jsonl` (baseline = 21 lines). GOTCHA: `ld_home_t2.log` reads
  0 bytes while running (PowerShell `*>` buffering) — that is NOT a crash; check
  `Get-Process python` (2 procs = healthy) instead.
- At completion: parse the new ledger line → Home Gate Log entry in the living doc
  (composite/hit80 vs baseline, verdict). Do NOT launch a second trial without deciding the
  assignment explicitly (entries 1–4 are box-owned; queue file says so).

**2. BOX wave-2 + explore** (`ssh -p 26061 root@76.67.137.57`, 4× RTX 3060)
- Repo clone is at **`/workspace/Lifelong-Learning`** (NOT `~` — a prior session misdiagnosed
  a "wipe" by looking in `~`). Branch `autoresearch-run-20260706` @ cc15528.
- Running since 16:32Z, one sweep per GPU, 2 sequential cells each (~85 min/cell), done
  ~19:15–19:45Z: `sweeps/confirm_ld_w2_cc` (critic_code seeds 2,3), `w2_af` (adamflush 2,3),
  `w2_gg` (gradgate 2,3), `explore_ld_b1` (gradgate_gain s1 + critic_code_gradgate s1).
- Completion signal: `summary.csv` appears in each dir; `AUTO_PUSH=1` pushes each group to
  the `results` branch automatically. Verify pushes landed (`git fetch origin results`);
  push failures = HARD-STOP issue (ephemeral disk).

## Wave-1 verdicts already adjudicated (in the living doc)
vs frozen control n=8 **0.5534 [0.5478, 0.5588]** (LOOP-0004, same preset/GPU class):
critic_code 0.5500 / adamflush 0.5401 / gradgate 0.5387 survive; **auxcode 0.5347 KILLED**
(pre-registered 2σ kill line 0.5374). Nobody above control mean yet.

## Gate rules (pre-registered — do not re-derive; source: research-log 0007 on the run branch)
Per candidate at n=3 (wave 1 + wave 2 seeds): mean ≤ 0.5534 → reject
(`primary_score_did_not_improve`); > 0.5534 → holdout (seeds 11/23/37, zero regression);
> 0.5588 → additionally extend to n=8 (seeds 4–8). Explore hits (> 0.5534 at n=1) → Leads
Queue, not confirmation. Zero-idle backfill order is in the living doc's 08:20Z Master
Decision; still-unrun explores: `brain_neuromod_auxcode_hi`, `brain_neuromod_adamflush_lo`.
OPEN QUESTION for holdout-on-box: verify `fast_switch_holdout_v1` fixed_train_args match the
scoutv2 preset before running holdout as scoutv2 cells with seeds 11/23/37 (check
`get_frozen_benchmark("fast_switch_holdout_v1")`); otherwise run holdout through the home
benchmark harness where its cached baseline lives.

## Key decisions & findings (why things are the way they are)
- All 5 queue entries + surface extension (`ppo.py`/`train.py` optional, flag-guarded
  default-off) were human-directed 2026-07-06; register rows updated in the same commit
  (6484e67 on `Auto-Research`).
- Box compares against LOOP-0004's n=8 control instead of re-running control cells.
- Conditions implemented by the master (worktree `C:\Users\elija\Desktop\Projects\LL-run-20260706`,
  branch `autoresearch-run-20260706`, commits 11a9259 + cc15528): gradgate (backward-pass
  mask), critic_code (code → critic input), auxcode (aux prediction loss, KILLED), adamflush
  (optimizer reset on ‖code‖ spikes), + 4 explore flag-combos. Mechanism arguments:
  `docs/research-notes/0003-code-directed-plasticity-gating.md` + LOOP-0005 note.
- Gradgate diagnostic quirk: `policy_kl_vs_unmasked`/`value_delta` read ~0 BY CONSTRUCTION
  (mask left the forward pass). Not a dead run — check mask_mean/mask_std.

## Gotchas (each cost real time today)
- **Process ownership:** harness background tasks are session-scoped. Anything long-lived
  must be OS-detached (`Start-Process` on Windows; `run_sweep.sh` self-nohups on the box)
  and watched by monitors YOU own — subagent watchers die with their session (caused a 7-h
  idle box + a dead home trial this morning; see living-doc Incidents).
- **Box python:** non-interactive ssh misses the venv — prefix `PATH=/venv/main/bin:$PATH`
  for any remote launch, or you get `python: command not found`.
- **Results pushes race:** concurrent `push_results.sh` can fail on non-fast-forward
  ("git pull before pushing" in sweep.out) — re-run
  `bash scripts/cloud/push_results.sh sweeps/<dir>` for the missed group.
- **`ls <path> 2>/dev/null` hides missing-dir errors** — verify paths before declaring
  remote state (root cause of the false "wipe" diagnosis).
- **No git commit/push Mon–Fri 09:00–17:30 local** (user rule). Box AUTO_PUSH to `results`
  is running with explicit user awareness as a data-loss exception; your own code/doc
  commits wait for evening.

## Next steps
1. Re-arm both watchers from YOUR session (5–10 min poll): home = exitcode/ledger/process
   checks above; box = count of the 4 summary.csv via ssh + GPU util stall check.
2. At box completion (~19:30Z): pull the 4 summaries, compute per-candidate n=3 means
   (wave-1 seed-1 cells are in `results/confirm_ld_g0..g3` on the `results` branch), apply
   the gate rules, write the Box Gate Log + Verdicts entries in the living doc, launch
   holdout/backfill so no GPU idles (explores auxcode_hi/adamflush_lo are next in queue).
3. At home completion: gate entry per above; decide next home assignment (or hold the 5070
   for holdout duty).
4. When all verdicts land: fill research-log 0007's Outcome section, update note 0003 with
   the gradgate result, finalize `docs/autoresearch-loops/LOOP-0006-learning-dynamics-campaign.md`
   (curate from the living doc, incl. the process-ownership lessons for cloud-setup.md),
   merge the run worktree branch, recommend box destroy (human-only), commit after 17:30.
5. If every candidate + explore closes rejected: start the next brainstorm loop (LOOP-0007)
   per user directive — new family OUTSIDE code→inner-agent routing; deliver queue
   candidates + research note (LOOP-0005 note is the template for how).

## References
`autoresearch/live/RUN-20260706.md` (living doc — the run's ground truth) ·
`AUTORESEARCH.md` · `docs/autoresearch-loops/LOOP-0006-learning-dynamics-campaign.md` ·
research-log 0007 (run branch) · `docs/research-notes/0003-*.md` ·
`config/hypothesis_queue.md` · `docs/plans/cloud-setup.md` ·
worktree: `C:\Users\elija\Desktop\Projects\LL-run-20260706` (branch `autoresearch-run-20260706`)
