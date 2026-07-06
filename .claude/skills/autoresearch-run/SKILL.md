---
name: autoresearch-run
description: Orchestrate the dual-box autoresearch loop — home GPU does continual exploratory scout screening, a Vast.ai box battle-tests promising leads with multi-seed sweeps and spends idle compute on extra exploration. Two background subagents share context through a living markdown document and synchronize at master-defined Gates. Use when the user invokes /autoresearch-run with a box spec and a handoff doc.
argument-hint: "<Ngpu>x<vram-gb>/<vcpu> <path-to-handoff-doc>"
---

# Autoresearch Run — dual-box gated orchestration

You are the **master agent**. Parse the two arguments, derive the compute-allocation
constants, seed the living document, spawn the two subagents, then run the gate loop:
reconcile at every gate, reallocate compute, and surface human-only decisions to the user.

## Arguments

1. **Box spec** `"{N}x{VRAM}/{vCPU}"` (e.g. `4x12/80` = 4 GPUs × 12 GB, 80 vCPU) — the
   Vast.ai box the user has rented (or will rent; ask for the SSH string if you don't have
   one). The home box is always the local RTX 5070 (12 GB).
2. **Handoff doc** — path to the doc seeding this run's context (prior campaign state, open
   leads, verdicts). Read it fully before anything else.

## Compute-allocation constants (derive at start, write into the living doc)

Measured cell footprints (AUTORESEARCH.md, cloud notes + 3060 sizing):

| Cell type | vCPU | VRAM | wall-clock |
| --- | --- | --- | --- |
| scout screening (n=1 trial: agent+tests+scoutv2 bench) | ~6 | ~6.5 GB | ~85 min |
| confirmation cell (scoutv2 preset, 1 cond × 1 seed) | ~6 | ~6.5 GB | ~75 min |

- `CELLS_PER_GPU = floor(VRAM_GB / 6.5)` (12 GB → 1; 24 GB → 3)
- `BOX_CAPACITY = N_GPU × CELLS_PER_GPU` parallel cells
- **`BOX_CONFIRM_SHARE = 0.75`** — battle-testing known scout leads is funded FIRST: at each
  box gate, fill up to 75% of `BOX_CAPACITY × gate_hours` with Confirmation Queue cells.
- **`BOX_EXPLORE_SHARE = the remainder plus ALL idle capacity`** — whenever confirmation
  waves leave GPUs idle (queue empty, uneven waves), backfill immediately with exploratory
  single-seed scout cells (AUTORESEARCH.md compute-check rule: idle box = pure waste).
- **`HOME_SHARE = 100% exploration`** — the 5070 runs the local screening loop
  (`scripts/run_local_loop.ps1`) continuously; it never runs confirmations.
- Budget rails: record in the living doc and enforce — `MAX_BOX_HOURS_PER_DAY` (default 12),
  `MAX_AGENT_SPEND_PER_GATE` (default $15 ≈ 4 trials), hard-stop if `results`-branch pushes
  fail (never accumulate unpushed results on an ephemeral disk).

## Gates

A gate = a synchronization boundary where a subagent MUST write its gate entry to the living
doc, read the Master Decisions section, and only then continue.

- **HOME_GATE — one exploratory scout run** (~85 min): after every single local trial
  completes (ledger append), write `{trial, hypothesis, composite vs baseline, hit_rate_80
  vs baseline, verdict, signature?}`. A trial shows the **signature** when hit_rate_80
  improves ≥ +5 pts while composite is within noise — flag it into the Leads Queue.
- **BOX_GATE — 4–8 hours**: round to whole confirmation waves
  (`gate_hours ≈ ceil(cells_queued / BOX_CAPACITY) × 1.5h`, clamped to [4, 8]). At the gate:
  pull `summary.csv`s, write per-condition `composite` and `hit_rate_80` mean ± bootstrap CI
  into the doc, list idle-capacity usage, push results.

## The living document (shared context)

Create `autoresearch/live/RUN-<UTC-stamp>.md` — under `autoresearch/`, which is gitignored
AND snapshot-ignored, so continuous writes never poison a running trial's audit. Fixed
section contract (owners in parentheses; logs are append-only):

```
# Autoresearch Run <stamp>
## Constants          (master; from box spec + this skill)
## Allocation         (master; current confirm/explore split, next gate times)
## Leads Queue        (home agent appends; master promotes → Confirmation Queue)
## Confirmation Queue (master only; {condition, seeds, priority, prediction})
## Home Gate Log      (home agent, one entry per HOME_GATE)
## Box Gate Log       (box agent, one entry per BOX_GATE)
## Master Decisions   (master only; dated; subagents re-read at every gate)
## Verdicts           (master; n=8 CI results → promote / extend / close)
```

Subagents communicate ONLY through this document — never directly with each other.

## Procedure (master)

1. **Create a fresh run worktree FIRST** — before touching anything else:
   `git worktree add ../LL-run-<UTC-stamp> -b autoresearch-run-<UTC-stamp>` from the repo root.
   ALL master-side repo edits for this run (code changes, condition flags, research-log
   entries, doc updates) happen in the worktree, never in the main tree: the home loop's
   snapshot audit attributes any main-tree edit to whatever trial is in flight and rejects
   it, rolling YOUR files back. The main tree belongs to the home loop for the duration of
   the run; the worktree branch merges back at shutdown (step 6). Tests in the worktree need
   `PYTHONPATH=<worktree>/src` (the venv's editable install points at the main tree).
   The living doc + gitignored state stay under the MAIN tree's `autoresearch/` (both trees
   share it conceptually; use the main-tree path so the home agent sees the same doc).
2. Read the handoff doc; create the living doc; fill Constants + initial Allocation; seed the
   Leads/Confirmation queues from the handoff (e.g. open items in
   `config/hypothesis_queue.md` and any prior signature leads).
3. **Prerequisite check:** confirmed leads can only be swept if they exist as flag-guarded
   `CONDITIONS` in `scripts/run_seed_sweep.py` with a `scoutv2` preset. If missing, implement
   them in the run worktree (step 1) and push the branch — the box clones/pulls that branch
   directly, so confirmation sweeps never wait on the home loop.
4. Spawn both subagents via the Agent tool (`run_in_background: true`), each prompt
   containing: its role brief below, the living doc path, its gate definition, the budget
   rails, and the handoff doc path.
5. Gate loop: on each subagent gate notification — reconcile (promote signature leads into
   the Confirmation Queue with a registered prediction; move CI results into Verdicts; route
   each verdict extend/escalate/close), update Allocation, append a Master Decision, and
   relay anything human-only to the user. **Register rule:** if any decision changes a
   register truth (scorer, gates, benchmark versions, surfaces, budget rails), update the
   AUTORESEARCH.md register row in the SAME worktree commit as the change — never later.
6. Shutdown: when both queues are empty and the user confirms — final results pull, verdicts
   into `docs/research-log/` + `docs/research-notes/` (per repo conventions), recommend box
   destroy, then **finalize this run's loop note** in `docs/autoresearch-loops/` (curated
   from the living doc; its *pickup state* section is what the next run starts from —
   handoff docs are retired for autoresearch work). Record sibling-loop cross-links if other
   loops ran concurrently. Merge the run worktree branch.

## Subagent role briefs

**HOME-SCOUT (local 5070):** Run the screening loop one session at a time via
`scripts/run_local_loop.ps1` (detached; watch `autoresearch/<RunName>.exitcode` and the
manifest ledger). One HOME_GATE entry per trial. Flag signature leads. On agent-429: log the
reset time in your gate entry, sleep until reset, resume — do not burn gate entries retrying.
Never edit managed repo files; never touch the benchmark/scorer surface; trial-code changes
happen only through the loop's own audited trials.

**VAST-BOX:** If the box isn't bootstrapped: `scripts/cloud/upload_secrets.sh`, clone the
designated branch, `scripts/cloud/bootstrap.sh` (Pascal boxes additionally need the cu126
torch downgrade — docs/plans/cloud-setup.md gotcha). Battle-testing: launch Confirmation
Queue cells with the proven pattern — one `run_sweep.sh` invocation per GPU,
`CUDA_VISIBLE_DEVICES=N MAX_PARALLEL=<CELLS_PER_GPU> RESUME=1 AUTO_PUSH=1`, preset `scoutv2`,
out dirs `sweeps/confirm_g{N}`. Exploration on idle capacity: single-seed `scoutv2` cells for
Leads Queue conditions (or next hypothesis_queue items) — clearly labeled `explore_*` out
dirs. Push to the `results` branch at every gate (the box disk is ephemeral). Report SPS
health; kill-and-resume stuck cells (`pkill -9 -f "[t]rain_brain.py"` bracket trick).

## Human-only actions (relay, never execute autonomously)

Renting/destroying a box; merging condition branches into the main branch; declaring a
research note resolved beyond the loop's automatic verdict-append; raising budget rails;
editing the frozen benchmark/scorer surface.
