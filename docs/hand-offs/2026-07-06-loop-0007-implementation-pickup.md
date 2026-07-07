# Hand-off: LOOP-0007 implementation phase — 5 code-free mechanisms + box loop (2026-07-06)

## Goal
LOOP-0006 (code-directed learning-dynamics neuromod family) is **complete and closed NULL at
n=8**. The user directed a pivot to **LOOP-0007**: a new family of *code-free* plasticity &
stability mechanisms. The brainstorm + curation are done; this session ended at the start of
the **implementation** phase. Your job: implement the 5 curated mechanisms on the inner-agent
surface, screen them at **n≥8** on the Vast box, and keep the box running continuously.

**Read first, in order:** `autoresearch/live/RUN-20260706.md` (living doc — the run's ground
truth, full verdict history), `docs/research-notes/0004-code-free-plasticity-stability.md`
(the LOOP-0007 family: mechanisms, assumptions, predictions), `AUTORESEARCH.md` (register).

## State
- **Branch:** `Auto-Research` (main tree). Run worktree: `C:\Users\elija\Desktop\Projects\LL-run-20260706`
  on branch `autoresearch-run-20260706` @ `d32815a` (this is what the box clones/pulls).
- **Done & verified:**
  - **LOOP-0006 → NULL at n=8.** Full arc in the living doc Verdicts. Bottom line: no condition
    in the code-directed family beats the frozen control (0.5534 / hit80 0.829) at n=8. The two
    n=3 "reliability leads" washed out: critic_code hit80 0.86→0.835, auxcode_hi 0.85→0.819;
    gradgate_gain (composite lead) failed holdout. All results on `origin/results` branch.
  - **Reliability fork** adopted (user-directed): accept on EITHER (A) composite improves OR
    (B) composite non-degraded AND hit80 > control CI-upper. **BUT apply only at n≥8** — n=3
    over-selected lucky seed triples and produced the two false leads. Saved as memory
    `reliability-fork-acceptance`.
  - **LOOP-0007 brainstorm committed** (`d9177f3` on Auto-Research): research note 0004 +
    `docs/autoresearch-loops/LOOP-0007-brainstorm.md`. User curated **ALL 5 candidates** and
    signed off the surface for the two that touch the train loop (cands 2 & 4).
- **In progress:** Box is running a **LOOP-0006 dose-response holding wave** — `auxcode_010`
  seeds 2–5 (`sweeps/confirm_ld_aux010_s{2,3,4,5}`), started ~04:43Z, ETA ~06:00Z. Watcher
  `b0wp96r4j` (SESSION-SCOPED — it dies with the session that launched it; **re-arm your own**).
- **Uncommitted:** none of consequence. `docs/hand-offs/` is untracked (expected). The 3
  LOOP-0006 follow-up conditions in the worktree's `run_seed_sweep.py` are already committed
  (`d32815a`) and pushed.
- **Pending (not started):** (a) **LOOP-0007 implementation** (5 mechanisms, below);
  (b) **LOOP-0006 close-out** — update the AUTORESEARCH.md register "Gate refinement (hit80 as
  second criterion)" row to ADOPTED (fork), write research-log 0007 Outcome = NULL, update
  research-note 0003 with the null result, finalize the LOOP-0006 loop note, merge the
  `autoresearch-run-20260706` worktree branch. Register-rule: the fork adoption + register row
  must land in the SAME commit.

## The 5 LOOP-0007 candidates + grounded integration points
Full argument/predictions in research note 0004. All are **code-free** (never feed the regime
code to the inner agent — that's the dead end both closed families share). Plumbing pattern
(follow LOOP-0006): `scripts/train_brain.py` arg → `train.py`/`meta_env.py` mechanism →
`scripts/run_seed_sweep.py` `CONDITIONS` dict entry. Integration points I already traced:
- **Optimizer / param-groups:** `src/lifelong_learning/agents/ppo/train.py:237` already splits
  Adam into param groups (the trainable-decoder case) — the template for critic-LR & two-timescale.
- **Brain's LR channel updates `param_groups[0]` ONLY** at `meta_env.py:428` and `train.py:392`
  (+ `eval_brain.py:246`). **Any new param group's LR must be mirrored at those sites** or you
  silently break the Brain's proven-causal LR lever.
- **Network heads:** `src/lifelong_learning/agents/ppo/network.py` — `self.encoder`,
  `self.actor_head`, `self.critic_head` (LayerNorm for cand 3; param-tagging for cands 1/5).
- **Existing neuromod flags** for the pattern: `train_brain.py:394-397` + `846-861`.

1. **Decoupled critic LR** (cheapest, top prior) — critic head in its own param group, LR =
   Brain-LR × `critic_lr_scale`. **OPEN DESIGN CHOICE (user leaned to (b)):** (a) fixed scale
   independent of the Brain LR vs **(b) track Brain-LR × scale** (mirror at the 2 sites above).
   Default (b), scale 0.5. Condition e.g. `brain_neuromod_critic_lr_lo`.
2. **Dormant-neuron reset (ReDo, Sokar 2023)** — reset ~dormant units; probe dormant-fraction.
   Touches train loop/optimizer (user signed off).
3. **Plasticity normalization** — static LayerNorm on encoder (Lyle 2023); also the A1 test.
4. **Surprise-triggered exploration spike** — TD-error change-point (NOT the code) spikes
   ent/intrinsic at detected switches. Touches train loop (user signed off).
5. **Two-timescale encoder/heads** (cheap) — encoder LR scaled below heads; code-free version
   of the null `gradgate`. Same param-group + LR-mirror mechanics as cand 1.

Suggested order: 1 → 5 (share mechanics) → 3 → 2 → 4. Add a unit test per mechanism; screen
each at **n≥8 seed 1..8** on the box. Cheapest first so the box gets LOOP-0007 work soon.

## Box: how to drive it (this is the continuous-compute engine)
- `ssh -p 26061 -o StrictHostKeyChecking=no root@76.67.137.57`; repo at
  **`/workspace/Lifelong-Learning`** (NOT `~`), branch `autoresearch-run-20260706`.
- **Launch pattern** (per GPU, self-detaches, auto-pushes to `origin/results`):
  `cd /workspace/Lifelong-Learning && export PATH=/venv/main/bin:$PATH && CUDA_VISIBLE_DEVICES=<g> MAX_PARALLEL=1 RESUME=1 AUTO_PUSH=1 bash scripts/cloud/run_sweep.sh scoutv2 "<condition>" "<seed>" <out_dir>`
- New conditions reach the box via commit+push on `autoresearch-run-20260706` then
  `git pull --ff-only` on the box.
- **User standing directive: keep the box running CONSTANTLY** (12h/day rail lifted; destroy is
  human-only and OFF the table until they say). Memory `never-idle-box-rerun-exploration`.
  Always have the next wave queued. When the current holding wave finishes (~06:00Z), backfill
  immediately — finish `auxcode_010` n=8 (seeds 6,7,8) to buy implementation time, then switch
  to LOOP-0007 conditions as they land.
- **HOME 5070 is HELD IDLE** (user-directed; queue exhausted). Do not restart it without new
  human-curated hypotheses.

## Gotchas (each cost real time this session)
- **Session-scoped watchers:** background poll tasks die with their session; the OS-detached
  box sweeps + home trials survive. Re-arm watchers from YOUR session. Pattern used: a bash
  loop polling `summary.csv` count via ssh, launched with `run_in_background`.
- **SSH login banner** ("Welcome to vast.ai…") pollutes stdout — parse box output by GREPPING
  a marker line (`grep -oE 'SUMMARY_COUNT=[0-9]+'`), never by line position.
- **Box venv:** non-interactive ssh misses it — prefix `PATH=/venv/main/bin:$PATH` or you get
  `python: command not found`.
- **Box git auth:** `.git-credentials` store is configured; `AUTO_PUSH=1` pushes work without
  `GH_TOKEN` in the env. No HARD-STOP as long as `origin/results` keeps receiving commits.
- **No commit/push Mon–Fri 09:00–17:30 local** (memory `no-commit-business-hours`). It is
  currently after-hours (Mon ~22:00 local); commits are fine now.
- **The n=3 trap:** never promote/confirm on n=3 (or n=1). It produced two false leads this
  campaign. Screen LOOP-0007 at n≥8 from the start.
- **summary.csv columns:** composite = col 5, hit_rate_80_mean = col 25 (after collapsing the
  quoted `"[…]"` seeds field with `sed 's/"\[[^]]*\]"/SEEDS/'`).

## Next steps
1. **Re-arm the box watcher** immediately (poll the 4 `confirm_ld_aux010_s{2,3,4,5}` summary.csv;
   fire on all-4-present or GPU stall). On completion, pull results, then **backfill**
   `auxcode_010` seeds 6,7,8 (finish the dose-response n=8; keeps box busy per the directive).
2. **Implement candidate 1** (decoupled critic LR) in the worktree using the integration points
   above; resolve the (a)-vs-(b) choice as (b); add a unit test; commit+push; `git pull` on box;
   screen `brain_neuromod_critic_lr_lo` at n=8. Then candidate 5, 3, 2, 4.
3. Gate each LOOP-0007 condition with the fork **at n≥8**; record in the living doc.
4. **LOOP-0006 close-out** (can interleave): register fork row + research-log 0007 NULL outcome
   + note 0003 update (same commit for register+fork), finalize LOOP-0006 loop note, merge the
   run worktree branch.

## References
- `autoresearch/live/RUN-20260706.md` — living doc (ground truth; full verdict/incident log).
- `docs/research-notes/0004-code-free-plasticity-stability.md` — LOOP-0007 family (note).
- `docs/autoresearch-loops/LOOP-0007-brainstorm.md` — loop note (5 candidates).
- `docs/autoresearch-loops/LOOP-0005-brainstorm.md` — template + closed-family guards.
- Commits: `d9177f3` (LOOP-0007 brainstorm, Auto-Research); `d32815a` (LOOP-0006 follow-up
  conditions, run branch); results on `origin/results` (tip `8576032`).
- Skill `/autoresearch-run` (`.claude/skills/autoresearch-run/SKILL.md`) — dual-box orchestration.
- Prior hand-off: `docs/hand-offs/2026-07-06-loop-0006-dual-box-gate-loop.md`.
