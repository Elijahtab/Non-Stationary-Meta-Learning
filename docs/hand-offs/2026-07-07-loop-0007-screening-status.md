# Hand-off: LOOP-0007 screening — 4 of 5 candidates down, ReDo is the live thread (2026-07-07)

## Goal
LOOP-0006 (code-directed neuromod family) closed **NULL** and the pivot to **LOOP-0007**
(code-free plasticity & stability family) is well underway: all 5 candidates are implemented
and screening at n≥8 on the Vast box. Job now: **drive the last two candidates (ReDo, surprise-
spike) to their n=8 verdicts, decide LOOP-0007's outcome, and keep the box productive** — while
holding the n≥8 discipline that LOOP-0006 taught us the hard way.

> This work spans multiple sessions. The **living doc `autoresearch/live/RUN-20260706.md` is the
> real-time ground truth** — read its Master Decisions + Box Gate Log + Verdicts tails first;
> this hand-off is a point-in-time snapshot (16:16Z) from the artifacts, and the box may have
> advanced since.

## State
- **Branch:** run branch `autoresearch-run-20260706` (worktree `..\LL-run-20260706`) @ `bae342c`;
  main working branch `Auto-Research` @ `0e845cc`. Box is on `bae342c`.
- **Done & verified (across sessions):**
  - **LOOP-0006 fully closed NULL** — register fork row adopted + null outcome (`0893f3f`),
    loop note finalized (`bae342c`). No condition beat control at n=8; the two n=3 "reliability
    leads" (critic_code, auxcode_hi) washed out at n=8; gradgate_gain failed holdout.
  - **All 5 LOOP-0007 candidates implemented, tested, pushed** (55 tests pass): cand1
    `critic_lr_lo` (`7d4c651`), cand5 `encoder_lr_lo` (`0477a63`), cand3 `plasticity_norm`
    (`e35b5a3`), cand4 `surprise_spike` (`f758a45`), cand2 `redo` (`fa864a1`). All code-free,
    flag-guarded default-off.
  - **LOOP-0007 verdicts so far (n=8 unless noted):**
    - cand1 `critic_lr_lo`: **REJECT** (comp 0.5490, hit80 0.8248 — neither fork path).
    - cand5 `encoder_lr_lo`: **REJECT** (comp 0.5531 flat, hit80 0.7589 **−7pts DEGRADED**);
      two-timescale retired.
    - cand3 `plasticity_norm`: **KILLED at n=2** (pre-registered kill) — implies A1 (plasticity
      loss is the bottleneck) is likely FALSE on this task.
- **In progress (box RUNNING, ~99% all 4 GPUs at snapshot):** finishing n=8 for the last two —
  currently `redo` seeds 5,6 + `surprise_spike` seeds 7,8.
  - **`redo` partial (n=2): comp 0.5586, 0.5562 — BOTH ABOVE control mean 0.5534.** hit80
    0.8125/0.8304 (not above baseline). ⚠️ **This is the ONLY live signal — and it is exactly
    the small-n pattern that faked us out in LOOP-0006. DO NOT get excited until n=8.**
  - `surprise_spike` partial (n=5: seeds 1,2,3,4,6): comp mean ≈ 0.546, hit80 ≈ 0.821 — both
    **below control → trending NULL.**
- **Uncommitted:** worktree clean. `docs/hand-offs/` untracked (expected).

## Key decisions & findings
- **The one hard-won rule: screen/confirm at n≥8, never n=3.** LOOP-0006 produced two *false*
  holdout-"confirmed" leads purely because n=3 over-selected lucky seed triples. The reliability
  fork (accept if composite non-degraded AND hit80 > control CI-upper, else composite improves)
  still stands — but only at n≥8. Applies doubly to `redo`'s n=2 composite blip.
- **Two families now closed** (forward-modulation LOOP-0002/04; code-directed LOOP-0005/06 null).
  LOOP-0007 tests whether attacking the *measured* pathologies (critic whiplash, plasticity loss)
  code-free helps. So far cands 1/3/5 say no; the plasticity-loss premise (A1) looks false
  (plasticity_norm killed, encoder two-timescale hurt).
- **Box ownership has changed hands repeatedly** — watchers are session-scoped; the OS-detached
  sweeps + AUTO_PUSH survive. Stale duplicate watchers have fired before (`bqc87t6xp`,
  `b0wp96r4j`). **Establish a single owner: re-arm ONE box watcher and don't double-drive.**

## Gotchas
- **Box:** `ssh -p 26061 root@76.67.137.57`, repo at `/workspace/Lifelong-Learning`. Prefix
  `PATH=/venv/main/bin:$PATH` on non-interactive ssh. Launch pattern:
  `CUDA_VISIBLE_DEVICES=<g> MAX_PARALLEL=1 RESUME=1 AUTO_PUSH=1 bash scripts/cloud/run_sweep.sh scoutv2 "<cond>" "<seed>" <out>`.
- Parse box output by grepping a marker, never line position (ssh login banner pollutes stdout).
- summary.csv: composite = col 5, hit_rate_80_mean = col 25 (after `sed 's/"\[[^]]*\]"/SEEDS/'`).
- **No commit/push Mon–Fri 09:00–17:30 local** (memory `no-commit-business-hours`) — it is
  currently within business hours (~09:16 local Tue), so hold commits until 17:30.
- User standing directive: **keep the box running constantly** (rail lifted; destroy is
  human-only). But LOOP-0007 is trending null — see decision point below.

## Next steps
1. **Re-arm a single box watcher** on the in-flight dirs (`confirm_redo_*`, `confirm_ss_*`);
   don't double-drive if another session is active.
2. **Gate `redo` at n=8** (the live thread). If its composite mean stays > 0.5534 across all 8
   seeds → it's the first real signal; route to holdout (seeds 11/23/37). If it regresses to the
   band like every prior n=3 blip → REJECT. **Hold the n≥8 line.**
3. **Gate `surprise_spike` at n=8** — expected REJECT on current trend.
4. **If all 5 reject → LOOP-0007 closes NULL.** That's three closed families; surface to the
   user a strategic check-in (the "run box constantly" directive was premised on promising
   results — a null trend warrants deciding: new family / wind down / change instrument). If
   `redo` survives, pursue it (holdout, then n=8 combos) before any new brainstorm.
5. Keep the box busy per directive until that decision (finish any missing seeds; don't idle).

## References
- `autoresearch/live/RUN-20260706.md` — living doc (ground truth; screening plan at Master
  Decisions 06:12Z W1–W10, verdicts at 11:32Z / 12:53Z).
- `docs/research-notes/0004-code-free-plasticity-stability.md` — LOOP-0007 family + predictions.
- `docs/autoresearch-loops/LOOP-0007-brainstorm.md`; `docs/autoresearch-loops/LOOP-0006-*.md`
  (closed null).
- Results: `origin/results` (LOOP-0007 dirs `confirm_redo_s*`, `confirm_ss_s*`,
  `confirm_lr_crit_s*`, `confirm_lr_enc_s*`, plus the LOOP-0006 `confirm_ld_*`).
- Prior hand-offs: `2026-07-06-loop-0007-implementation-pickup.md` (now largely executed),
  `2026-07-06-loop-0006-dual-box-gate-loop.md`.
