# Hand-off: memory campaign complete (LOOP-0009→0015 closed), paper v2, strategy fork open (2026-07-16)

## Goal

Continue the neuromodulation/catastrophic-forgetting research program after a five-day campaign
(2026-07-14→16) that decomposed the +0.25 restoration ceiling, built the program's first learned
memory method, and adjudicated its limits. **Orient first:** read `AUTORESEARCH.md` (register +
loop list) and the pickup states of `docs/autoresearch-loops/LOOP-0015-fingerprint-selection.md`
and `LOOP-0014-step-trigger.md` — the loop notes are the canonical autoresearch hand-off; this
file only frames the cross-loop picture and the open strategy fork.

## State

- **Branch:** `Auto-Research` @ `e629ea0`, pushed; clean tree. (`doubleRL` is the repo default
  but the program lives on `Auto-Research`.)
- **Done (all pre-registered, all $0/home, loops closed same-day):**
  - LOOP-0009 replication CONFIRMED (+0.0455 pooled, n=16/arm) — note 0007; Brains archived on
    the `results` branch (`results/loop9_final/`).
  - LOOP-0011: ceiling decomposed — **heads ~90% / WM null / optimizer ~4%**, protocol-robust
    (+0.2434); K=3 premium flat → scaling axis dead — note 0012.
  - LOOP-0012/0013: **the method** — K=2 head bank + self-supervised value-loss trigger =
    **+0.1188 (n=16, p=4.7e-7, 52% of the oracle heads slice, zero oracle bits)**; precision
    hardening moves nothing — note 0013.
  - LOOP-0014: per-step trigger cuts lag 2,048→280 steps but **loses** (+0.0916) — false flips
    self-generate collapses (churn > lag savings) — note 0013 addendum.
  - LOOP-0015: drift-robust fingerprints fail on a third identified mechanism — the live reward
    head adapts within the trigger lag, so "no switch" wins even at true switches. **Selection
    family closed: 3 probes / 3 mechanisms (drift, churn, live-side adaptation)** — note 0013
    final addendum.
  - **Paper v2** (`paper/main.tex`, no local LaTeX — build on Overleaf): new §6 with Table 3 +
    Fig 3, abstract result (4), updated discussion/limitations. Never compiled end-to-end.
- **In progress:** nothing. No runs live, GPU idle, no box exists.
- **Uncommitted:** only this hand-off file.

## Key decisions & findings a new session should not re-derive

- The learned-WHICH is **adjudicated dead on this instrument** — do not build a fourth selector
  here. The recorded escape hatches: oracle identity, or an instrument whose regime signal
  outlives the learner's adaptation (regimes differing in *dynamics*, not just reward).
- The mem-Brain restore-gate premise is **bounded small by measurement** (same per-update lag
  floor as the static trigger; fire suppression worth +0.0007) — LOOP-0013/0014. Don't spend
  the box on it without a new rationale.
- MoWM (world-model banking) failed at its oracle upper bound; H4 (autoresearch screening) died
  its registered power-analysis death (note 0011); scaling (K=3, 16×16) lost its motivating
  hypothesis (note 0012 §K=3).
- LOOP-0012's original "value-error selection null" was an implementation artifact (missing
  slot allocation), caught and publicly corrected — the honest re-run is LOOP-0013's.
- Evidence locations: raw eval dirs are **local-only** (`evals/`, gitignored; ~1 GB); scores
  in `evals/wave1_scores.json`; the scoring path is `scripts/score_wave1.py` (batches: ladder,
  k3, g3, g3re, t15, stp, fp) on top of the validated `scripts/score_eval_dir.py` adapter.

## Gotchas

- Python: `myenv/Scripts/python.exe` with `PYTHONPATH=src`; full suite 161 tests ~100 s.
- Never edit `src/`/`scripts/` while an eval batch is in flight (evals import the live tree).
- Home box: 12 GB VRAM / 31 GB RAM → evals run sequentially (~7.5 min each); one Brain
  training at a time max (21 GB RAM/run).
- The Vast box from LOOP-0009 (63.142.193.28) is unreachable; **destroy was never confirmed in
  the Vast console** (stopped instances still bill storage) — worth a human check if not done.
- Pre-registration discipline is load-bearing here: gates go in `docs/research-log/` and are
  committed BEFORE results; register/index rows update in the same commit as any truth change.

## Next steps (strategy fork — user's call, none pre-approved)

1. **Paper polish + venue:** compile v2 on Overleaf (first full build of §6/Fig 3/enumitem),
   fix TODOs (affiliation, repo link), venue retarget (CoLLAs-tier per note 0008 §Venue).
2. **New-instrument port:** a dynamics-differing regime pair (the recorded route to a live
   learned-WHICH and a transferable ladder) — a *benchmark design* decision, master-level.
3. **Mem-Brain / box work:** only with a new rationale (see bounds above); ~$20–35, re-provision
   guide in LOOP-0009's note (RAM rule: ≥24 GB/GPU).
4. **Small loose end:** W0c encoder-dormancy probe (paper claim C4's blind spot) was never run —
   the last unfinished Wave-0 item (action tree status block).

## References

`AUTORESEARCH.md` · loop index `docs/autoresearch-loops/README.md` (0011–0015 are this
campaign) · notes `docs/research-notes/0011..0013` · pre-regs `docs/research-log/0008..0013` ·
paper `paper/` (README has the build/provenance map) · plan `docs/plans/2026-07-09-research-action-tree.md`
(status block current) · prior hand-off `2026-07-08-loop-0008-findings-strategic-fork.md`.
