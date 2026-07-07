# Session plan — 2026-07-07: redo endgame, recovery pass, and the oracle rung

**Context:** three mechanism families closed/trending null (~25+ interventions); `redo` is the
sole live thread and at n=6 passes BOTH fork paths (composite 0.5538 > control mean 0.5534;
hit80 0.8527 > CI-upper 0.846). W9 (redo seeds 7,8 → n=8 + 9,10 → n=10) lands ~18:15Z.
Full background: `docs/hand-offs/2026-07-07-loop-0007-screening-status.md` +
`docs/hand-offs/2026-07-07-three-family-null-strategic-inflection.md` + living doc
`autoresearch/live/RUN-20260706.md`.

**Standing-rules delta (this session):** the business-hours commit blocker is **RESCINDED**
by the user (2026-07-07) — commits/pushes allowed at any time. Still in force: n≥8 gate
discipline, never-idle-box, box destroy human-only, immutable benchmark/scorer surface.

**Organizing idea:** the program has been testing mechanisms on a ladder —
**no-Brain static floor (never scored on this instrument) → static mechanisms (tested → null)
→ oracle-timed (never tested, cheap) → Brain-learned (never tested, expensive)** — and has
skipped both the bottom and middle rungs. The bottom rung (D0) re-validates the paper-era
founding result (Table 2: static 0.4638 vs Brain 0.7307 avg success on the same fast-switch
schedule) at current scorer + n≥8 standards; the oracle rung (O1/O2) adjudicates the
strategic fork ((a) write up the null vs (b) promote an adaptive lever) far cheaper than
jumping to learned levers.

---

## Phase A — offline, free, start immediately (no box contact)

- **A1. Brain policy adaptivity probe** (prerequisite for option (b), Finding B).
  Pushed artifacts already contain a trained `brain_model.pt` per confirm cell (plus
  LOOP-0004 control runs). Load checkpoints locally and sweep the 19-dim observation over
  plausible post-switch ranges (success-rate collapse, `steps_since_surprise_spike`,
  `value_loss` spike); measure how much the policy mean actually moves per action dim.
  **Question:** is the trained Brain's policy meaningfully obs-dependent, or ~static?
  If ~static, a promoted lever would be used statically too (already-tested regime) and the
  binding constraint is sensing/cadence, not lever count. Scratch analysis only — no repo
  surface. Deliverable: **research note 0005** with the verdict.
- **A2. Pre-register the ladder designs** (research-notes convention: predictions before
  results): D0 + O1 + O2 below, with explicit predictions and kill/accept rules. For D0 also
  extract the tuned-static lever vector from pushed control `brain_trends` (offline, free).

## Phase B — box contact resumes at W9 completion (~18:15Z; on user go-ahead)

- **B1.** Re-arm **ONE** box watcher (single-owner rule; stale duplicates have fired before).
- **B2. Gate redo at n=8** with the fork; read n=10 as robustness. Caveat recorded: the n=8
  verdict is conditioned on a favorable 6-seed prefix — a truly-null redo still has ~coin-flip
  odds of passing — so **a pass routes to holdout (seeds 11/23/37), it does not become a claim.**
- **B3. Recovery pass** (regardless of verdict; I/O only, GPUs keep working):
  - Pull the **redo dormant-fraction probe** from box inner-run logs
    (`runs/<inner>/…`, keys `brain_neuromod/redo_dormant_fraction_{actor,critic}`) — the
    pre-registered A1 adjudicator for cands 2 AND 3; it exists nowhere else and box disks are
    ephemeral. Probe never moved → A1 false and redo's blip is suspect regardless of score.
  - **Re-push the ~20 result dirs missing from `origin/results`** (AUTO_PUSH race losses):
    `confirm_redo_s3/s4`, `confirm_ss_s5/s8`, `confirm_lr_crit_s1/s4/s6`,
    `confirm_lr_enc_s3/s5/s7`, `confirm_ld_cc_ext_s8`, `confirm_ld_auxhi_ext_s6`,
    `confirm_ld_gg_gain_s2`, `confirm_ld_cc_hold_s23/s37`, and any others found by a
    box-vs-branch diff. Verdicts are unaffected (gates read the box), but the durable per-seed
    archive for the paper is currently incomplete.
  - Tar/archive the inner-run metric streams for at least the redo cells.
- **B4. Branch on the redo verdict:**
  - **REJECT** → LOOP-0007 closes NULL (5/5). Box's next work = O1 screen (Phase D).
  - **PASS** → launch holdout seeds 11/23/37 immediately (3 cells, ~85 min). Holdout + probe
    movement together decide whether redo is the program's first real mechanism win.

## Phase C — infra + bookkeeping commits (interleave with B/D waves)

- **C1. Fix the AUTO_PUSH race** in `scripts/cloud/run_sweep.sh`: fetch + rebase + retry with
  backoff before push; fail loudly. This silently dropped ~20 dirs across two campaigns.
- **C2. Make registered probes durable:** write mechanism-probe scalars (e.g., dormant
  fractions) into `summary.json`/`runs.json` so no future gate is probe-blind (Finding A
  class of failure).
- **C3. LOOP-0007 close-out:** loop-note verdicts, note 0004 outcome updates (no stale
  predictions), register rows, commit `docs/hand-offs/` + this plan.

## Phase D — bottom-rung control + the oracle rung (cheap decisive tests, gate before the fork)

- **D0. Bottom-rung control — does the Brain beat a tuned static baseline AT ALL on this
  instrument?** (Highest interpretive leverage on the board; run FIRST.)
  Context: the founding paper-era result (§4.3 Table 2, same 800k/100k fast-switch schedule:
  static 0.4638 avg success and never reaching 80% vs Brain 0.7307) predates the composite
  redesign and the n≥8 discipline, and its static baseline's tuning status is unknown — the
  Brain-vs-static gap conflates *static tuning value* with *adaptive control value*. No
  no-Brain/static-HP run has ever been scored with the scout composite (all 31 sweep
  conditions are `brain_*` variants; the 0.5534 control is itself a Brain run).
  - **`brain_constant_action` condition:** override the Brain's action with a fixed vector
    set to the mean lever values trained control Brains settle at (from pushed `brain_trends`
    `brain_hyperparams/mean_inner_*` — i.e., a *tuned* static baseline, the fair comparison).
    Same MetaEnv/scorer pipeline → directly comparable to control 0.5534. Flag-guarded,
    default-off; screen n=8 (~2 waves).
  - Optional backing: `scripts/eval_inner_hyperparams.py` (the purpose-built "Brain's static
    HPs on their own" tool, apparently never run) with scout-matched args.
  - **Pre-registered meaning:** Brain ≫ constant-action → paper-era finding is REAL at
    today's standards; the controller extracts adaptive value; the three-family null means
    the margin *above* a working controller is thin; option (b) extends something that works.
    Brain ≈ constant-action → the Brain adds ~no adaptive value here; the three-family null
    was over-determined (no adaptive signal to amplify); story shifts to write-up and/or
    instrument pivot (reading ① unified with ②) — and the paper's Table-2 hierarchy needs a
    correction at the new standards.
  - Cross-validates with A1: a ~static Brain policy predicts Brain ≈ constant-action; an
    obs-dependent policy makes D0 measure whether that adaptivity buys composite points.
- **D1. O1 — oracle-timed critic-LR damp** (`critic_lr_oracle`). The inner loop already
  detects true regime switches (anchor-model snapshot in Phase A of `run_inner_update`), so
  oracle timing needs no new information channel: on detected switch, scale the critic LR
  group ×0.5 for K≈15 updates, else 1.0. Flag-guarded, default-off, opened surface
  (`train.py`). One variant only (no multiplicity). **Pre-registered meaning:** a
  perfectly-timed damp is an upper bound on any Brain-learned damp — if it fails at n=8, A2
  is dead and option (b)'s critic lever is dead with it; if it wins, it's the first positive
  result in three families AND a guaranteed-achievable target for the learned version.
  Adjacent prior: surprise_spike (a self-timed *exploration* transient) HURT — reading ②
  predicts O1 also fails. Cost: ~2 waves ≈ 3 h box for n=8.
- **D2. O2 — policy-swap headroom topline** (diagnostic harness, never a paper mechanism).
  Snapshot the inner actor-critic per regime; on revisit, restore that regime's last
  snapshot. Measures the instrument's achievable post-switch ceiling. Control ≈ topline →
  reading ① (no headroom) is *proven*, and the null becomes a property of the instrument —
  write-up with confidence. Large gap → headroom exists and the mechanism search failed for
  findable reasons. Flag-guarded `train.py` addition; scorer/benchmark untouched.
  **Needs explicit user surface sign-off before implementation.** n≈4–8 cells.
- **D3.** Screen on the box under the never-idle rule, priority order **D0 → O1 → O2** (D0 is
  cheapest-to-interpret and every fork branch reads differently depending on its outcome).

## Phase E — strategic fork adjudication (user decision, evidence on the table)

| Evidence | favors (a) write up null | favors (b) adaptive lever | favors (c/d) instrument/metric |
| --- | --- | --- | --- |
| **D0 Brain vs tuned-static** | ≈ equal → null was over-determined, +a | Brain ≫ static → controller works, +b | ≈ equal → instrument may not reward adaptation, +c/d |
| redo n=8/n=10 + holdout | fails → +a | passes → pursue redo before any pivot | — |
| dormant probe (A1) | never moved → A1 false, +a | moved + score held → mechanism real | — |
| A1 adaptivity probe | Brain ~static → −b, +a | obs-dependent policy → +b | Brain ~static → sensing/cadence first |
| O1 oracle damp | null → A2 dead, +a | wins → +b with achievable target | — |
| O2 topline gap | ≈0 → ceiling proven, +a | large → −a | large → +c/d worth weighing |

Notes for the fork: the metric reading (④) is stronger than first ranked — research-log 0001
pre-registered the intent to add a reliability/speed facet (`hit_rate_95`/`steps_to_95`) for
the *final* benchmark; the whole discovery campaign has run on the narrowed calibration
instrument. A metric evolution is program history, not goalpost-moving. And regardless of
branch, the **methods contribution** (n=3→n=8 washout reproduced 3×, push-race and
probe-durability lessons) is paper-worthy on its own.

## Timeline sketch (times Z)

- **now–18:15** — Phase A (adaptivity probe, pre-registration note); draft C1/C2 patches.
- **~18:15** — W9 lands → Phase B (gate, recovery, holdout-or-close).
- **evening** — Phase C commits; D0 + O1 implementation; box screens holdout and/or D0→O1 waves.
- **later / next session** — D0/O1/O2 gates → Phase E fork decision with the user.
