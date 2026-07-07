# Hand-off: Three-family null — strategic inflection & the untested interface frontier (2026-07-07)

## Goal
Capture a **strategic brainstorm** (no code/box work) on what the accumulating null across three
mechanism families *means* for the neuromodulation paper, and where the program should go next.
The prior hand-off (`2026-07-07-loop-0007-screening-status.md`) covers the live screening state;
**this doc is the interpretation layer on top of it** — the through-line, the competing readings,
two concrete findings surfaced this session (a lost probe + a static-vs-adaptive confound), and
the decision fork. Reader: fresh agent, full repo access, zero access to this conversation.

## State
- **Branch:** working `Auto-Research` @ `0e845cc`. **The inner/Brain code referenced below lives on
  the run branch `autoresearch-run-20260706`** (box code) — not on `Auto-Research`. Check out /
  `git show autoresearch-run-20260706:<path>` to see the line numbers cited here.
- **This session was read-only:** no edits, no runs, no box contact. The only network action was a
  `git fetch origin results` (GitHub remote where the box auto-pushes — not the Vast box itself).
- **Box / data status:** `origin/results` is **unchanged since 15:31Z** — tip `confirm_ss_s6`,
  plus `confirm_redo_s1/s2`. **redo n=8 and surprise_spike n=8 are NOT yet pushed.** The box is
  either finishing quietly, done, or stalled — *not investigated* (no box contact per instruction).
- **Experimental state** otherwise unchanged from the prior hand-off: cands 1/5 REJECT (n=8),
  cand 3 plasticity_norm KILLED (degenerate op), cand 4 surprise_spike trending REJECT (n=6),
  **cand 2 redo the sole live thread** (n=4: composite ≈ control, hit80 ≈ 0.857 — the trap shape).
- **Uncommitted:** this hand-off (untracked). `docs/hand-offs/` untracked (expected). Nothing else.

## The three-family null (the through-line)
Against the frozen control (composite **0.5534 [0.5478, 0.5588]**, hit80 **0.829 [0.815, 0.846]**):

| Family | What it did to the Brain→inner interface | n=8 verdict |
| --- | --- | --- |
| Forward-modulation (LOOP-0002/04) | made the **8-d neuromod code** reshape the forward pass (20 variants) | **null** |
| Code-directed learning-dynamics (LOOP-0005/06) | rerouted the **same 8-d code** into grads / critic input / aux loss / optimizer | **null** (two n=3 "reliability leads" washed out; gradgate_gain failed holdout) |
| Code-free plasticity/stability (LOOP-0007) | added static inner mechanisms **beside** the interface (LR ratios, LayerNorm, ReDo, surprise detector) | 4/5 rejected; **trending null** |

The negative *escalates*: the code doesn't help the forward pass → doesn't help the learning
dynamics by any injection → even attacking the measured pathologies directly, code-free, doesn't
help. ~25+ diverse interventions, one answer. This is systematic elimination, not random failure.

## Four readings of what it means (with lean)
1. **Ceiling / no headroom.** When *diverse* mechanisms all fail identically, the usual common
   cause is the instrument, not the mechanisms. A well-tuned frozen agent on a 2-regime, short-
   horizon 8×8 switch may already capture nearly all achievable post-switch performance. Three
   families of null is strong circumstantial evidence. **Best-supported, most deflationary.**
2. **The measured pathologies are functional, not pathological.** The crispest *positive* finding
   hiding in the nulls. A2 pre-registered the risk ("if the critic re-fit is adaptive, damping it
   will HURT") — and critic_lr_lo was neutral, encoder-slowdown *hurt* reliability. Reading: the
   critic "whiplash" is the agent correctly re-fitting to a new regime, not a bug to suppress.
3. **Plasticity loss may simply be absent here (A1 false) — and we never checked.** Cands 2–3
   borrow from many-regime long-horizon continual-RL literature (Sokar 2023, Lyle 2023). On a
   2-regime fast switch the agent may never live long enough to go dormant. The real A1 test is
   the ReDo dormant-fraction probe — **which is not recoverable from the pushed results** (below).
4. **The metric is blind.** Composite = `mean_post_switch_window_success_rate`, a narrow window.
   Some interventions may change dynamics (asymptotic recovery, sample-efficiency-to-threshold) in
   ways this window can't see. Weakest-supported; the only reading that says "keep the mechanisms,
   change the ruler." Would touch the immutable scorer.

**Lean:** ① and ② are load-bearing — the program has fairly rigorously shown that *on this
instrument* an outer controller can't beat a well-tuned frozen inner agent by any imagined route,
and produced one genuine mechanistic result (critic re-fit is functional). ③ is a cheap loose end;
④ is the escape hatch if you're unwilling to accept ①.

## Finding A — the redo dormant-fraction probe is not in any pushed artifact
- redo at n=4 is the **canonical trap shape**: composite blipped above the mean at n=2 (0.559,
  0.556) then regressed to ≈ control by n=4, while hit80 popped to ~0.857 (> CI-upper 0.846). This
  is bit-for-bit the critic_code / auxcode_hi signature — both washed out at n=8 (0-for-2). The
  disciplined prior on redo surviving n=8 is **low**; hold the n≥8 line.
- redo's distinguishing feature was supposed to be a **mechanism probe**: `redo_reset_heads` in
  `train.py` returns `{head: dormant_fraction}` and `train.py:914-918` logs
  `brain_neuromod/redo_dormant_fraction_{actor,critic}` every 50 updates. When redo n=8 lands, the
  *first* number to read is the probe, not the composite — it adjudicates A1 for cands 2 AND 3
  regardless of score (probe moved + score held = real; probe never moved = A1 false, blip is noise).
- **But the probe is logged to the inner-PPO scalar stream, not to any pushed artifact.** Verified
  on `confirm_redo_s1`: absent from `summary.json`/`runs.json` (what the gate reads), **0 matches**
  in the raw `.log`, and absent from `brain_trends/…_data.json`. It survives only in the inner run's
  metrics under `runs/<inner>/` on the box — ephemeral, never pushed to `origin/results`.
- **Consequence:** the pre-registered A1 discipline ("probe must move or cands 2/3 fail regardless
  of score", note 0004) is **currently unenforceable from the results — the gate is blind to A1.**
  Recovering it needs the box's inner-run logs (box contact; may not survive a destroy).
- Minor flag: redo s1's scored `mean_post_switch_value_delta_abs = 0.283` — far below the **0.917**
  "critic whiplash" figure that motivates the whole critic-damping line (note 0004). One seed,
  redo-on, possibly a different normalization — but if the *scored* post-switch |ΔV| is ~0.3, the
  pathology is smaller than the headline, further weakening A2's premise. Worth reconciling.

## Finding B (open question) — the levers were screened STATIC, not Brain-controlled
The Brain action space is a **fixed 15-d vector** and has been since the paper config
(`neuromod.py:14-17`: `SCALAR_BRAIN_ACTION_DIM = 7`, `CONTEXT_CODE_DIM = 8`, `BRAIN_ACTION_DIM = 15`):
- `action[0:7]` — 7 scalar HP levers (`meta_env.py:444` `_apply_action`): `lr`, `ent_coef`,
  `intrinsic_coef`, `imagined_horizon`, `replay_ratio`, `replay_prioritization`, `anchoring_weight`.
- `action[7:15]` — the **8-d neuromodulation context code** (`BRAIN_CONTEXT_SLICE`).

`critic_lr_scale` / `encoder_lr_scale` are **not** in that action space — they're config floats
(default `1.0`), hard-set to `0.5` for the screens (`run_seed_sweep.py:293-294`). Mechanically
(`train.py:394-420`): scale ≠ 1.0 carves the critic/encoder into its own optimizer group at
`lr = main_lr × scale`; `main_lr` *is* Brain-controlled (`action[0]`) so the critic LR *tracks* the
Brain's lever, **but the 0.5 ratio is fixed** — the Brain cannot choose the asymmetry or time it to
the switch.

**So the cand 1/5 nulls falsify "a constant 2:1 critic/actor LR ratio helps" — NOT "the Brain can
learn to damp the critic at the switch, when it helps."** The adaptive hypothesis — the project's
actual thesis — was never tested. Same holds by construction for plasticity_norm and ReDo (both
wired to constants). **This is the single most important unresolved question on the board.**

## The interface frontier (what's actually unexplored)
Every family stayed **inside the fixed 15-d interface**, and two of three fought over the 8-d code
channel specifically. Even LOOP-0007, meant to break out, added mechanisms as **static config, never
as new Brain levers**. The meta-learning premise (Brain *learns* to adapt its control) has only ever
been exercised on the original, frozen channel set. Three axes have been held constant:
1. **Action dimensionality / new levers** — expose a *new* Brain action (critic-LR asymmetry,
   encoder/head timescale, a plasticity-reset trigger) and train the Brain to use it. **LOOP-0007
   already built every one of these mechanisms — it just wired them to constants.** Promoting one
   from config to `action[15]` is cheap and is the *only* test of the adaptive thesis.
2. **Sensing** — the Brain acts on a fixed signal set (`NUM_SIGNALS`); it may be acting *blind* to
   the switch. (surprise_spike conceded this, then built the detector *outside* the Brain.)
3. **Cadence** — `decision_interval = 10` caps response speed; never varied.

## Strategic fork & next-step options
The "keep the box running constantly" directive was **explicitly premised on promising results**
(living doc 20:20Z; WAVE-G note 04:43Z already flagged the premise falsified). Three closed families
make the marginal inner-mechanism variant a poor bet. Options:
- **Close cleanly first (cheap, do regardless):** let redo finish n=8, **read the dormant-fraction
  probe** (recover it from the box inner-run logs if the box is still up), close LOOP-0007.
- **Then choose:**
  - **(a) Write up the null.** Three-family null + the n=3→n=8 washout reproduced 3× (critic_code,
    auxcode_hi, redo) is a genuine *methods* contribution to a field littered with n=3 claims —
    possibly the most novel, defensible result the project has. Stop burning compute.
  - **(b) Pivot altitude, not mechanism.** Promote an existing static mechanism (critic_lr_scale
    above all) to a real Brain-controlled lever and train it. Reuses code already written/tested;
    the only experiment that probes the actual thesis. **Recommended**, gated on Finding B.
  - **(c/d) Instrument or metric pivot** (more regimes / longer horizon / recovery-curve metric).
    These respond to readings ① / ④ but **touch the immutable benchmark/scorer surface** — big,
    deliberate decisions, not autoresearch moves.
- **Lean:** close redo + read the probe → then **(b)**, unless Finding B comes back "the adaptive
  version was already tested" (it wasn't, per the code above). Surface the falsified-premise inflection
  to the user before committing more box compute.

## Gotchas
- **Code line numbers are on branch `autoresearch-run-20260706`, not `Auto-Research`.**
- The dormant probe is **not** in `origin/results`; only in ephemeral box inner-run logs.
- **n≥8 discipline** is non-negotiable — n=3 produced false leads three times.
- Standing rules: no commits Mon–Fri 09:00–17:30 local; box destroy is human-only; "never idle the
  box" — **but that directive's premise (promising results) is now in question**; flag before spending.
- Box access + parse recipes: see prior hand-off `2026-07-07-loop-0007-screening-status.md` "Gotchas".

## Next steps
1. When `origin/results` advances: gate **redo at n=8** with the reliability fork (n≥8 only). Expect
   regression to the band per the trap-shape prior; if it holds, route to holdout (seeds 11/23/37).
2. **Read redo's dormant-fraction probe** — from the box inner-run logs while the box is still up
   (`runs/<inner>/…`), since it is not in the pushed results. This adjudicates A1 for cands 2 & 3.
3. If redo rejects → **LOOP-0007 closes null** → surface the strategic fork above to the user
   (write-up vs adaptive-lever pivot). Do not auto-launch another inner-mechanism wave.
4. If pursuing option (b): the cheapest test is **critic_lr_scale as a Brain action dim** — the
   mechanism (decoupled optimizer group) already exists in `train.py`; the work is exposing it in
   the action space (`neuromod.py` dims + `meta_env.py._apply_action`) and re-screening at n≥8.

## References
- Prior hand-off (live state): `docs/hand-offs/2026-07-07-loop-0007-screening-status.md`.
- Living doc (ground truth): `autoresearch/live/RUN-20260706.md` (Master Decisions, Verdicts).
- Family note + predictions: `docs/research-notes/0004-code-free-plasticity-stability.md`;
  `docs/autoresearch-loops/LOOP-0007-brainstorm.md`; LOOP-0006 note (closed null).
- Code (branch `autoresearch-run-20260706`): `src/lifelong_learning/agents/brain/neuromod.py:14-17`
  (action dims) · `.../brain/meta_env.py:444` (`_apply_action` lever map) ·
  `.../agents/ppo/train.py:58-60,394-420` (LR-group carving), `:914-918` (dormant probe log) ·
  `scripts/run_seed_sweep.py:293-301` (LOOP-0007 condition defs).
- Results: `origin/results` dirs `confirm_redo_s*`, `confirm_ss_s*` (redo/surprise), plus the
  LOOP-0006 `confirm_ld_*`.
