# Confirmation Sweep: Actor-Only & Gain-Mask vs Frozen (multi-seed, on-box)

**Goal:** Adjudicate the two hit_rate_80 signals from the v2 scout campaign (actor-only +8.9pts,
gain-α0.5 +10.7pts at n=1) with a 3-arm × 8-seed sweep on a rented 4× RTX 3060 box.
**Status:** proposed 2026-07-05

## Summary

Both signal-showing mechanisms exist only as rolled-back trial diffs; Stage 1 re-implements them
as flag-guarded options (defaults = paper-faithful frozen behavior, exactly the
`trainable_neuromod` precedent) plus a sweep preset that mirrors `fast_switch_scout_v2`'s config
byte-for-byte. Because the local machine is mid-campaign (trials 3–5 screening), all edits happen
in a **git worktree on a side branch** the box clones directly — the local supervisor's snapshot
audit never sees them. The sweep itself follows the proven `run_sweep.sh` +
`CUDA_VISIBLE_DEVICES` pattern with results auto-pushed to the `results` branch.

## Decisions & answers

| Decision | Answer |
| --- | --- |
| Condition set | **3 arms × 8 seeds** (frozen / actor-only / gain α=0.5) = 24 cells |
| Box | **4× RTX 3060 12GB, 80 vCPU** (user-provided SSH; Ampere → stock torch works) |
| Local loop meanwhile | Trials 3–5 keep screening at n=1; a 4th arm is backfilled onto the box only if one of them shows the hit80 signature |

Flagged assumptions:

- **A1 — Preset mirrors the benchmark, not calib8x8:** the existing `calib8x8` preset
  ([run_seed_sweep.py:121](../../scripts/run_seed_sweep.py)) uses 30 sync Brain episodes — a
  different distribution than the benchmark's 4 episodes × 4 async envs. The new `scoutv2`
  preset copies `fast_switch_scout_v2.fixed_train_args` exactly (incl. `reward_mode: recovery`)
  plus its scoring params, so the sweep samples the same distribution the signal came from.
- **A2 — Two orthogonal flags, not one enum:** `--neuromod_actor_only` (bool; gates only the
  actor-head input in `forward_with_mask`, [network.py:120](../../src/lifelong_learning/agents/ppo/network.py))
  and `--neuromod_gain_alpha` (float, 0.0 = off; `1 + α·s·tanh(decoder(code))` in
  `decode_context_code`, [neuromod.py:123](../../src/lifelong_learning/agents/brain/neuromod.py)).
  Composable later for a factorial arm without new code.
- **A3 — Frozen control re-runs on the box** (8 fresh seeds) rather than importing local scores —
  same hardware, same wave, apples-to-apples.
- **A4 — Packing:** ~6.5 GB VRAM per async cell → plan 1 cell/GPU (6 waves ≈ 7–9 h); if measured
  VRAM ≤ 5.5 GB, opportunistically pack 2/GPU per the AUTORESEARCH backfill rule. 80 vCPU is
  never the constraint (~6 cores/cell).
- **A5 — hit_rate_80 is the registered secondary endpoint**; `run_seed_sweep` already emits
  bootstrap CIs for it ([run_seed_sweep.py:224](../../scripts/run_seed_sweep.py)). Composite
  stays primary. Prediction registered in research-log 0005 **before** results.

## Stages

### Stage 1 — Conditions + preset on a side branch (local worktree; ~1 h)

Cannot edit the main tree mid-campaign (managed-file edits would land in the running trial's
audit diff). So: `git worktree add ../LL-confirm confirmation-sweep-conditions`.

- `--neuromod_actor_only` + `--neuromod_gain_alpha` args in
  [train_brain.py:814](../../scripts/train_brain.py) region, threaded through
  [network.py:26](../../src/lifelong_learning/agents/ppo/network.py) →
  [neuromod.py:105](../../src/lifelong_learning/agents/brain/neuromod.py) constructors
  (`trainable_neuromod` at [train_brain.py:385](../../scripts/train_brain.py) is the template).
- Actor-only forward: mask multiplies the actor-head input only; critic reads raw features
  (~6 lines, per the trial's own notes). Gain: tanh branch in `decode_context_code`; zero-code
  → identity preserved in both modes.
- `CONDITIONS` entries ([run_seed_sweep.py:195](../../scripts/run_seed_sweep.py)):
  `brain_neuromod_actor_only`, `brain_neuromod_gain05`.
- `scoutv2` preset per A1.
- Tests (in-worktree, main venv via absolute path): actor-only → critic output invariant to
  context code; gain → mask ∈ [1−α, 1+α], zero-code identity; defaults → byte-identical
  behavior to current frozen path.
- Push branch. **Do not merge** into `trainable-vs-frozen-sweep` until the local campaign ends.

**Verify:** new tests green + existing `test_network.py`/`test_dyna_logic.py` green in the
worktree; a 1-episode pilot smoke run with each flag.

### Stage 2 — Box bootstrap (~20 min, needs user SSH string)

- `upload_secrets.sh` → clone repo `-b confirmation-sweep-conditions` → `bootstrap.sh`
  (Ampere: reuse image torch; smoke validates GPU end-to-end). Check
  `workspace_is_volume` and free disk as usual.

**Verify:** bootstrap smoke sweep scores; `nvidia-smi` clean.

### Stage 3 — Launch + registered prediction (~7–9 h wall-clock)

- Research-log **0005** first: prediction = actor-only and gain-α0.5 each beat frozen on
  hit_rate_80 with CI separation at n=8; composite within noise either way. Decision rule:
  CI-separated hit80 gain + no composite regression → promote to paper analysis (and consider
  the factorial arm); otherwise close both as "n=1 artifacts" and the inertness thesis stands.
- One `run_sweep.sh` invocation per GPU (`CUDA_VISIBLE_DEVICES=N`, `MAX_PARALLEL=1`,
  `RESUME=1`, `AUTO_PUSH=1`), conditions split across GPUs, seeds 1–8, out dirs
  `sweeps/confirm_g{0..3}`. Detached; survives SSH exit; auto-pushes to `results`.
- Monitor from local via SSH SPS checks; backfill spare capacity per the compute-check rule
  (4th arm from local trials 3–5 if a signal appears, else extra seeds on the leading arm).

**Verify:** healthy SPS per cell; `runs.csv` rows appearing; first wave scores within ~75 min.

### Stage 4 — Pull, analyze, record (~30 min)

- `pull_results.sh`; per-condition mean ± bootstrap CI for composite AND hit_rate_80.
- Research note **0003** (next free number): hypothesis → prediction → result → interpretation.
- Research-log 0005 Outcome filled; hypothesis_queue.md items 1–2 annotated with the n=8
  verdicts; box destroyed (no volume; everything on `results`).
- Merge `confirmation-sweep-conditions` into the main branch once the local campaign is done.

**Verify:** note + log committed; `results` branch has all 4 sweep dirs; box destroyed.

## Files touched

| File | Change |
| --- | --- |
| scripts/train_brain.py | two new CLI flags → constructor args |
| src/lifelong_learning/agents/ppo/network.py | `actor_only_neuromod` forward path |
| src/lifelong_learning/agents/brain/neuromod.py | `gain_alpha` decode branch |
| scripts/run_seed_sweep.py | `scoutv2` preset + 2 CONDITIONS entries |
| tests/test_network.py | invariance/range/default-equivalence tests |
| docs/research-log/0005-*.md (+README) | registered prediction, then outcome |
| docs/research-notes/0003-*.md (+README) | the confirmation-sweep lab note |
| config/hypothesis_queue.md | n=8 verdict annotations (Stage 4) |

## Risks & alternatives

- **Seed-override in run_frozen_benchmark.py rejected**: it's immutable surface and its frozen
  seeds are the point; the mirrored preset achieves the same sampling without touching it.
- **Distribution mismatch risk (A1)** if the preset drifts from the spec — mitigated by copying
  `fixed_train_args` programmatically in the preset definition comment and a test asserting
  preset == spec args.
- **VRAM estimate off** (6.5 GB is inferred, not measured on 3060): wave 1 measures it; worst
  case stays at 1 cell/GPU (the planned schedule).
- **Local campaign interference**: fully avoided via worktree + box-side execution; the only
  shared resource is the ledger-free `results` branch.

## Open questions

- After local trials 3–5 finish, keep the local loop running further free-pick screening
  trials (~$3.6 + ~85 min each), or pause screening until the confirmation verdict? (Doesn't
  block anything; default: pause after trial 5 — the queue is empty of untested priorities.)
