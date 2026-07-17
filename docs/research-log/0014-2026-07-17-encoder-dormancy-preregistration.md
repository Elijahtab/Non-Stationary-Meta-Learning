# 0014 — 2026-07-17 — Encoder-dormancy probe (W0c): pre-registration (LOOP-0016)

**Registered BEFORE any results exist.** User direction 2026-07-17: launch a new autoresearch
run from the post-memory-campaign state, $0/home only, no box. Context: paper claim **C4**
("plasticity loss is absent at this horizon — dormant-unit fraction falls 0.9→0.35") rests on
a **heads-only** probe (`redo_reset_heads`, LOOP-0007); encoder dormancy has never been
measured — the action tree flagged this as C4's blind spot and made it the last Wave-0 item
(W0c). This rung closes it before the paper ships C4.

## Mechanism (committed pre-launch, `dormancy_probe_interval`)

A **probe-only** dormancy measurement — no resets, no optimizer writes, nothing fed back into
training — flag-guarded, default-off, in `train.py`. Every `dormancy_probe_interval` updates
(the arm uses 1 = every update, ~390 points/run), one extra no-grad forward pass on the
current obs batch (`s.obs_t`, the same reference batch the LOOP-0007 probe used) with forward
hooks captures post-ReLU activations at five sites:

- **encoder conv1/conv2/conv3** (32/64/64 channels): per-**channel** normalized mean-abs
  activation over batch × spatial positions (the Sokar 2023 conv convention);
- **actor/critic head hidden layers** (256 units each): per-**unit** score, byte-identical to
  the `redo_reset_heads` statistic that produced C4's trajectory.

Score = mean-abs / (layer-mean mean-abs + 1e-9); dormant iff score ≤ τ. Primary τ = 0.025
(= `REDO_TAU`, C4's threshold); τ = 0.1 logged as descriptive-only secondary
(`dormancy10_*`). Logged as `brain_neuromod/dormancy_{conv1,conv2,conv3,actor,critic}` into
the standard eval `*_data.json`.

## Arm (8 evals, home 5070, ~1.1 h, $0)

`dorm` × eval seeds 1–8: the standard eval protocol (8×8, 800k steps, 100k/regime, K=2,
16 envs), LOOP-0009 seed-1 ep130 Brain, **no mechanism flags** — the probe rides an
otherwise-plain trained-Brain arm, so trajectories describe the paper's baseline
configuration. Reference: archived `loop9_s1_model_e*` (n=16, same Brain/protocol/seeds) for
the composite-inertness check. Scored by `scripts/score_wave1.py dorm`.

## Statistics

Per run × layer: split the τ=0.025 trajectory into 8 equal contiguous windows; f_w = mean
dormant fraction in window w. Two derived quantities:

- **rise** = f_8 − min(f_1..f_4) — late dormancy vs the early-training trough. The trough
  reference (not f_1) is deliberate: orthogonal init makes *initial* dormancy high (C4's own
  trajectory starts at 0.9), and the claim under test is *accumulation across regime
  switches*, not the init transient.
- **fall** = f_8 − f_1 — the C4 shape (should be negative for heads).

## Pre-registered gates

- **P-W0c2 (instrument integrity — adjudicated FIRST; if it fails, P-W0c1 is not read):**
  (a) heads replicate C4's falling shape: median-over-runs fall < 0 for **both** actor and
  critic; (b) the probe is inert: dorm-arm composite vs archived control, Welch p > 0.05
  AND |Δmean| < 0.02. Failure of (b) means the "measurement-only" flag leaked into training —
  fix and re-run; nothing is interpretable.
- **P-W0c1 (encoder no-accumulation — the C4 scope check):** for **each** of conv1, conv2,
  conv3 at τ=0.025: median-over-runs rise ≤ +0.05. The bar is 10× smaller than the documented
  heads fall (0.55 absolute) and above window-mean noise.
- **Readings.** W0c1 pass → C4's "plasticity loss absent" extends to the encoder; the claim
  ships full-scope with the blind spot closed; branch F stays retired. W0c1 fail (any layer)
  → C4 gets a **scope correction** in the paper (heads-only) + note 0011 update, and branch F
  (encoder-targeted ReDo / shrink-and-perturb) acquires a live target at 8×8 — any
  intervention run is a NEW registration, human call. Either way the result is a paper
  sentence, not a method claim.
- **Extension rule:** any deciding median within ±0.02 of its bar → extend to n=16 before
  adjudicating.

## Registered risks

The probe batch is 16 envs × one timestep (~1,024 samples/channel at conv resolution, 16 at
the flattened heads — same as C4's probe); per-probe noise is real at τ=0.025 but the
window-mean over ~49 probes/window integrates it away. The normalized score is
scale-invariant *within* a layer; cross-layer comparisons are within-convention only
(channels vs units). Regime composition of `obs_t` varies across the run — dormant fractions
here are "dormant on the current task distribution," the same operational definition C4
already uses.
