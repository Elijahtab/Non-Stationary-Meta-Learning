# 0014 — Encoder dormancy accumulates in the first conv layer; C4 scope-corrected

**Status:** ✅ RESOLVED 2026-07-17 (LOOP-0016, same-day, $0) — **P-W0c2 PASS / P-W0c1 FAIL.**
**Hypothesis (as registered):** dormancy does not accumulate in the encoder — paper claim C4
("plasticity loss is absent at this horizon"), measured heads-only until now, extends to the
full network.
**Result: falsified for the first conv layer.** Median across 8 seeds at τ=0.025: conv1
dormant fraction **rises 0.125 → 0.479** over the run (rise vs early trough **+0.318**, 6×
the +0.05 bar; threshold-robust: 0.219 → 0.516 at τ=0.1) while conv2 (+0.008) and conv3
(−0.001) stay flat and the heads **fall** (actor 0.789 → 0.516, critic 0.788 → 0.605 —
replicating C4's falling shape at the eval protocol). The instrument is valid: probe-only
composite delta vs archived control +0.0115 (Welch p=0.586, n=8 vs 16); the extension list
is empty — adjudication final at n=8.
**Owner:** Elijah · **Relates to:** [log 0014 (pre-reg)](../research-log/0014-2026-07-17-encoder-dormancy-preregistration.md)
· [LOOP-0016](../autoresearch-loops/LOOP-0016-encoder-dormancy-probe.md) · [note 0011](./0011-wave0-desk-probes.md)
(Wave-0; W0c was the last item) · [note 0008](./0008-paper-skeleton.md) (claim C4) ·
[note 0012](./0012-ceiling-decomposition.md) (the encoder share this may explain).

## Method (one paragraph)

`dormancy_probe_interval=1` (probe-only: no resets, no optimizer writes, no RNG consumption —
unit-tested) on `dorm_e1..8`, the standard eval protocol with the LOOP-0009 seed-1 ep130
Brain and no mechanism flags. Five sites per update (~390 probes/run): encoder conv1/2/3
per-channel and actor/critic head hidden units per-unit, normalized mean-abs post-ReLU score,
dormant iff ≤ τ. Windowed stats (8 equal windows; f1/f8 = first/last window means) and gates
per the pre-registration; adjudicator `scripts/score_wave1.py dorm`; raw trajectories in each
eval's `*_data.json`, adjudicated numbers in `evals/wave1_scores.json` (local).

## Reading

- **C4 is scope-corrected, not withdrawn.** "Plasticity loss is absent at this horizon" is
  true of the actor/critic heads (where the ReDo intervention was aimed and where the claim
  was measured) and of conv2/3 — it is **false for conv1**, where nearly half the channels
  go dormant by run end. Paper sites updated: abstract result (2), contributions bullet,
  related-work sentence, §Plasticity probe, statistical-scope limitation.
- **Whether conv1 dormancy costs anything is untested.** No intervention has been run; the
  probe arm matches control by design, and every arm in the paper's ladder (including the
  +0.25 oracle ceiling) trained under this same accumulation. What CAN be said: it does not
  prevent the system from reaching the ceiling when knowledge is restored.
- **Interpretive link (not a claim):** the G-DECOMP ladder found heads+encoder restores
  +96.3% of the ceiling vs +89.6% for heads alone (note 0012) — a ~7pp encoder share.
  First-layer dormancy accumulation is a candidate mechanism for exactly that slice: worth
  one sentence when the paper discusses the decomposition, no more.
- **Branch F acquires a live target** — the action tree's contingency ("activates ONLY if
  probes show rising dormancy") now has its trigger condition met, at 8×8, layer-locally.
  Any encoder-targeted intervention (probe-guided ReDo on conv1, shrink-and-perturb) is a
  **new registration and a human call** (per log 0014's reading) — nothing launched.
- **Assumption made explicit:** the probe reads dormancy on the *current* obs batch (the
  same operational definition C4's heads probe used); conv1 channels dormant on the current
  task distribution could in principle re-activate under other inputs. The C4 correction
  inherits this definition — it compares like with like.

## Numbers (τ=0.025 medians over 8 runs, 8-window trajectories)

| site | f1 | f8 | rise vs early trough | fall (f8−f1) |
| --- | --- | --- | --- | --- |
| conv1 | 0.125 | 0.479 | **+0.318** | +0.318 |
| conv2 | 0.110 | 0.029 | +0.008 | −0.071 |
| conv3 | 0.090 | 0.036 | −0.001 | −0.049 |
| actor | 0.789 | 0.516 | −0.133 | −0.282 |
| critic | 0.788 | 0.605 | −0.124 | −0.186 |

(τ=0.1: conv1 0.219→0.516; conv2/3 and heads fall — same picture. f1 is the first-eighth
mean, so it already sits below the fresh-init values the LOOP-0007 scout probe reported.)
