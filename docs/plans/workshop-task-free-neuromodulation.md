# Plan — Workshop Paper: Task-Free Learned Neuromodulation for Continual RL

**Status:** active · **Created:** 2026-06-16 · **Owner:** Elijah Tabachnik
**Target:** first arXiv drop via a NeurIPS-2026 continual-learning / meta-learning workshop
(~4–8 weeks). Full version later → CoLLAs / TMLR.

This plan extends the **Meta-Learning ("Brain") subsystem** of the catastrophic-forgetting
project into a standalone, publishable result. Architecture background lives in
[../spec/](../spec/) (start with [01-research-flow.md](../spec/01-research-flow.md) and
[05-neuromodulation.md](../spec/05-neuromodulation.md)).

---

## 1. The claim (one sentence)

> A **task-free** outer RL controller learns to emit a feature-gating **context code** that
> infers **hidden** regime switches from inner training dynamics and accelerates post-switch
> recovery — with **no task labels** and **a single network** (no per-regime models).

Supporting claims:
- The Brain's scalar levers (lr / entropy / curiosity / replay / anchoring) are meta-learned
  *plasticity control*; neuromodulation adds *learned routing* on top.
- The learned context code is **regime-informative** even though it was trained without labels
  (the headline analysis result).

## 2. Why it's novel (positioning)

Closest neighbors and the differentiator we must defend:

| Prior work | What it does | Our difference |
| --- | --- | --- |
| ANML (Beaulieu 2020), Backpropamine (Miconi 2019) | neuromodulatory gating of activations | supervised / task boundaries → **we use online RL with hidden boundaries + an RL meta-controller** |
| FiLM (Perez 2018), Hypernets-CL (von Oswald 2020), SupSup (Wortsman 2020), PackNet (Mallya 2018) | condition computation on a **task ID** | **we condition on training *dynamics*, not identity** |
| LPG (Oh 2020), Meta-SGD | meta-learn *what/how to optimize* | **we meta-learn *how to modulate computation* online** |

**Citable gap:** *task-free, dynamics-conditioned neuromodulation produced by an outer RL agent.*
The reviewer's first question will be "isn't this ANML?" — the answer (RL, hidden regimes,
dynamics-conditioned, no task labels) must be airtight in the intro and supported by the oracle
baseline + decoding probe.

## 3. Baseline reality

From paper §4.3 (single seed, fast-switch): static `0.4638` → heuristic neuromod `0.6989` →
non-neuromod Brain `0.7307` → neuromod Brain `0.7498`. Suggestive, **not yet evidence** — the
Brain-vs-Brain gap is within plausible seed noise. The workshop paper's job: turn one sharp
claim into something a skeptic can't wave away (seeds + CIs + baselines + a mechanism).

## 3a. Verified finding (2026-06-16): the decoder is frozen

A gradient check confirms the code→mask **decoder never trains** — `set_context_code` decodes
under `no_grad` and the mask enters `forward` as a detached buffer, so `decoder.weight.grad is
None` after an inner backward (encoder/head grads are populated). The decoder is a **fixed
random orthogonal projection**; the **Brain's context code is the only learned part** of the
modulation pathway (reservoir-like). Implications:

- **Framing:** strengthens the "task-free *code* is what's learned" story; the oracle baseline
  also passes its one-hot through the same frozen decoder.
- **New ablation / novelty lever (cheap, high value):** a **trainable decoder** (recompute the
  mask in-graph, or train it under a small auxiliary objective). "Frozen vs trainable decoder"
  is a clean comparison and a plausible accuracy win — add it to §6 #5.

## 4. Compute envelope

Budget: **~$200/mo** cloud GPU. MiniGrid models are tiny and largely CPU-bound, so target a
cheap card on an interruptible marketplace + Brain-checkpoint resume.

| Option | ~$/hr | GPU-hr / $200 |
| --- | --- | --- |
| RunPod RTX 4090 (community) | $0.34 | ~590 |
| Vast.ai RTX 4090 (interruptible) | ~$0.29–0.40 | ~500–690 |
| Vast.ai RTX 3090 / 3090-Ti | ~$0.08–0.20 | ~1,000–2,500 |

Plan around **~500 GPU-hr/mo** (4090-conservative). Throughput multipliers: run several seeds
concurrently per GPU; use interruptible + resume.

### Measured (timing pilot, 2026-06-16, RTX 5070, single worker)
Inner Dyna-PPO throughput **≈ 1,000 inner steps/sec wall-clock** (full loop: collect + PPO +
WM-train + dream + replay), ~40 s per 40k-step episode, ~11 s startup. Extrapolated:

| Run type | config | est. wall-clock | est. GPU-hr |
| --- | --- | --- | --- |
| 5×5 Brain run | 800k inner × 50 episodes, `brain_num_envs=4` | ~3 h | **~3.5** |
| 8×8 Brain run | 800k inner × 50 episodes, `brain_num_envs=4` | ~5 h | **~5–7** |
| No-Brain baseline / oracle | single 800k inner run | ~13 min | **~0.25** |

### ⚠️ Corrected by calibration (2026-06-17) — earlier estimate was ~4–5× too optimistic
A real `scout5x5` run measured **~15.5 hr/cell** (50 *serial* 800k-step inner runs @ ~1,100 s
each), and **`brain_num_envs=4 async` gave ~no speedup** on a single box — the work is
**CPU-bound** MiniGrid env-stepping, so total core count (not GPU) is the throughput lever. My
3.5 h figure wrongly assumed async ≈ 4×. Implications: full local 5-seed `scout5x5` sweep ≈
**1–2 weeks** → impractical on one box.

**First real data point** (`scout5x5`, `brain_neuromod`, seed 0,
[cell1 score](../../sweeps/calib_scout5x5/cell1_neuromod_score.json)): **composite 0.8077**,
post-switch-window success 0.710, **hit_rate_80 = 0.996** (recovers to 80% after 99.6% of
1,400 switches; median 18.8k steps), hit_rate_95 = 0.96. Beats the 0.75 target and the paper's
0.7498 — encouraging, but n=1.

**Throughput fix (adopted):** run on a **many-core cloud box** with cells **concurrent**
(`run_seed_sweep.py --max-parallel`, default `nproc/8`), plus a lighter **`calib5x5`** preset
(400k inner / 30 episodes / `brain_num_envs=1`) for fast iteration. See
[cloud-setup.md](./cloud-setup.md): `calib5x5` 5-seed ≈ ~11 hr/~$17, `scout5x5` 5-seed ≈ ~40
hr/~$60 on a 64-vCPU box — both within budget.

### Config presets (`scripts/run_seed_sweep.py`) — use the right one for the right job

| Preset | Purpose | Grid / inner | Brain training | Comparable to paper? |
| --- | --- | --- | --- | --- |
| `calib5x5` | **iteration ONLY** — fast *relative* signal | 5×5, 400k/100k (3 switches) | 1 env sync, 30 ep, no pretrain, `decision_interval=10` | **No** (easier task, under-trained Brain) |
| `scout5x5` | intermediate / matches the frozen scout benchmark | 5×5, 800k/100k (7 switches) | 4 env async, 50 ep | Partly (still 5×5) |
| `paper8x8` | **paper-faithful — final headline numbers** | 8×8, 800k/100k (7 switches), `inner_num_envs=16` | 8 env async, **`decision_interval=1`**, pretrain 1, ep_mem 100k | **Yes** — matches `runs/brain_2_regimes_8x8_neuromod` (0.7498 ref); only `brain_episodes` differs (50 vs paper's resumed 130) |

**Workflow:** iterate on `calib5x5` to find which conditions clearly separate → **confirm the
survivors on `paper8x8`** for the headline table (directly comparable to the paper). `calib5x5`
numbers are for *relative* comparison only and must NOT be reported against the paper's absolute
scores (5×5 is easier — e.g. the n=1 `scout5x5` 0.8077 looks better than the paper's 8×8 0.7498
simply because the task is easier).

> **`paper8x8` cost caveat:** faithful = expensive. `decision_interval=1` + `brain_num_envs=8`
> async means one cell already uses ~8×16 envs and runs ~20–40 hr, so keep `--max-parallel` at
> 1–2 even on a big box. Use `paper8x8` for a *final confirmation* of the surviving conditions
> with fewer seeds (e.g. 3), not the full ablation matrix. Bump `brain_episodes` 50→130 only for
> a single best-config confirmation.

## 5. Experiment design (Phase 1 = the paper, 5×5)

### Headline comparison — 5 seeds each
| Condition | Role | Brain? | How |
| --- | --- | --- | --- |
| Static Dyna-PPO (tuned) | honest strong baseline | no | `train_ppo.py`, tuned static HPs |
| Online EWC / SI | task-free CL baseline | no | inner-agent regularizer (to add) |
| Experience replay / A-GEM | task-free CL baseline | no | replay levers (have) / A-GEM (to add) |
| Brain, neuromod **OFF** | ablation | yes | `--disable_neuromodulation` |
| **Brain, neuromod ON** | **method** | yes | default |
| Brain, **random/frozen** code | info-content control | yes | `--context_code_source random` |
| Task-ID-conditioned modulation | **oracle upper bound** | no/fixed | `--context_code_source oracle` |
| Heuristic neuromod | reference (have) | no | pretrain heuristic only |

### Ablations — 3 seeds each
- Modulation site: actor-only / critic-only / shared.
- Context dim: 8 / 16 / 32.
- Mask form: suppressive `(0,1]` vs **affine (γ,β)** (to add in `neuromod.py`).
- (cheap) decision_interval sensitivity.

### Metrics
- Primary: post-switch **recovery speed** (`median_steps_to_80/95`) and
  `mean_post_switch_window_success_rate` — reuse `score_brain_run` so it's identical to the
  frozen scorer.
- CL metrics: GEM-style **BWT / FWT** (forward/backward transfer) adapted to RL.
- Report mean ± **bootstrap 95% CI** across seeds + a significance test.

### Analysis (≈free — reuses existing `*_data.json` logs)
1. **Regime-decoding probe** (money figure): logistic regression `context code → hidden regime`;
   report k-fold accuracy vs chance. Trained-without-labels code that decodes the hidden regime
   = the result.
2. **Mask clustering** by regime (silhouette / 2-D projection).
3. **Causal lesion**: clamp the code to the wrong regime at eval → predictable collapse ⇒ routing,
   not noise.
4. Quantify the §4.3.1 "value head moves more than policy" effect across conditions.

### Phase 2 (full version, later)
Harder env for headline conditions only (8×8 / 4 regimes / random vs cyclic schedules; ideally
one non-MiniGrid env), input-conditioned (FiLM-style) modulation upgrade, larger seed counts.

## 6. Engineering tasks (priority order)

1. **Seed-sweep runner + aggregation** — `scripts/run_seed_sweep.py`: loop conditions×seeds,
   invoke `train_brain` (or `train_ppo`), score each run dir, emit tidy CSV/JSON with mean ±
   bootstrap CI. *(Unlocks everything.)* ✅ DONE — incl. `--max-parallel` concurrency and
   `--resume` (auto per-cell resume from latest checkpoint + skip-rescore completed cells;
   spot-safe). Presets: `pilot` / `calib5x5` / `scout5x5` / `paper8x8`.
2. **Oracle + random-code paths** — additive `context_code_source ∈ {brain,random,oracle,zero}`
   flag in `MetaEnv` + `train_brain` CLI; defaults preserve current behavior. ← build now (b)
3. **Regime-decoding probe** — `scripts/analyze_regime_decoding.py`: torch logistic regression
   (no sklearn), k-fold CV, chance baseline, per-run + aggregate. ← build now (b)
4. **Task-free CL baselines** — online EWC / SI / A-GEM on the inner agent.
5. **Affine (γ,β) mask** variant in `neuromod.py` (editable surface; novelty bump).
   Also add a **trainable-decoder** variant (see §3a) — frozen vs trainable is a clean ablation.
6. **Lesion + mask-clustering** analysis scripts.
7. **BWT/FWT** computation from logs.

## 7. Immediate next steps

- [x] **(a)** Timing pilot — done 2026-06-16: ~1,000 inner steps/s; 5×5 Brain run ≈ 3.5 GPU-hr;
      Phase-1 ≈ 170–250 GPU-hr (see §4). Budget confirmed sufficient.
- [ ] **(b)** Land tasks #1–#3 above as runnable scaffolding (+ a smoke test for the new flag).
- [ ] Tune the static Dyna-PPO baseline (so the strong baseline is genuinely strong).
- [ ] Draft intro + related-work positioning around the ANML/FiLM differentiator.
- [ ] Confirm the target workshop + deadline.

## 8. Risks / watch-items

- **Effect size**: the Brain-vs-Brain gap may not survive seeds. Mitigation: the decoding probe
  + oracle gap + lesion give a mechanism story even if the headline number is modest.
- **Compute creep**: meta-training is inner-runs-deep; keep Phase 1 on 5×5, parallelize seeds.
- **"Just ANML?"**: defend with the task-free/RL/dynamics-conditioned framing + oracle baseline.
- **Code vs paper drift**: action mapping is linear-to-absolute-bounds, not multiplicative
  (see [04-outer-brain-and-metaenv.md](../spec/04-outer-brain-and-metaenv.md)); describe the
  code's real behavior in the paper.
