# 0010 — Related-work literature map: what the field says about "meta-control helps a little; memory is the bottleneck"

**Status:** 📚 SURVEY 2026-07-09 — literature groundwork for the paper skeleton
([note 0008](./0008-paper-skeleton.md) §2 Related Work). Not a hypothesis note; this is the
annotated bibliography + threat model for claims C1–C6.
**Owner:** Elijah (produced by a two-pass multi-agent verified web sweep, 2026-07-09).
**How produced:** pass 1 = 108-agent deep-research harness (5 search angles → 26 sources
fetched → 124 claims extracted → top 25 adversarially verified by 3-vote panels: 24 confirmed,
1 refuted); pass 2 = 52 papers individually web-verified for exact title/venue/year/arXiv/DOI
and mapped to C1–C6. Every entry below survived verification unless flagged ⚠.

## TL;DR for the Related Work section

1. **The tension the section must resolve (threat to C1's framing):** the canonical
   HP-meta-control results — Xu et al. meta-gradients (+30–80 pts median HNS on Atari),
   STAC/STACX (243%→364% median HNS), PBT, Agent57's bandit — all report **large** gains, and
   STAC's ablation shows performance improving *monotonically* as more HPs go under
   meta-control. **Every one of those results is stationary single-task RL.** No published
   head-to-head of dynamic HP control vs. a tuned-static baseline in a continual/forgetting
   regime was found in either pass. That is simultaneously our escape hatch, our burden of
   proof, and our novelty claim: C1+C6 appear to be the first *controlled* measurement of that
   contrast under regime switches.
2. **The strongest structural ally (C1-smallness + C2):** XdG (Masse et al., PNAS 2018) —
   context gating **alone** collapses to 61.4% over 100 tasks, stabilization alone to
   70.8–82.3%, the **combination** holds 95.4%. Gating-without-memory has a low ceiling —
   exactly our oracle_code null (C3) + O2 ceiling (C2) story, in supervised form. Reinforced
   by Eimer et al. (ICML 2023: well-tuned static HPO often suffices) and the AutoRL survey's
   open problem ("it is often not clear which hyperparameters need to be optimized dynamically
   and which are best optimized statically").
3. **Memory-side support (C2) is broad:** CLEAR (replay beats EWC/P&C while simpler); CORA
   (CLEAR forgetting 0.7±0.1 vs EWC 1.6±0.1); Continual-Dreamer (replay-buffer *composition*
   is the decisive lever on MiniGrid/MiniHack — and task-aware L2 anchoring to previous-task
   weights "performs poorly", direct external support for our weak anchoring lever); WMAR;
   life-long world models; SupSup/hypernetworks (weight-space memory ⇒ ~zero forgetting over
   hundreds–thousands of tasks); and the plasticity-loss survey's observation that **hard
   resets work only because the replay buffer restores the discarded knowledge** — memory sets
   the restoration ceiling, verbatim our framing.
4. **The plasticity-loss literature is C4's foil, and our null is clean:** Sokar (origin of
   the dormant-fraction metric; LR tuning reduces dormancy but *doesn't fix it* — ReDo at
   default LR wins); Dohare (Nature 2024: plasticity lost **for all constant step-sizes**;
   unassisted PPO degrades progressively under periodic friction changes; Adam/dropout/norm
   make it *worse*); Abbas (CReLU; the effect needs ~2B-frame scale); Nikishin (injection as
   *diagnostic*; plasticity loss bites only in a subset of envs — consistent with our
   short-horizon absence). Our dormant-fraction-falls result (C4) is a horizon-scoped null
   against this literature, not a contradiction of it.
5. **Threats to carry honestly** — see §Threat table: Continual World (naive replay fails in
   robotic CRL), Kessler et al. (in *our exact* MiniGrid goal-flip regime, replay
   **interferes**), CORA's own caveat (cyclic protocols structurally favor replay — our
   benchmark cycles 2 regimes), ANML (meta-learned gating alone claims 600-class continual
   learning — supervised, but a counterexample to "gating has a low ceiling"), and Dohare's
   L2/shrink-and-perturb exceptions ("LR alone is weak" ≠ "all HP control is weak").

## Novelty scan (bucket 6): is there a direct precedent for the Brain?

Nearest neighbors found, and why each is *not* the Brain:

| Prior work | Mechanism | Difference from the Brain |
| --- | --- | --- |
| PBT (Jaderberg 2017) / PB2 (Parker-Holder 2020) | evolutionary population / GP-bandit over parallel runs | outer loop is selection across a population, not a trained RL policy inside one continual run |
| Meta-gradients (Xu 2018) / STAC-STACX (Zahavy 2020) | differentiate through the inner update | gradient-based, tunes return/loss params; no outer RL agent, stationary tasks |
| Agent57 (Badia 2020) | sliding-window UCB bandit over exploration policies | bandit (not trained policy), exploration HPs only, stationary per-game |
| **BiERL (Wang, ECAI 2023)** | **bilevel outer loop online-tunes inner ERL learner's HPs within one agent** | **closest published prior art — must cite & differentiate**: evolutionary meta-level, stationary MuJoCo/Box2D, no forgetting benchmark, no neuromod channel |
| RL² (Duan 2016) / L2RL (Wang 2016) | inner learner lives in RNN activations | inner "algorithm" is implicit; no explicit HP interface |
| L2L-by-GD (Andrychowicz 2016) / LPG (Oh 2020) | learned optimizer / learned update rule | replaces the update rule; not online control of a fixed learner |
| ANML/OML (2019–20) | meta-learn a gating fn / representation for CL | second-order meta-learning, supervised CL, no online controller |

**No paper found that trains an outer RL policy to control an inner learner's plasticity HPs
online, within a single continual-RL run, against a forgetting benchmark.** The novelty
sentence is claimable; BiERL is the mandatory differentiation.

## Threat table (what to defuse, claim by claim)

| Claim | Threatening paper(s) | The threat | Defusal in the paper |
| --- | --- | --- | --- |
| C1 "helps a little" | Xu 2018; STAC/STACX; PBT; Agent57; BiERL | large meta-control gains, monotone in #HPs controlled | all stationary; no forgetting; our contrast is the first controlled one under switches |
| C2 "memory is the bottleneck" | Continual World (Perfect Memory 0.12 avg perf, −1.34 fwd transfer) | naive replay *fails* in SAC robotic CRL | O2 is a *knowledge-restoration* ceiling, not replay advocacy; selection/trigger is the open half (note 0009) |
| C2 (framing) | Continual World's thesis | field over-weights forgetting; forward transfer is the frontier | our benchmark cycles regimes → forgetting/restoration *is* the relevant axis; say so |
| C2 (protocol) | CORA limitations §: cyclic protocols structurally favor replay | our 2-regime cycle could inflate the memory ceiling | O2 restores *snapshots*, not buffer data; the ceiling is protocol-native; still: acknowledge |
| C3 pathway inert | ANML (600 classes, no replay); HAT (45–80% forgetting reduction); FiLM/CBN | gating channels *can* be powerful | those gates are meta-learned per-input or supervised per-task; our code is a low-dim outer-emitted signal — and oracle_code shows even perfect information doesn't help *this* pathway |
| C4 no plasticity loss | Dohare; Abbas; Sokar; Lyle; Kumar | plasticity loss is real and general | scale/horizon: effects need thousands of tasks or ~2B frames; we measure the diagnostic and report the trajectory (falls 0.9→0.35) |
| C1 (weak-lever reading) | Dohare's exceptions: L2, shrink-and-perturb work; primacy-bias resets give big gains | *some* HP-mediated interventions do work | cite as "constant-LR alone is weak", not "all HP control is weak"; resets gain in sample-efficiency regime, and their recovery is replay-mediated (survey) |
| C1 motivation | DreamerV3 (one fixed HP config, 150+ domains) | robust *algorithm design* is an alternative to learned HP control | scope: single-domain mastery, not non-stationary regimes |
| any small-Δ claim | Henderson; Colas (p=0.031 comparing an algorithm to itself at n=5); rliable | small-n deltas are unreliable | this *is* our C5 contribution: n=32/arm, convergence-vs-decay, control ladder |

**Verification kill:** the claim "dynamic HP adaptation beats any tuned-static configuration"
was **refuted 1–2** when sourced to the AutoRL survey. Do not cite the survey for the strong
version; Eimer et al. 2023 actively supports tuned-static sufficiency.

---

## Annotated bibliography (62 papers, by Related-Work bucket)

Stance = relative to the thesis ("meta-control helps a little; memory is the bottleneck"):
✅ supports · ⚔ threatens · ◐ mixed · ○ neutral/background.

### Bucket 1 — Meta-learned optimizers & hyperparameter control in RL

| Paper | Venue, year | arXiv | Stance | Why it matters here |
| --- | --- | --- | --- | --- |
| Xu, van Hasselt & Silver, *Meta-Gradient Reinforcement Learning* | NeurIPS 2018 | 1805.09801 | ⚔ | Canonical online HP control (γ, λ via meta-gradients); +30–80 *absolute points* median HNS over fixed-HP IMPALA (e.g. 211.9→292.9). Large gain, stationary. |
| Zahavy et al., *A Self-Tuning Actor-Critic Algorithm* (STAC/STACX) | NeurIPS 2020 | 2002.12928 | ⚔ | Self-tunes **all** differentiable loss HPs; monotone improvement with more HPs; ALE 243%→364% (the 364 is **STACX**, cite as STAC/STACX). Sharpest counter-evidence — stationary only. |
| Parker-Holder et al., *AutoRL: A Survey and Open Problems* | JAIR 74, 2022 | 2201.03916 | ◐ | Organizes the bucket; flags dynamic-vs-static as an **open problem** (citable cover for C1). Do **not** cite for "dynamic always beats static" (refuted). |
| Jaderberg et al., *Population Based Training* | arXiv 2017 | 1711.09846 | ◐ | Canonical outer-loop schedule discovery; "fixed settings are generally sub-optimal". Evolutionary population, not a trained controller. |
| Parker-Holder et al., *Population-Based Bandits* (PB2) | NeurIPS 2020 | 2002.02518 | ○ | Provably-efficient online HPO (sublinear regret), small populations; strong tuned-baseline machinery. |
| Badia et al., *Agent57* | ICML 2020 | 2003.13350 | ◐ | Bandit meta-controller over exploration/horizon → first ≥human on all 57 games; large gains but leans on episodic-memory backbone (NGU) — resonates with C2. |
| Eimer, Lindauer & Raileanu, *Hyperparameters in RL and How To Tune Them* | ICML 2023 | 2306.01324 | ◐ | Tuned-**static** HPO with seed discipline often suffices; underwrites C6 and the C5 methodology. |
| Andrychowicz et al., *What Matters in On-Policy RL* | ICLR 2021 | 2006.05990 | ◐ | >250k agents, >50 design choices: HPs matter a lot, and a well-tuned static PPO is a strong baseline. |
| Hafner et al., *DreamerV3 / Mastering Diverse Domains through World Models* | Nature 2025 (arXiv 2023) | 2301.04104 | ✅ | One **fixed** HP config masters 150+ domains — robust algorithm design as the alternative remedy to learned HP control. |
| Oh et al., *Discovering RL Algorithms* (LPG) | NeurIPS 2020 | 2007.08794 | ○ | The maximal form of the axis (learn the whole update rule); frames the design space. |
| Goldie et al., *How Should We Meta-Learn RL Algorithms?* | RLC 2025 | 2507.17668 | ○ | 2025 design-space comparison (evolution vs LLM-proposed code) for meta-learning RL components; recent-coverage anchor. |
| Wang et al., *BiERL: Meta Evolutionary RL via Bilevel Optimization* | ECAI 2023 | 2308.01207 | ◐ | **Closest prior art** to the Brain: bilevel outer loop online-tunes inner learner's HPs in one agent. Evolutionary, stationary — differentiate explicitly. |

### Bucket 2 — Neuromodulation, context gating, feature-wise modulation

| Paper | Venue, year | arXiv | Stance | Why it matters here |
| --- | --- | --- | --- | --- |
| Beaulieu et al., *Learning to Continually Learn* (ANML) | ECAI 2020 | 2002.09571 | ◐ | Closest architectural analog: NM net gates the forward pass element-wise over the final-conv latent; gate doubles as plasticity controller via chain rule; 600 classes / 9,000 updates, **no replay** — partial threat to C2 (supervised). |
| Miconi et al., *Backpropamine* | ICLR 2019 | 1811.06308 ⚠ | ✅ | Canonical neuromodulated-plasticity citation (network-generated signal gates Hebbian plasticity, end-to-end). ⚠ verify the id at OpenReview: one sweep link said 2002.10585 — 1811.06308 is canonical. |
| Miconi, Stanley & Clune, *Differentiable Plasticity* | ICML 2018 | 1804.02464 | ◐ | Predecessor of Backpropamine; shows plasticity channels *can* be powerful — sharpens our contrasting oracle_code null. |
| Perez et al., *FiLM* | AAAI 2018 | 1709.07871 | ○ | The modulation primitive: our sigmoidal gate is restricted FiLM (γ-only, β=0) — a relationship the FiLM paper itself discusses. |
| de Vries et al., *Modulating early visual processing by language* (CBN) | NeurIPS 2017 | 1707.00683 | ○ | Origin of conditional-BN feature-wise conditioning; lineage citation. |
| Dumoulin et al., *Feature-wise transformations* | Distill 2018 | doi 10.23915/distill.00011 | ○ | The unifying review of the mechanism family. |
| Masse, Grant & Freedman, *XdG* | PNAS 115(44), 2018 | 1802.01569 | ✅ | **The structural ally:** gating alone 61.4%, SI alone 82.3%, EWC alone 70.8%, gating+stabilization 95.4% @100 tasks — neither channel alone suffices. |
| von Oswald et al., *Continual learning with hypernetworks* | ICLR 2020 | 1906.00695 | ✅ | Task-conditioned weight generation from a **low-dim embedding** (precedent for the 8-d code); near-oracle retention in a compressive regime — weight-space memory sets the ceiling. |
| Wortsman et al., *Supermasks in Superposition* (SupSup) | NeurIPS 2020 | 2006.14769 | ✅ | Frozen random net + per-task binary mask: thousands of tasks, ~zero forgetting; entropy-based task inference = a regime-detection mechanism; gating trades plasticity for stability, doesn't raise the ceiling. |
| Serrà et al., *Hard Attention to the Task* (HAT) | ICML 2018 | 1801.01423 | ✅ | Learned per-task near-binary unit masks; 45–80% forgetting reduction; the supervised working analogue of our (inert) gate. |
| Mallya & Lazebnik, *PackNet* | CVPR 2018 | 1711.05769 | ✅ | Parameter isolation by iterative pruning; ~zero forgetting via frozen capacity — memory-preservation, not dynamics tuning. |
| Javed & White, *Meta-Learning Representations for Continual Learning* (OML) | NeurIPS 2019 | 1905.12588 | ✅ | Meta-learned sparse representations resist interference; plain SGD on them rivals rehearsal methods (a strong-baseline lesson for C6). |

### Bucket 3 — Catastrophic forgetting, continual RL, loss of plasticity

| Paper | Venue, year | arXiv | Stance | Why it matters here |
| --- | --- | --- | --- | --- |
| Kirkpatrick et al., *EWC* | PNAS 114(13), 2017 | 1612.00796 | ✅ | Canonical consolidation; ancestor of our anchoring/KL lever; 10 Atari games sequentially. |
| Zenke, Poole & Ganguli, *Synaptic Intelligence* | ICML 2017 | 1703.04200 | ✅ | Online per-parameter importance; EWC's sibling; the regularization family our anchoring lever meta-controls. |
| Rusu et al., *Progressive Neural Networks* | arXiv 2016 | 1606.04671 | ✅ | Zero forgetting by frozen per-task columns — structural memory. |
| Schwarz et al., *Progress & Compress* | ICML 2018 | 1805.06370 | ✅ | Online-EWC knowledge base + active column; constant parameters, no stored data. |
| Kaplanis, Shanahan & Clopath, *Complex Synapses* | ICML 2018 | 1802.07239 | ✅ | Multi-timescale Benna–Fusi consolidation mitigates forgetting **without task-switch knowledge** — task-free like our setting. |
| Kaplanis et al., *Policy Consolidation* | ICML 2019 | 1902.00255 | ✅ | Multi-timescale policy cascade regularized by its own history; beats EWC/online-EWC in continual RL, boundary-free. |
| Khetarpal et al., *Towards Continual RL: Review & Perspectives* | JAIR 75, 2022 | 2012.13490 | ○ | The framing survey: scope/driver non-stationarity taxonomy legitimizes the goal-flip MDP; families: explicit retention 18.5% / shared structure 40.8% / learning-to-learn 40.7% — our thesis compares two of the three; replay assessed most successful to date (with off-policy-drift caveat). |
| Zuffer et al., *Advancements and Challenges in Continual RL* | arXiv 2025 | 2506.21899 | ○ | Recent comprehensive review; recency anchor. |
| Dohare et al., *Loss of plasticity in deep continual learning* | Nature 632:768–774, 2024 | (Nature; arXiv id ⚠ verify) | ✅ | Plasticity lost **for all constant step-sizes**; PPO degrades progressively on Slippery Ant after the first friction change; Adam/dropout/norm worsen it; continual backprop (reinit diversity) fixes it. Premier citation for the baseline premise. Carry both caveats: constant-LR only; L2/shrink-and-perturb are HP-mediated exceptions. |
| Abbas et al., *Loss of Plasticity in Continual Deep RL* | CoLLAs 2023 | 2303.07507 | ○ | Plasticity loss in Atari game cycles at ~2B-frame scale; activation collapse mechanism; CReLU mitigates; **reset-agent baseline** = control-ladder precedent (C5); "effective reuse of past experience… is not likely to be resolved by a new activation function" — direct C2 support. |
| Sokar et al., *The Dormant Neuron Phenomenon* (ReDo) | ICML 2023 | 2302.12902 | ✅/○ | Origin of our C4 diagnostic; **target** non-stationarity is the driver; replay ratio correlates with dormancy (mechanistic backing for that Brain lever); LR tuning helps but ReDo at default LR wins — HP control has the lower ceiling. |
| Nikishin et al., *The Primacy Bias in Deep RL* | ICML 2022 | 2205.07802 | ◐ | Simple periodic resets give big gains (sample-efficiency regime) — a foil to "dynamics levers are weak"; recovery is replay-mediated. |
| Nikishin et al., *Plasticity Injection* | NeurIPS 2023 | 2305.15555 | ◐ | Injection as **diagnostic** (anticipates our oracle/probe methodology); plasticity loss bites only in a subset of envs — consistent with our short-horizon null. |
| Lyle et al., *Understanding and Preventing Capacity Loss in RL* | ICLR 2022 | 2204.09560 | ○ | Capacity-loss lens (target-fitting ability); InFeR; the phenomenon C4 tests for. |
| Lyle et al., *Understanding Plasticity in Neural Networks* | ICML 2023 | 2303.01486 | ○ | Plasticity loss tracks curvature, often *without* dead units; layer-norm preserves it. |
| Kumar et al., *Implicit Under-Parameterization* | ICLR 2021 | 2010.14498 | ○ | Feature-rank collapse under bootstrapping — the pathological-critic-dynamics foil to our "whiplash is functional" (C3). |
| Klein et al., *Plasticity Loss in Deep RL: A Survey* | arXiv 2024 | 2411.04832 | ✅ | Single Related-Work anchor for the fast-moving line; taxonomy of mitigation "knobs" maps ~1:1 onto the Brain's action space; two gems: general SL regularizers beat purpose-built interventions, and **hard resets work because the replay buffer is the memory that restores discarded knowledge**. |

### Bucket 4 — Memory, replay, episodic & model-based approaches

| Paper | Venue, year | arXiv | Stance | Why it matters here |
| --- | --- | --- | --- | --- |
| Sutton, *Dyna* | ICML 1990 | (in bib) | ○ | The inner learner's lineage (already cited). |
| Rolnick et al., *CLEAR* | NeurIPS 2019 | 1811.11682 | ✅ | Canonical continual-RL replay: V-trace replay + behavioral cloning, task-boundary-free, beats EWC/P&C while simpler; the fixed-schedule counterpart to our learned replay-ratio lever. |
| Blundell et al., *Model-Free Episodic Control* | arXiv 2016 | 1606.04460 | ✅ | Non-parametric episodic memory → near-instant re-exploitation of past strategies — the restoration capability O2 oracles. |
| Pritzel et al., *Neural Episodic Control* | ICML 2017 | 1703.01988 | ✅ | Differentiable Neural Dictionary; orders-of-magnitude early data-efficiency from memory, not dynamics tuning. |
| Isele & Cosgun, *Selective Experience Replay* | AAAI 2018 | 1802.10269 | ✅ | Buffer **selection strategy** (distribution-matching best) is where the gains live — grounds the replay-prioritization lever and note 0009's selection question. |
| Shin et al., *Deep Generative Replay* | NeurIPS 2017 | 1705.08690 | ✅ | Root of generative replay (hippocampal framing). |
| Yue et al., *t-DGR* | CoLLAs 2024 | 2401.02576 | ✅ | Trajectory-level generative replay; SOTA average success on Continual World — replay-pathway upgrades keep paying. |
| Kessler et al., *Effectiveness of World Models for Continual RL* (Continual-Dreamer) | CoLLAs 2023 | 2211.15944 | ✅/⚔ | Buffer composition (reservoir sampling) is the decisive lever on MiniGrid/MiniHack; DreamerV2 0.72 vs CLEAR 0.03 at equal budget; task-aware L2 anchoring "performs poorly". **⚔ In our exact regime (FourRooms, changed goal/reward): replay INTERFERES — only one task ever solved.** Naive replay is not the answer; restoration/selection is (note 0009). |
| Yang et al., *WMAR* | arXiv 2024 | 2401.16650 | ✅ | Distribution-matching replay for DreamerV3 at fixed buffer size → substantially better forgetting; memory strategy dominates at fixed capacity. |
| Pan et al., *Life-Long World Model* | ECML-PKDD (arXiv 2023) | 2303.06572 | ✅ | Mixture-of-Gaussians task-specific latent dynamics + generative replay through the world model — the MoWM-adjacent instance of "mixture-of-world-models for non-stationarity". |

### Bucket 5 — Evaluation methodology & benchmarks

| Paper | Venue, year | arXiv | Stance | Why it matters here |
| --- | --- | --- | --- | --- |
| Agarwal et al., *Statistical Precipice* (rliable) | NeurIPS 2021 (Outstanding Paper) | 2108.13264 | ✅ | IQM + stratified bootstrap CIs + performance profiles; rankings flip under proper uncertainty — the toolkit under our small-Δ claims. |
| Henderson et al., *Deep RL That Matters* | AAAI 2018 | 1709.06560 | ✅ | Seeds/HPs/codebase swing results; two seed-groups of the *same* algorithm can differ "significantly" — the mirage-section anchor. |
| Colas, Sigaud & Oudeyer, *How Many Random Seeds?* | arXiv 2018 | 1806.08295 | ✅ | Power analysis: n=5 gave p=0.031 on identical algorithms; type-II ≈51% at n=5 on their case study; prescriptions (Welch's t-test, α<0.05, n≥10–20) — justifies our n=32/arm discipline. |
| Wołczyk et al., *Continual World* | NeurIPS 2021 | 2105.10919 | ◐ | Canonical CRL benchmark + baseline control-ladder + 20-seed/90%-CI protocol; ⚔ Perfect Memory/A-GEM perform *poorly* (critic regularization) and the paper argues the field over-weights forgetting — frame O2 as restoration, not replay. |
| Powers et al., *CORA* | CoLLAs 2022 | 2110.10067 | ○/⚔ | Standard metric suite (Continual Eval / Isolated Forgetting / 0-shot Transfer); CLEAR dominates baselines (0.7 vs EWC 1.6 forgetting); ⚔ own caveat: **cyclic protocols structurally favor replay** — acknowledge for our 2-regime cycle. |
| Yu et al., *Meta-World* | CoRL 2019 | 1910.10897 | ○ | The substrate of CW10/CW20; broad-distribution evaluation argument. |
| Chevalier-Boisvert et al., *Minigrid & Miniworld* | NeurIPS D&B 2023 | 2306.13831 | ○ | Our env's citable paper (already in bib); design philosophy legitimizes custom goal-flip tasks. |

### Bucket 6 — Two-level / bilevel systems, meta-RL foundations

| Paper | Venue, year | arXiv | Stance | Why it matters here |
| --- | --- | --- | --- | --- |
| Duan et al., *RL²* | arXiv 2016 | 1611.02779 | ○ | Slow outer RL meta-learns a fast inner RL in RNN activations — the canonical outer/inner precedent. |
| Wang et al., *Learning to Reinforcement Learn* | arXiv 2016 / CogSci 2017 | 1611.05763 | ○ | The companion deep meta-RL paper (Harlow task; PFC framing). |
| Finn, Abbeel & Levine, *MAML* | ICML 2017 | 1703.03400 | ○ | Meta-learned initialization; background positioning only. |
| Andrychowicz et al., *Learning to learn by gradient descent by gradient descent* | NeurIPS 2016 | 1606.04474 | ○ | The learned-optimizer paradigm the Brain instantiates in RL form. |

(Cross-listed in bucket 6: PBT, PB2, Agent57, BiERL, LPG, Goldie 2025 — see bucket 1.)

## Citation-hygiene flags (fix before camera-ready)

- **Backpropamine arXiv id:** cite **1811.06308** (ICLR 2019); a fetched link said 2002.10585
  — reconcile against OpenReview.
- **STAC vs STACX:** the 243%→364% ALE number belongs to **STACX** (auxiliary heads); plain
  STAC gains less. Cite "STAC/STACX".
- **Xu et al. margins** are *absolute percentage points* of median human-normalized score,
  not relative gains.
- **Dohare et al.:** cite the Nature version (632:768–774, 2024, doi 10.1038/s41586-024-07711-7);
  the arXiv id was not re-verified in these passes.
- **Andrychowicz 2021** ICLR camera-ready is retitled *"What Matters for On-Policy Deep
  Actor-Critic Methods? A Large-Scale Study"*.
- **DreamerV3** Nature 2025 title differs from the arXiv title ("…diverse control tasks…").
- One gap-fill verdict (Continual World) passed with a stance disagreement across agents
  (mixed vs supports) — read the paper's §5 before quoting its replay numbers.

## Leads seen but NOT verified (candidate follow-ups)

- Moalla et al., arXiv 2405.00662 — PPO representation degradation under non-stationarity.
- Juliani & Ash, NeurIPS 2024 — plasticity loss in continual learning (corroboration cite).
- "CL+LSTM" follow-up reported to beat ANML crediting sparsity — would sharpen the ANML defusal.
- BG-PBT / Multiple-Frequencies PBT — PBT lineage descriptions ("evolutionary outer loop").
- Ash & Adams, *shrink-and-perturb* — the Dohare exception's source.
- Oracle/ceiling methodology as its own literature: nothing found that matches O2's
  policy-swap ceiling exactly (nearest: Abbas's reset-agent, plasticity injection as
  diagnostic, Continual World's reference-transfer). Worth one targeted pass; if it stays
  empty, the control-ladder methodology (C5) is itself a contribution.
- Regime/context detection & task-free CL beyond SupSup's entropy inference — thin coverage;
  relevant to note 0009's learned trigger.

## Links

- Paper skeleton this feeds: [note 0008](./0008-paper-skeleton.md) §2.
- Claim numbers cited from: [note 0006](./0006-controls-axis-thesis-relocated.md).
- Memory-phase relevance (Kessler interference, Isele selection): [note 0009](./0009-memory-levers-preregistration.md).
- BibTeX for all entries: [docs/references/references.bib](../references/references.bib)
  (expanded 2026-07-09; author lists in "first-author and others" form — complete at
  camera-ready).
