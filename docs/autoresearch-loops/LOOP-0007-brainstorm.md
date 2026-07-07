# LOOP-0007 — brainstorm + code-free campaign (2026-07-07, CLOSED NULL)

**Goal:** Generate the next hypothesis family after the code-directed learning-dynamics family
(LOOP-0005/0006) closed **NULL** at n=8. User-directed pivot (2026-07-07, via RUN-20260706
gate loop) to a family **outside code→inner-agent routing**.
**Verdict:** **CLOSED NULL 2026-07-07 — 0/5 at n≥8.** critic_lr_lo REJECT (0.5490/0.8248);
encoder_lr_lo REJECT (0.5531/0.7589, reliability −7 pts — two-timescale retired);
plasticity_norm KILLED n=2 (degenerate LayerNorm(4096) operationalization); surprise_spike
REJECT (0.5427/0.8103 — over-explores); redo REJECT — n=8 Path-B pass (0.5522/0.8527) failed
n=10 (0.5480/0.8393), holdout n=3 passed the mechanical bar (0.5472/0.8363) on ONE carrying
seed (2/3 below baseline; n=4 with s41 ≈ baseline), and the **pre-registered dormant-fraction
probe adjudicated A1 FALSE** (dormancy falls 0.9→0.35 over the run, never accumulates —
note 0005 addendum). redo = the third and deepest-surviving small-n reliability mirage
(after critic_code, auxcode_hi). Full trail: RUN-20260706 living doc W1–W10 +
[note 0004 outcome addendum](../research-notes/0004-code-free-plasticity-stability.md).

## The forcing evidence
Two families now closed: forward-modulation (LOOP-0002/0004, 20 variants, 0 wins) and
code-directed learning-dynamics (LOOP-0005/0006, NULL at n=8 — both n=3 "reliability leads"
critic_code/auxcode_hi washed out, gradgate_gain failed holdout). **Through-line: injecting the
regime `code` into the inner agent, by any pathway, does not help.** The proven-causal channel
is the Brain's scalar-HP control (research-log 0001). So LOOP-0007 stops routing the code and
attacks the two measured post-switch pathologies directly.

## Brainstorm output (2026-07-07) — new family: code-free plasticity & stability maintenance
Full argument, assumptions (A1 plasticity-loss present?, A2 critic-whiplash causal?), and
registered predictions: **[research note 0004](../research-notes/0004-code-free-plasticity-stability.md).**

Candidate one-liners (human copies chosen ones into `config/hypothesis_queue.md`; ranked):
1. **Decoupled critic LR** — scalar `critic_lr_scale` lever to damp measured critic whiplash
   (|ΔV|≈0.917); code-free analogue of the null `critic_code`. Cheapest first screen.
2. **Dormant-neuron reset (ReDo)** — reset ~dormant units to restore plasticity; probe:
   dormant-fraction ↓. Needs surface sign-off.
3. **Plasticity-preserving normalization** — static LayerNorm on the encoder (Lyle 2023);
   also the cleanest test of assumption A1.
4. **Surprise-triggered exploration spike** — TD-error change-point (not the code) transiently
   spikes ent/intrinsic at detected switches, faster than `decision_interval`.
5. **Two-timescale encoder-vs-heads (uniform)** — code-free version of the null `gradgate`;
   isolates whether two-timescale itself carries value. Cheap first screen.

## Method discipline (carried from LOOP-0006's mistake)
Screen at **n≥8** — n=3 over-selected lucky seed triples and produced two false leads. The
reliability fork stands (composite-non-degrade AND hit_80 > control CI-upper) but applies only
at n≥8. Register predictions before results; probe-based candidates (2,3) must move their probe.

## Sibling loops
[LOOP-0006](./LOOP-0006-learning-dynamics-campaign.md) (predecessor; closed null) — this loop
starts from its null verdict + the diagnostic pathologies it left unaddressed.

## Pickup state
**CLOSED — nothing to pick up here.** The program's live thread is
[LOOP-0008](./LOOP-0008-controls-campaign.md): the bottom-rung/oracle control campaign
(D0/O1/O2 + oracle_code) pre-registered in
[note 0005](../research-notes/0005-untested-controller-bottom-rung-oracle.md), which also
records this loop's biggest retrospective finding — the scout Brain is effectively untrained,
so all three closed families screened their mechanisms under a noise controller.

## Links
[research note 0004](../research-notes/0004-code-free-plasticity-stability.md) ·
`config/hypothesis_queue.md` (closed-family guards + format) ·
[LOOP-0005](./LOOP-0005-brainstorm.md) (template for this note) ·
spec [05-neuromodulation.md](../spec/05-neuromodulation.md) (mechanism ground truth).
