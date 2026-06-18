# 01 — Research Flow

This is the map. It shows how the pieces connect at three zoom levels:

1. **The two-level RL system** — Brain ⟷ inner Dyna-PPO ⟷ MiniGrid.
2. **The inner Dyna-PPO update cycle** — what one inner step actually does.
3. **The research lifecycle** — how a **run** (training) becomes an **eval**, and how the
   **autoresearch supervisor** drives bounded research trials over a frozen benchmark.

All diagrams are [Mermaid](https://mermaid.js.org/) and render in GitHub and VS Code.

---

## 1. The two-level (double-RL) system

The **outer Brain** treats one full **inner training run** as a single RL episode. Each
outer step ([`MetaEnv.step`](../../src/lifelong_learning/agents/brain/meta_env.py#L218))
runs `decision_interval` inner PPO updates, then hands the Brain a 19-dim summary and
takes a 15-dim action back.

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px', 'textColor': '#111827', 'lineColor': '#4b5563', 'edgeLabelBackground': '#ffffff'}}}%%
flowchart TB
    subgraph OUTER["OUTER LOOP — the Brain (a PPO meta-agent)"]
        BRAIN["Brain policy<br/>2-layer MLP actor-critic<br/>tanh-squashed Gaussian<br/>(meta_agent.py)"]
    end

    subgraph META["MetaEnv — one inner run = one Brain episode (meta_env.py)"]
        SIG["SignalExtractor<br/>19-dim normalized observation<br/>(signals.py)"]
        APPLY["_apply_action<br/>map 15-dim action → absolute inner HPs"]
        REWARD["meta-reward<br/>auc / recovery / curriculum"]
    end

    subgraph INNER["INNER LOOP — Dyna-PPO continual learner (train.py)"]
        STEP["run_inner_update()<br/>× decision_interval"]
        AC["CNNActorCritic<br/>(+ neuromodulation mask)<br/>network.py"]
        WM["SimpleWorldModel<br/>world_model.py"]
        MEM["EpisodicMemory<br/>episodic_memory.py"]
    end

    subgraph ENV["ENVIRONMENT — non-stationary MiniGrid (envs/)"]
        E["MultiGoal/DualGoal<br/>+ RegimeGoalSwapWrapper<br/>regime flips every steps_per_regime"]
    end

    BRAIN -- "action a ∈ [-1,1]^15<br/>(7 levers + 8-dim context code)" --> APPLY
    APPLY -- "lr, ent, intrinsic, horizon,<br/>replay ratio/prio, anchoring" --> STEP
    APPLY -- "context code → set_context_code" --> AC
    STEP --> AC
    AC -- "act / evaluate" --> E
    E -- "obs, reward, regime_id" --> STEP
    STEP <--> WM
    STEP <--> MEM
    STEP -- "inner training stats" --> SIG
    STEP -- "Δsuccess, Δreturn, failure" --> REWARD
    SIG -- "obs s ∈ [-10,10]^19" --> BRAIN
    REWARD -- "scalar reward r" --> BRAIN

    classDef default color:#111827;
    classDef outer fill:#e8f0fe,stroke:#4285f4,color:#111827;
    classDef inner fill:#fef7e0,stroke:#f9ab00,color:#111827;
    classDef env fill:#e6f4ea,stroke:#34a853,color:#111827;
    class BRAIN outer;
    class STEP,AC,WM,MEM inner;
    class E env;
```

**Key interface facts** (all in
[meta_env.py](../../src/lifelong_learning/agents/brain/meta_env.py)):

- Observation space: `Box(-10, 10, shape=(19,))` — see [signals.py `SIGNAL_NAMES`](../../src/lifelong_learning/agents/brain/signals.py#L14).
- Action space: `Box(-1, 1, shape=(15,))` — layout in [neuromod.py](../../src/lifelong_learning/agents/brain/neuromod.py#L11-L17): indices `0:7` are scalar levers, `7:15` are the neuromodulation context code.
- The action maps **linearly to absolute hyperparameter bounds** ([`_apply_action`](../../src/lifelong_learning/agents/brain/meta_env.py#L325)), *not* multiplicatively — see [04-outer-brain-and-metaenv.md](./04-outer-brain-and-metaenv.md).
- A Brain episode `terminated` when the inner run reaches its `total_timesteps` ([`done`](../../src/lifelong_learning/agents/brain/meta_env.py#L265)).

---

## 2. The inner Dyna-PPO update cycle

One call to
[`run_inner_update`](../../src/lifelong_learning/agents/ppo/train.py#L321) is the atomic
unit the Brain controls. It runs these phases in order:

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px', 'textColor': '#111827', 'lineColor': '#4b5563', 'edgeLabelBackground': '#ffffff'}}}%%
flowchart LR
    A["Phase A — Collect real experience<br/>num_steps × num_envs<br/>surprise = CE(next-state) + MSE(reward)<br/>intrinsic = clip(surprise·intrinsic_coef)<br/>buffer ← extrinsic + intrinsic"]
    GAE["GAE<br/>advantages + returns"]
    B["Phase B — PPO on real data<br/>update_epochs × minibatches<br/>(+ optional anchoring KL)"]
    B2["Phase B2 — Replay<br/>if replay_ratio > 0.01:<br/>mix fresh + episodic memory<br/>(prioritized to older regimes)"]
    C["Phase C — Train world model<br/>CE(state) + MSE(reward)"]
    ARCH["Archive rollout<br/>→ EpisodicMemory"]
    D["Phase D — Dream<br/>WM rolls out imagined_horizon steps<br/>PPO update on imagined data<br/>(gradients blocked to WM)"]
    LOG["Log charts/* + brain/*<br/>checkpoint every N updates<br/>return stats dict"]

    A --> GAE --> B --> B2 --> C --> ARCH --> D --> LOG

    classDef default fill:#f8fafc,stroke:#94a3b8,color:#111827;
```

- A **regime switch** is detected inside Phase A and snapshots the
  [`anchor_model`](../../src/lifelong_learning/agents/ppo/train.py#L391-L395) used by the
  anchoring penalty.
- `"passive"` mode zeroes the surprise/intrinsic signal
  ([train.py](../../src/lifelong_learning/agents/ppo/train.py#L422)); `"dyna"` mode is the
  full loop above.
- Detail and the meaning of every signal returned: [03-inner-dyna-ppo.md](./03-inner-dyna-ppo.md).

---

## 3. The research lifecycle: run → eval, and autoresearch

### 3a. A single run and its eval

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px', 'textColor': '#111827', 'lineColor': '#4b5563', 'edgeLabelBackground': '#ffffff'}}}%%
flowchart LR
    TB["scripts/train_brain.py<br/>(optionally) imitation pretrain<br/>then PPO over brain_episodes"]
    RUNS[("runs/&lt;name&gt;/<br/>config.txt<br/>brain_model.pt ← TRAINED ARTIFACT<br/>brain_trends/<br/>episode_N/ep_envI/ (inner logs)")]
    EB["scripts/eval_brain.py<br/>--brain_checkpoint runs/&lt;name&gt;/brain_model.pt<br/>inference-only: frozen Brain drives one long inner run"]
    EVALS[("evals/&lt;name&gt;/<br/>config.txt<br/>eval_..._/ (inner eval logs)<br/>NO brain training")]

    TB -- "writes (TRAINING)" --> RUNS
    RUNS -- "brain_model.pt loaded by" --> EB
    EB -- "writes (EVALUATION)" --> EVALS

    classDef default color:#111827;
    classDef train fill:#e8f0fe,stroke:#4285f4,color:#111827;
    classDef eval fill:#fce8e6,stroke:#ea4335,color:#111827;
    class TB,RUNS train;
    class EB,EVALS eval;
```

This is the distinction this spec hammers on: **`runs/` is where the Brain is trained;
`evals/` is where a trained Brain is measured.** Full directory contract in
[06-runs-and-evals.md](./06-runs-and-evals.md).

### 3b. The frozen benchmark

[`scripts/run_frozen_benchmark.py`](../../scripts/run_frozen_benchmark.py) is a *scored*
training run: for each frozen seed it calls `train_brain()` (so the training output still
lands in `runs/`), then scores that run and writes the report to `benchmarks/`.

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px', 'textColor': '#111827', 'lineColor': '#4b5563', 'edgeLabelBackground': '#ffffff'}}}%%
flowchart LR
    RFB["run_frozen_benchmark.py --benchmark fast_switch_scout_v1"]
    SPEC["FrozenBenchmarkSpec<br/>frozen env/HP args + seeds<br/>(benchmarking.py)"]
    TBF["train_brain() per seed"]
    RUNS2[("runs/&lt;spec&gt;_&lt;ts&gt;_seedN/")]
    SCORE["score_brain_run()<br/>recovery metrics +<br/>composite_score"]
    BENCH[("benchmarks/&lt;spec&gt;_&lt;ts&gt;/<br/>summary.json + seed_N_score.json")]

    SPEC --> RFB --> TBF -- "writes (TRAINING)" --> RUNS2
    RUNS2 -- "read + scored" --> SCORE --> BENCH

    classDef default fill:#f8fafc,stroke:#94a3b8,color:#111827;
```

### 3c. The autoresearch supervisor (Karpathy-style bounded loop)

[`AutoresearchSupervisor`](../../src/lifelong_learning/research/autoresearch.py#L553)
drives an **external coding agent** (Codex) through bounded, append-only trials. The agent
may edit only a narrow **editable surface** (neuromodulation code); the benchmark, scorer,
and environment are **immutable**. Each trial that fails any gate is rolled back.

```mermaid
%%{init: {'themeVariables': {'fontSize': '14px', 'textColor': '#111827', 'lineColor': '#4b5563', 'edgeLabelBackground': '#ffffff'}}}%%
flowchart TB
    START["run_autoresearch.py --manifest research_manifest.toml --program program_neuromod.md"]
    BASE["Baseline: run primary (+holdout) benchmark<br/>(fingerprint-cached)"]
    SNAP["Snapshot repo"]
    AGENT["Render trial prompt + context →<br/>invoke external agent (Codex)<br/>edits neuromod.py / network.py"]
    DIFF["Diff + AUDIT against editable/immutable surface<br/>enforce max_new_python_files + max_net_new_lines"]
    TESTS["Run test suite"]
    PRIM["Run primary (scout) benchmark → composite_score"]
    IMP{"improved over best?"}
    HOLD["Run holdout benchmark(s)"]
    REG{"holdout not regressed?"}
    ACCEPT["ACCEPT: keep change<br/>update best score"]
    ROLLBACK["ROLLBACK: restore snapshot"]
    LEDGER[("append trial record →<br/>autoresearch/trial_results.jsonl")]
    STOP{"max_trials / target_score /<br/>max_stale_trials?"}

    START --> BASE --> SNAP --> AGENT --> DIFF
    DIFF -- "violation" --> ROLLBACK
    DIFF -- "ok" --> TESTS
    TESTS -- "fail" --> ROLLBACK
    TESTS -- "pass" --> PRIM --> IMP
    IMP -- "no" --> ROLLBACK
    IMP -- "yes" --> HOLD --> REG
    REG -- "no" --> ROLLBACK
    REG -- "yes" --> ACCEPT
    ACCEPT --> LEDGER
    ROLLBACK --> LEDGER
    LEDGER --> STOP
    STOP -- "no" --> SNAP
    STOP -- "yes" --> DONE["session_summary.json/.md"]

    classDef default fill:#f8fafc,stroke:#94a3b8,color:#111827;
    classDef gate fill:#fef7e0,stroke:#f9ab00,color:#111827;
    class IMP,REG,STOP gate;
```

Each trial writes its artifacts under `autoresearch/<session>/trial_<NNN>/`
(`research_prompt.md`, `trial_context.json`, `agent_notes.md`, `codex_final_message.md`,
logs). Full mechanics, the editable/immutable surface, and the manifest schema:
[07-benchmarking-and-autoresearch.md](./07-benchmarking-and-autoresearch.md).

---

## How the layers nest (one sentence each)

- **Environment** flips the rewarded goal on a fixed schedule → forgetting becomes visible.
- **Inner Dyna-PPO** tries to relearn fast after each flip without erasing old skills.
- **Outer Brain** watches the inner learner and tunes its plasticity (HPs + neuromodulation) to recover faster.
- **Frozen benchmark** turns "did the Brain recover faster?" into a single reproducible `composite_score`.
- **Autoresearch** lets an external agent propose neuromodulation changes and keeps only the ones the frozen benchmark says helped.
