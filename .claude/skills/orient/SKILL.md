---
name: orient
description: Orient in the Lifelong-Learning Meta-RL repo (outer "Brain" controlling an inner Dyna-PPO continual learner, plus a frozen-benchmark + autoresearch layer). Use at the start of a task in this repo — when you need a mental map of the architecture, the runs-vs-evals distinction, the research flow, or where a subsystem lives. Reads docs/spec as a non-canonical aid, then grounds claims in the source.
---

# Orient

Build an accurate mental model of this repository before doing work in it. The
[`docs/spec/`](../../../docs/spec/) directory is a **map**, not the territory: use it to find
things fast, then **verify against the source code**, which is the only source of truth.

## What this repo is (the 10-second version)

A two-level RL system for catastrophic forgetting — the **Meta-Learning / "double-RL"** strategy
from `Combating Catastrophic Forgetting.pdf` (§3.7):

- **inner** Dyna-PPO agent learns a MiniGrid task whose rewarded goal flips periodically, and
- **outer** "Brain" (PPO) treats the inner training run as its environment and tunes the inner
  learner's plasticity (7 hyperparameter levers + an 8-dim neuromodulation context code),
- with a **frozen benchmark + scorer** and an **autoresearch supervisor** on top that drives an
  external coding agent through bounded research trials.

**Vocabulary that matters everywhere:** a **run** = an outer-Brain *training* run (`runs/`); an
**eval** = a *trained*-outer-Brain *evaluation* (`evals/`). A run produces `brain_model.pt`; an
eval consumes one.

## Procedure

1. **Read the map.** Read [docs/spec/README.md](../../../docs/spec/README.md) (index + framing +
   vocabulary) and [docs/spec/01-research-flow.md](../../../docs/spec/01-research-flow.md) (the
   diagrams that connect every layer). This alone gives you the architecture.

2. **Drill into the relevant subsystem.** Pick the spec doc(s) matching the task:
   - environments / regime switching → [02-environments.md](../../../docs/spec/02-environments.md)
   - inner learner (PPO, world model, curiosity, dreaming, replay, anchoring) → [03-inner-dyna-ppo.md](../../../docs/spec/03-inner-dyna-ppo.md)
   - outer Brain, MetaEnv, the 19-dim obs / 15-dim action, reward modes → [04-outer-brain-and-metaenv.md](../../../docs/spec/04-outer-brain-and-metaenv.md)
   - neuromodulation (context code → mask → diagnostics) → [05-neuromodulation.md](../../../docs/spec/05-neuromodulation.md)
   - output directories / what a folder is → [06-runs-and-evals.md](../../../docs/spec/06-runs-and-evals.md)
   - benchmark, scorer, composite score, autoresearch → [07-benchmarking-and-autoresearch.md](../../../docs/spec/07-benchmarking-and-autoresearch.md)
   - scripts and flags → [08-cli-reference.md](../../../docs/spec/08-cli-reference.md)

3. **Verify before you rely.** The spec links to `file.py#Lnn` for every non-trivial claim.
   Before acting on a fact (a tensor shape, a default, a reward formula, an output path), open
   the cited source and confirm it still holds. If the spec and code disagree, **the code wins**
   — note the drift and, if appropriate, fix the spec line.

4. **Respect the research boundary.** If the task touches autoresearch, remember the editable
   surface is narrow (`neuromod.py`, `network.py`; optionally `signals.py`, `meta_agent.py`) and
   the benchmark/scorer/env are immutable. See `research_manifest.toml` and
   [program_neuromod.md](../../../program_neuromod.md).

5. **Report your orientation.** Summarize for the user: what the task touches, the relevant
   files (with paths), any spec-vs-code drift you found, and your proposed entry point. Don't
   dump the whole spec back — give the map *for this task*.

## Notes

- The spec is intentionally **non-canonical**. Treat recalled facts as hypotheses to confirm in
  the source, especially shapes, defaults, and file paths.
- One known spec-vs-paper nuance to watch for: the Brain action maps **linearly to absolute HP
  bounds** in code (`MetaEnv._apply_action`), even though the paper/PLAN describe "multiplicative
  scaling." When in doubt, read the code.
- If `docs/spec/` is missing or stale, fall back to: `README.md`, `PLAN.md`, `AUTORESEARCH.md`,
  the PDF (§3.7, §4.3, §4.3.1), and a direct read of `src/lifelong_learning/`.
