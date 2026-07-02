# 0001 — Reduce artifacts on the GPU box + auto send-back

- **Date:** 2026-07-01 · **Status:** ✅ Shipped (reductions + auto-push mechanism); auto-push goes
  live once a `GH_TOKEN` is set on the box.

## Shipped (2026-07-01)

Verified on the box with a fresh pilot run (2 episodes): inner checkpoints **2 → 0**, PNGs
**5 → 0**, `data.json` now compact, run dir **~110 MB → 1.8 MB**, and the cell **still scored**
(summary.csv written). Full manifest suite: **60 passed**. Confirmed `logger.py` and ppo
`train.py` are **not** in `[immutable_surface]`, so these are safe human infra edits.

- **PNG gate** — `logger.py:plot()` skips rendering unless `LL_RENDER_CHARTS=1`; new
  `scripts/render_charts.py` rebuilds figures locally from `*_data.json`. (~1.4 GB saved.)
- **Compact JSON** — `logger.py` dumps with `separators=(",",":")` (measured 2.8× smaller, lossless).
- **Checkpoint leak fixed** — ppo `train.py` no longer force-saves a final inner checkpoint when
  periodic saving is off (`save_checkpoints=False`, the sweep default). No new flag; uses the
  existing `save_every_updates` sentinel. (~12 GB saved.)
- **Auto send-back (A)** — `scripts/cloud/push_results.sh` pushes the light bundle to a `results`
  branch; `run_sweep.sh` calls it on completion (`AUTO_PUSH=1`). Verified live: `tvf_g0..g3`
  pushed to the `results` branch on GitHub.
- **Token policy** — the PAT lives ONLY in the gitignored `.secrets/gh_token` on the local machine
  (never committed, never in an image). `scripts/cloud/upload_secrets.sh` pushes it to a box on
  start (token sent via stdin), writing a git credential store entry so push works in any shell.
  `bootstrap.sh` also arms from a `GH_TOKEN` env var if you prefer a Vast launch var. Verified:
  cleared box creds → `upload_secrets.sh` → push auth restored.

### Correction to the original item 2

The plan proposed gating the raw per-step traces off by default. **That is unsafe:** the scorer
(`benchmarking.py:469–486`) reads `charts/success_rate` and `charts/regime_id` to compute
`composite_score`, and scoring runs *on the box*. So `data.json` is kept **complete**; we only
compacted it. Further shrinking (post-score pruning of `debug/*` step traces) is a possible future
step but is deliberately not done here to avoid touching scorer inputs.

- **Motivation:** a completed 8-cell sweep leaves **~19 GB** in `runs/` on the Vast box, most of
  it derived or step-level detail we never pull. We want (1) results to come back automatically
  when a run finishes, and (2) far less junk written in the first place.
- **Relates to:** [handoff 2026-07-01](../handoffs/2026-07-01-trainable-vs-frozen-cloud-sweep.md),
  `scripts/cloud/pull_results.sh` (the manual light-pull this plan automates).

## Where the 19 GB actually goes (measured, per 8-cell sweep)

| Artifact | Size | Written by | Needed for | Reconstructible? |
| --- | --- | --- | --- | --- |
| `episode_*/…/inner_checkpoints/*.pt` | **~12 GB** (54 MB × 30 ep × cells) | ppo `train.py` | resume/extend only | no (weights) |
| `episode_*/…/epNN_env0_data.json` | **~5 GB** (20 MB × 30 ep) | `logger.py:plot()` | deep per-step analysis | — (raw traces) |
| `.png` charts (`ep*_charts.png`, `ep*_success_rate.png`, brain_trends) | **~1.4 GB** (61/run) | `logger.py:plot()` | eyeballing | **yes — 100% derived** |
| summaries, logs, `brain_trends/*_data.json`, `brain_model.pt`, config | **~10 MB** | sweep runner | everything we actually analyze | n/a (this is the source) |

Key facts established while pulling this sweep:
- **The PNGs are pure output.** `logger.py:plot()` (src/lifelong_learning/utils/logger.py:28–127)
  writes `_data.json` **and** renders `_charts.png` + `_success_rate.png` from the same `self.data`.
  Every chart can be regenerated locally from its `_data.json`. → dropping PNGs loses nothing.
- **Whole-run Brain data is already compact:** `brain_trends/<run>_data.json` (~39 KB) holds the
  full per-episode series (reward, all losses, all 5 HP adjustments + resulting HPs, success rate).
  This is what we need to distill paper graphs — **we have it.**
- **The 20 MB per-episode JSON is 95% raw step-level traces** (`charts/regime_id`,
  `charts/episodic_return`, `debug/*` at 26k–50k length). The *research-relevant* coarse series in
  the same file (per-update `loss/*`, `ppo/*`; per-brain-decision `brain_neuromod/*` incl.
  `decoder_weight_norm`, 64 `channel_mean_*`, `policy_kl_vs_unmasked`) total only a few KB.
- **`indent=4` alone bloats the JSON 2.8×** (38.5 → 13.8 KB on the brain_trends sample); the ratio
  is worse on the big step-trace files.

## Plan — reduce what's written (ordered by impact ÷ effort)

1. **Stop rendering PNGs on the box; render on demand locally.** In `logger.py:plot()`, gate the
   two `savefig` blocks behind `if os.environ.get("LL_RENDER_CHARTS", "0") == "1":` (default off on
   cloud). Add a tiny `scripts/render_charts.py <run_dir>` that rebuilds the same figures from
   `_data.json` for local use. **Saves ~1.4 GB, zero information loss.** *(This is the "a lot of it
   is just graph images" win.)*
2. **Make the JSON dump compact + split coarse vs raw.**
   - `json.dump(..., separators=(",", ":"))` instead of `indent=4` (≥2.8× smaller, free).
   - Always write a small `epNN_metrics.json` = the per-update + per-brain-decision series only.
     Gate the raw per-step traces (`regime_id`, `episodic_return`, `debug/*`) behind
     `LL_DUMP_STEP_TRACES=1` (default off). **Cuts the ~5 GB JSON to tens of MB** while keeping
     every series we actually plot.
3. **Thin the inner checkpoints.** In ppo `train.py`, keep only `last-N` (e.g. 1) + optional
   `best` inner checkpoint per run instead of one per episode, or gate all-but-final behind
   `LL_KEEP_ALL_CKPT=1`. **Cuts ~12 GB → ~50 MB/run.** Final checkpoint is enough to resume/extend.

Net: an 8-cell sweep drops from ~19 GB to well under ~100 MB on the box, and the ~10 MB we
actually analyze is unchanged.

## Plan — auto send-back on completion (item 1)

Constraint: the **local machine is behind NAT** — the cloud box can't push *to* it. So either the
box pushes to durable storage, or the local side pulls on a trigger. Recommended:

- **A (robust, survives box death) — cloud pushes the light bundle to a durable store on
  completion.** Add a post-sweep step to `scripts/cloud/run_sweep.sh` (after `run_seed_sweep.py`
  exits) that builds the same light bundle as `pull_results.sh` and uploads it:
  - simplest: commit the ~10 MB bundle to a dedicated `results/<sweep>` branch on the existing
    GitHub remote (`Non-Stationary-Meta-Learning`) and push; local `git fetch` gets it. Or
  - `rclone`/`rsync` to object storage (S3/B2) if we'd rather keep git clean.
- **B (interim, no new infra) — local poll-and-pull.** Run `pull_results.sh` on a loop that checks
  for `sweeps/<name>/summary.csv` and pulls once it appears (e.g. via the `/loop` skill or Task
  Scheduler). Requires the local machine on and the box alive; fine as a stopgap.

Recommendation: ship **A** — it decouples results from the ephemeral/flaky box (we lost the SSH
endpoint twice this session). Keep `pull_results.sh` as the manual fallback.

## Caveats / open questions

- **Check `config/research_manifest.toml [immutable_surface]` before editing** `logger.py` /
  ppo `train.py` — if either is listed, changing output format is a human infra decision that may
  invalidate cached fingerprints (same rule that bit the scorer change in
  [research-log 0001](../research-log/0001-2026-06-18-calibration-scoring.md)).
- Every change above is **flag-gated with cloud-lean defaults**, so local/debug runs can still get
  full PNGs + step traces by exporting the env vars.
- Decide whether step-level traces are ever needed for the paper. If not, they can default off
  permanently rather than behind a flag.

## Follow-ups

- [x] Confirm immutable_surface status of `logger.py` and ppo `train.py` — both editable (safe).
- [x] Implement (1) PNG gate + `render_charts.py`; verified re-render rebuilds the chart from JSON.
- [x] Implement (2) compact JSON (kept complete for the scorer; trace-gating dropped as unsafe).
- [x] Implement (3) checkpoint leak fix.
- [x] Wire auto send-back (A) into `run_sweep.sh`; validated against a local bare repo.
- [x] Arm auto-push + verify (real push of `tvf_g0..g3` to the `results` branch, 2026-07-01).
- [x] **Token policy:** store PAT in gitignored `.secrets/gh_token`; `upload_secrets.sh` pushes it
      to each box on start. This is the standing policy — do NOT commit the token or bake it into an
      image. (On-box `~/.git-credentials` dies on destroy; the local file is the source of truth.)
- [ ] Rotate the PAT — it was pasted into a chat/transcript (fine-grained + expiry limits risk).
- [ ] Add "run `upload_secrets.sh` on start; auto-push/`pull_results.sh` before teardown" to the
      handoff template.
- [ ] (Optional) Post-score pruning of `debug/*` step traces to shrink `data.json` further.
</content>
