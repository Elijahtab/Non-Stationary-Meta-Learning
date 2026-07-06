> **SUPERSEDED for pickup (2026-07-05):** handoff docs are retired for autoresearch work —
> new sessions start from `AUTORESEARCH.md` + the latest note in `docs/autoresearch-loops/`
> (pickup-state section). This document is preserved as the historical record of the
> 2026-07-04/05 dual-box campaign. Its location note no longer applies.

# Hand-off: Dual-box autoresearch — v2 campaign + pending confirmation sweep (2026-07-05)

> Location note: this file lives in gitignored `autoresearch/` because the local campaign was
> mid-trial when it was written — writing to `docs/` would have landed in the running trial's
> audit diff and poisoned it (see Gotchas #1). Once the campaign ends, COPY it to
> `docs/handoffs/` (repo convention) and commit.

## Goal

Find a neuromodulation mechanism that beats the frozen baseline on the catastrophic-forgetting
benchmark, via an autonomous research loop: Claude (Fable) implements one bounded hypothesis
per trial, the supervisor (`scripts/run_autoresearch.py`) audits/tests/benchmarks it. Two-tier
design: the home RTX 5070 screens hypotheses at n=1 on `fast_switch_scout_v2` (8×8); a rented
Vast.ai box (4× RTX 3060 12GB / 80 vCPU — user provides SSH) confirms signals with multi-seed
sweeps. Long-term the flow is orchestrated by the `/autoresearch-run` skill (parked, see below).

## State

- **Branches:** `trainable-vs-frozen-sweep` @ `32c8676` (main line, pushed);
  `confirmation-sweep-conditions` @ `9991565` (pushed; worktree at `../LL-confirm`) — contains
  the new flags/conditions/preset + supervisor crash-safety. Merge into the main line only
  AFTER the local campaign ends.
- **Local campaign RUNNING:** session `20260704-203740` (launcher `scripts/run_local_loop.ps1`,
  detached; done-marker `autoresearch/v2_campaign1.exitcode`; ledger
  `autoresearch/trial_results.jsonl`). Trials 3–5 pending as of writing.
- **Parked in scratchpad** (`C:\Users\elija\AppData\Local\Temp\claude\C--Users-elija-Desktop-Projects-Lifelong-Learning\38ecf6a9-448b-4d07-8800-6d7b84dd7c7d\scratchpad\`),
  install into the repo + commit ONLY after the campaign ends:
  `PARKED-SKILL-autoresearch-run.md` → `.claude/skills/autoresearch-run/SKILL.md`;
  `PARKED-confirmation-sweep-plan.md` → `docs/plans/2026-07-05-confirmation-sweep-actor-only-gain.md`;
  this file → `docs/handoffs/`.
- **Box:** NOT yet rented/connected. User will paste an SSH string (`ssh -p PORT root@IP`).
- **Uncommitted in main tree:** only the running trial's own edits (roll back automatically).

## Home-box trial record (the evidence so far)

v1 scout (`fast_switch_scout_v1`, 5×5, baseline composite 0.6622, hit80 0.991 — SATURATED,
see research-log 0004; verdicts are v1-scoped prior evidence, not retirement):
six rejections, all −1–3%: actor-only 0.6541 · gain α=1.0 0.6552 · FiLM 0.6470 ·
split actor/critic masks 0.6411 · context-dim-16 0.6622 (near-no-op) · channel gates 0.6507.

v2 scout (`fast_switch_scout_v2`, 8×8, baseline composite **0.5499**, hit80 **0.786**;
holdout baseline 0.5477/hit80 0.839), session `20260704-203740`:
- Trial 1 actor-only: scout **0.5505** (beat baseline!) hit80 **0.875**; holdout 0.5475 vs
  0.5477 → rejected `holdout_regressed` by zero-tolerance on a −0.0002 delta. hit80 UP on both.
- Trial 2 gain α=0.5: 0.5491, hit80 **0.893** → rejected on composite, hit80 +10.7pts.
- **The "signature": composite flat/noise, hit_rate_80 up ≥ +5pts.** Two mechanisms show it.
  This is what the confirmation sweep adjudicates. Trials 3–5: check the ledger for
  session `20260704-203740`, `trial_index >= 3` — if any shows the signature, add it as a
  4th sweep arm (needs a flag+condition first, same pattern as commit `9991565`).

## Key decisions & findings

- Scout v1→v2 switch rationale + registered prediction: `docs/research-log/0004-*.md`. The
  prediction partially fired (a v1-rejected mechanism cleared the v2 scout bar).
- The unconditional tanh mask from the smoke-pilot "accept" was REVERTED (user decision) — its
  prompt had been corrupted by a since-fixed encoding bug and it failed the real benchmark.
  Never trust the pilot ledger (`trial_results_pilot.jsonl`) as science; it's plumbing-smoke.
- Trial-history injection is benchmark-scoped; science verdicts never age out; note numbers are
  reserved per trial; accepted trials get auto-appended note verdicts (all in `trial_prompt.py`
  / `autoresearch.py` on the main line).
- Crash safety (branch `9991565`): write-ahead journal + startup recovery + `autoresearch/STOP`
  sentinel for graceful kills. Before this, killing the supervisor mid-trial silently
  contaminated the next baseline with stranded agent edits.
- Trial agent = local `claude` CLI via `scripts/invoke_claude_exec.ps1` (Fable, subscription
  auth, no API key; ~$3.60 + ~85 min per trial incl. benchmark).

## Gotchas

1. **NEVER write/edit snapshot-managed files (.py/.md/.toml/.png etc. outside `autoresearch/`,
   `runs/`, `benchmarks/`) in the MAIN tree while a local trial is in flight** — the diff audit
   attributes your edit to the running trial, rejects it, and rolls YOUR file back. Use the
   `../LL-confirm` worktree or wait for the session's `.exitcode` file.
2. Worktree testing: the venv has an editable install pointing at the MAIN tree — always run
   `PYTHONPATH=<worktree>/src` or imports silently resolve to the wrong copy.
3. Claude subscription 429 ("session limit"): trials fail fast as `research_command_failed`
   (retryable); relaunch `run_local_loop.ps1` after the stated reset time.
4. `tests_command` uses a `{python}` placeholder resolved to `sys.executable` — never hardcode
   interpreters in manifests. Windows shell=True is cmd.exe (no forward-slash exe paths).
5. Vast boxes: `workspace_is_volume` usually false — push results (`AUTO_PUSH=1` →`results`
   branch) religiously; Pascal GPUs need the cu126 torch downgrade (docs/plans/cloud-setup.md);
   3060/Ampere needs nothing.
6. Keep the home machine awake during campaigns (a sleep once killed a run mid-episode).

## Next steps

1. **When `autoresearch/v2_campaign1.exitcode` appears:** read trials 3–5 verdicts; install the
   three parked files (paths above) + commit; merge `confirmation-sweep-conditions`; fill
   research-log 0004's Outcome; update `config/hypothesis_queue.md` with v2 verdicts.
2. **When the user provides the box SSH:** `HOST=<ip> PORT=<port> bash
   scripts/cloud/upload_secrets.sh`; on box: `git clone -b confirmation-sweep-conditions
   https://github.com/Elijahtab/Non-Stationary-Meta-Learning.git /workspace/Lifelong-Learning`
   → `bash scripts/cloud/bootstrap.sh`. Then write research-log 0005 (registered prediction:
   both arms beat frozen on hit80 with CI separation at n=8, composite within noise) and launch
   per GPU: `CUDA_VISIBLE_DEVICES=N MAX_PARALLEL=1 RESUME=1 bash scripts/cloud/run_sweep.sh
   scoutv2 "<one-of: brain_neuromod | brain_neuromod_actor_only | brain_neuromod_gain05>"
   "<seed subset of 1..8>" confirm_gN` — 24 cells ≈ 6 waves ≈ 7–9 h. Details in the parked plan.
3. **After the sweep:** `pull_results.sh`, CIs for composite + hit_rate_80, research note 0003,
   verdict → extend/escalate/close, destroy the box.
4. Optionally register the `/autoresearch-run` skill flow (parked SKILL.md) for the standing
   dual-box loop, seeded with THIS file as its handoff argument.

## References

`AUTORESEARCH.md` (architecture + ops); `docs/plans/2026-07-03-autoresearch-review-remediation.md`
(hardening history); `docs/research-log/0004-2026-07-04-scout-v2-8x8.md`;
`docs/research-notes/0001-*,0002-*` (why the mechanism family is the open direction);
`config/hypothesis_queue.md`; `scripts/run_local_loop.ps1`; prior handoff
`docs/handoffs/2026-07-01-trainable-vs-frozen-cloud-sweep.md` (cloud ops patterns).
