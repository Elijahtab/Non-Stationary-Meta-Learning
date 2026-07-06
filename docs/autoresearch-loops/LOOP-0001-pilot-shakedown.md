# LOOP-0001 — pilot-shakedown (2026-07-04)

**Goal:** Validate the Claude (Fable) trial harness end-to-end on the tiny cpu pilot benchmark.
**Verdict:** Harness validated after two infrastructure fixes; all gate paths exercised.

## Hardware
Home RTX 5070 only (pilot benchmark runs on cpu; agent local).

## Code state
`trainable-vs-frozen-sweep` @ `cb05fe2`→`ce4c819`; `config/research_manifest_pilot.toml`;
agent = `claude -p` (claude-fable-5) via `scripts/invoke_claude_exec.ps1`.

## Runs
3 pilot sessions (ledger `autoresearch/trial_results_pilot.jsonl`, sessions `20260704-004958`,
`-005857`, `-010600`): baseline ~51 s cpu, agent ~7 min/$3.60/33 turns per trial.

## Obstacles
1. Trial 1 rejected `unauthorized_surface_violation` for writing a research note — the repo's
   own CLAUDE.md convention conflicted with the manifest → `docs/research-notes` added to the
   editable surface (`97f3e6a`).
2. Trial 2 rejected `tests_failed`: `tests_command` used a forward-slash exe path; cmd.exe
   can't run it → backslash fix (`ce4c819`); later generalized to `{python}` (research-log 0006 era).
3. Trial 3 ACCEPTED (two-sided tanh mask, 0.5014 vs 0.4398 smoke) — later revealed: its prompt
   had been mojibaked by PS 5.1's US-ASCII `$OutputEncoding` (fixed `6ae4e42`), and the diff was
   deliberately reverted (`1ca17f0`). **Pilot-ledger accepts are plumbing signals, not science.**

## Statistics & verdicts
Gate-path coverage: audit-reject ✓, tests-reject ✓, accept ✓ (science-reject exercised in
LOOP-0002's re-validation pilot). Smoke scores are meaningless by design (12k steps, n=1).

## Cost
~$11 agent fees; negligible compute.

## Sibling loops
None (sequential precursor to LOOP-0002).

## Pickup state
(historical) Harness trusted; proceed to real campaigns.

## Links
`autoresearch/trial_results_pilot.jsonl`; review remediation plan
`docs/plans/2026-07-03-autoresearch-review-remediation.md` (the 16-finding hardening that
preceded this loop).
