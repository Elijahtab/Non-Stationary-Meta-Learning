# Autoresearch Review Remediation

**Goal:** Fix the 16 confirmed findings from the 2026-07-03 xhigh code review of the autoresearch
harness (branch `trainable-vs-frozen-sweep`), so the real 5-trial scout loop can launch on a
trustworthy foundation.
**Status:** proposed 2026-07-03

## Summary

The review confirmed six loop-integrity bugs (any of which can corrupt or silently waste
unattended trials), a set of science-validity hazards around the AI-authored mask change, config
debt from fixing the interpreter bug at the wrong altitude, and process gaps in the research-note
lifecycle. Per decisions below, Fable's uncommitted tanh-mask diff is **reverted** (the fixed loop
will re-derive it under the scout benchmark, where acceptance means something), all fixes land
**before** the real loop launches, and the supervisor gains **automatic note-verdict resolution**.
Five stages, each leaving the repo working; Stage 5 is the end-to-end revalidation and launch.

## Decisions & answers

| Decision | Answer |
| --- | --- |
| Fable's sigmoid→tanh mask diff (uncommitted) | **Revert entirely** — discard diff + note; the real loop re-derives it as a proper trial |
| Real-loop launch gate | **After all stages** complete and the pilot revalidates |
| Research-note verdicts | **Supervisor auto-appends** a dated verdict to notes created by accepted trials |

Flagged assumptions (veto anytime):

- **A1 — Timeout fix is a process-group kill** inside `default_command_runner`
  ([autoresearch.py:449](../../src/lifelong_learning/research/autoresearch.py)): replace
  `subprocess.run` with `Popen` + `CREATE_NEW_PROCESS_GROUP`/`taskkill /F /T` on Windows,
  `start_new_session=True` + `os.killpg` on POSIX. No new dependencies (no psutil).
- **A2 — Interpreter resolution via `{python}` placeholder** in `tests_command`, substituted with
  `sys.executable` at command-build time (the mechanism `default_benchmark_runner` already uses at
  [autoresearch.py:511](../../src/lifelong_learning/research/autoresearch.py)). All five manifests
  switch to the placeholder; bare commands without the placeholder keep working (format is opt-in).
- **A3 — Both wrappers converge on the ps1's null-`result` behavior** (final message falls back to
  raw stdout) since it preserves diagnostics; the sh side changes.
- **A4 — Heatmap fix is mask-mode-agnostic**: diverging colormap centered at 1.0 with
  `vmin=0, vmax=2` at [plot_high_scale.py:342](../../scripts/plot_high_scale.py) — correct for
  suppress-only today and for any future gain-mask trial (agents cannot fix this file themselves).
- **A5 — Rejected trials' notes need no verdict-append** — they are rolled back with the trial;
  the ledger + injected history are their record. Auto-resolution applies to accepted trials only.
- **A6 — `.png` is the only suffix added** to `MANAGED_FILE_SUFFIXES`
  ([autoresearch.py:28](../../src/lifelong_learning/research/autoresearch.py)) — the notes
  convention uses PNG figures only.

## Stages

### Stage 0 — Revert the tanh trial + correct the queue

Clean the working tree so every later stage diffs against paper-faithful semantics.

- `git checkout --` the three modified files:
  [neuromod.py](../../src/lifelong_learning/agents/brain/neuromod.py) (sigmoid mask restored),
  [network.py](../../src/lifelong_learning/agents/ppo/network.py) (docstrings),
  [docs/research-notes/README.md](../research-notes/README.md) (index line); delete untracked
  `docs/research-notes/0003-two-sided-gain-mask.md`. The pilot ledger + trial dir under
  `autoresearch/` keep the full record locally.
- Reword [config/hypothesis_queue.md](../../config/hypothesis_queue.md) item 2: drop the
  "accepted" claim; state honestly that a 12k-step n=1 smoke (with a corrupted prompt — see
  Stage 1) previously favored it, i.e. a weak prior, untested on the real benchmark.
- Note: the real manifest's ledger (`autoresearch/trial_results.jsonl`) contains no gain-mask
  record, so history injection will not mark queue item 2 as resolved. Correct behavior.

**Verify:** `git status` shows only intended files; full local suite
(`myenv\Scripts\python.exe -m pytest tests/... -q`, the 60-test set) passes.

### Stage 1 — Loop integrity (the six confirmed harness bugs)

1. **Orphan-process race** — rework `default_command_runner`
   ([autoresearch.py:449–489](../../src/lifelong_learning/research/autoresearch.py)) per A1 so a
   research-timeout kills the whole tree before the supervisor rolls back / snapshots the next
   trial (review finding #1).
2. **ps1 false success** — in [invoke_claude_exec.ps1](../../scripts/invoke_claude_exec.ps1):
   after the invocation, `if ($null -eq $exitCode) { $exitCode = 1 }` (finding #2, reproduced:
   caller saw exit 0 for a never-launched agent).
3. **Prompt mojibake** — same file: set `$OutputEncoding = [System.Text.Encoding]::UTF8` before
   the pipe and read the prompt with `Get-Content -Encoding UTF8` (finding #3; default is
   US-ASCII → every non-ASCII char became `?`; the prompt file is BOM-less UTF-8 so the read side
   is also wrong today). Fold-in: write output files BOM-less via `[System.IO.File]::WriteAllText`
   (deferred finding, free here).
4. **sh wrong-directory hazard** — [invoke_claude_exec.sh:52](../../scripts/invoke_claude_exec.sh):
   `cd "$REPO_ROOT" || exit 1` (finding #8, verified on-box). Also align null-`result` fallback
   with ps1 per A3 (finding #15).
5. **Queue decode crash** — [trial_prompt.py:26](../../src/lifelong_learning/research/trial_prompt.py):
   `errors="replace"`, matching the module's other two readers (finding #9).
6. **PNG snapshot blindness** — add `.png` to `MANAGED_FILE_SUFFIXES`
   ([autoresearch.py:28](../../src/lifelong_learning/research/autoresearch.py)) so agent-written
   figures roll back and appear in the audit (sweep finding).

**Verify:** new targeted tests — (a) a command-runner timeout test whose child spawns a
grandchild that touches a sentinel file after the timeout: assert the sentinel never appears;
(b) a snapshot/restore round-trip containing a `.png`; (c) wrapper encoding check: pipe `α≈−`
through the ps1 pattern into a hexdump stub, assert UTF-8 bytes arrive. Then the two harness
suites (`test_autoresearch.py`, `test_trial_prompt.py`).

### Stage 2 — Interpreter resolution at the right altitude

- Implement A2 in the supervisor: substitute `{python}` → `sys.executable` where `tests_command`
  is executed ([autoresearch.py:886](../../src/lifelong_learning/research/autoresearch.py)).
- Update all **five** manifests to `tests_command = "{python} -m pytest ..."`:
  [research_manifest.toml](../../config/research_manifest.toml),
  [research_manifest_pilot.toml](../../config/research_manifest_pilot.toml),
  [research_manifest_cloud.toml](../../config/research_manifest_cloud.toml),
  [research_manifest_cloud_pilot.toml](../../config/research_manifest_cloud_pilot.toml), and the
  previously-missed [research_manifest_085.toml](../../config/research_manifest_085.toml)
  (finding #7 — live, spec-documented, still broken). This deletes the per-OS interpreter
  duplication entirely (findings #7/#12).

**Verify:** supervisor `--dry-run` against local pilot + (over SSH) cloud pilot manifests shows
the resolved command; run the pilot tests_command verbatim under `cmd /c` locally and `bash -c`
on the box.

### Stage 3 — Diagnostics & docs validity

- **Heatmap** per A4 at [plot_high_scale.py:342](../../scripts/plot_high_scale.py) (finding #4).
  This file is in the baseline-cache fingerprint — cached baselines invalidate, which is fine:
  no scout baseline exists yet, and landing this *before* Stage 5's baseline avoids a re-run.
- **Stale trainable diagnostics** — `describe_neuromodulation`
  ([network.py:128](../../src/lifelong_learning/agents/ppo/network.py)) reads
  `neuromodulator.active_mask().detach()` instead of the snapshot buffer; frozen mode is
  byte-identical per the comment at network.py:157 (finding #5).
- **Spec 05** ([docs/spec/05-neuromodulation.md](../spec/05-neuromodulation.md)): post-revert the
  sigmoid formula is true again; fix the remaining stale claim (lines ~40–48: "decoder is frozen
  … never trains" predates `trainable_neuromod`) and note the mask-bound statement is
  mode-conditional (finding #10, narrowed).

**Verify:** network/neuromod test files; regenerate one dashboard from an existing run dir and
eyeball the mask panel; grep spec 05 for the corrected claims.

### Stage 4 — Note lifecycle & history quality

- **Auto-verdict on accepted trials** (decision 3, A5): in the accept path
  ([autoresearch.py:977–987](../../src/lifelong_learning/research/autoresearch.py)), append a
  dated `**Resolved:** accepted (primary_improved) — composite X vs baseline Y (session S,
  trial N)` section to each `docs/research-notes/*.md` in `diff.new_paths` (finding #11).
- **Note-number reservation** — compute the next free `NNNN` from `docs/research-notes/` in
  [run_research_trial.py](../../scripts/run_research_trial.py) and inject it into the prompt's
  Required Output section ("use number NNNN for any new note") (finding #14).
- **Retirement horizon** — in `collect_trial_history`
  ([trial_prompt.py:95](../../src/lifelong_learning/research/trial_prompt.py)): never evict
  science-verdict records (accepted / `primary_score_did_not_improve` / `holdout_regressed`);
  cap only infrastructure failures at 5 (finding #13).
- Update [AUTORESEARCH.md](../../AUTORESEARCH.md) to document all of the above.

**Verify:** `test_trial_prompt.py` + `test_autoresearch.py`; a rendered prompt against the pilot
ledger shows science verdicts retained past the cap and the reserved note number.

### Stage 5 — Revalidate end-to-end, then launch (decision 2)

- Re-run the **pilot** (1 trial, cpu) with the fixed harness: confirm prompt bytes are clean
  UTF-8, a forced-timeout dry test leaves no orphan, ledger verdict sane.
- Launch the **real loop**: scout baseline (~1.2 h local cuda) + 5 trials against
  [research_manifest.toml](../../config/research_manifest.toml), detached, monitored; box stays
  reserved for multi-seed confirmation sweeps of anything accepted.

**Verify:** pilot session ledger record; then the real session's baseline record and first trial
audit/tests/benchmark chain.

## Files touched

| File | Change |
| --- | --- |
| src/lifelong_learning/research/autoresearch.py | process-group kill; `.png` suffix; `{python}` substitution; accept-path note verdict |
| scripts/invoke_claude_exec.ps1 | null-exit guard; UTF-8 in/out; BOM-less writes |
| scripts/invoke_claude_exec.sh | `cd` guard; null-result fallback parity |
| src/lifelong_learning/research/trial_prompt.py | queue read `errors=replace`; retirement-horizon fix |
| scripts/run_research_trial.py | next-note-number injection |
| config/research_manifest*.toml (×5) | `{python}` placeholder |
| config/hypothesis_queue.md | item-2 honesty fix |
| scripts/plot_high_scale.py | diverging mask colormap, vmax=2 |
| src/lifelong_learning/agents/ppo/network.py | `describe_neuromodulation` uses `active_mask()` (after Stage-0 revert) |
| docs/spec/05-neuromodulation.md | trainable-mode correction |
| docs/research-notes/{README.md,0003-*} + neuromod.py | Stage-0 revert |
| AUTORESEARCH.md | document new mechanics |
| tests/ | new: timeout-tree kill, png snapshot round-trip, wrapper encoding |

## Risks & alternatives

- **Process-tree kill on Windows** relies on `taskkill /T` semantics; the new test in Stage 1 is
  the guard. Alternative (psutil) rejected: new dependency for one call site.
- **`{python}` placeholder** changes the manifest contract; mitigated by opt-in formatting (bare
  commands still run verbatim) and dry-run verification on both OSes.
- **plot_high_scale.py edit** invalidates baseline-cache fingerprints — deliberate, sequenced
  before the first scout baseline so nothing is re-run.
- **Reverting the tanh diff** discards a pilot "accept"; accepted risk — the evidence was n=1,
  12k steps, with a corrupted prompt, and queue item 2 preserves the idea for the real loop.
- Deliberately deferred (tracked, not planned): consolidating both wrappers into one Python
  launcher; `--add-dir` vs cwd parity question; ps1 `is_error` null-compare (schema-drift only);
  extracting the shared `Resolve-*` helpers into a module.

## Open questions

- None blocking. If the real loop's first accepted trial re-derives a gain-style mask, decide
  then whether it lands flag-guarded (the `trainable_neuromod` precedent) or unconditional —
  that decision was explicitly not pre-made here.
