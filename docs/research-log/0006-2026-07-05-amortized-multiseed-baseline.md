# 0006 — Amortized 3-seed scout baseline (`baseline_primary`)

- **Date:** 2026-07-05
- **Status:** Accepted
- **Touches:** `src/lifelong_learning/research/benchmarking.py` (**immutable_surface** — new
  spec `fast_switch_scout_v2_baseline` ADDED, sharing `fast_switch_scout_v2.fixed_train_args`
  by reference; nothing existing modified), `src/lifelong_learning/research/autoresearch.py`
  (`[benchmark].baseline_primary` manifest field), `config/research_manifest{,_cloud}.toml`.

## Context

The 2026-07-05 confirmation sweep (research-log 0005) revealed that the entire v2 "hit80
signature" was regression to the mean around the **n=1 scout baseline**: seed 0 drew
hit_rate_80 = 0.786 against a true frozen-control mean of ~0.829 [0.815, 0.846] at n=8. Every
variant compared against that low anchor looked reliability-improving; three n=1 scout "wins"
burned ~11 h of holdout compute adjudicating what the n=8 CIs showed was nothing.

## Decision

Split the baseline anchor from trial scoring. Trials stay on the cheap n=1
`fast_switch_scout_v2` (screening speed is the scout's whole point), but the baseline they
must beat is `fast_switch_scout_v2_baseline` — identical config, **seeds (0, 1, 2)**, scored
as the 3-seed mean. Wired via a new optional manifest field `[benchmark].baseline_primary`
(defaults to `primary`, so existing manifests are unaffected). The cloud manifest also caught
up from scout_v1 → v2 (missed in the 0004 switch).

**Amortization:** the existing fingerprint-keyed baseline cache already reuses baselines
across sessions while the fingerprint files are unchanged, so the 3× baseline cost (~3.8 h on
the 5070) is paid once per code-state, not per session. The cache lookup now also validates
the cached entry's benchmark name, so stale n=1 baselines can't satisfy a `baseline_primary`
manifest.

## Rationale

A single-seed anchor puts the acceptance bar at the mercy of one draw — tonight it sat ~1σ
low and generated three false promotions. A 3-seed mean cuts anchor variance by √3 and, more
importantly, was the cheapest change that directly kills the observed failure mode without
slowing per-trial screening. (Raising trial-side seeds was rejected: 3× per-trial cost defeats
the scout's screening role; a variance-derived acceptance margin ε remains an open register
decision — see the AUTORESEARCH.md restructure.)

## Outcome

_(fill after the first campaign runs against the 3-seed anchor)_

## Follow-ups

- First post-merge campaign pays the one-time 3-seed baseline (~3.8 h local), then amortizes.
- Open: variance-derived ε for the scout gate and holdout tolerance (needs the register).
