# Paper draft — "Meta-Control Helps a Little, Memory Decides"

Full working draft (v1, 2026-07-11). No LaTeX toolchain on this machine — build on
Overleaf (upload this folder) or locally:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Provenance (where every number comes from)

| Content | Source of truth |
| --- | --- |
| Control ladder (Table 1, Fig 1) | `docs/research-notes/0006` |
| March matched contrast | `docs/research-notes/0006` (T-series, n=32/arm) |
| Multi-seed replication (Table 2, Fig 2) | `docs/research-notes/0007` §Results (LOOP-0009, n=16/arm) |
| Related work + citation hygiene | `docs/research-notes/0010` (62 verified papers) |
| Structure / claim map | `docs/research-notes/0008` (skeleton) |
| Instrument details (levers, bounds, signals, reward) | `src/.../brain/{neuromod,meta_env,signals}.py` (verified against source 2026-07-11) |

- `references.bib` is a **copy** of `docs/references/references.bib` (source of truth) —
  re-copy after edits there.
- Figures regenerate from committed numbers: `python paper/figures/make_figures.py`.
- Venue: drafted venue-agnostic (article class); retarget to the venue class (e.g.
  CoLLAs/PMLR) at submission. Remaining TODOs in the tex: affiliation, anonymized repo link.

## Pre-submission checklist (from note 0010 §Citation-hygiene)

- [ ] Verify Backpropamine arXiv id (1811.06308) against OpenReview
- [ ] Dohare et al.: cite Nature version (632:768-774, 2024)
- [ ] Complete "and others" author lists in the bib at camera-ready
- [ ] Optional: encoder-dormancy probe before finalizing the C4 sentence (action-tree Wave 0)
- [ ] Fig 3 candidate (dormant-fraction trajectory) — data on `origin/results`; add if wanted
