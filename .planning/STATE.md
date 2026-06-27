# Project State

**Project:** Targeted ABSA Demo Website
**Phase:** Initialized — ready to plan Phase 1
**Last Updated:** 2026-06-27

## Current Status

- [x] PROJECT.md written
- [x] REQUIREMENTS.md written (19 v1 requirements)
- [x] Research completed (4 domain areas)
- [x] ROADMAP.md written (3 phases)
- [ ] Phase 1: Inference API — not started
- [ ] Phase 2: Frontend Foundation — not started
- [ ] Phase 3: Examples & Polish — not started

## Active Phase

None. Run `/gsd-plan-phase 1` to begin.

## Key Context

- Brownfield: existing ML training code in root; demo website is additive
- No `requirements.txt` yet — Phase 1 creates it
- Checkpoint path: `checkpoints/<group>/<run>/best_model.pt` — user must provide trained model
- Best model config: `phobert_large` + `target_conditioned_attention` + `multitask_aspect_sentiment`
- No word segmentation needed (confirmed by code + data inspection)

## Research Artifacts

| File | Focus |
|------|-------|
| `.planning/research/api-server.md` | FastAPI patterns, model loading, concurrency, pitfalls |
| `.planning/research/frontend-demo.md` | React component structure, target selection UX, Vietnamese handling |
| `.planning/research/domain-context.md` | 14 curated examples, encoder recommendation, font/Unicode guidance |
| `.planning/research/deployment.md` | Project structure, requirements.txt, Makefile, HF Spaces path |

---
*State initialized: 2026-06-27*
