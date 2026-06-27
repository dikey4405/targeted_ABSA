# Targeted ABSA Demo Website

## What This Is

An interactive frontend demo for a Vietnamese Targeted Aspect-Based Sentiment Analysis (ABSA) system. Users paste Vietnamese review text, select or highlight a target phrase, and the system predicts the aspect category (e.g., `ROOMS#CLEANLINESS`) and sentiment (`POSITIVE`, `NEGATIVE`, `NEUTRAL`). The site showcases the trained PhoBERT/XLM-RoBERTa multi-task model in an accessible, shareable format.

## Core Value

A user can paste any Vietnamese review, highlight a target phrase, and instantly see the model's aspect + sentiment prediction with a confidence score.

## Requirements

### Validated

- ✓ Multi-task transformer model (`TargetedABSAModel`) predicts aspect category and sentiment from (text, target) pairs — existing
- ✓ PhoBERT-large encoder with target-conditioned attention achieves best performance (aspect F1 ~43%, sentiment F1 ~72%) — existing
- ✓ YAML-based experiment config system supports hot-swapping encoders, attention strategies, loss functions — existing
- ✓ Vocabulary builds label maps from JSONL datasets; supports 4 Vietnamese domains (hotel, restaurant, mobile, general) — existing
- ✓ Training, evaluation, and checkpointing pipeline produces `best_model.pt` — existing

### Active

- [ ] REST API endpoint wrapping `TargetedABSAModel` inference (FastAPI or Flask) — accepts `{text, target}` JSON, returns `{aspect, sentiment, probabilities}`
- [ ] Frontend single-page app with text input and interactive target phrase selection
- [ ] Prediction results display: aspect label, sentiment label, confidence scores
- [ ] Pre-loaded example sentences from each domain (hotel, restaurant, mobile) for one-click demo
- [ ] Model selector UI to switch between encoder variants (phobert_base, phobert_large, xlm_roberta_base)
- [ ] Responsive layout deployable as a static + API stack

### Out of Scope

- Training UI — training stays CLI/script-only
- Multi-language support — Vietnamese only, no translation
- User authentication — this is a public demo, no accounts
- Batch file upload — single-annotation interactive mode only
- Real-time training or fine-tuning from UI — inference only

## Context

- **ML backend**: PyTorch + HuggingFace Transformers, Python 3.9+. Model loaded from `best_model.pt` checkpoint. No `requirements.txt` exists — dependencies managed manually or via Kaggle notebook.
- **Data**: 4 Vietnamese review domains. Labels follow `ASPECT_CATEGORY#SUBCATEGORY#SENTIMENT` format (e.g., `ROOMS#CLEANLINESS#POSITIVE`). Multi-task heads predict aspect category and sentiment independently.
- **Best model config**: `phobert_large` encoder + `target_conditioned_attention` + `multitask_aspect_sentiment` label structure + `weighted_multitask_ce` loss.
- **Existing artifacts**: `reports/test_metrics.json` has benchmark scores; `config/` YAML files define all model variants.

## Constraints

- **Stack**: Python (FastAPI/Flask) for inference API; frontend in React or plain HTML/JS (lightweight preferred for demo)
- **Inference latency**: PhoBERT-large is ~1s on CPU; GPU preferred but optional — UI should show a loading state
- **No existing API layer**: Must build inference server from scratch wrapping `TargetedABSAModel`
- **Checkpoint dependency**: Frontend demo requires a trained `best_model.pt` to be present at a known path

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FastAPI for inference backend | Async-ready, auto-docs, easy JSON schema | — Pending |
| React (Vite) for frontend | Fast dev cycle, component-based UI, broad ecosystem | — Pending |
| Single model loaded at startup | Avoids per-request load latency; swap via config flag | — Pending |
| Target phrase selection via text highlight | Most natural UX for span-level annotation demo | — Pending |

---
*Last updated: 2026-06-27 after initialization*
