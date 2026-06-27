# Requirements: Targeted ABSA Demo Website

**Defined:** 2026-06-27
**Core Value:** A user can paste any Vietnamese review, highlight a target phrase, and instantly see the model's aspect + sentiment prediction with a confidence score.

## v1 Requirements

### Inference API

- [ ] **API-01**: `POST /predict` accepts `{"text": str, "target": str}` and returns `{"aspect": str, "sentiment": str, "aspect_probs": {label: float}, "sentiment_probs": {label: float}, "latency_ms": int}`
- [ ] **API-02**: `GET /health` returns `{"status": "ok", "model_loaded": bool, "device": str}`
- [ ] **API-03**: `GET /models` returns available encoder configs and which is active
- [ ] **API-04**: CORS headers configured so the frontend (localhost:5173 + production origin) can call the API
- [x] **API-05**: Validate that `target` is an exact substring of `text` (required by `_build_target_mask`); return 422 with message if not; return 503 if model not loaded; return 500 with message for inference errors
- [x] **API-06**: Model and `Vocabulary` load from checkpoint + JSONL data files on server startup; API returns 503 until both are ready
- [x] **API-07**: Input text and target normalized to Unicode NFC before inference (prevents tokenization mismatch from iOS/macOS copy-paste NFD input)

### Frontend — Input

- [ ] **FE-01**: Textarea for Vietnamese review text (multi-line, resizable, placeholder text in Vietnamese)
- [ ] **FE-02**: Target phrase selection: user can highlight text in the textarea and the selected span is captured as the target input
- [ ] **FE-03**: Target phrase field shows the currently selected span; user can also type it manually
- [ ] **FE-04**: "Predict" submit button disabled when text or target is empty; shows spinner during API call
- [ ] **FE-05**: Error state displayed inline when the API call fails (e.g., "Model not loaded — please wait")

### Frontend — Results

- [ ] **FE-06**: Predicted aspect label displayed as a styled chip/badge (e.g., `ROOMS#CLEANLINESS`)
- [ ] **FE-07**: Predicted sentiment displayed with color coding (green=POSITIVE, red=NEGATIVE, gray=NEUTRAL)
- [ ] **FE-08**: Confidence scores shown as a bar chart or progress bars for top-5 aspect predictions
- [ ] **FE-09**: Confidence score shown for the predicted sentiment

### Frontend — Examples

- [ ] **FE-10**: At least 6 pre-loaded example cards (2 per domain: hotel, restaurant, mobile) with (text, target) pairs
- [ ] **FE-11**: Clicking an example card populates the input fields and auto-submits prediction
- [ ] **FE-12**: Examples are annotated with their expected domain so users understand the taxonomy

### Frontend — Layout

- [ ] **FE-13**: Responsive single-page layout works on desktop (≥768px) and mobile (<768px)
- [ ] **FE-14**: Page header with project title, brief description, and link to GitHub repo
- [ ] **FE-15**: Loading state (skeleton or spinner) shown during API latency

### Project Setup

- [x] **SETUP-01**: `requirements.txt` covering all inference server dependencies (fastapi, uvicorn, torch, transformers, etc.)
- [ ] **SETUP-02**: `README.md` updated with demo setup instructions (install, checkpoint path, start commands)
- [ ] **SETUP-03**: `dev.sh` or Makefile target to run API + Vite dev server concurrently with a single command

## v2 Requirements

### Advanced Demo Features

- **V2-01**: Model selector dropdown in UI to switch between encoder variants (phobert_base, phobert_large, xlm_roberta_base) — requires API reload or multiple model instances
- **V2-02**: Batch mode: paste multiple sentences and get predictions for all
- **V2-03**: Attention visualization showing which tokens the model attended to for the target phrase
- **V2-04**: Feedback mechanism (thumbs up/down) to collect corrections

### Deployment

- **V2-05**: HuggingFace Spaces or Render deployment with public URL
- **V2-06**: Docker compose setup for reproducible deployment
- **V2-07**: GitHub Actions CI to run lint + type-check on push

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training UI | Training stays CLI-only; model weights are pre-computed |
| User authentication | Public demo; no accounts needed |
| Multi-language support | Vietnamese only per training data |
| Batch file upload | Interactive annotation mode only |
| Real-time fine-tuning from UI | Inference only; no training loop |
| PhoBERT word segmentation | **Not needed** — training data is raw Vietnamese; adding segmentation creates train/inference distribution mismatch |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| API-01–06 | Phase 1: Inference API | Pending |
| FE-01–05 | Phase 2: Frontend Input | Pending |
| FE-06–09 | Phase 2: Frontend Results | Pending |
| FE-10–12 | Phase 3: Examples & Polish | Pending |
| FE-13–15 | Phase 2: Frontend Layout | Pending |
| SETUP-01–03 | Phase 1: Inference API | Pending |

**Coverage:**

- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2026-06-27*
*Last updated: 2026-06-27 after initialization*
