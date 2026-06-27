# Roadmap: Targeted ABSA Demo Website

**Project:** Targeted ABSA Demo Website
**Version:** v1.0
**Goal:** Interactive frontend demo where users paste Vietnamese reviews, highlight a target phrase, and get aspect + sentiment predictions from the trained PhoBERT model.

---

## Milestone 1: Demo Website v1.0

### Phase 1: Inference API

**Goal:** A working FastAPI server that loads the ABSA model and serves predictions via REST.

**Delivers:**

- `api/` directory with FastAPI app, inference engine, and Pydantic schemas
- `requirements.txt` covering all inference dependencies
- Health check, predict, and model-info endpoints
- Model + Vocabulary loaded at startup; graceful 503 until ready

**Requirements:** API-01, API-02, API-03, API-04, API-05, API-06, API-07, SETUP-01

**Plans:** 2/3 plans executed

Plans:

- [x] 01-01-PLAN.md — Pydantic schemas (api/schemas.py) + requirements.txt, .env.example, checkpoints directory
- [x] 01-02-PLAN.md — InferenceEngine class (api/__init__.py, api/inference.py) with model loading, asyncio.Lock, NFC normalization, predict()
- [ ] 01-03-PLAN.md — FastAPI app (api/main.py) with lifespan, CORS, /predict /health /models endpoints + Makefile

**UAT:**

- `curl -X POST http://localhost:8000/predict -d '{"text":"Phòng khách sạn rất sạch sẽ","target":"Phòng"}' -H "Content-Type: application/json"` returns `{"aspect":"ROOMS#CLEANLINESS","sentiment":"POSITIVE","aspect_probs":{...},"sentiment_probs":{...},"latency_ms":...}`
- `curl http://localhost:8000/health` returns `{"status":"ok","model_loaded":true,"device":"cpu"}`
- Request where `target` is not substring of `text` returns HTTP 422 with descriptive message
- Server starts with `uvicorn api.main:app --workers 1` (single worker due to model memory)

**Key constraints from research:**

- Use `model.load_state_dict(torch.load(path, weights_only=True)['model_state_dict'])` — `train.py` saves state-dict only
- `Vocabulary` scans JSONL files at startup — point `DATA_DIR` env var to `Data/`
- Single asyncio.Lock() for inference — phobert-large is CPU-bound ~1s/request
- Mount static files LAST in FastAPI (wildcard catch-all eats API routes)
- `allow_credentials=True` + `allow_origins=["*"]` is rejected by browsers — use explicit origins

---

### Phase 2: Frontend Foundation

**Goal:** React/Vite SPA with text input, target phrase selection, and prediction result display.

**Delivers:**

- `frontend/` Vite + React project scaffold
- `ReviewInput` component: textarea + selectionStart/End-based target capture
- `TargetSelector` component: shows selected span, allows manual override
- `PredictButton` with loading state and error handling
- `PredictionResult` component: aspect chip, sentiment badge with color, confidence bars
- Vite proxy config to forward `/predict` → `localhost:8000`
- Page header with project title and GitHub link

**Requirements:** FE-01, FE-02, FE-03, FE-04, FE-05, FE-06, FE-07, FE-08, FE-09, FE-13, FE-14, FE-15

**UAT:**

- User types Vietnamese text in textarea, highlights a target phrase, clicks Predict → result displays in < 5s
- Predict button is disabled when text or target field is empty
- Loading spinner shown during API call
- POSITIVE result shows green badge, NEGATIVE shows red, NEUTRAL shows gray
- Confidence bars show top-5 aspect predictions with percentages
- Layout works on 320px mobile viewport and 1280px desktop

**Key constraints from research:**

- Use `selectionStart`/`selectionEnd` from textarea — NOT contenteditable (breaks Vietnamese diacritics)
- Send target as string to API, not character offsets
- Normalize to NFC before sending: `text.normalize('NFC')`
- Use native `fetch` with `AbortSignal.timeout(15_000)` — no Axios needed
- Vite proxy eliminates CORS in dev; no CORSMiddleware needed until production deployment

---

### Phase 3: Demo Examples & Polish

**Goal:** Pre-loaded examples, UX polish, and dev workflow automation.

**Delivers:**

- 6 curated example cards (2 hotel, 2 restaurant, 2 mobile) with one-click predict
- Domain labels on example cards (hotel/restaurant/mobile)
- Responsive layout polish and font: `Be Vietnam Pro` from Google Fonts
- `dev.sh` / `Makefile` to run API + Vite dev server with one command
- `README.md` update with demo setup instructions and checkpoint path
- `.env.example` documenting `CHECKPOINT_PATH` and `DATA_DIR`

**Requirements:** FE-10, FE-11, FE-12, SETUP-02, SETUP-03

**UAT:**

- 6 example cards visible below the input form
- Clicking any example card pre-fills text + target and triggers prediction
- Cards show domain tag (e.g., "🏨 Hotel", "🍴 Restaurant", "📱 Mobile")
- `make dev` starts both API and frontend with one command
- README includes step-by-step: install deps → copy checkpoint → run demo
- Vietnamese text renders correctly with diacritics on Chrome, Firefox, Safari

**Curated examples (from domain research):**

1. Hotel: `"Phòng khách sạn rất sạch sẽ và thoáng mát"` / target: `"Phòng"` → ROOMS#CLEANLINESS / POSITIVE
2. Hotel: `"Khăn tắm cũ và có vết bẩn"` / target: `"Khăn tắm"` → ROOM_AMENITIES#* / NEGATIVE
3. Restaurant: `"Món ăn rất ngon, phục vụ nhanh"` / target: `"Món ăn"` → FOOD#* / POSITIVE
4. Restaurant: `"Nhân viên phục vụ thái độ kém"` / target: `"Nhân viên"` → SERVICE#* / NEGATIVE
5. Mobile: `"Thiết kế đẹp, cầm rất vừa tay"` / target: `"Thiết kế"` → DESIGN / POSITIVE
6. Mobile: `"Loa ngoài nhỏ, nghe không rõ"` / target: `"Loa ngoài"` → FEATURES / NEGATIVE

---

## Backlog (v2)

- **Model selector UI** — switch between encoder variants at runtime (requires API multi-model loading)
- **Batch mode** — predict for multiple annotations at once
- **HuggingFace Spaces / Gradio deployment** — one-file `app.py` for instant public URL
- **Docker Compose** — reproducible deployment stack
- **Attention visualization** — token-level heatmap for target attention

---

## Phase Summary

| Phase | Focus | Key Deliverable | Requirements |
|-------|-------|-----------------|--------------|
| 1 | 2/3 | In Progress|  |
| 2 | Frontend Foundation | `frontend/` React app | FE-01–09, FE-13–15 |
| 3 | Examples & Polish | Demo cards + dev workflow | FE-10–12, SETUP-02–03 |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Checkpoint not found at startup | Medium | High | Graceful 503 with clear error message; README checkpoint setup step |
| PhoBERT-large CPU latency > 3s | Medium | Medium | Show loading spinner; consider xlm_roberta_base for demo (278MB, faster) |
| `_build_target_mask` returns all-zeros (target not found in tokens) | Medium | High | Validate target is exact substring before inference; return 422 |
| Vocabulary cold-start (scans 6MB JSONL) | Low | Medium | Pre-build and pickle Vocabulary, or cache on first load |
| Vietnamese NFD diacritics from iOS | Low | Medium | NFC normalize on both frontend and API |

---
*Roadmap defined: 2026-06-27*
*Last updated: 2026-06-27 after initialization*
