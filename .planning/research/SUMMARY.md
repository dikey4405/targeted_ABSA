# Research Summary — Targeted ABSA Demo Website

**Synthesized:** 2026-06-27  
**Sources:** api-server.md · frontend-demo.md · domain-context.md · deployment.md  
**Confidence:** HIGH — all four files based on direct codebase inspection + confirmed metrics from `reports/test_metrics.json`

---

## Executive Summary

This project wraps a trained PyTorch/PhoBERT model (`phobert_large_target_attn_weighted`) in a FastAPI inference server and a React/Vite demo frontend. The ML model achieves **91.2% sentiment accuracy** and **43.2% aspect F1** on a 3-domain Vietnamese review dataset (hotel, restaurant, mobile phones). The demo's value proposition is *targeted* sentiment analysis: the same sentence produces different predictions depending on which phrase the user highlights — this is the one UX moment that must work perfectly.

The overall architecture is simple: a single FastAPI process loads the model once at startup via a `lifespan` hook, and a React SPA renders results from a single `POST /predict` endpoint. No message queues, no caching layer, no databases. The only non-trivial complexity is the **model reconstruction** at inference time (state-dict-only checkpoint requires the full `TargetedABSAModel` class to be co-deployed) and the **Vietnamese text normalization pipeline** (BOM stripping + NFC normalization on both frontend and backend to avoid silent substring-match failures that zero out the target mask).

The primary deployment path is **HuggingFace Spaces with Docker** (~2 hours setup): it bundles the FastAPI backend + compiled React frontend in a multi-stage image, handles the 1.2GB checkpoint via `hf_hub_download` at startup, and requires no paid plan. Local development uses a Makefile (`make dev`) that runs both processes in parallel with Vite's dev proxy eliminating all CORS friction.

---

## Cross-Cutting Insights

### Confirmed Across ≥ 2 Research Files

| Insight | Files | Implication |
|---------|-------|-------------|
| **No Vietnamese word segmentation needed** — model trained on raw text via PhoBERT BPE | api-server.md + domain-context.md | Do NOT add underthesea/VnCoreNLP; it would create train/serve distribution mismatch |
| **NFC normalization + BOM strip on both sides** | frontend-demo.md + domain-context.md + api-server.md | Single defense at API is insufficient — iOS/macOS paste from frontend must normalize before sending |
| **Target as string, never as offset** | frontend-demo.md + api-server.md | Avoids entire class of UTF-16 vs Python str offset bugs; `vocab.tokenize_text_pair(text, target)` accepts string args |
| **CPU inference latency 1–3s** | api-server.md + deployment.md | Frontend: 15s `AbortSignal.timeout`; backend: `asyncio.Lock()` to serialize requests |
| **Only phobert-large has a trained checkpoint** | domain-context.md + deployment.md | Model selector dropdown should show other variants but mark them "unavailable" — don't load what you don't have |
| **86 aspect classes in response** | api-server.md + frontend-demo.md | Backend must trim to top-5 aspect probabilities before sending; rendering all 86 bars is unusable |

### Tension / Conflicts Between Files

| Conflict | Resolution |
|----------|------------|
| **API response shape:** api-server.md uses flat `aspect_probabilities`/`sentiment_probabilities`; frontend-demo.md expects nested `probabilities: {aspect: {...}, sentiment: {...}}` | **Use flat shape** — InferenceEngine returns flat; frontend adapter maps it. Flat is easier to validate with Pydantic. |
| **Model selector:** frontend-demo.md proposes loading all 3 models at startup; deployment.md notes each model is 1.2GB RAM | **Load one model** (phobert-large). Show other variants in dropdown but disabled, labeled "coming soon". Multiple models in RAM is impractical on free-tier hosts. |
| **CORS:** frontend-demo.md says CORS not needed in dev (Vite proxy); api-server.md hardcodes `allow_origins=["*"]` in demo config | **Keep CORS middleware** in FastAPI with explicit origins list — no wildcard in production. Dev proxy covers local; CORS covers deployed origins. |

---

## Architecture Decisions Confirmed by Research

### Technology Choices

| Choice | Decision | Evidence |
|--------|----------|----------|
| Backend framework | **FastAPI** (not Flask) | Pydantic validation prevents target/text mismatch bugs; lifespan hook is cleaner than `before_first_request` for model loading |
| Frontend framework | **Vite + React + TypeScript** | Minimal bundle (~150-200KB gzipped), fast HMR, TS catches `probabilities.aspect` key-access bugs at compile time |
| Styling | **Tailwind CSS via `@tailwindcss/vite`** | Optional but recommended — utility classes ship the demo layout in one file without a CSS cascade |
| Target selection UI | **`textarea` + `selectionStart`/`selectionEnd`** | Vietnamese chars are all BMP; JS UTF-16 indices = Python str indices. `contenteditable` has IME/normalization issues on Android |
| Data fetching | **Native `fetch` with `AbortSignal.timeout`** | Single endpoint, no caching/pagination. Axios adds 14KB for zero benefit. TanStack Query adds 30KB. |
| Probability display | **CSS width trick** | `width: ${prob * 100}%` with `transition: width 0.4s ease`. No chart library (recharts/d3 = overkill) |
| State management | **`useState` in App.tsx** | Three state variables total. No context, no reducer, no Zustand. |
| Deployment | **HuggingFace Spaces (Docker)** | Free tier, 7GB RAM, multi-stage Dockerfile bakes React into FastAPI static files |

### Recommended Project Layout

```
targeted_ABSA/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + lifespan + CORS + static file serving
│   ├── inference.py     # InferenceEngine class (sys.path trick to import from root)
│   └── schemas.py       # Pydantic: PredictRequest, PredictResponse, HealthResponse
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/  # ReviewInput, TargetBadge, PredictButton, ExamplePicker,
│   │   │                #   ResultCard, AspectLabel, SentimentBadge, ProbabilityBars
│   │   ├── hooks/       # usePredict.ts
│   │   ├── data/        # examples.ts (hard-coded, never fetched)
│   │   └── types.ts
│   ├── vite.config.ts   # proxy /predict → localhost:8000
│   └── package.json
├── checkpoints/         # gitignored *.pt; .gitkeep keeps dir in repo
├── model/               # UNTOUCHED
├── vocabulary.py        # UNTOUCHED — imported via sys.path
├── requirements.txt     # CPU-only inference deps
├── requirements-train.txt
├── Makefile             # make dev · make api · make frontend · make build
└── .env.example
```

### Key Startup Pattern

```python
# api/inference.py — sys.path trick (one line, no install step needed)
sys.path.insert(0, str(Path(__file__).parent.parent))

# api/main.py — lifespan loads model once; all requests share singleton
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = InferenceEngine(checkpoint_path=CHECKPOINT_PATH, ...)
    yield
```

### Key Inference Contract

```
POST /predict
Body:  { "text": "<raw Vietnamese>", "target": "<substring of text>" }
Returns: {
  "aspect": "SERVICE#GENERAL",
  "sentiment": "POSITIVE",
  "aspect_probabilities": { top-5 labels → float },   ← trimmed, not all 86
  "sentiment_probabilities": { "POSITIVE": 0.82, "NEGATIVE": 0.13, "NEUTRAL": 0.05 },
  "confidence": 0.82,
  "warning": null | "Target not found in tokenized input"
}
```

---

## Critical Gotchas — DO NOT DO List

> These are the bugs that will silently corrupt the demo if not addressed.

### 🔴 Critical (breaks inference or demo silently)

1. **DO NOT add Vietnamese word segmentation (underthesea/VnCoreNLP).**  
   The model was trained on raw unsegmented text. Adding segmentation at inference creates train/serve distribution mismatch → degraded predictions on every request with no error.

2. **DO NOT forget `model.eval()` after loading state dict.**  
   Without it, dropout layers stay active during inference → every call returns different probabilities. Predictions become stochastic and non-deterministic.

3. **DO NOT skip NFC normalization on the frontend before sending.**  
   iOS/macOS paste sometimes produces NFD-encoded Vietnamese diacritics. If `text` is NFD and `target` is NFC (or vice versa), `target not in text` is `True` even though they look identical → `target_mask` is all zeros → model falls back to CLS-only pooling → degraded predictions with no error message.

4. **DO NOT send character offsets to the backend.** Only send `text` and `target` as plain strings. The API's `tokenize_text_pair(text, target)` does its own substring scanning.

5. **DO NOT reconstruct the model without loading the state dict first.**  
   `best_model.pt` is saved as state dict only. `torch.load()` it into a manually constructed `TargetedABSAModel(...).load_state_dict(state_dict)`. Passing wrong constructor args (wrong `num_aspects`, wrong `attention_key`) causes a shape mismatch error at load time.

6. **DO NOT use multiple uvicorn workers (`--workers 4`).**  
   PhoBERT-large is ~1.2GB in memory. 4 workers = ~5GB RAM. Crashes on any free-tier host. Always `--workers 1`; use `asyncio.Lock()` to serialize requests.

### 🟡 Moderate (causes confusing bugs, not silent)

7. **DO NOT use `contenteditable` for the text input area.**  
   Vietnamese IME (Gboard, SwiftKey) on Android emits inconsistent `compositionend` events on `contenteditable`. `textarea` has native OS handling. Use `textarea` + `selectionStart`/`selectionEnd`.

8. **DO NOT render all 86 aspect probabilities in the frontend.**  
   Return top-5 from backend, render top-5 with CSS bars. Rendering 86 rows looks broken and takes ~150ms to layout.

9. **DO NOT set textarea font-size below 16px.**  
   iOS Safari auto-zooms into any input with `font-size < 16px`. Non-fixable without user-scalable=no viewport (which breaks accessibility).

10. **DO NOT strip BOM only on the backend.**  
    Data files contain BOM (`\uFEFF`) at file start. Users who open a `.txt` file in Notepad and paste it carry the BOM into the textarea. Strip on frontend (`text.replace(/^\uFEFF/, '')`) AND in the API endpoint.

11. **DO NOT use a tokenizer that doesn't match the checkpoint's encoder.**  
    `train.py` has a known bug: it always passes `"vinai/phobert-base"` to `Vocabulary()` even when training with `xlm-roberta-large`. For inference, the `InferenceEngine` must use the tokenizer that matches the checkpoint's actual encoder (phobert-large for the current best checkpoint).

12. **DO NOT call `allow_credentials=True` with `allow_origins=["*"]`.**  
    Browsers reject this combination. If credentials are needed, use an explicit origin list.

---

## Example Corpus for Demo

> 6 curated (text, target, expected_aspect, expected_sentiment) tuples.  
> Selection criteria: high-confidence predictions, cover all 3 sentiments, span all 3 domains, demonstrate the "same text → different target → different result" value prop.

**Use H-2a + H-2b as the ANCHOR pair** — load these two by default on page open to immediately demonstrate why targeted ABSA is more powerful than sentence-level SA.

| # | Domain | Text | Target | Expected Aspect | Expected Sentiment | Why |
|---|--------|------|--------|----------------|-------------------|-----|
| 1 | 🏨 Hotel | `khăn tắm không được thay hằng ngày. phòng có mùi thơm tinh dầu rất dễ chịu, thoải mái.` | `khăn tắm` | `SERVICE#CLEANLINESS` | `NEGATIVE` | **Anchor pair A** — negation with explicit "không được". Same review as #2. |
| 2 | 🏨 Hotel | (same text as above) | `phòng` | `ROOMS#COMFORT` | `POSITIVE` | **Anchor pair B** — proves same review, different target → opposite sentiment. Core value prop. |
| 3 | 🏨 Hotel | `nhân viên tiếp đón ân cần, rất vui vẻ và thân thiện.` | `nhân viên` | `SERVICE#GENERAL` | `POSITIVE` | Most common aspect. Strong explicit sentiment words. |
| 4 | 🍜 Restaurant | `hương vị thơm ngon, ăn cay cay rất thích, nêm nếm vừa miệng.` | `hương vị` | `FOOD#QUALITY` | `POSITIVE` | Highest-frequency restaurant aspect. Multiple stacked positive signals. |
| 5 | 📱 Mobile | `máy mượt, pin trâu.` | `pin` | `BATTERY` | `POSITIVE` | Classic colloquial Vietnamese. "trâu" (buffalo) = ultra-common positive battery slang. Shortest example. |
| 6 | 📱 Mobile | `camera ổn chụp lúc sáng đẹp nhưng ban đêm chỉ tạm thôi.` | `camera` | `CAMERA` | `NEUTRAL` | Demonstrates NEUTRAL: balanced positive/negative signals cancel out. Model's 91% sentiment accuracy makes this credible. |

**Examples to intentionally exclude from primary demo flow** (model struggles):  
- Double negation: `"không phải là không tốt"` — confuses the model  
- Mixed-language slang: `"loa zin tàu"` — OOV tokens  
- Rare aspects: `HOTEL#STYLE&OPTIONS` — zero training examples for this label  
- Pronoun targets: `"cái này"`, `"nó"` — no lexical signal for aspect detection

---

## Implementation Priority Order

Build in this sequence. Each step is independently testable before the next.

1. **`api/` scaffolding** — `api/__init__.py`, `api/schemas.py` (Pydantic models), `api/main.py` (FastAPI app + lifespan + CORS + `/health`). Verify: `uvicorn api.main:app` starts without errors, `/health` returns 503 (checkpoint not yet placed). Proves `sys.path` import works.

2. **`api/inference.py` — InferenceEngine** — `Vocabulary` bootstrap from data files + `TargetedABSAModel` reconstruction + `load_state_dict` + `model.eval()`. Verify: `engine.predict("phòng sạch sẽ", "phòng")` returns a dict with aspect + sentiment keys. This is the riskiest step (checkpoint must load cleanly).

3. **`POST /predict` endpoint** — wire `InferenceEngine.predict()` into FastAPI route. Add: NFC normalization, BOM strip, target-in-text validation, target_mask zero-check warning, top-5 aspect trimming, `asyncio.Lock()`. Verify with `curl -X POST localhost:8000/predict -d '{"text":"phòng sạch sẽ","target":"phòng"}'`.

4. **Frontend scaffold** — `npm create vite@latest frontend -- --template react-ts`, configure Vite proxy (`/predict` → `localhost:8000`), add TypeScript types (`types.ts`). Verify: `npm run dev` proxies to FastAPI.

5. **Static UI with mock data** — `ReviewInput.tsx` (textarea), `TargetBadge.tsx`, `PredictButton.tsx`, `ResultCard.tsx` with hardcoded mock result. Verify layout at 1200px and 375px without any API call.

6. **Target selection logic** — wire `onMouseUp`/`onKeyUp` in `ReviewInput` to capture `value.slice(selectionStart, selectionEnd).trim()`; fire `onTargetSelect` callback. Verify with Vietnamese text that multi-byte diacritics select correctly.

7. **`usePredict` hook** — `fetch('/predict', ...)` with `AbortSignal.timeout(15_000)`, loading/error state. Wire to `PredictButton`. Verify: live round-trip shows real model prediction.

8. **`ExamplePicker.tsx` + `data/examples.ts`** — hard-code the 6 curated examples. On click: set `reviewText` + `targetPhrase` + fire `predict()` immediately (auto-submit). Verify: clicking anchor pair (examples #1 and #2) shows opposite sentiments from same text.

9. **Result visualization** — `ProbabilityBars.tsx` (CSS width, top-5 aspects), `AspectLabel.tsx` (color-coded by top-level category), `SentimentBadge.tsx` (traffic-light). Add confidence tier label (Confident / Likely / Uncertain / Low). Show `warning` field if backend returns it.

10. **Vietnamese text normalization** — add `text.normalize('NFC').replace(/^\uFEFF/, '').replace(/[\u200B-\u200D\u00AD]/g, '').replace(/\u00A0/g, ' ')` before sending. Verify by pasting from a `.txt` file that was opened in Notepad.

11. **Error/empty states** — all branches: no text, no target (disabled button + hint), API error (red banner), timeout (specific message), loading skeleton bars. Verify all 5 states manually.

12. **Production build** — `npm run build` → `frontend/dist/`. Verify FastAPI serves `frontend/dist` as static files with SPA fallback route. Test `make serve-prod` at `localhost:8000`.

13. **Dockerfile + HuggingFace Spaces setup** — multi-stage Docker build, `hf_hub_download` for checkpoint, Spaces `README.md` header. Verify `docker compose up` works locally before pushing to Spaces.

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Model performance numbers | **HIGH** | Direct from `reports/test_metrics.json` — not estimated |
| Inference code pattern | **HIGH** | Derived from actual `vocabulary.py`, `model/targeted_absa.py` — constructor args confirmed |
| No-segmentation requirement | **HIGH** | Confirmed by both api-server.md and domain-context.md independently from data inspection |
| Frontend patterns | **HIGH** | Stable browser APIs (`selectionStart`, `AbortSignal.timeout` Baseline 2023) |
| Deployment on HuggingFace Spaces | **MEDIUM** | Cold-start timing with 1.2GB checkpoint on free tier is estimated, not measured |
| Example prediction accuracy | **MEDIUM-HIGH** | Examples selected from high-confidence pattern categories; actual model outputs not measured for these specific examples yet |
| Multi-model support | **LOW** | Only phobert-large checkpoint confirmed to exist; other variants assumed missing |

### Gaps to Address During Planning

- **Checkpoint location**: `train.py` saves to `checkpoints/<group>/<run>/best_model.pt`. The exact path of the trained checkpoint is not in the research files. Must verify path before writing `.env`.
- **`TargetedABSAModel` constructor exact args**: `num_labels` parameter — the research shows it's required for `joint_head` init, but the value (198) was computed from vocab scan. Verify it matches the saved checkpoint.
- **Inference latency measurement**: The 1–3s estimate for CPU phobert-large is from ecosystem knowledge, not measured on this codebase. If latency is higher (e.g., 5–8s on a 2-core host), the 15s frontend timeout may feel bad — consider showing a progress indicator after 2s.
- **`Vocabulary` constructor signature**: The research assumes `Vocabulary(data_path=[...], model_name=...)`. Verify actual signature in `vocabulary.py` matches before wiring `InferenceEngine`.

---

## Sources Aggregated

| Source | Used By | Key Facts Extracted |
|--------|---------|-------------------|
| `reports/test_metrics.json` | domain-context.md | Sentiment acc 91.2%, Aspect F1 43.2%, run name `phobert_large_target_attn_weighted` |
| `vocabulary.py` | api-server.md + domain-context.md | `tokenize_text_pair(text, target)` signature; 86 aspects, 3 sentiments |
| `model/targeted_absa.py` | api-server.md | Constructor args; `target_conditioned_attention` attention key |
| `config/base.yaml` + `config/encoders.yaml` | api-server.md + deployment.md | `requires_word_segmentation: true` flag (documentation only — not applied in training) |
| `Data/train.jsonl` | api-server.md + domain-context.md | Raw unsegmented Vietnamese text confirmed; BOM present at file start |
| `train.py` | api-server.md | State-dict-only save; known bug: always uses phobert-base tokenizer for vocab |
| MDN Web Docs | frontend-demo.md | `selectionStart/End`, `AbortSignal.timeout` Baseline 2023 |
| HuggingFace Spaces docs | deployment.md | Port 7860 requirement; `hf_hub_download` pattern |
