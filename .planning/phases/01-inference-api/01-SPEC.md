# Phase 1: Inference API — Specification

**Created:** 2026-06-27
**Ambiguity score:** 0.13 (gate: ≤ 0.20) ✓
**Requirements:** 8 locked

## Goal

A FastAPI server starts up, loads `TargetedABSAModel` + `Vocabulary` from existing checkpoint and data files, and serves three REST endpoints (`/predict`, `/health`, `/models`) so the React frontend can obtain ABSA predictions for Vietnamese (text, target) pairs.

## Background

**Current state:** No `api/` directory exists. No `requirements.txt`. The ML model (`model/targeted_absa.py`) and `Vocabulary` (`vocabulary.py`) are implemented and used by `train.py`, but solely for training. The checkpoint saved by `train.py` (line 314) is a raw state-dict written with `torch.save(model.state_dict(), path)` — there is no wrapping dict with a `"model_state_dict"` key.

**Gap:** Zero inference infrastructure. The frontend cannot call anything today.

**Trigger:** The user wants a demo website; the demo requires a running inference API before any frontend code can work end-to-end.

## Requirements

1. **POST /predict**: Returns ABSA prediction for (text, target) pair.
   - Current: Endpoint does not exist.
   - Target: `POST /predict` accepts `{"text": str, "target": str}`, returns `{"aspect": str, "sentiment": str, "aspect_probs": {label: float, ...}, "sentiment_probs": {label: float, ...}, "latency_ms": int}` with top-5 aspect probabilities trimmed.
   - Acceptance: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"Phòng khách sạn rất sạch sẽ","target":"Phòng"}' ` returns HTTP 200 with `aspect` string, `sentiment` in `["POSITIVE","NEGATIVE","NEUTRAL"]`, `aspect_probs` dict with 5 entries, `latency_ms` integer.

2. **GET /health**: Exposes server and model-load status.
   - Current: Endpoint does not exist.
   - Target: `GET /health` returns `{"status": "ok", "model_loaded": bool, "device": str}`. During startup (before model ready), `model_loaded` is `false` and overall HTTP status is 503.
   - Acceptance: `curl http://localhost:8000/health` returns HTTP 200 with `model_loaded: true` after startup completes; returns HTTP 503 before model loads.

3. **GET /models**: Exposes active model configuration.
   - Current: Endpoint does not exist.
   - Target: `GET /models` returns `{"active": str, "encoder": str, "attention_key": str, "label_structure": str}` reflecting the loaded config.
   - Acceptance: `curl http://localhost:8000/models` returns HTTP 200 with non-null `encoder` and `attention_key` fields.

4. **CORS configuration**: Frontend can call the API without browser rejection.
   - Current: No CORS headers. Any cross-origin request would be blocked.
   - Target: `CORSMiddleware` configured with explicit origins `["http://localhost:5173", "http://127.0.0.1:5173"]` for dev. `allow_credentials=False` (never `True` with wildcard origins — rejected by browsers).
   - Acceptance: `curl -H "Origin: http://localhost:5173" -v http://localhost:8000/health 2>&1 | grep "access-control-allow-origin"` shows the header present with the correct value.

5. **Input validation + error responses**: Prevents silent degradation.
   - Current: No validation layer exists.
   - Target: (a) Return HTTP 422 if `text` or `target` is empty/whitespace. (b) Return HTTP 422 if `target` is not an exact Python `str` substring of `text` (after NFC normalization on both). (c) Return HTTP 503 with `{"detail": "Model not ready"}` if model hasn't loaded. (d) Return HTTP 500 with `{"detail": str}` for unexpected inference errors.
   - Acceptance: `POST /predict` with `{"text":"hello","target":"xyz"}` returns HTTP 422; POST with empty `target` returns HTTP 422; POST before model loads returns HTTP 503.

6. **Startup loading (model + Vocabulary)**: Both components load once at startup.
   - Current: Loading logic exists in `train.py` but not as a reusable service.
   - Target: On startup, `Vocabulary` is constructed by scanning JSONL files from `DATA_DIR` env var; `TargetedABSAModel` is instantiated with config from `config/base.yaml` and weights loaded via `model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))`. A single `asyncio.Lock()` serializes concurrent inference requests. Multiple requests during model load receive HTTP 503.
   - Acceptance: Server startup log shows "Model loaded" within 60s on CPU. Second concurrent request during slow inference is queued (not rejected). Server starts with `uvicorn api.main:app --workers 1`.

7. **Unicode normalization**: Prevents tokenization mismatch from copy-paste input.
   - Current: Not implemented anywhere in the inference path.
   - Target: Both `text` and `target` are processed with `unicodedata.normalize('NFC', s).replace('\ufeff', '')` before substring validation and before tokenization.
   - Acceptance: A request with NFD-encoded Vietnamese diacritics (e.g., iOS copy-paste) produces the same prediction as the NFC equivalent.

8. **requirements.txt**: Reproducible inference environment.
   - Current: No `requirements.txt` exists in the repository.
   - Target: `requirements.txt` at project root covers all inference server dependencies: `fastapi`, `uvicorn[standard]`, `torch`, `transformers`, `pydantic`, `pyyaml`, `scikit-learn`, `tqdm` with pinned major.minor versions.
   - Acceptance: `pip install -r requirements.txt && uvicorn api.main:app` succeeds in a fresh venv.

## Boundaries

**In scope:**
- `api/` directory: `main.py` (FastAPI app), `inference.py` (`InferenceEngine` class), `schemas.py` (Pydantic models)
- `requirements.txt` at project root
- Environment variable support: `CHECKPOINT_PATH`, `DATA_DIR`, `DEVICE` (`cpu`/`cuda`)
- CORS configuration for `localhost:5173`
- Three endpoints: `/predict`, `/health`, `/models`
- Single model loaded at startup (no hot-swap, no multi-model)

**Out of scope:**
- React frontend — Phase 2
- Training or fine-tuning endpoints — training stays in `train.py`
- Authentication/API keys — public demo, no auth
- Batch predict endpoint — single annotation interactive mode only
- Docker/containerization — Phase 3+
- Word segmentation (VnCoreNLP/underthesea) — NOT added; training data is raw Vietnamese, segmentation would degrade predictions
- GPU setup or CUDA configuration — optional; falls back to CPU

## Constraints

- **Single worker only**: `--workers 1` mandatory — PhoBERT-large is ~1.3GB; multiple workers each load a separate copy
- **asyncio.Lock()**: CPU-bound inference blocks the event loop; a single lock serializes requests without crashing
- **Checkpoint format**: `train.py:314` saves raw `model.state_dict()` (not a dict with `model_state_dict` key) — load with `model.load_state_dict(torch.load(path, weights_only=True))`
- **Vocabulary cold start**: `Vocabulary.__init__` scans all JSONL files; `DATA_DIR` must point to `Data/` directory at startup
- **Top-5 trimming**: Return only the top-5 aspect probabilities (86 total classes) to avoid broken frontend rendering
- **Response language**: All error messages in English (for frontend display)

## Acceptance Criteria

- [ ] `POST /predict` with valid `{"text":"Phòng khách sạn rất sạch sẽ","target":"Phòng"}` returns HTTP 200 with `aspect` string, `sentiment` in `["POSITIVE","NEGATIVE","NEUTRAL"]`, `aspect_probs` with 5 entries, `latency_ms` integer
- [ ] `GET /health` returns HTTP 200 with `model_loaded: true` after startup; HTTP 503 before model loads
- [ ] `GET /models` returns HTTP 200 with non-null `encoder` and `attention_key` fields
- [ ] `curl -H "Origin: http://localhost:5173"` to any endpoint shows `access-control-allow-origin: http://localhost:5173` header
- [ ] `POST /predict` with `target` not substring of `text` returns HTTP 422
- [ ] `POST /predict` with empty `text` or empty `target` returns HTTP 422
- [ ] `POST /predict` before model loads returns HTTP 503 with `{"detail": "Model not ready"}`
- [ ] Two concurrent `POST /predict` requests complete without error (second queued behind asyncio.Lock)
- [ ] NFD-encoded Vietnamese diacritics in request produce same prediction as NFC-equivalent
- [ ] `requirements.txt` exists at project root; `pip install -r requirements.txt` succeeds
- [ ] `uvicorn api.main:app --workers 1` starts successfully with `CHECKPOINT_PATH` and `DATA_DIR` set

## Edge Coverage

**Coverage:** 10/14 applicable edges resolved · 4 dismissed (SETUP-01 edges: not applicable to file content)

| Category | Requirement | Status | Resolution / Reason |
|----------|-------------|--------|---------------------|
| empty | API-01 | ✅ covered | AC: empty text/target → HTTP 422 (see AC-06) |
| encoding | API-01 | ✅ covered | Python str comparison after NFC normalization; see API-07 |
| unclassified | API-02 | ⛔ dismissed | Health endpoint is read-only status; no state to probe |
| unclassified | API-03 | ⛔ dismissed | Models endpoint returns static config; no state to probe |
| concurrency | API-04 | ✅ covered | CORS is stateless headers; inference concurrency handled by asyncio.Lock (see AC-08) |
| empty | API-05 | ✅ covered | AC: empty/whitespace target → HTTP 422 before inference (see AC-06) |
| encoding | API-05 | ✅ covered | Substring check uses Python str after NFC normalization on both sides |
| concurrency | API-06 | ✅ covered | 503 during startup; asyncio.Lock serializes concurrent requests (see AC-08) |
| empty | API-07 | ✅ covered | Empty text after normalization → validation → 422 (see AC-06) |
| encoding | API-07 | ✅ covered | `unicodedata.normalize('NFC', text).replace('\ufeff','')` before all processing |
| adjacency | SETUP-01 | ⛔ dismissed | pip version pinning; merging semantics not applicable |
| empty | SETUP-01 | ⛔ dismissed | requirements.txt will have ≥8 entries; never empty |
| ordering | SETUP-01 | ⛔ dismissed | Alphabetical convention; no functional ordering requirement |
| concurrency | SETUP-01 | ⛔ dismissed | File read; not a concurrency concern |

---
*Spec created: 2026-06-27 (auto mode — ambiguity 0.13, gate passed on initial assessment)*
