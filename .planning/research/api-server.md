# API Server Research: Targeted ABSA Inference Endpoint

**Project:** Targeted ABSA Demo Website  
**Researched:** 2026-06-27  
**Domain:** ML inference server (PyTorch + HuggingFace → HTTP API → demo frontend)  
**Confidence:** HIGH — based on direct codebase analysis + FastAPI/PyTorch ecosystem patterns

---

## 1. Recommended Approach

**Use FastAPI with a lifespan-managed singleton model.** Flask is viable but requires more boilerplate for request validation, async, and docs. FastAPI gives you Pydantic validation, auto-generated OpenAPI docs, async request handling, and startup/shutdown hooks — all needed here.

**Architecture in one sentence:** single FastAPI process, model loaded once at startup via `lifespan`, request handler calls `vocab.tokenize_text_pair()` + `model.forward()` inside `torch.no_grad()`, response serializes softmax probabilities with decoded label strings.

---

## 2. FastAPI vs Flask Decision

| Criterion | FastAPI | Flask |
|-----------|---------|-------|
| Request schema validation | Pydantic (`BaseModel`) — built-in | Manual or Flask-Marshmallow |
| Response schema | Auto-serialized from Pydantic | `jsonify()` manual dict |
| Startup hook (model load) | `lifespan` context manager | `before_first_request` / app factory |
| Async request handling | Native `async def` | Requires Quart or greenlets |
| Auto-generated API docs | `/docs` (Swagger) + `/redoc` | Requires Flask-RESTX or Flasgger |
| CORS | `fastapi.middleware.cors` | `flask-cors` |
| Static file serving | `StaticFiles` mount | Blueprint / `send_from_directory` |
| Cold start with heavy model | Same as Flask | Same as FastAPI |
| Community ML pattern | Dominant in 2024+ | Legacy, still common |

**Verdict: FastAPI.** The Pydantic validation alone prevents dozens of edge-case bugs (empty strings, wrong types, missing fields). The lifespan hook is cleaner than Flask's app factory pattern for a single-model server.

---

## 3. Model Loading: The Only Correct Pattern for This Codebase

The model checkpoint (`best_model.pt`) was saved as **state dict only** (`torch.save(self.model.state_dict(), ...)`  — `train.py:314`). This means you **cannot** `torch.load()` it directly into a model instance without first reconstructing the architecture.

### 3.1 The Vocabulary Bootstrap Problem

`Vocabulary.__init__` scans JSONL data files to build label maps. This is fine for training but awkward for inference. You have two options:

**Option A: Re-build Vocabulary from data files at startup** (simplest, no extra artifacts)
```python
# Requires Data/ directory to exist at the API server's working directory
vocab = Vocabulary(
    data_path=["Data/train.jsonl", "Data/dev.jsonl"],
    model_name="vinai/phobert-large"   # must match the checkpoint's encoder
)
```

**Option B: Serialize label maps to JSON and load at startup** (portable, no data dependency)
```python
# vocab_config.json — generated once from vocabulary, committed to repo
{
  "aspects": ["AMBIENCE#GENERAL", "BATTERY", "CAMERA", ...],   # 86 items
  "sentiments": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
  "labels": ["AMBIENCE#GENERAL#NEGATIVE", ...],                # 198 items
  "encoder_name": "vinai/phobert-large"
}
```
Then at startup, build `aspect2id`, `sentiment2id`, `id2aspect`, etc. from the JSON instead of scanning data files.

**Recommendation: Option B** for production demo. The data files are 4× the startup time cost and shouldn't be a required deployment artifact for inference. Generate `vocab_config.json` once as a build step. However, **Option A works perfectly** if data files ship with the demo.

### 3.2 Complete Startup Pattern

```python
# api/model_loader.py
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager
from fastapi import FastAPI

from vocabulary import Vocabulary
from model.targeted_absa import TargetedABSAModel

# Global singletons — populated by lifespan
MODEL = None
VOCAB = None
DEVICE = None

CHECKPOINT_PATH = "checkpoints/best_model.pt"  # adjust to actual path
ENCODER_NAME    = "vinai/phobert-large"         # must match checkpoint
DATA_FILES      = ["Data/train.jsonl", "Data/dev.jsonl"]  # for vocab bootstrap
MAX_LENGTH      = 128

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, VOCAB, DEVICE

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Build vocabulary (label maps + tokenizer)
    VOCAB = Vocabulary(data_path=DATA_FILES, model_name=ENCODER_NAME)

    # 2. Reconstruct model architecture (MUST match training config exactly)
    MODEL = TargetedABSAModel(
        encoder_name=ENCODER_NAME,
        num_aspects=VOCAB.num_aspects,           # 86
        num_sentiments=VOCAB.num_sentiments,     # 3
        num_labels=VOCAB.num_labels,             # 198 — required for joint_head init
        label_structure_key="multitask_aspect_sentiment",
        attention_key="target_conditioned_attention",
        dropout=0.1,                             # must match training dropout
        classifier_hidden_size=256,
    )

    # 3. Load state dict — weights_only=True is safer (PyTorch ≥ 2.0)
    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    MODEL.load_state_dict(state_dict)
    MODEL.to(DEVICE)
    MODEL.eval()  # CRITICAL: disables dropout and batch norm

    print(f"Model loaded on {DEVICE}. Aspects: {VOCAB.num_aspects}, Sentiments: {VOCAB.num_sentiments}")
    yield
    # Cleanup (optional for demo)
    del MODEL, VOCAB

app = FastAPI(title="ABSA Inference API", lifespan=lifespan)
```

**Critical detail:** `MODEL.eval()` must be called. Without it, dropout layers remain active during inference, producing different predictions on every call.

---

## 4. Inference Handler: Request → Response

### 4.1 Pydantic Schemas

```python
# api/schemas.py
from pydantic import BaseModel, Field
from typing import Dict

class InferenceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="Vietnamese review text")
    target: str = Field(..., min_length=1, max_length=200,
                        description="Target phrase to analyze (substring of text)")

class InferenceResponse(BaseModel):
    aspect: str
    sentiment: str
    aspect_probabilities: Dict[str, float]
    sentiment_probabilities: Dict[str, float]
    # Convenience composite score
    confidence: float  # max softmax probability for sentiment
```

### 4.2 Inference Function

```python
# api/inference.py
import torch
import torch.nn.functional as F

def run_inference(text: str, target: str) -> dict:
    """Thread-safe inference. Call inside torch.no_grad()."""
    # Tokenize using the same method as training
    encoded = VOCAB.tokenize_text_pair(text, target, max_length=MAX_LENGTH)

    # Move to device and add batch dimension
    input_ids      = encoded["input_ids"].unsqueeze(0).to(DEVICE)
    attention_mask = encoded["attention_mask"].unsqueeze(0).to(DEVICE)
    target_mask    = encoded["target_mask"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=target_mask,
        )

    # Decode aspect
    aspect_probs = F.softmax(outputs["aspect_logits"], dim=-1).squeeze(0)
    aspect_id    = int(torch.argmax(aspect_probs).item())
    aspect_label = VOCAB.id2aspect[aspect_id]
    aspect_prob_dict = {
        VOCAB.id2aspect[i]: round(float(p), 4)
        for i, p in enumerate(aspect_probs)
    }

    # Decode sentiment
    sentiment_probs = F.softmax(outputs["sentiment_logits"], dim=-1).squeeze(0)
    sentiment_id    = int(torch.argmax(sentiment_probs).item())
    sentiment_label = VOCAB.id2sentiment[sentiment_id]
    sentiment_prob_dict = {
        VOCAB.id2sentiment[i]: round(float(p), 4)
        for i, p in enumerate(sentiment_probs)
    }

    return {
        "aspect": aspect_label,
        "sentiment": sentiment_label,
        "aspect_probabilities": aspect_prob_dict,
        "sentiment_probabilities": sentiment_prob_dict,
        "confidence": round(float(sentiment_probs.max().item()), 4),
    }
```

### 4.3 FastAPI Route with Concurrency Lock

```python
# api/routes.py
import asyncio
from fastapi import HTTPException
from api.schemas import InferenceRequest, InferenceResponse
from api.inference import run_inference

_inference_lock = asyncio.Lock()  # serialize concurrent requests on single model

@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    # Validate target is a substring (prevents confusing off-distribution inputs)
    if req.target not in req.text:
        raise HTTPException(
            status_code=422,
            detail="target must be a substring of text"
        )

    async with _inference_lock:
        try:
            result = run_inference(req.text, req.target)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail="GPU OOM — retry with shorter text")
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"Label encoding error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return result
```

**Why the lock?** PyTorch CPU inference is not thread-safe for the same model when `model.eval()` + `no_grad()` is used *without* explicit locking. With a single model instance on CPU, serializing requests avoids race conditions on internal buffers. On GPU, `model.eval()` + `no_grad()` is safe for concurrent calls, but the lock costs nothing in a demo context.

---

## 5. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Demo: allow all. Production: ["https://yourdomain.com"]
    allow_credentials=False,      # False when allow_origins="*"
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)
```

**Gotcha:** If you ever switch to `allow_credentials=True`, you must replace `allow_origins=["*"]` with an explicit list. Browsers reject the combination of wildcard origin + credentials.

---

## 6. ⚠️ PhoBERT Word Segmentation — Critical Gotcha

The `encoders.yaml` explicitly marks `requires_word_segmentation: true` for `phobert_base` and `phobert_large`. However, **the actual training data is raw unsegmented Vietnamese text** (confirmed from `Data/train.jsonl` inspection: `"em mua ip7plus được 3 tháng nhưng dung lượng pin chỉ còn 98%"` — spaces are character-level, not word-segmented).

**This means the model was trained on raw Vietnamese input** processed by PhoBERT's BPE tokenizer directly. The tokenizer handles subword splitting without pre-segmentation. **Do NOT add Vietnamese word segmentation at the API layer unless you also retrain with the same segmentation pipeline** — doing so would create a train/inference distribution mismatch.

However, if you decide to add word segmentation for quality improvement in a retrain, the correct library is:

```python
# Install: pip install underthesea
from underthesea import word_tokenize

def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text for PhoBERT. Only use if model was trained with this."""
    return word_tokenize(text, format="text")  # returns "em mua ip7_plus được 3 tháng..."

# underthesea word_tokenize is CPU-only, ~50ms per sentence
```

**Bottom line for this codebase:** Pass text directly to `vocab.tokenize_text_pair()` without segmentation. Be consistent with training.

---

## 7. Serving Static Frontend from the Same FastAPI App

For a demo context, serving the frontend from the same origin as the API is simplest (no CORS needed, single process deployment).

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Mount after all API routes to avoid route shadowing
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/dist/index.html")

# Catch-all for SPA routing (if using React Router / client-side routing)
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    index = Path("frontend/dist/index.html")
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(status_code=404)
```

**Ordering matters:** Mount `/static` AFTER all `@app.post`/`@app.get` API routes. FastAPI evaluates routes top-down; a wildcard catch-all registered first will intercept API calls.

**Alternative: Separate origins.** Run `vite preview` on port 5173 and FastAPI on port 8000. This is simpler during development but requires CORS. Fine for demo deployments on the same machine.

---

## 8. Error Handling Matrix

| Scenario | Cause | HTTP Code | Handling |
|----------|-------|-----------|----------|
| `CHECKPOINT_PATH` doesn't exist | Missing `best_model.pt` | 503 on startup | Check in `lifespan`; print clear error and exit |
| `target not in text` | User input error | 422 | Pydantic/route validation |
| Empty string input | Tokenizer edge case | 422 | Pydantic `min_length=1` |
| Text > max_length tokens | HF truncates silently | — | No error; truncation is expected behavior |
| `KeyError` in vocab encode | Unseen label during decode | 422 | Try/except in inference handler |
| `torch.cuda.OutOfMemoryError` | Very long text on GPU | 503 | Catch + `cuda.empty_cache()` + retry guidance |
| Model not yet warmed up | First request cold start | — | No special handling needed; just log timing |
| `token_type_ids` missing on XLM-R | Encoder mismatch | — | `TargetedABSAModel.forward` skips `token_type_ids` if `None` — safe |
| Concurrent requests | Single model, multiple threads | — | `asyncio.Lock()` serializes |

### Startup Validation

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path(CHECKPOINT_PATH).exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run training first or copy best_model.pt to the expected path."
        )
    # ... rest of loading
```

---

## 9. Recommended Libraries and Versions

```toml
# requirements.txt (or pyproject.toml [tool.poetry.dependencies])
fastapi>=0.111.0          # lifespan support stable since 0.93
uvicorn[standard]>=0.29.0 # ASGI server; [standard] includes watchfiles for --reload
pydantic>=2.7.0           # v2 is default in fastapi>=0.100
torch>=2.1.0              # weights_only=True stable; torch.compile available
transformers>=4.40.0      # supports phobert-large without deprecation warnings
python-multipart>=0.0.9   # needed for form data (optional for this API)
```

**Dev only:**
```
httpx>=0.27.0             # for FastAPI TestClient in tests
pytest>=8.0
```

**If adding word segmentation later:**
```
underthesea>=6.8.4        # Vietnamese NLP toolkit; CPU-only
```

---

## 10. Minimal Working Server Layout

```
targeted_ABSA/
├── api/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + lifespan + CORS + static files
│   ├── schemas.py       # InferenceRequest, InferenceResponse
│   ├── inference.py     # run_inference() function
│   └── routes.py        # POST /predict, GET /health
├── model/               # existing model code (unchanged)
├── vocabulary.py        # existing (unchanged)
├── Data/                # needed at runtime for vocab bootstrap
│   ├── train.jsonl
│   └── dev.jsonl
├── checkpoints/
│   └── best_model.pt    # required; not committed to git
└── frontend/
    └── dist/            # built frontend static files
```

### Startup Command

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Production (no reload, multiple workers — BUT only 1 worker for single model):
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Important: `--workers 1` only.** Multiple workers would each load the full PhoBERT model into RAM/VRAM (~1.3GB for phobert-large). For a demo, a single worker is correct. If you need throughput, use a request queue (e.g., Celery) instead of multiple workers.

---

## 11. Health Check Endpoint

Always add a health check. Load balancers, Docker healthchecks, and Render/Railway deployments need it.

```python
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "encoder": ENCODER_NAME,
        "num_aspects": VOCAB.num_aspects if VOCAB else None,
        "num_sentiments": VOCAB.num_sentiments if VOCAB else None,
    }
```

---

## 12. Pitfalls Specific to This Project

### ❌ Pitfall 1: Wrong dropout value at inference load
The `TargetedABSAModel` constructor accepts `dropout=0.2` as default but training used `dropout=0.1` (from `train.py:389`). **You must pass `dropout=0.1` when reconstructing the model** — otherwise the `MLPHead` and `AttentionPooling` layers will have different parameter shapes and `load_state_dict` will raise a shape mismatch error.

Actually: dropout value doesn't affect state dict shape (no parameters to mismatch). **But** it matters for inference: if `dropout > 0` and `model.eval()` is NOT called, predictions will be stochastic. Always call `model.eval()`.

### ❌ Pitfall 2: Tokenizer/Encoder Mismatch (Known Bug in Codebase)
`train.py:334` always creates `Vocabulary` with `"vinai/phobert-base"` tokenizer even when training with `xlm-roberta-large`. For inference, **use the tokenizer that matches the checkpoint encoder**. The checkpoint path often encodes the encoder name — use it to infer which tokenizer to load.

### ❌ Pitfall 3: target_mask is required for target_conditioned_attention
The best model uses `attention_key="target_conditioned_attention"`. Its `build_representation()` calls `self.masked_mean_pooling(hidden_states, target_mask)`. If `target_mask` is all zeros (target not found in tokenized sequence), it falls back to `cls_vec`. This is handled in `_build_target_mask`'s reverse-scan logic — but if the target string appears nowhere in the tokenized input (due to truncation or segmentation differences), prediction quality degrades silently.

**Mitigation:** Validate that `target_mask.sum() > 0` after tokenization. If all zeros, return a warning in the response.

```python
# After tokenization
if encoded["target_mask"].sum() == 0:
    # Target was truncated or not found — still predict but warn
    response["warning"] = "Target phrase not located in tokenized input; prediction may be degraded"
```

### ❌ Pitfall 4: State dict key prefix mismatch (PyTorch DataParallel)
If the checkpoint was saved from a `DataParallel`-wrapped model, all keys will have `module.` prefix (e.g., `module.encoder.embeddings.weight`). The base `TargetedABSAModel` expects unprefixed keys.

```python
# Safe load that strips DataParallel prefix if present
state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
if any(k.startswith("module.") for k in state_dict):
    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
MODEL.load_state_dict(state_dict)
```

### ❌ Pitfall 5: Blocking inference in async route without lock
`run_inference()` is CPU-bound. Running it directly in an `async def` route blocks the event loop for ~1s (PhoBERT-large on CPU). The `asyncio.Lock()` approach serializes requests but doesn't unblock the event loop.

For a proper async solution: `asyncio.get_event_loop().run_in_executor(thread_pool, run_inference, text, target)`. For a demo with low traffic, the lock + direct call is acceptable.

### ❌ Pitfall 6: Vocabulary requires data files at inference time (Option A)
If you deploy the API without `Data/train.jsonl` and `Data/dev.jsonl`, startup will fail. **Either ship data files with the deployment OR implement Option B** (serialized vocab JSON). The data files are ~2MB total, so shipping them is fine for a demo.

### ❌ Pitfall 7: 86 aspect classes in response
The `aspect_probabilities` dict will contain **86 entries**. For the frontend, rendering all 86 is impractical. Consider returning only top-K:

```python
import heapq
top5_aspects = dict(
    heapq.nlargest(5, aspect_prob_dict.items(), key=lambda x: x[1])
)
```

Or always return the full dict and let the frontend filter.

---

## 13. Complete File Reference (copy-pasteable)

### `api/main.py`

```python
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Optional
import torch.nn.functional as F
import asyncio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # project root on path

from vocabulary import Vocabulary
from model.targeted_absa import TargetedABSAModel

# ─── Configuration ────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
ENCODER_NAME    = "vinai/phobert-large"
DATA_FILES      = ["Data/train.jsonl", "Data/dev.jsonl"]
MAX_LENGTH      = 128
ATTENTION_KEY   = "target_conditioned_attention"
LABEL_STRUCTURE = "multitask_aspect_sentiment"

# ─── Globals ──────────────────────────────────────────────────────────────────
MODEL: Optional[TargetedABSAModel] = None
VOCAB: Optional[Vocabulary]        = None
DEVICE: Optional[torch.device]     = None
_lock = asyncio.Lock()

# ─── Schemas ──────────────────────────────────────────────────────────────────
class InferenceRequest(BaseModel):
    text:   str = Field(..., min_length=1, max_length=2000)
    target: str = Field(..., min_length=1, max_length=200)

class InferenceResponse(BaseModel):
    aspect:                  str
    sentiment:               str
    aspect_probabilities:    Dict[str, float]
    sentiment_probabilities: Dict[str, float]
    confidence:              float
    warning:                 Optional[str] = None

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, VOCAB, DEVICE
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB  = Vocabulary(data_path=DATA_FILES, model_name=ENCODER_NAME)
    MODEL  = TargetedABSAModel(
        encoder_name         = ENCODER_NAME,
        num_aspects          = VOCAB.num_aspects,
        num_sentiments       = VOCAB.num_sentiments,
        num_labels           = VOCAB.num_labels,
        label_structure_key  = LABEL_STRUCTURE,
        attention_key        = ATTENTION_KEY,
        dropout              = 0.1,
        classifier_hidden_size = 256,
    )
    sd = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    MODEL.load_state_dict(sd)
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"[ABSA] ready on {DEVICE} | aspects={VOCAB.num_aspects} sentiments={VOCAB.num_sentiments}")
    yield
    del MODEL, VOCAB

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Targeted ABSA API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "device": str(DEVICE), "encoder": ENCODER_NAME}

@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest):
    if req.target not in req.text:
        raise HTTPException(422, "target must be a substring of text")

    async with _lock:
        try:
            encoded = VOCAB.tokenize_text_pair(req.text, req.target, MAX_LENGTH)
            input_ids      = encoded["input_ids"].unsqueeze(0).to(DEVICE)
            attention_mask = encoded["attention_mask"].unsqueeze(0).to(DEVICE)
            target_mask    = encoded["target_mask"].unsqueeze(0).to(DEVICE)

            warning = None
            if encoded["target_mask"].sum() == 0:
                warning = "Target phrase not found in tokenized input after truncation; prediction may be less accurate"

            with torch.no_grad():
                out = MODEL(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    target_mask=target_mask,
                )

            a_probs = F.softmax(out["aspect_logits"],    dim=-1).squeeze(0)
            s_probs = F.softmax(out["sentiment_logits"], dim=-1).squeeze(0)

            return InferenceResponse(
                aspect    = VOCAB.id2aspect[int(a_probs.argmax())],
                sentiment = VOCAB.id2sentiment[int(s_probs.argmax())],
                aspect_probabilities    = {VOCAB.id2aspect[i]: round(float(p), 4) for i, p in enumerate(a_probs)},
                sentiment_probabilities = {VOCAB.id2sentiment[i]: round(float(p), 4) for i, p in enumerate(s_probs)},
                confidence = round(float(s_probs.max()), 4),
                warning    = warning,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(503, "GPU out of memory — try shorter text")
        except Exception as e:
            raise HTTPException(500, str(e))

# ─── Static frontend (mount last) ─────────────────────────────────────────────
_frontend = Path("frontend/dist")
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(str(_frontend / "index.html"))
```

---

## 14. Summary Recommendations

| Topic | Recommendation |
|-------|---------------|
| Framework | **FastAPI** with `uvicorn[standard]` |
| Model loading | Lifespan context manager; state dict + explicit `TargetedABSAModel` reconstruction |
| Vocab bootstrap | Option A (data files) for simplicity; Option B (JSON) for portability |
| Concurrency | `asyncio.Lock()` — single inference at a time; sufficient for demo |
| CORS | `allow_origins=["*"]` for demo |
| PhoBERT segmentation | **Do NOT add** — model trained on raw text; segmenting at inference creates distribution shift |
| target_mask | Validate post-tokenization; include `warning` field in response |
| Static files | Mount last in FastAPI; catch-all SPA fallback route |
| Workers | **1 only** — single model, no VRAM waste from forking |
| Response shape | Return all 86 aspect probs in dict; let frontend trim to top-K |

---

*Analysis based on direct inspection of: `model/targeted_absa.py`, `vocabulary.py`, `dataloader.py`, `train.py`, `config/encoders.yaml`, `config/label_structures.yaml`, `Data/train.jsonl` (3 samples), `reports/test_metrics.json`, `.planning/codebase/ARCHITECTURE.md`*
