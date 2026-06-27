# Deployment & Project Structure Research

**Project:** Targeted ABSA Demo Website  
**Researched:** 2026-06-27  
**Scope:** Monorepo layout, FastAPI inference server, React+Vite frontend, local dev workflow, deployment options  
**Overall confidence:** HIGH (based on direct codebase analysis + standard ecosystem patterns)

---

## 1. Recommended Project Structure

The guiding constraint: `vocabulary.py`, `dataloader.py`, and `model/` are at the project root with no packaging. The API must import them without a refactor. The solution is a top-level `api/` module that inserts the project root into `sys.path` at startup.

```
targeted_ABSA/
│
├── api/                            # ← NEW: FastAPI inference server
│   ├── __init__.py                 # empty, marks api/ as a package
│   ├── main.py                     # FastAPI app, CORS, lifespan, /health + /predict
│   ├── inference.py                # InferenceEngine class (loads vocab + model)
│   └── schemas.py                  # Pydantic request/response models
│
├── frontend/                       # ← NEW: React + Vite SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── ReviewInput.jsx     # Textarea for Vietnamese review text
│   │   │   ├── TargetSelector.jsx  # Text highlight → sets target phrase
│   │   │   ├── PredictionResult.jsx# Shows aspect + sentiment + probabilities
│   │   │   └── ExampleCards.jsx    # Pre-loaded samples per domain
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── public/
│   │   └── favicon.ico
│   ├── index.html
│   ├── vite.config.js              # proxy /api → localhost:8000
│   └── package.json
│
├── checkpoints/                    # ← NEW directory (gitignored for *.pt)
│   └── .gitkeep                    # keeps dir in git without committing model files
│
├── model/                          # EXISTING — untouched
│   ├── attn_pooling.py
│   ├── conditional_attn.py
│   ├── gated_fusion.py
│   ├── mlp_head.py
│   └── targeted_absa.py
│
├── config/                         # EXISTING — untouched
│   ├── base.yaml
│   ├── encoders.yaml
│   ├── label_structures.yaml
│   ├── attention.yaml
│   ├── losses.yaml
│   └── experiments.yaml
│
├── Data/                           # EXISTING — untouched
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
│
├── vocabulary.py                   # EXISTING — imported by api/inference.py
├── dataloader.py                   # EXISTING — imported for training only
├── train.py                        # EXISTING — training only, untouched
├── evaluate.py                     # EXISTING (empty) — implement separately
│
├── requirements.txt                # ← NEW: inference server + API deps (CPU)
├── requirements-train.txt          # ← NEW: full training deps (GPU torch)
├── Makefile                        # ← NEW: `make dev`, `make api`, `make install`
├── .env.example                    # ← NEW: CHECKPOINT_PATH, CORS_ORIGINS, etc.
├── .gitignore                      # UPDATE: add checkpoints/*.pt, frontend/node_modules
└── README.md                       # UPDATE: add Demo Setup section
```

### Why this layout

- **`api/` and `frontend/` at root level** — not inside a `src/` wrapper. Keeps `python -m uvicorn api.main:app` runnable from root without any install step.
- **`vocabulary.py` stays at root** — no refactor needed. `api/inference.py` does `sys.path.insert(0, str(Path(__file__).parent.parent))` once at import time.
- **`model/` has no `__init__.py`** (confirmed) — Python 3.3+ namespace packages handle this; relative imports inside `model/*.py` work as long as the project root is in `sys.path`.
- **`checkpoints/` at root** — matches where `train.py` writes files (`checkpoints/<group>/<run>/best_model.pt`). The API config points to the specific trained checkpoint path.

---

## 2. Key File Contents

### `api/__init__.py`
```python
# empty
```

### `api/schemas.py`
```python
from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str    # Full Vietnamese review sentence
    target: str  # Target phrase (substring of text)

class AspectSentimentResult(BaseModel):
    aspect: str
    sentiment: str
    aspect_probabilities: dict[str, float]
    sentiment_probabilities: dict[str, float]
    model_ready: bool = True

class HealthResponse(BaseModel):
    ready: bool
    encoder: str
    num_aspects: int
    num_sentiments: int
```

### `api/inference.py`
```python
import sys
from pathlib import Path

# Allow importing vocabulary.py, model/, config/ from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from vocabulary import Vocabulary
from model.targeted_absa import TargetedABSAModel

ENCODER_MAP = {
    "phobert_base":       "vinai/phobert-base",
    "phobert_large":      "vinai/phobert-large",
    "xlm_roberta_base":   "xlm-roberta-base",
    "xlm_roberta_large":  "xlm-roberta-large",
}

class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "config/base.yaml",
        data_paths: list[str] = None,
    ):
        self.ready = False
        self.encoder_key = "unknown"
        self.device = torch.device("cpu")

        if data_paths is None:
            data_paths = ["Data/train.jsonl", "Data/dev.jsonl"]

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.encoder_key = cfg["model"]["encoder_key"]
        encoder_name = ENCODER_MAP[self.encoder_key]

        # Vocabulary must use the SAME tokenizer as the encoder that was trained
        # (train.py hardcodes phobert-base tokenizer — match that for existing checkpoints)
        self.vocab = Vocabulary(data_paths, encoder_name)

        if not Path(checkpoint_path).exists():
            print(f"[InferenceEngine] WARNING: checkpoint not found at {checkpoint_path!r}. "
                  "API will start but /predict returns 503 until checkpoint is placed.")
            return

        self.model = TargetedABSAModel(
            encoder_name=encoder_name,
            num_aspects=self.vocab.num_aspects,
            num_sentiments=self.vocab.num_sentiments,
            label_structure_key=cfg["model"]["label_structure_key"],
            attention_key=cfg["model"]["attention_key"],
            dropout=0.0,  # disable dropout at inference
            classifier_hidden_size=cfg["model"]["classifier_hidden_size"],
        )
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.ready = True
        print(f"[InferenceEngine] Loaded checkpoint: {checkpoint_path}")

    @torch.no_grad()
    def predict(self, text: str, target: str) -> dict:
        if not self.ready:
            raise RuntimeError("Model checkpoint not loaded.")

        encoded = self.vocab.tokenize_text_pair(text, target, max_length=256)

        input_ids      = encoded["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = encoded["attention_mask"].unsqueeze(0).to(self.device)
        target_mask    = encoded.get("target_mask")
        if target_mask is not None:
            target_mask = target_mask.unsqueeze(0).to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_mask=target_mask,
        )

        asp_probs  = torch.softmax(outputs["aspect_logits"][0], dim=-1)
        sent_probs = torch.softmax(outputs["sentiment_logits"][0], dim=-1)

        return {
            "aspect":    self.vocab.id2aspect[asp_probs.argmax().item()],
            "sentiment": self.vocab.id2sentiment[sent_probs.argmax().item()],
            "aspect_probabilities": {
                self.vocab.id2aspect[i]: round(p.item(), 4)
                for i, p in enumerate(asp_probs)
            },
            "sentiment_probabilities": {
                self.vocab.id2sentiment[i]: round(p.item(), 4)
                for i, p in enumerate(sent_probs)
            },
        }
```

### `api/main.py`
```python
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from api.schemas import PredictRequest, AspectSentimentResult, HealthResponse
from api.inference import InferenceEngine

engine: InferenceEngine | None = None

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best_model.pt")
CONFIG_PATH     = os.getenv("CONFIG_PATH", "config/base.yaml")
CORS_ORIGINS    = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:4173").split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = InferenceEngine(
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
    )
    yield

app = FastAPI(title="Targeted ABSA Demo", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    if engine is None:
        return HealthResponse(ready=False, encoder="unknown", num_aspects=0, num_sentiments=0)
    return HealthResponse(
        ready=engine.ready,
        encoder=engine.encoder_key,
        num_aspects=engine.vocab.num_aspects if engine.vocab else 0,
        num_sentiments=engine.vocab.num_sentiments if engine.vocab else 0,
    )

@app.post("/predict", response_model=AspectSentimentResult)
def predict(req: PredictRequest):
    if not engine or not engine.ready:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready. Place checkpoint at: {CHECKPOINT_PATH}"
        )
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    if not req.target.strip():
        raise HTTPException(status_code=422, detail="target must not be empty")
    if req.target not in req.text:
        raise HTTPException(status_code=422, detail="target must be a substring of text")
    try:
        result = engine.predict(req.text, req.target)
        return AspectSentimentResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Production: serve compiled React build from FastAPI
FRONTEND_DIST = Path("frontend/dist")
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="static")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(status_code=404)
```

### `frontend/vite.config.js`
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // All /predict and /health calls go to FastAPI — no CORS needed in dev
      '/predict': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
```

> **Note:** With Vite's proxy, React calls `/predict` (relative), Vite forwards to `http://localhost:8000/predict`. FastAPI CORS is only needed for direct cross-origin calls (e.g., from a deployed CDN). In dev, CORS is irrelevant because the proxy rewrites the origin.

---

## 3. `requirements.txt` — Inference Server (CPU)

This file is for the demo API server only (not for training). Uses CPU torch to minimize size for deployment.

```
# ── Web framework ──────────────────────────────────────────────────
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9     # for form data if needed later

# ── ML inference (CPU-only torch) ──────────────────────────────────
# Install with:
#   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
# Then install the rest:
#   pip install -r requirements.txt
torch==2.3.1
transformers==4.44.2
sentencepiece==0.2.0         # required for vinai/phobert-base tokenizer
protobuf==4.25.3             # required by tokenizer serialization

# ── Config & utilities ─────────────────────────────────────────────
pyyaml==6.0.2
numpy==1.26.4

# ── NOT needed at inference (training only) ────────────────────────
# scikit-learn, tqdm — remove from inference deploy to save ~100MB
```

**Install command sequence (CPU deploy):**
```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Why CPU torch?** PhoBERT-large on CPU takes ~1-2s per request — acceptable for a demo. CPU torch is 300MB vs 2.5GB for CUDA torch, which matters for Render/Railway free tiers.

---

## `requirements-train.txt` — Training (full GPU stack)

```
# Full training stack — use this on Kaggle / Colab / local GPU machine
torch>=2.1.0                 # GPU version; install via: pip install torch (picks CUDA automatically)
transformers==4.44.2
sentencepiece==0.2.0
protobuf==4.25.3
scikit-learn==1.5.2
tqdm==4.66.5
pyyaml==6.0.2
numpy==1.26.4
```

---

## 4. `Makefile` — Local Development Commands

```makefile
.PHONY: dev api frontend install install-frontend help

PYTHON   := python
UVICORN  := uvicorn
NPM      := npm

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Run FastAPI + Vite dev server concurrently (Ctrl+C to stop both)
	@echo "Starting API on http://localhost:8000 and frontend on http://localhost:5173"
	@trap 'kill %1 %2 2>/dev/null; exit 0' SIGINT; \
	$(UVICORN) api.main:app --reload --port 8000 & \
	cd frontend && $(NPM) run dev & \
	wait

api: ## Run FastAPI only (with hot reload)
	$(UVICORN) api.main:app --reload --port 8000

frontend: ## Run Vite dev server only
	cd frontend && $(NPM) run dev

build: ## Build React for production (output: frontend/dist)
	cd frontend && $(NPM) run build

install: ## Install Python inference deps (CPU torch)
	pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

install-frontend: ## Install frontend npm deps
	cd frontend && $(NPM) install

install-all: install install-frontend ## Install everything

serve-prod: build ## Build frontend then serve both via FastAPI on port 8000
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000
```

---

## 5. `.env.example`

```bash
# Path to trained checkpoint — copy this file to .env and fill in
CHECKPOINT_PATH=checkpoints/phobert_large_target_attn_weighted/best_model.pt

# Config to load (controls encoder, attention, label structure)
CONFIG_PATH=config/base.yaml

# Comma-separated allowed CORS origins (production)
CORS_ORIGINS=http://localhost:5173,http://localhost:4173,https://your-app.onrender.com
```

Load with `python-dotenv` in `api/main.py` or pass as shell env vars.

---

## 6. Deployment Options — Ranked by Ease

### ⭐ Option 1: HuggingFace Spaces (Gradio) — EASIEST, ~30 min

Best for: Quick shareable demo, no frontend code required.

**How it works:** Create a Space with SDK=Gradio. Push `app.py` + `requirements.txt`. Spaces runs `app.py` automatically. Free CPU tier, ~7GB RAM.

**`app.py` (Gradio wrapper):**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from api.inference import InferenceEngine

engine = InferenceEngine(
    checkpoint_path="checkpoints/best_model.pt",
    config_path="config/base.yaml",
)

EXAMPLES = [
    ["em mua ip7plus được 3 tháng nhưng dung lượng pin chỉ còn 98%", "pin"],
    ["phòng sạch sẽ, thoáng mát, nhân viên phục vụ rất nhiệt tình", "phòng"],
    ["đồ ăn ngon nhưng giá hơi đắt so với chất lượng", "đồ ăn"],
]

def predict(text: str, target: str):
    if not engine.ready:
        return "⚠️ Model not loaded.", {}
    result = engine.predict(text, target)
    label = f"**{result['aspect']}** → **{result['sentiment']}**"
    probs = {
        **{f"aspect:{k}": v for k, v in result["aspect_probabilities"].items()},
        **{f"sentiment:{k}": v for k, v in result["sentiment_probabilities"].items()},
    }
    return label, probs

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Vietnamese review text", placeholder="e.g. phòng sạch sẽ, nhân viên nhiệt tình"),
        gr.Textbox(label="Target phrase", placeholder="e.g. phòng"),
    ],
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Label(label="Probabilities", num_top_classes=5),
    ],
    examples=EXAMPLES,
    title="Targeted ABSA Demo",
    description="Predict aspect category and sentiment for a target phrase in a Vietnamese review.",
)

if __name__ == "__main__":
    demo.launch()
```

**`requirements.txt` for Spaces:**
```
fastapi>=0.100
gradio>=4.0
torch>=2.0
transformers>=4.30
sentencepiece
protobuf
pyyaml
```
Spaces pre-installs torch 2.x, so `torch>=2.0` without a URL works fine.

**Deploy steps:**
1. `pip install huggingface_hub`
2. `huggingface-cli login`
3. `huggingface-cli repo create <your-space-name> --type space --space_sdk gradio`
4. `git push` the repo (or use the Spaces web UI to upload files)
5. Upload `checkpoints/best_model.pt` via `huggingface_hub.upload_file()` or git-lfs

**Limitation:** No custom React UI. Target phrase must be typed manually (not highlighted by click).

---

### ⭐⭐ Option 2: HuggingFace Spaces (Docker) — MODERATE, ~2 hours

Best for: Full React frontend with rich UX (text highlight → target selection).

**`Dockerfile` (multi-stage, builds React then serves everything via FastAPI):**
```dockerfile
# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python inference server
FROM python:3.11-slim
WORKDIR /app

# Install CPU torch first (separate step for Docker layer caching)
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY model/ ./model/
COPY config/ ./config/
COPY Data/ ./Data/
COPY vocabulary.py dataloader.py ./
COPY api/ ./api/

# Copy compiled React into FastAPI's static directory
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# HuggingFace Spaces requirement: port 7860
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Spaces `README.md` header (required by HuggingFace):**
```yaml
---
title: Targeted ABSA Demo
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
```

**Model file in Spaces:** Use `huggingface_hub.hf_hub_download()` to pull the checkpoint from a separate model repo at startup, rather than committing 1.2GB to the Space repo.

```python
# In api/inference.py, add before loading checkpoint:
from huggingface_hub import hf_hub_download
if not Path(checkpoint_path).exists():
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="your-username/absa-checkpoints",
        filename="best_model.pt",
        local_dir=str(Path(checkpoint_path).parent),
    )
```

---

### ⭐⭐⭐ Option 3: Render — EASY BUT COLD STARTS HURT

**Problem:** PhoBERT-large weights are 1.2GB. Loading from disk on a cold start takes ~30-45s on Render free tier. Users will see a blank spinner. The free tier spins down after 15 min of inactivity.

**Mitigation:** Use `xlm_roberta_base` (smaller, 278MB) for the demo deployment config instead of `phobert_large`. Update `CHECKPOINT_PATH` env var to point to XLM-R checkpoint.

**Backend (Render Web Service):**
- Build command: `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt`
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Env vars: `CHECKPOINT_PATH`, `CONFIG_PATH`, `CORS_ORIGINS`

**Frontend (Render Static Site):**
- Root dir: `frontend`
- Build command: `npm install && npm run build`
- Publish dir: `frontend/dist`
- Add rewrite rule: `/* → /index.html` (200) for SPA routing

**Or serve frontend from FastAPI (simpler, 1 service):**
- `make build` before deploy → `frontend/dist` copied into container
- FastAPI serves `frontend/dist` as static files (already implemented in `api/main.py` above)
- One service = one URL, no CORS needed

---

### ⭐⭐⭐ Option 4: Railway — MODERATE, PAID

Same as Render but `$5/month` keeps the service warm (no cold starts). `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn api.main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

Add `Procfile` as fallback:
```
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

---

### ⭐⭐⭐⭐ Option 5: Local Docker — ZERO INFRA KNOWLEDGE, BEST REPRODUCIBILITY

For "clone and run" demos shared via GitHub.

**`docker-compose.yml`:**
```yaml
services:
  api:
    build:
      context: .
      target: prod
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints  # mount checkpoint dir (not baked in)
    environment:
      - CHECKPOINT_PATH=/app/checkpoints/best_model.pt
      - CONFIG_PATH=/app/config/base.yaml
```

```bash
docker compose up --build
# Open http://localhost:8000
```

---

## 7. Local Development Workflow (No Docker)

**Prerequisites:**
```
Python 3.9+
Node.js 18+
A trained checkpoint at checkpoints/<group>/<run>/best_model.pt
```

**Step-by-step setup (first time):**
```bash
# 1. Clone repo
git clone https://github.com/dikey4405/targeted_ABSA.git
cd targeted_ABSA

# 2. Create Python virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install CPU torch + API deps
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Install frontend deps
cd frontend && npm install && cd ..

# 5. Configure checkpoint path
cp .env.example .env
# Edit .env: set CHECKPOINT_PATH to your actual checkpoint path

# 6. Run both servers
make dev
```

**Verify:**
```bash
# In a third terminal
curl http://localhost:8000/health
# → {"ready":true,"encoder":"phobert_large","num_aspects":N,"num_sentiments":M}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"phòng sạch sẽ thoáng mát","target":"phòng"}'
# → {"aspect":"ROOMS#CLEANLINESS","sentiment":"POSITIVE",...}
```

**Open browser:** http://localhost:5173

---

## 8. Critical Issues Found in Codebase

These must be resolved before the API works correctly:

### 🔴 Issue 1: Checkpoint path is nested, not flat

`train.py` saves to `checkpoints/<group_name>/<run_name>/best_model.pt`, not `checkpoints/best_model.pt`.

The best existing checkpoint from `reports/test_metrics.json` is:
```
D:\Python\ABSA_targeted\source\checkpoints\best_model.pt
```
This is a Windows absolute path from Kaggle training. It needs to be re-placed at a known relative path.

**Fix:** After training, copy the checkpoint:
```bash
mkdir -p checkpoints
cp checkpoints/high_score_experiments/phobert_large_target_attn_weighted/best_model.pt \
   checkpoints/best_model.pt
```
Or update `CHECKPOINT_PATH` in `.env` to the nested path.

### 🔴 Issue 2: `train.py` hardcodes PhoBERT tokenizer for ALL encoders

In `train.py:334`, `Vocabulary` is always initialized with `"vinai/phobert-base"` regardless of the experiment's `encoder_key`. The `api/inference.py` above fixes this by reading `encoder_key` from the config and passing the correct `encoder_name` to `Vocabulary`.

**Impact on demo:** If the deployed checkpoint was trained with the PhoBERT tokenizer (which it was), the API must also use the PhoBERT tokenizer. Passing `encoder_name = ENCODER_MAP["phobert_large"]` (`"vinai/phobert-large"`) to `Vocabulary` is correct for the best existing checkpoint.

### 🟡 Issue 3: PhoBERT requires Vietnamese word segmentation

`config/encoders.yaml` flags `requires_word_segmentation: true` for PhoBERT. PhoBERT was pre-trained on VnCoreNLP-segmented text (words joined with `_`). If training data was raw text (as the sample data suggests), this may not matter — but for production, segmented input would improve accuracy.

**For the demo:** Skip segmentation for simplicity. Note in UI: "Input does not require word segmentation; segmentation may improve results."

**If you want segmentation:** Add `underthesea` to `requirements.txt` and wrap the input:
```python
from underthesea import word_tokenize
text_segmented = word_tokenize(text, format="text")  # joins with spaces
target_segmented = word_tokenize(target, format="text")
```

### 🟡 Issue 4: `model/` has no `__init__.py`

The model package works as a Python 3.3+ namespace package. **Do not add `__init__.py`** — it's not needed and would require testing. The `sys.path` approach in `api/inference.py` handles this correctly.

### 🟡 Issue 5: `torch.load()` security warning

PyTorch 2.0+ emits a `FutureWarning` for `torch.load()` without `weights_only=True`. The `api/inference.py` above adds `weights_only=True` which is correct for loading state dicts only.

---

## 9. Files to Create (Prioritized Build Order)

| Priority | File | Notes |
|----------|------|-------|
| 1 (blocker) | `api/__init__.py` | Empty |
| 1 (blocker) | `api/schemas.py` | Pydantic models |
| 1 (blocker) | `api/inference.py` | InferenceEngine |
| 1 (blocker) | `api/main.py` | FastAPI app |
| 1 (blocker) | `requirements.txt` | CPU torch + FastAPI |
| 2 (dev QoL) | `Makefile` | `make dev` command |
| 2 (dev QoL) | `.env.example` | Checkpoint path template |
| 2 (dev QoL) | `checkpoints/.gitkeep` | Keep dir in git |
| 3 (frontend) | `frontend/package.json` | React + Vite |
| 3 (frontend) | `frontend/vite.config.js` | Proxy config |
| 3 (frontend) | `frontend/src/App.jsx` | Main layout |
| 3 (frontend) | `frontend/src/components/ReviewInput.jsx` | Text input |
| 3 (frontend) | `frontend/src/components/TargetSelector.jsx` | Highlight UX |
| 3 (frontend) | `frontend/src/components/PredictionResult.jsx` | Result display |
| 3 (frontend) | `frontend/src/components/ExampleCards.jsx` | Pre-loaded examples |
| 4 (deploy) | `app.py` | Gradio Space (fastest path to deployed demo) |
| 4 (deploy) | `Dockerfile` | Docker/HF Spaces Docker option |
| 5 (docs) | `README.md` additions | Demo setup section |
| 5 (docs) | `.gitignore` updates | Add `checkpoints/*.pt`, `frontend/node_modules/`, `.env` |

---

## 10. `.gitignore` Updates Needed

Add these lines to the existing `.gitignore`:
```gitignore
# Demo additions
checkpoints/*.pt
checkpoints/**/*.pt
frontend/node_modules/
frontend/dist/
.env
.venv/
__pycache__/
*.pyc
.DS_Store
```

---

## 11. README Demo Section Template

```markdown
## Demo Setup

### Requirements
- Python 3.9+
- Node.js 18+
- Trained checkpoint (see Training section below, or download from HuggingFace)

### Local Setup
\`\`\`bash
# Install Python deps (CPU)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Install frontend
cd frontend && npm install && cd ..

# Set checkpoint path
cp .env.example .env
# Edit .env: set CHECKPOINT_PATH to your best_model.pt path

# Run
make dev
\`\`\`

Open **http://localhost:5173** in your browser.  
API docs: **http://localhost:8000/docs**

### Online Demo
Try the live demo on HuggingFace Spaces: [link]

### Example inputs
| Text | Target | Expected |
|------|--------|----------|
| phòng sạch sẽ thoáng mát | phòng | ROOMS#GENERAL → POSITIVE |
| pin chỉ dùng được nửa ngày | pin | BATTERY → NEGATIVE |
| giá hơi đắt so với chất lượng | giá | PRICE → NEGATIVE |
```

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|------------|-------|
| Project structure | HIGH | Direct codebase analysis; follows standard Python-API-React monorepo patterns |
| requirements.txt | HIGH | Imports enumerated from source files; torch CPU URL confirmed stable |
| Vite proxy config | HIGH | Official Vite docs pattern; eliminates CORS for dev |
| FastAPI inference code | HIGH | Direct analysis of `vocabulary.py` API and `TargetedABSAModel.forward()` signature |
| HF Spaces Gradio | HIGH | Standard pattern; Spaces SDK handles dep installation |
| Render/Railway deployment | MEDIUM | Cold start severity depends on instance size; phobert_large is 1.2GB |
| PhoBERT segmentation | MEDIUM | Flag in config but not enforced; may or may not affect demo quality |
