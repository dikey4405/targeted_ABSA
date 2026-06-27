# RUNBOOK — Vietnamese ABSA Demo

## Prerequisites

- Python 3.9+ (Homebrew: `brew install python`)
- Node.js + npm (Homebrew: `brew install node`)
- uv (Homebrew: `brew install uv`)
- Trained checkpoint at `checkpoints/best_model.pt`

---

## First-time setup

```bash
# 1. Clone / cd into repo
cd /path/to/targeted_ABSA

# 2. Create Python venv và install deps (only once)
uv sync

# 3. Install frontend deps (only once)
cd frontend && npm install && cd ..
```

---

## Run (every time)

Open **2 terminals** in the repo root.

### Terminal 1 — Backend API

```bash
cd /path/to/targeted_ABSA
PYTHONPATH=. uv run uvicorn api.main:app --workers 1 --port 8000
```

Wait for: `Application startup complete.`

### Terminal 2 — Frontend

```bash
cd /path/to/targeted_ABSA/frontend
npm run dev
```

Open **http://localhost:5173**

---

## Environment variables (optional overrides)

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_PATH` | `checkpoints/best_model.pt` | Path to trained model |
| `DATA_DIR` | `Data` | Directory containing JSONL data files |
| `MODEL_ENCODER_KEY` | `phobert_large` | Encoder: phobert_base / phobert_large / xlm_roberta_base |
| `MODEL_ATTENTION_KEY` | `target_conditioned_attention` | Attention mechanism |
| `DEVICE` | auto (cpu/cuda) | Force device: `cpu` or `cuda` |

Example with overrides:
```bash
CHECKPOINT_PATH=checkpoints/other_model.pt MODEL_ENCODER_KEY=phobert_base \
  PYTHONPATH=. uv run uvicorn api.main:app --workers 1 --port 8000
```

---

## API endpoints

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Phòng khách sạn rất sạch sẽ","target":"Phòng"}'

# Active model config
curl http://localhost:8000/models
```

---

## Troubleshooting

**`No module named 'fastapi'`**
→ Chưa install deps:
```bash
uv sync
```

**`RuntimeError: size mismatch`**
→ Vocab file mismatch. Make sure `DATA_DIR` points to the same data used during training (default `Data/` uses only `train.jsonl`).

**`model_loaded: false` on /health**
→ Checkpoint missing. Copy it to `checkpoints/best_model.pt` or set `CHECKPOINT_PATH`.

**Model loads slow (~30-60s)**
→ Normal — PhoBERT-large is 1.4GB on CPU. Wait for `Model ready on cpu.` in backend log.

**Frontend shows "Lỗi 503"**
→ Backend still loading. Wait and retry.
