# Phase 1: Inference API — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-27
**Phase:** 01-inference-api
**Areas discussed:** config-sourcing, import-path, vocab-data, model-arch-recovery

---

## Config & Checkpoint Sourcing

| Option | Description | Selected |
|--------|-------------|----------|
| Env vars with defaults | CHECKPOINT_PATH defaults to checkpoints/best_model.pt, DATA_DIR defaults to Data/ | ✓ |
| Env vars required | Server refuses to start without both vars set | |
| api/config.yaml | Dedicated config file user edits before running | |

**User's choice:** Env vars with defaults
**Notes:** Also confirmed to include `.env.example` documenting all variables.

---

## Import Path Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| PYTHONPATH=. | Run from project root, Python resolves imports naturally | ✓ |
| sys.path.insert | Add project root at top of inference.py | |
| pip install -e . | Make project a package with pyproject.toml | |

**User's choice:** PYTHONPATH=. uvicorn api.main:app
**Notes:** Zero changes to existing ML code required.

---

## Vocabulary Data Scan

| Option | Description | Selected |
|--------|-------------|----------|
| train + dev + test splits | Exactly what training used; guaranteed vocab match | ✓ |
| Domain source files | hotel.jsonl + restaurant.jsonl + mobile.jsonl | |
| train.jsonl only | Fastest startup; may miss some labels | |

**User's choice:** train.jsonl + dev.jsonl + test.jsonl
**Notes:** Matches training pipeline exactly for 100% label coverage.

---

## Model Architecture Recovery

| Option | Description | Selected |
|--------|-------------|----------|
| Env vars with best-run defaults | MODEL_ENCODER_KEY=phobert_large, MODEL_ATTENTION_KEY=target_conditioned_attention | ✓ |
| Sidecar config.json | User manually places model_config.json next to checkpoint | |
| CLI args to uvicorn | Pass config keys as --env-var overrides at launch | |

**User's choice:** Env vars with best-run defaults
**Notes:** Defaults match reports/test_metrics.json best run. Documented in .env.example.

---

## the agent's Discretion

- Response format for `aspect_probs`: list of `{label, score}` objects (top-5, sorted desc) rather than flat dict — better for frontend bar chart rendering
- `InferenceEngine` class in `api/inference.py` wraps model state and asyncio.Lock
- FastAPI `lifespan` context manager for startup/shutdown lifecycle

## Deferred Ideas

- Multi-model hot-swap (v2 backlog)
- Vocabulary pickle cache (v2 backlog)
- Gradio/HuggingFace Spaces deploy (v2 backlog)
- ONNX export for CPU latency (v2 backlog)
