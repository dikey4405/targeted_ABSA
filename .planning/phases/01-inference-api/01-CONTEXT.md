# Phase 1: Inference API — Context

**Gathered:** 2026-06-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the FastAPI inference server (`api/`) that loads `TargetedABSAModel` and `Vocabulary` at startup and serves `/predict`, `/health`, and `/models` endpoints. The React frontend (Phase 2) depends on this being live first.

</domain>

<spec_lock>
## Requirements (locked via SPEC.md)

**8 requirements are locked.** See `01-SPEC.md` for full requirements, boundaries, and acceptance criteria.

Downstream agents MUST read `01-SPEC.md` before planning or implementing. Requirements are not duplicated here.

**In scope (from SPEC.md):**
- `api/` directory: `main.py`, `inference.py`, `schemas.py`
- `requirements.txt` at project root
- Environment variable support: `CHECKPOINT_PATH`, `DATA_DIR`, `DEVICE`, `MODEL_ENCODER_KEY`, `MODEL_ATTENTION_KEY`, `MODEL_LABEL_STRUCTURE`
- CORS configuration for `localhost:5173`
- Three endpoints: `/predict`, `/health`, `/models`
- Single model loaded at startup (no hot-swap, no multi-model)

**Out of scope (from SPEC.md):**
- React frontend — Phase 2
- Training/fine-tuning endpoints
- Authentication/API keys
- Batch predict endpoint
- Docker/containerization
- Word segmentation (not needed — training data is raw Vietnamese)
- GPU setup or CUDA configuration

</spec_lock>

<decisions>
## Implementation Decisions

### Configuration & Checkpoint Sourcing
- **D-01:** Use environment variables with sensible defaults. `CHECKPOINT_PATH` defaults to `checkpoints/best_model.pt`; `DATA_DIR` defaults to `Data/`. Server starts without requiring vars to be set (user copies checkpoint once to the default path, or sets the var).
- **D-02:** Provide `.env.example` documenting all env vars with their defaults and descriptions.

### Import Path Strategy
- **D-03:** Run the server from the project root using `PYTHONPATH=. uvicorn api.main:app --workers 1`. Python's import resolution finds `vocabulary.py`, `model/`, and `config/` naturally. Zero changes to existing ML code.

### Vocabulary Data Files
- **D-04:** Pass `[Data/train.jsonl, Data/dev.jsonl, Data/test.jsonl]` to `Vocabulary()` — exactly the same files the training pipeline used. Guarantees the vocab matches the trained model. Paths resolved relative to `DATA_DIR` env var.

### Model Architecture Recovery
- **D-05:** Env vars with best-run defaults:
  - `MODEL_ENCODER_KEY=phobert_large` → resolves to `vinai/phobert-large`
  - `MODEL_ATTENTION_KEY=target_conditioned_attention`
  - `MODEL_LABEL_STRUCTURE=multitask_aspect_sentiment`
  - These defaults match the best-performing checkpoint in `reports/test_metrics.json`.
  - Documented in `.env.example` so users know they can override for a different checkpoint.

### the agent's Discretion
- Response format for `aspect_probs`: use a list of `{label, score}` objects sorted descending (top-5) rather than a flat dict — easier to render as a sorted bar chart in the frontend.
- `models` endpoint returns the active config values as strings, not resolved model IDs.
- `InferenceEngine` class (in `api/inference.py`) wraps all model state and the asyncio.Lock — `main.py` instantiates it once via FastAPI's `lifespan` context manager.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Locked Requirements
- `.planning/phases/01-inference-api/01-SPEC.md` — Locked requirements, boundaries, constraints, and acceptance criteria. Read this FIRST.

### Research (already done — do not re-research)
- `.planning/research/api-server.md` — FastAPI patterns, model loading, asyncio.Lock, CORS, top-5 trimming, pitfalls (638 lines). Especially: §Model Loading, §Concurrency, §Critical Pitfalls.
- `.planning/research/deployment.md` — Project structure, requirements.txt contents, Makefile, `.env.example` format (856 lines).

### Existing ML Code (read before implementing inference.py)
- `vocabulary.py` — `Vocabulary(data_path, model_name)` constructor. Uses `utf-8-sig` encoding (BOM handled). `tokenize_text_pair(text, target, max_length)` is the inference entry point.
- `model/targeted_absa.py` — `TargetedABSAModel` constructor signature and supported `attention_key` / `label_structure_key` values.
- `config/encoders.yaml` — Maps `encoder_key` strings to HuggingFace model IDs (e.g., `phobert_large` → `vinai/phobert-large`).
- `reports/test_metrics.json` — Confirmed best checkpoint: `phobert_large_target_attn_weighted`.

### Project Config
- `.planning/PROJECT.md` — Project overview, constraints, key decisions.
- `.planning/REQUIREMENTS.md` — Full v1 requirements list (API-01 through SETUP-03).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Vocabulary.tokenize_text_pair(text, target, max_length=256)` — The exact inference tokenization entry point. Returns `{input_ids, attention_mask, target_mask, token_type_ids?}` dict. Use `max_length=128` for inference (same as eval config).
- `Vocabulary.id2aspect`, `Vocabulary.id2sentiment` — Used to convert model output logit indices back to label strings.
- `ENCODER_MAP` in `train.py` — Maps encoder key strings to HuggingFace IDs. **Duplicate this mapping in `api/inference.py`** rather than importing from train.py (avoid pulling in all training deps).

### Established Patterns
- **Tokenizer always initialized in Vocabulary**: `Vocabulary.__init__` loads `AutoTokenizer.from_pretrained(model_name)`. The API must pass the correct `model_name` to match the encoder — use `ENCODER_MAP[MODEL_ENCODER_KEY]`.
- **BOM handling**: `_read_data()` opens files with `utf-8-sig` encoding — BOM in JSONL is handled transparently. Frontend NFC normalization handles the other Unicode edge case.
- **State dict only**: `torch.save(self.model.state_dict(), path)` at `train.py:314` — raw state dict, no wrapper. Load with `model.load_state_dict(torch.load(path, weights_only=True, map_location=device))`.
- **dropout=0.2 at training time**: The SPEC notes the default dropout value; do NOT override unless the checkpoint was trained with a different value.

### Integration Points
- `api/inference.py` imports: `from vocabulary import Vocabulary` and `from model.targeted_absa import TargetedABSAModel` — works when run from project root with `PYTHONPATH=.`.
- The frontend (Phase 2) will call `POST http://localhost:8000/predict` via Vite proxy. CORS must cover `localhost:5173`.

</code_context>

<specifics>
## Specific Ideas

- Default checkpoint path is `checkpoints/best_model.pt` (flat — user runs `cp checkpoints/encoder_comparison/.../best_model.pt checkpoints/best_model.pt` once).
- `.env.example` should include: `CHECKPOINT_PATH`, `DATA_DIR`, `DEVICE`, `MODEL_ENCODER_KEY`, `MODEL_ATTENTION_KEY`, `MODEL_LABEL_STRUCTURE`.
- Start command: `PYTHONPATH=. uvicorn api.main:app --workers 1 --host 0.0.0.0 --port 8000`
- Makefile target: `make api` runs the above with `.env` loaded via `export $(shell cat .env | xargs)`.

</specifics>

<deferred>
## Deferred Ideas

- **Multi-model hot-swap** — switching encoders at runtime; too complex for Phase 1, deferred to v2 backlog.
- **Vocabulary pickle cache** — serialize Vocabulary to avoid JSONL scan on restart; ~1-2s startup cost acceptable for demo; defer to v2 if cold-start matters.
- **Gradio interface** — single-file HuggingFace Spaces deploy; belongs in deployment phase (v2).
- **ONNX export** — for faster CPU inference on demo host; deferred until latency is measured.

</deferred>

---

*Phase: 01-inference-api*
*Context gathered: 2026-06-27*
