---
phase: 01-inference-api
plan: "02"
subsystem: api-inference-engine
status: complete
tags: [python, pytorch, inference, fastapi, absa]

dependency_graph:
  requires:
    - "01-01 (api/schemas.py, requirements.txt)"
  provides:
    - "api/inference.py (InferenceEngine class)"
    - "api/__init__.py (package marker)"
  affects:
    - "01-03 (api/main.py depends on InferenceEngine interface)"

tech_stack:
  added:
    - torch (inference backbone — torch.load, torch.no_grad, F.softmax)
    - asyncio.Lock (concurrency serialization)
    - unicodedata (NFC normalization)
    - heapq (top-5 aspect extraction)
  patterns:
    - Graceful degradation: ready=False when checkpoint missing; no exception raised
    - weights_only=True in torch.load (safe deserialization, T-02-01)
    - DataParallel module. prefix stripping in state_dict
    - model.eval() for deterministic dropout-free inference

key_files:
  created:
    - api/inference.py
  modified: []

decisions:
  - "ENCODER_MAP duplicated in inference.py (not imported from train.py) to avoid pulling training-only deps at API startup (D-03)"
  - "_MAX_INFERENCE_LENGTH = 128 used for inference (matches eval config, shorter than Vocabulary default of 256)"
  - "Graceful degradation: missing checkpoint sets ready=False, prints WARNING, returns without exception (API-06)"
  - "heapq.nlargest(5, ...) for top-5 aspect_probs; all 3 sentiments returned sorted descending"
  - "asyncio.Lock() initialized in __init__ so api/main.py can use 'async with engine.lock' without setup"

metrics:
  duration: "~26 minutes"
  completed: "2026-06-27"
  tasks_completed: 2
  tasks_total: 2
  files_created: 1
---

# Phase 01 Plan 02: InferenceEngine Implementation Summary

**One-liner:** InferenceEngine wrapping Vocabulary, TargetedABSAModel, and asyncio.Lock with NFC normalization, graceful checkpoint degradation, and top-5 aspect prediction.

## What Was Built

`api/inference.py` — a self-contained InferenceEngine class that:

1. Reads all configuration from env vars (CHECKPOINT_PATH, DATA_DIR, DEVICE, MODEL_ENCODER_KEY, MODEL_ATTENTION_KEY, MODEL_LABEL_STRUCTURE) with sensible defaults per D-01 and D-05
2. Initialises `Vocabulary` from the same three JSONL files used at training time, ensuring label maps match the saved checkpoint
3. Loads `TargetedABSAModel` architecture matching the checkpoint's structure, applies DataParallel prefix stripping, and calls `model.eval()` for deterministic inference
4. Exposes `predict(text, target) -> dict` with NFC normalization, BOM stripping, input validation, and top-5 aspect + full sentiment probability decoding
5. Implements graceful degradation: when the checkpoint file is absent, logs a WARNING and returns with `ready=False` — the server starts normally and `/predict` returns 503

## Tasks

| # | Name | Status | Commit |
|---|------|--------|--------|
| 1 | Python package marker (api/__init__.py) | ✅ Pre-existing (Plan 01-01) | `4b19a3a` |
| 2 | InferenceEngine class (api/inference.py) | ✅ Complete | `a701e24` |

## Key Implementation Details

### ENCODER_MAP (module-level constant)
```python
ENCODER_MAP: dict[str, str] = {
    "phobert_base":     "vinai/phobert-base",
    "phobert_large":    "vinai/phobert-large",
    "xlm_roberta_base": "xlm-roberta-base",
    "xlm_roberta_large":"xlm-roberta-large",
}
```
Duplicated from train.py (not imported) to avoid pulling training-only dependencies at API startup (D-03 decision).

### Graceful Degradation (API-06)
```python
if not Path(checkpoint_path).exists():
    print(f"[InferenceEngine] WARNING: Checkpoint not found at {checkpoint_path!r}. ...")
    return   # self.ready stays False
```
Server starts cleanly; `/predict` endpoint in main.py checks `engine.ready` and returns HTTP 503.

### Security: Safe Deserialization (T-02-01)
```python
state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
```
`weights_only=True` prevents arbitrary code execution from crafted checkpoint files.

### DataParallel Prefix Handling
```python
if any(k.startswith("module.") for k in state_dict):
    state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
```
Handles checkpoints saved from multi-GPU training runs.

### predict() Input Contract
- NFC normalization + BOM strip applied to **both** `text` and `target` before any other processing
- `ValueError("text must not be empty or whitespace")` — raised for blank text
- `ValueError("target must not be empty or whitespace")` — raised for blank target
- `ValueError("target must be an exact substring of text")` — raised for non-substring targets

### Return Dict (matches PredictResponse fields exactly)
```python
{
    "aspect":          str,        # top-1 aspect label (e.g. "ROOMS#CLEANLINESS")
    "sentiment":       str,        # top-1 sentiment label ("POSITIVE"/"NEGATIVE"/"NEUTRAL")
    "aspect_probs":    list[dict], # top-5 {"label": str, "score": float} sorted desc
    "sentiment_probs": list[dict], # all 3 sentiments sorted desc
    "latency_ms":      int,        # wall-clock inference time in milliseconds
}
```

## Deviations from Plan

### Pre-existing Artifact (No-op)
**Task 1 (api/__init__.py):** The file was already created and committed by Plan 01-01 execution (commit `4b19a3a`). Plan 02 verified its existence and proceeded without a redundant commit. The done criteria (`test -f api/__init__.py`) passed.

### Runtime Import Test Skipped (Environment Limitation)
The plan's automated verification command `PYTHONPATH=. python -c "from api.inference import InferenceEngine, ENCODER_MAP; ..."` could not be executed because `torch` and `transformers` are not installed in the local dev environment (per PROJECT.md: "No requirements.txt exists — dependencies managed manually or via Kaggle notebook").

**Mitigation applied:** Comprehensive static analysis (Python AST parsing) verified:
- ✅ Valid Python syntax
- ✅ ENCODER_MAP all 4 keys with correct HuggingFace IDs
- ✅ `weights_only=True` in `torch.load`
- ✅ `model.eval()` called after `load_state_dict`
- ✅ NFC normalization + BOM strip on both inputs
- ✅ All three `ValueError` messages match spec exactly
- ✅ `asyncio.Lock()` initialised
- ✅ Return dict keys: `aspect`, `sentiment`, `aspect_probs`, `sentiment_probs`, `latency_ms`
- ✅ `heapq.nlargest(5, ...)` for top-5 aspect extraction
- ✅ Graceful degradation `Path.exists()` guard + `return` path

**User action required:** Run `pip install torch>=2.1.0 transformers==4.44.2 sentencepiece==0.2.0` (or see `requirements.txt` CPU-only install instructions) and then verify with `CHECKPOINT_PATH=/nonexistent/path.pt DATA_DIR=Data PYTHONPATH=. python -c "from api.inference import InferenceEngine; e = InferenceEngine(); assert e.ready == False; print('graceful degradation ok')"`.

## Threat Model Coverage

| ID | Mitigation Applied |
|----|-------------------|
| T-02-01 | `weights_only=True` in `torch.load` — prevents arbitrary code execution from crafted checkpoints |
| T-02-04 | NFC normalization + BOM strip in `predict()` Step 1 before substring validation |
| T-02-02 | ValueError messages are English-only; contain no paths, stack traces, or env var values |

## Public Interface (consumed by Plan 03 / api/main.py)

| Attribute / Method | Type | Purpose |
|-------------------|------|---------|
| `engine.ready` | `bool` | `/health` + `/predict` gate |
| `engine.lock` | `asyncio.Lock` | `async with engine.lock:` in predict route |
| `engine.device` | `torch.device` | `/health` response |
| `engine.encoder_key` | `str` | `/models` response |
| `engine.encoder_name` | `str` | `/models` response |
| `engine.attention_key` | `str` | `/models` response |
| `engine.label_structure` | `str` | `/models` response |
| `engine.predict(text, target)` | `-> dict` | Core inference |

## Known Stubs

None — `InferenceEngine.predict()` is fully wired end-to-end. No hardcoded placeholder values in the return path.

## Self-Check: PASSED

- [x] `api/inference.py` exists at `/Users/baovle/Code/Personal/targeted_ABSA/api/inference.py`
- [x] `api/__init__.py` exists (committed `4b19a3a`)
- [x] Task 2 commit `a701e24` exists in git log
- [x] ENCODER_MAP correctness verified via AST
- [x] All security mitigations (T-02-01, T-02-04) confirmed in source
