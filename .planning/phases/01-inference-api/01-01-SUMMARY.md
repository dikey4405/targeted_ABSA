---
phase: 01-inference-api
plan: "01"
subsystem: api-contracts
tags: [pydantic, schemas, requirements, env-config, gitignore]
dependency_graph:
  requires: []
  provides:
    - api/schemas.py (PredictRequest, PredictResponse, AspectProb, HealthResponse, ModelsResponse)
    - requirements.txt (10 inference deps)
    - .env.example (7 env vars with defaults)
    - checkpoints/.gitkeep
  affects:
    - api/inference.py (Plan 02 — field names must match InferenceEngine.predict() return dict)
    - api/main.py (Plan 03 — imports all five schema classes)
tech_stack:
  added:
    - pydantic>=2.7.0 (Pydantic v2 BaseModel, Field)
    - fastapi==0.115.0 (HTTP layer, declared in requirements.txt)
    - uvicorn[standard]==0.30.6 (ASGI server, declared in requirements.txt)
    - torch>=2.1.0 (inference runtime, CPU wheel)
    - transformers==4.44.2 (PhoBERT tokenizer)
    - sentencepiece==0.2.0 (tokenizer backend)
    - protobuf==4.25.3 (transformer dep)
    - pyyaml==6.0.2 (config YAML loading)
    - numpy>=1.26.4 (inference numerics)
    - python-multipart==0.0.9 (FastAPI form data support)
  patterns:
    - Pydantic v2 BaseModel with Field validation (min_length, max_length)
    - List[AspectProb] list-of-objects format for sorted bar chart rendering
key_files:
  created:
    - api/__init__.py
    - api/schemas.py
    - requirements.txt
    - .env.example
    - checkpoints/.gitkeep
  modified:
    - .gitignore (appended *.pt, checkpoints/*.pt, .env entries)
decisions:
  - "AspectProb uses list-of-objects format (not flat dict) per D-05 for sorted bar chart rendering"
  - "requirements.txt excludes scikit-learn and tqdm (training-only) to keep inference env lightweight (~300 MB CPU torch)"
  - "checkpoints/.gitkeep tracks directory in git; *.pt files are gitignored to prevent large binary commits"
metrics:
  duration: "5 minutes"
  completed: "2026-06-27"
  tasks_completed: 2
  tasks_total: 2
  files_created: 5
  files_modified: 1
status: complete
---

# Phase 01 Plan 01: Pydantic Schemas, Requirements, and Env Config Summary

**One-liner:** Pydantic v2 API contracts (`api/schemas.py`) plus `requirements.txt`, `.env.example`, and `.gitignore` rules establishing the inference dependency and configuration baseline.

## What Was Built

Five Pydantic v2 `BaseModel` classes define the complete HTTP data contract for the three inference endpoints before either `api/main.py` (Plan 03) or `api/inference.py` (Plan 02) are implemented — enabling parallel Wave-1 development.

### Files Created

| File | Description |
|------|-------------|
| `api/__init__.py` | Empty package marker — makes `api/` importable as `from api.schemas import ...` |
| `api/schemas.py` | Five Pydantic v2 schema classes (see below) |
| `requirements.txt` | 10 inference-only Python dependencies with CPU torch install note |
| `.env.example` | Documents 7 env vars with defaults matching D-01/D-05 decisions |
| `checkpoints/.gitkeep` | Tracks `checkpoints/` in git; user copies `best_model.pt` here |

### Files Modified

| File | Change |
|------|--------|
| `.gitignore` | Appended `checkpoints/*.pt`, `*.pt` (model binaries too large for git), `.env` (secrets) |

### Schema Classes (api/schemas.py)

| Class | Fields | Validation |
|-------|--------|------------|
| `PredictRequest` | `text: str`, `target: str` | `text`: min 1, max 2000 chars; `target`: min 1, max 200 chars |
| `AspectProb` | `label: str`, `score: float` | Shared by `aspect_probs` and `sentiment_probs` |
| `PredictResponse` | `aspect`, `sentiment`, `aspect_probs` (List[AspectProb] top-5), `sentiment_probs` (List[AspectProb] all 3), `latency_ms: int` | Field names match `InferenceEngine.predict()` return dict keys exactly |
| `HealthResponse` | `status: str`, `model_loaded: bool`, `device: str` | — |
| `ModelsResponse` | `active`, `encoder`, `attention_key`, `label_structure` | All `str` reflecting active config per D-05 |

## Verification Passed

```
PYTHONPATH=. python3 -c "from api.schemas import ..." → schemas ok
Empty text → ValidationError ✓
text > 2000 chars → ValidationError ✓
Empty target → ValidationError ✓
target > 200 chars → ValidationError ✓
AspectProb, HealthResponse, ModelsResponse instantiable ✓
PredictResponse with list fields ✓
requirements.txt: fastapi==0.115.0, torch>=2.1.0, transformers==4.44.2 ✓
.env.example: all 7 env vars with correct defaults ✓
checkpoints/.gitkeep exists ✓
.gitignore: checkpoints/*.pt, *.pt, .env ✓
best_model.pt at root correctly gitignored ✓
```

## Decisions Made

1. **List-of-objects format for `aspect_probs`** — Per D-05 discretion: `List[AspectProb]` sorted descending (not flat dict) is easier for the frontend to render as a sorted bar chart without additional client-side processing.

2. **Training deps excluded from requirements.txt** — `scikit-learn` and `tqdm` are training-only; excluding them keeps the inference environment ~300 MB lighter (CPU torch only) and prevents dependency conflicts in the demo deployment.

3. **Flat `checkpoints/` layout** — `CHECKPOINT_PATH` defaults to `checkpoints/best_model.pt` (flat, not nested). User runs `cp best_model.pt checkpoints/` once, matching D-01.

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — this plan creates static configuration files and schema definitions. No UI rendering or data wiring occurs at this stage.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary file access patterns introduced. All created files are developer-controlled configuration artifacts.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| `api/__init__.py` | ✅ FOUND |
| `api/schemas.py` | ✅ FOUND |
| `requirements.txt` | ✅ FOUND |
| `.env.example` | ✅ FOUND |
| `checkpoints/.gitkeep` | ✅ FOUND |
| `01-01-SUMMARY.md` | ✅ FOUND |
| Commit `4b19a3a` (Task 1 — schemas) | ✅ FOUND |
| Commit `dcba19c` (Task 2 — env/config) | ✅ FOUND |

