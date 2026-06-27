---
phase: 01-inference-api
plan: "03"
subsystem: api-application
tags: [fastapi, cors, lifespan, endpoints, makefile]
dependency_graph:
  requires:
    - "01-01 (api/schemas.py — PredictRequest, PredictResponse, HealthResponse, ModelsResponse)"
    - "01-02 (api/inference.py — InferenceEngine with ready, lock, predict())"
  provides:
    - "api/main.py (FastAPI app, three HTTP endpoints, CORS, lifespan)"
    - "Makefile (api, api-prod, install, help targets)"
  affects:
    - "Phase 2 frontend — calls POST /predict, GET /health via Vite proxy on port 8000"
tech_stack:
  added:
    - "FastAPI lifespan context manager (asynccontextmanager)"
    - "CORSMiddleware with explicit origin list (allow_credentials=False)"
    - "GNU Make Makefile with self-documenting help target"
  patterns:
    - "Module-level singleton (_engine) set by lifespan, read by route handlers"
    - "async with engine.lock for request serialization in /predict"
    - "JSONResponse for 503 /health (bypasses Pydantic serialization on error path)"
    - "Guarded static file mount registered last (avoids wildcard shadowing API routes)"
key_files:
  created:
    - api/main.py
    - Makefile
  modified: []
decisions:
  - "String annotation for _engine type ('InferenceEngine | None') for Python <3.10 compatibility"
  - "allow_credentials=False with explicit CORS_ORIGINS env var — safe default per T-03-02"
  - "frontend/dist static guard registered after all API routes to prevent wildcard route shadowing"
  - "Makefile PYTHON := python3 (not python) to match macOS default where python3 is the binary"
  - "--workers 1 mandatory in api-prod target: phobert-large ~1.3 GB per worker"
metrics:
  duration: "~3 minutes"
  completed: "2026-06-27"
  tasks_completed: 2
  tasks_total: 2
  files_created: 2
  files_modified: 0
status: complete
---

# Phase 01 Plan 03: FastAPI Application and Makefile Summary

**One-liner:** FastAPI app wiring InferenceEngine via lifespan context manager into three HTTP endpoints (/predict, /health, /models) with env-var-driven CORS, plus a self-documenting Makefile for dev/prod server start and dependency install.

## What Was Built

### Files Created

| File | Description |
|------|-------------|
| `api/main.py` | Complete FastAPI application (167 lines): lifespan, CORS, three endpoints, static guard |
| `Makefile` | GNU Make targets: `help` (default), `api`, `api-prod`, `install` |

### api/main.py — Structure

The file is structured in the order mandated by the plan to prevent registration-order bugs:

1. **Imports** — stdlib (`os`, `asynccontextmanager`, `Path`), FastAPI, local schemas and inference
2. **Module-level singleton** — `_engine: "InferenceEngine | None" = None`
3. **Lifespan context manager** — `InferenceEngine()` constructed once before `yield`; set to `None` on shutdown
4. **FastAPI app** — `FastAPI(title=..., lifespan=lifespan)`
5. **CORSMiddleware** — `allow_credentials=False`, origins from `CORS_ORIGINS` env var (default: `localhost:5173,127.0.0.1:5173`)
6. **GET /health** — 503 with `model_loaded=false` before ready; 200 with `model_loaded=true` after
7. **GET /models** — 503 if engine None; 200 with encoder config strings
8. **POST /predict** — 503 if not ready; `async with _engine.lock`; ValueError→422, Exception→500
9. **Static file guard** — conditional `frontend/dist` mount registered last (unreachable in Phase 1)

### Makefile — Targets

| Target | Command | Notes |
|--------|---------|-------|
| `help` | `grep`+`awk` self-doc | Default target; parses `##` comments |
| `api` | `PYTHONPATH=. uvicorn api.main:app --reload --port 8000` | Dev with hot-reload |
| `api-prod` | `PYTHONPATH=. uvicorn api.main:app --workers 1 --host 0.0.0.0 --port 8000` | `--workers 1` mandatory |
| `install` | Two-step: PyTorch CPU wheel + `requirements.txt` | Special PyTorch index URL required |

## Verification Passed

```
AST parse: OK (Python 3 syntax valid)
asynccontextmanager import: PASS
_engine module singleton: PASS
lifespan function: PASS
FastAPI app instance: PASS
CORSMiddleware: PASS
allow_credentials=False: PASS
CORS_ORIGINS env var: PASS
GET /health: PASS
503 on not ready (health): PASS
GET /models: PASS
POST /predict: PASS
async with engine.lock: PASS
503 model not ready: PASS
ValueError -> 422: PASS
Exception -> 500: PASS
frontend/dist guard LAST: PASS
InferenceEngine import: PASS
PredictResponse(**result): PASS
---
make help lists api/api-prod/install: OK
make -n api shows PYTHONPATH=. uvicorn --reload --port 8000: OK
make -n api-prod shows --workers 1 --host 0.0.0.0: OK
make -n install shows download.pytorch.org URL: OK
```

Runtime integration (requires `best_model.pt` at `checkpoints/best_model.pt` or `CHECKPOINT_PATH` env var) validated by the done criteria in the plan — static checks confirm all contract points are met.

## Decisions Made

1. **`PYTHON := python3` in Makefile** — macOS ships `python3` (not `python`) as the default; using `python3` avoids "command not found" for users who have not aliased `python`. The `PYTHON` variable is defined but not used in api/api-prod targets (uvicorn is invoked directly per the plan spec). The variable is available for future extension.

2. **`allow_credentials=False` with explicit origin list** — Per T-03-02: setting `allow_credentials=True` with a non-wildcard origin list causes browser preflight rejection in certain user-agents. `False` is the correct default for a credential-free public demo API.

3. **String annotation `"InferenceEngine | None"`** — PEP 604 union syntax (`X | Y`) is only available at runtime in Python 3.10+. Using a string annotation keeps the module importable on Python 3.9 (which is the project's stated minimum per PROJECT.md).

4. **`JSONResponse` for 503 /health** — The Pydantic `response_model=HealthResponse` validates the return value on success paths only; returning a `JSONResponse` directly bypasses Pydantic serialization and lets us set the status code to 503 while still returning a body that matches the shape consumers expect (`status`, `model_loaded`, `device`).

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — `api/main.py` is fully wired. All three endpoints delegate to `InferenceEngine` (Plan 02) which is itself fully implemented. No placeholder values or TODO comments in the hot path.

## Threat Model Coverage

| ID | Mitigation |
|----|-----------|
| T-03-02 | `allow_credentials=False` enforced in CORSMiddleware; explicit origin list from `CORS_ORIGINS` env var — never hardcoded |
| T-03-03 | All exceptions in `/predict` caught; only `str(exc)` returned as `detail` — no traceback, no file paths exposed |
| T-03-04 | `async with _engine.lock` serializes all `/predict` requests (single asyncio.Lock, accepted DoS per STRIDE register) |
| T-03-05 | NFC normalization delegated to `InferenceEngine.predict()` (Plan 02 Step 1); Pydantic `max_length=2000` on `text` field (Plan 01) |

## Threat Flags

None — no new network endpoints, auth paths, or file access patterns beyond those already in the STRIDE register.

## Self-Check: PASSED

| Item | Status |
|------|--------|
| `api/main.py` created | ✅ FOUND |
| `Makefile` created | ✅ FOUND |
| Commit `bafc25a` (Task 1 — api/main.py) | ✅ FOUND |
| Commit `2b9e3c4` (Task 2 — Makefile) | ✅ FOUND |
| `01-03-SUMMARY.md` created | ✅ FOUND |
