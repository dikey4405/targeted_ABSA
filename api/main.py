"""FastAPI application — Targeted ABSA Inference API.

Wire up the three HTTP endpoints (/predict, /health, /models) with CORSMiddleware
and a lifespan context manager that loads InferenceEngine exactly once at startup.

Run from project root with PYTHONPATH=.:
    uvicorn api.main:app --workers 1 --host 0.0.0.0 --port 8000
or via Makefile:
    make api        # dev mode (--reload)
    make api-prod   # production mode (--workers 1, --host 0.0.0.0)
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS — all at top, no inline imports (except the guarded static-files
#    block at the bottom of this file which is never reachable in Phase 1).
# ---------------------------------------------------------------------------
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse, ModelsResponse, PredictRequest, PredictResponse
from api.inference import InferenceEngine


# ---------------------------------------------------------------------------
# 2. MODULE-LEVEL SINGLETON.
#    String annotation (forward-reference style) avoids type-checker issues
#    on Python < 3.10 where `X | Y` union syntax is not available at runtime.
# ---------------------------------------------------------------------------
_engine: "InferenceEngine | None" = None


# ---------------------------------------------------------------------------
# 3. LIFESPAN CONTEXT MANAGER — per API-06: single model load at startup.
#    FastAPI calls the body before `yield` on startup and after `yield` on
#    shutdown.  Storing the engine in a module global avoids the overhead of
#    a dependency-injection lookup on every request.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = InferenceEngine()   # reads all config from env vars internally
    yield
    _engine = None


# ---------------------------------------------------------------------------
# 4. FASTAPI APP INSTANCE.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Targeted ABSA Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 5. CORS MIDDLEWARE — per API-04.
#    allow_credentials=False MUST be used with an explicit origin list (not
#    wildcard).  Setting True with a non-wildcard list causes browser preflight
#    rejection in some user-agents — False is the safe default for a
#    credential-free public demo API (T-03-02).
# ---------------------------------------------------------------------------
_cors_raw: str = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
_cors_origins: list = [o.strip() for o in _cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,        # CRITICAL: must be False with explicit origin list
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


# ---------------------------------------------------------------------------
# 6. GET /health ENDPOINT — per API-02.
#    Returns HTTP 503 with model_loaded=false before engine is ready;
#    HTTP 200 with model_loaded=true after startup completes.
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    if _engine is None or not _engine.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "ok", "model_loaded": False, "device": ""},
        )
    return HealthResponse(status="ok", model_loaded=True, device=str(_engine.device))


# ---------------------------------------------------------------------------
# 7. GET /models ENDPOINT — per API-03.
#    Returns the active model configuration strings; HTTP 503 if engine has
#    not yet loaded (e.g., checkpoint missing — graceful degradation path).
# ---------------------------------------------------------------------------
@app.get("/models", response_model=ModelsResponse)
def models():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return ModelsResponse(
        active=_engine.encoder_key,
        encoder=_engine.encoder_name,
        attention_key=_engine.attention_key,
        label_structure=_engine.label_structure,
    )


# ---------------------------------------------------------------------------
# 8. POST /predict ENDPOINT — per API-01, API-05, API-06.
#    Acquires engine.lock before calling engine.predict() so that only one
#    inference runs at a time (AC-08 concurrency serialization).
#
#    Error mapping:
#      ValueError from engine.predict() → HTTP 422 (validation failure,
#        e.g. target not a substring of text, empty inputs after NFC norm)
#      Any other exception              → HTTP 500 (unexpected inference error)
#      engine not ready                 → HTTP 503 "Model not ready"
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # Pydantic has already validated min_length=1 on req.text and req.target.
    # Additional NFC normalization and exact substring check happen inside
    # engine.predict() — see api/inference.py Step 1-2.
    if _engine is None or not _engine.ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    async with _engine.lock:
        try:
            result = _engine.predict(req.text, req.target)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictResponse(**result)


# ---------------------------------------------------------------------------
# 9. STATIC FILE GUARD — Phase 2/3 integration (zero impact in Phase 1).
#    This block MUST be registered LAST because FastAPI evaluates routes
#    top-down; a wildcard catch-all registered before /predict would intercept
#    all API calls before they reach the inference route.
#    The guard import is intentionally inside the if-block to avoid importing
#    StaticFiles/FileResponse unconditionally in a pure-API deployment.
# ---------------------------------------------------------------------------
if Path("frontend/dist").exists():
    from fastapi.staticfiles import StaticFiles  # noqa: E402 — guarded import
    from fastapi.responses import FileResponse   # noqa: F811 — intentional shadow

    _frontend_dist = Path("frontend/dist")
    app.mount(
        "/assets",
        StaticFiles(directory=str(_frontend_dist / "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        idx = _frontend_dist / "index.html"
        if idx.exists():
            return FileResponse(str(idx))
        raise HTTPException(status_code=404)
