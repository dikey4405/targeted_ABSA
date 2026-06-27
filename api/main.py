import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse, ModelsResponse, PredictRequest, PredictResponse
from api.inference import InferenceEngine

_engine: "InferenceEngine | None" = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = InferenceEngine()
    yield
    _engine = None


app = FastAPI(title="Targeted ABSA API", version="1.0.0", lifespan=lifespan)

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Accept"])


@app.get("/health", response_model=HealthResponse)
def health():
    if _engine is None or not _engine.ready:
        return JSONResponse(status_code=503, content={"status": "ok", "model_loaded": False, "device": ""})
    return HealthResponse(status="ok", model_loaded=True, device=str(_engine.device))


@app.get("/models", response_model=ModelsResponse)
def models():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return ModelsResponse(active=_engine.encoder_key, encoder=_engine.encoder_name, attention_key=_engine.attention_key, label_structure=_engine.label_structure)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
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


# Static files — served only when frontend/dist exists (after `npm run build`)
if Path("frontend/dist").exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        idx = Path("frontend/dist/index.html")
        return FileResponse(str(idx)) if idx.exists() else HTTPException(status_code=404)
