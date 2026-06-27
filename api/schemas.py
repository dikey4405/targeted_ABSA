from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    target: str = Field(..., min_length=1, max_length=200)


class AspectProb(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    aspect: str
    sentiment: str
    aspect_probs: list[AspectProb]
    sentiment_probs: list[AspectProb]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelsResponse(BaseModel):
    active: str
    encoder: str
    attention_key: str
    label_structure: str
