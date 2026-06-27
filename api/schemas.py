"""Pydantic v2 request/response schemas for the Targeted ABSA inference API.

These models define the data contract between the HTTP layer (api/main.py) and
the inference engine (api/inference.py).  Field names here MUST match the keys
returned by InferenceEngine.predict() exactly.
"""

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Input body for POST /predict."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Full Vietnamese review text",
    )
    target: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Target phrase — must be an exact substring of text",
    )


class AspectProb(BaseModel):
    """A single label/probability pair used in both aspect_probs and sentiment_probs."""

    label: str = Field(..., description="Aspect label (e.g. 'ROOMS#CLEANLINESS') or sentiment string")
    score: float = Field(..., description="Softmax probability rounded to 4 decimal places")


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    aspect: str = Field(..., description="Top predicted aspect label")
    sentiment: str = Field(
        ...,
        description="Top predicted sentiment; one of POSITIVE, NEGATIVE, NEUTRAL",
    )
    aspect_probs: List[AspectProb] = Field(
        ...,
        description="Top-5 aspect predictions sorted descending by score",
    )
    sentiment_probs: List[AspectProb] = Field(
        ...,
        description="All 3 sentiment scores sorted descending by score",
    )
    latency_ms: int = Field(..., description="Inference wall time in milliseconds")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="Always 'ok'")
    model_loaded: bool = Field(..., description="True after model finishes loading at startup")
    device: str = Field(..., description="Compute device: 'cpu', 'cuda', or '' if not yet loaded")


class ModelsResponse(BaseModel):
    """Response body for GET /models — reflects the active model configuration."""

    active: str = Field(
        ...,
        description="Encoder key string, e.g. 'phobert_large' (per D-05)",
    )
    encoder: str = Field(
        ...,
        description="Resolved HuggingFace model name, e.g. 'vinai/phobert-large'",
    )
    attention_key: str = Field(
        ...,
        description="Attention mechanism key, e.g. 'target_conditioned_attention'",
    )
    label_structure: str = Field(
        ...,
        description="Label output structure, e.g. 'multitask_aspect_sentiment'",
    )
