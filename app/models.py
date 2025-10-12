"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    container_name: str
    backend_url: str
    health_url: str
    startup_timeout: int = 180


class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0


class ModelInfo(BaseModel):
    """OpenAI model info format."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    """OpenAI /v1/models response format."""
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    current_model: Optional[str] = None
    queue_size: int
    queue_capacity: int
    message: Optional[str] = None
