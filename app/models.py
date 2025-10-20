"""Pydantic models for request/response validation.

Note: ChatCompletionRequest uses minimal validation since LiteLLM
handles full OpenAI schema validation and forwarding.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any


class ModelConfig(BaseModel):
    """Configuration for a single backend model."""
    container_name: str
    backend_url: str
    health_url: str
    startup_timeout: int = 180


class ChatCompletionRequest(BaseModel):
    """Minimal chat completion request validation.

    LiteLLM handles full OpenAI schema validation. This class only
    validates what the proxy needs for routing decisions.
    """
    model_config = ConfigDict(extra='allow')  # Pass through all fields to LiteLLM

    # Required for routing
    model: str = Field(..., description="Model name for routing")
    messages: List[Dict[str, Any]] = Field(..., min_length=1, description="Chat messages")

    # Required for routing decision (stream vs queue)
    stream: Optional[bool] = Field(False, description="Enable streaming response")

    def model_dump(self, **kwargs):
        """Override to exclude None values when serializing."""
        kwargs.setdefault('exclude_none', True)
        return super().model_dump(**kwargs)


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
