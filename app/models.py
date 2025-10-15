"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal


class ModelConfig(BaseModel):
    """Configuration for a single model."""
    container_name: str
    backend_url: str
    health_url: str
    startup_timeout: int = 180


class ChatMessage(BaseModel):
    """OpenAI chat message format - supports all message types."""
    model_config = ConfigDict(extra='allow')  # Allow extra fields

    role: str  # system, user, assistant, tool, function, developer
    content: Optional[Union[str, List[Dict[str, Any]]]] = None  # Can be None for tool calls
    name: Optional[str] = None  # For function/tool messages
    tool_calls: Optional[List[Dict[str, Any]]] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool response messages
    function_call: Optional[Dict[str, Any]] = None  # Deprecated but still supported

    def model_dump(self, **kwargs):
        """Override to exclude None values when serializing."""
        kwargs.setdefault('exclude_none', True)
        return super().model_dump(**kwargs)


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format - full API compatibility."""
    model_config = ConfigDict(extra='allow')  # Allow extra/unknown fields to pass through

    # Required fields
    model: str
    messages: List[ChatMessage]

    # Sampling parameters
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = None

    # Token limits
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None

    # Penalties
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

    # Streaming
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None

    # Tool/Function calling (tools is modern, functions is deprecated)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    functions: Optional[List[Dict[str, Any]]] = None  # Deprecated
    function_call: Optional[Union[str, Dict[str, Any]]] = None  # Deprecated

    # Response control
    response_format: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None

    # Metadata & control
    seed: Optional[int] = None
    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    store: Optional[bool] = None

    # Advanced features
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    prediction: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None
    modalities: Optional[List[Literal["text", "audio"]]] = None

    # Service configuration
    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = None

    # OpenAI-specific (may not apply to llama.cpp but pass through anyway)
    prompt_cache_key: Optional[str] = None
    safety_identifier: Optional[str] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    web_search_options: Optional[Dict[str, Any]] = None

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
