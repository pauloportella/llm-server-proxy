"""LLM Queue Proxy with automatic model swapping."""

import asyncio
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

import aiohttp
import docker
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Config
from .models import (
    ChatCompletionRequest,
    ModelListResponse,
    ModelInfo,
    HealthResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
config = Config()
docker_client = docker.from_env()
request_queue = asyncio.Queue(maxsize=config.queue_size)
processing_lock = asyncio.Lock()
current_model: Optional[str] = None


async def wait_for_health(health_url: str, timeout: int = 180) -> bool:
    """Poll health endpoint until ready or timeout."""
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            try:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        logger.info(f"Model is healthy: {health_url}")
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            await asyncio.sleep(2)

    return False


async def switch_model(target_model: str) -> None:
    """Switch to target model by stopping current and starting new."""
    global current_model

    if current_model == target_model:
        logger.info(f"Model {target_model} already loaded")
        return

    model_config = config.get_model(target_model)

    # Stop current model if any
    if current_model:
        current_config = config.get_model(current_model)
        logger.info(f"Stopping model: {current_model}")
        try:
            container = docker_client.containers.get(current_config.container_name)
            container.stop(timeout=10)
            logger.info(f"Stopped container: {current_config.container_name}")
        except docker.errors.NotFound:
            logger.warning(f"Container not found: {current_config.container_name}")
        except Exception as e:
            logger.error(f"Error stopping container: {e}")

    # Start target model
    logger.info(f"Starting model: {target_model}")
    try:
        container = docker_client.containers.get(model_config.container_name)
        container.start()
        logger.info(f"Started container: {model_config.container_name}")
    except docker.errors.NotFound:
        raise HTTPException(
            status_code=500,
            detail=f"Container not found: {model_config.container_name}. Please create it first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting container: {e}")

    # Wait for model to be ready
    logger.info(f"Waiting for model to be ready (timeout: {model_config.startup_timeout}s)...")
    is_ready = await wait_for_health(model_config.health_url, model_config.startup_timeout)

    if not is_ready:
        raise HTTPException(
            status_code=504,
            detail=f"Model {target_model} did not become ready within {model_config.startup_timeout}s"
        )

    current_model = target_model
    logger.info(f"Model {target_model} is now active")


async def forward_request(backend_url: str, request_data: dict) -> dict:
    """Forward request to backend LLM server."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{backend_url}/chat/completions",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=config.request_timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Backend error: {error_text}"
                )
            return await response.json()


async def queue_worker():
    """Process requests from queue one at a time."""
    while True:
        request_future, model, request_data = await request_queue.get()

        try:
            async with processing_lock:
                # Switch to requested model
                await switch_model(model)

                # Forward request
                model_config = config.get_model(model)
                result = await forward_request(model_config.backend_url, request_data)

                request_future.set_result(result)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            request_future.set_exception(e)
        finally:
            request_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Start queue worker
    worker_task = asyncio.create_task(queue_worker())
    logger.info("LLM Queue Proxy started")

    yield

    # Shutdown
    worker_task.cancel()
    logger.info("LLM Queue Proxy shutting down")


app = FastAPI(
    title="LLM Queue Proxy",
    description="OpenAI-compatible proxy with automatic model swapping",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """List available models (OpenAI compatible)."""
    models = []
    for model_name in config.list_models():
        models.append(ModelInfo(
            id=model_name,
            created=int(time.time()),
            owned_by="local"
        ))

    return ModelListResponse(data=models)


async def stream_response(backend_url: str, request_data: dict):
    """Stream response from backend."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{backend_url}/chat/completions",
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=config.request_timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Backend error: {error_text}"
                )

            # Stream the response chunks
            async for chunk in response.content.iter_any():
                if chunk:
                    yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion with queueing and model swapping."""
    model = request.model

    # Validate model
    if model not in config.list_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {config.list_models()}"
        )

    # Handle streaming requests differently (direct processing with lock)
    if request.stream:
        logger.info(f"Streaming request for model: {model}")

        async with processing_lock:
            # Switch to requested model
            await switch_model(model)

            # Get model config and stream response
            model_config = config.get_model(model)
            return StreamingResponse(
                stream_response(model_config.backend_url, request.model_dump()),
                media_type="text/event-stream"
            )

    # Non-streaming: use queue pattern
    # Check queue capacity
    if request_queue.full():
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({request_queue.maxsize} requests). Try again later."
        )

    # Create future for result
    request_future = asyncio.Future()

    # Add to queue
    await request_queue.put((request_future, model, request.model_dump()))

    logger.info(f"Queued request for model: {model} (queue size: {request_queue.qsize()})")

    # Wait for result
    try:
        result = await asyncio.wait_for(request_future, timeout=config.request_timeout + 300)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {config.request_timeout + 300}s"
        )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        current_model=current_model,
        queue_size=request_queue.qsize(),
        queue_capacity=request_queue.maxsize,
        message="Proxy is operational"
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
