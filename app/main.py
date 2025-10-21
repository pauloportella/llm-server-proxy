"""LLM Queue Proxy with Redis-based reliable queue and automatic model swapping."""

import asyncio
import time
import logging
import os
import uuid
import json
from typing import Optional
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime

import aiohttp
import docker
import httpx
import litellm
from litellm import acompletion, aembedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from langfuse import Langfuse

from .config import Config
from .redis_queue import RedisQueue
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

# Configure Langfuse observability (SDK v3 with manual instrumentation)
# LiteLLM callbacks don't work properly with Langfuse SDK v3, so we use manual tracing
# Requires environment variables: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_OTEL_HOST
# Documentation: https://langfuse.com/docs/observability/sdk/python/overview
langfuse_client = None
if all([
    os.getenv("LANGFUSE_PUBLIC_KEY"),
    os.getenv("LANGFUSE_SECRET_KEY"),
    os.getenv("LANGFUSE_OTEL_HOST")
]):
    # Set environment variables for Langfuse SDK v3
    os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_OTEL_HOST")

    # Enable debug logging for Langfuse
    import logging as stdlib_logging
    stdlib_logging.getLogger("langfuse").setLevel(stdlib_logging.DEBUG)

    # Initialize Langfuse client
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_OTEL_HOST")
    )
    logger.info(f"Langfuse observability enabled (SDK v3, host: {os.getenv('LANGFUSE_OTEL_HOST')})")
else:
    logger.info("Langfuse observability disabled (missing required environment variables)")

# Global state
config = Config()
docker_client = docker.from_env()
model_switch_lock = asyncio.Lock()  # Protects Docker operations only
metrics_lock = asyncio.Lock()       # Protects metrics counters
current_model: Optional[str] = None
num_workers = int(os.getenv("NUM_WORKERS", "1"))  # Number of parallel queue workers

# Redis queue
redis_queue: Optional[RedisQueue] = None

# Metrics
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "cancelled_requests": 0,
    "retry_count": 0,
    "active_workers": 0,
    "peak_concurrent_requests": 0,
    "start_time": datetime.utcnow().isoformat(),
}

# Track pending futures for cancellation detection
pending_futures = {}  # job_id -> asyncio.Future


async def init_redis() -> None:
    """Initialize Redis queue."""
    global redis_queue
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD", "redis_password")
        redis_db = int(os.getenv("REDIS_DB", "0"))

        redis_queue = RedisQueue(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=redis_db,
        )
        await redis_queue.connect()
        logger.info("Redis queue initialized successfully")

        # Recover any stalled jobs from previous runs
        recovered = await redis_queue.cleanup_stalled_jobs()
        if recovered > 0:
            logger.info(f"Recovered {recovered} stalled jobs from Redis")

    except Exception as e:
        logger.error(f"Failed to initialize Redis queue: {e}")
        raise


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

    model_config = config.get_model(target_model)

    # Acquire lock for Docker operations and model state changes
    async with model_switch_lock:
        # Check if target model container is already running
        # This check is now inside the lock to prevent TOCTOU race conditions
        try:
            target_container = docker_client.containers.get(model_config.container_name)
            if target_container.status == 'running':
                logger.info(f"Model {target_model} already running")
                current_model = target_model
                return
        except docker.errors.NotFound:
            pass

        # Stop ALL model containers that might be running on port 8080
        # Don't trust in-memory state - check actual Docker state
        for model_name in config.list_models():
            if model_name == target_model:
                continue
            model_cfg = config.get_model(model_name)
            try:
                container = docker_client.containers.get(model_cfg.container_name)
                if container.status == 'running':
                    logger.info(f"Stopping running model: {model_name}")
                    container.stop(timeout=10)
                    logger.info(f"Stopped container: {model_cfg.container_name}")
            except docker.errors.NotFound:
                pass
            except Exception as e:
                logger.error(f"Error checking/stopping container {model_cfg.container_name}: {e}")

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
    """Forward request to backend LLM server using LiteLLM with cancellation support."""
    try:
        # Extract model name and add openai/ prefix for LiteLLM routing
        model = request_data.get("model", "unknown")
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        # Langfuse tracing (manual instrumentation for SDK v3)
        if langfuse_client:
            with langfuse_client.start_as_current_generation(
                name="llm-completion",
                model=request_data.get("model"),  # Original model name (not openai/ prefixed)
                input=request_data.get("messages"),
                metadata={
                    "backend_url": backend_url,
                    "stream": False,
                    **{k: v for k, v in request_data.items()
                       if k not in ["model", "messages", "stream"]}
                }
            ) as generation:
                # Use LiteLLM for standardized OpenAI-compatible forwarding
                response = await acompletion(
                    model=model,
                    messages=request_data.get("messages", []),
                    api_base=backend_url,
                    api_key="dummy-key",  # Required by LiteLLM but not validated by llama-server
                    stream=False,
                    timeout=config.request_timeout,
                    # Pass through all other OpenAI parameters
                    **{k: v for k, v in request_data.items()
                       if k not in ["model", "messages", "stream"]}
                )

                # Update generation with output and usage
                generation.update(
                    output=response.choices[0].message.content if response.choices else None
                )
                if response.usage:
                    generation.update(usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": response.usage.total_tokens
                    })
                # Flush to send trace immediately
                langfuse_client.flush()
        else:
            # No Langfuse tracing - direct LiteLLM call
            response = await acompletion(
                model=model,
                messages=request_data.get("messages", []),
                api_base=backend_url,
                api_key="dummy-key",  # Required by LiteLLM but not validated by llama-server
                stream=False,
                timeout=config.request_timeout,
                # Pass through all other OpenAI parameters
                **{k: v for k, v in request_data.items()
                   if k not in ["model", "messages", "stream"]}
            )

        # Convert ModelResponse to dict for consistency
        return response.model_dump() if hasattr(response, 'model_dump') else dict(response)

    except asyncio.CancelledError:
        logger.info("Request forwarding cancelled")
        raise
    except Exception as e:
        logger.error(f"LiteLLM forward error: {e}")
        # Re-raise as HTTPException for FastAPI
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Backend error: {str(e)}"
        )


async def forward_embedding(backend_url: str, request_data: dict) -> dict:
    """Forward embedding request to backend LLM server using LiteLLM with Langfuse tracing."""
    try:
        # Extract model name and add openai/ prefix for LiteLLM routing
        model = request_data.get("model", "unknown")
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        # Langfuse tracing (manual instrumentation for SDK v3)
        if langfuse_client:
            with langfuse_client.start_as_current_generation(
                name="llm-embedding",
                model=request_data.get("model"),  # Original model name (not openai/ prefixed)
                input=request_data.get("input"),
                metadata={
                    "backend_url": backend_url,
                    **{k: v for k, v in request_data.items()
                       if k not in ["model", "input"]}
                }
            ) as generation:
                # Use LiteLLM for standardized OpenAI-compatible embedding
                response = await aembedding(
                    model=model,
                    input=request_data.get("input", []),
                    api_base=backend_url,
                    api_key="dummy-key",  # Required by LiteLLM but not validated by backend
                    timeout=config.request_timeout,
                    # Pass through all other OpenAI parameters
                    **{k: v for k, v in request_data.items()
                       if k not in ["model", "input"]}
                )

                # Update generation with output (embedding count and dimensions)
                if response.data:
                    generation.update(
                        output=f"{len(response.data)} embeddings generated"
                    )
                if response.usage:
                    generation.update(usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": 0,  # Embeddings don't have output tokens
                        "total": response.usage.total_tokens
                    })
                # Flush to send trace immediately
                langfuse_client.flush()
        else:
            # No Langfuse tracing - direct LiteLLM call
            response = await aembedding(
                model=model,
                input=request_data.get("input", []),
                api_base=backend_url,
                api_key="dummy-key",
                timeout=config.request_timeout,
                # Pass through all other OpenAI parameters
                **{k: v for k, v in request_data.items()
                   if k not in ["model", "input"]}
            )

        # Convert EmbeddingResponse to dict for consistency
        return response.model_dump() if hasattr(response, 'model_dump') else dict(response)

    except asyncio.CancelledError:
        logger.info("Embedding request cancelled")
        raise
    except Exception as e:
        logger.error(f"LiteLLM embedding error: {e}")
        # Re-raise as HTTPException for FastAPI
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=500,
            detail=f"Backend embedding error: {str(e)}"
        )


async def stream_response(backend_url: str, request_data: dict):
    """Stream response from backend using LiteLLM."""
    try:
        # Extract model name and add openai/ prefix for LiteLLM routing
        model = request_data.get("model", "unknown")
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        # Langfuse tracing for streaming (manual instrumentation for SDK v3)
        # Note: Can't use context manager with streaming, so we manually start/end
        generation = None
        accumulated_content = []
        total_tokens = {"input": 0, "output": 0, "total": 0}

        if langfuse_client:
            # Start generation without context manager (manual lifecycle management)
            generation = langfuse_client.start_generation(
                name="llm-completion-stream",
                model=request_data.get("model"),  # Original model name
                input=request_data.get("messages"),
                metadata={
                    "backend_url": backend_url,
                    "stream": True,
                    **{k: v for k, v in request_data.items()
                       if k not in ["model", "messages", "stream"]}
                }
            )

        try:
            # Use LiteLLM for streaming
            response = await acompletion(
                model=model,
                messages=request_data.get("messages", []),
                api_base=backend_url,
                api_key="dummy-key",
                stream=True,
                stream_timeout=config.request_timeout,
                # Pass through all other OpenAI parameters
                **{k: v for k, v in request_data.items()
                   if k not in ["model", "messages", "stream"]}
            )

            # Stream chunks in SSE format
            async for chunk in response:
                if chunk:
                    # LiteLLM returns ModelResponse chunks, convert to JSON bytes
                    chunk_dict = chunk.model_dump() if hasattr(chunk, 'model_dump') else dict(chunk)

                    # Accumulate content for Langfuse
                    if generation and chunk_dict.get("choices"):
                        for choice in chunk_dict["choices"]:
                            delta = choice.get("delta", {})
                            if "content" in delta and delta["content"]:
                                accumulated_content.append(delta["content"])

                    # Track usage stats
                    if chunk_dict.get("usage"):
                        usage = chunk_dict["usage"]
                        total_tokens["input"] = usage.get("prompt_tokens", 0)
                        total_tokens["output"] = usage.get("completion_tokens", 0)
                        total_tokens["total"] = usage.get("total_tokens", 0)

                    # Format as SSE event
                    yield f"data: {json.dumps(chunk_dict)}\n\n".encode('utf-8')

            # Send completion signal
            yield b"data: [DONE]\n\n"

        finally:
            # Update and end Langfuse generation
            if generation:
                # Update with output
                if accumulated_content:
                    generation.update(output="".join(accumulated_content))
                # Update with usage
                if total_tokens["total"] > 0:
                    generation.update(usage_details={
                        "input": total_tokens["input"],
                        "output": total_tokens["output"],
                        "total": total_tokens["total"]
                    })
                # End the generation
                generation.end()
                # Flush to send trace immediately
                langfuse_client.flush()

    except asyncio.CancelledError:
        logger.info("Streaming cancelled")
        raise
    except Exception as e:
        logger.error(f"LiteLLM streaming error: {e}")
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')


async def enqueue_and_wait(
    job_id: str,
    model: str,
    request_data: dict,
    request: Request,
    job_type: str = "completion"
) -> dict:
    """
    Unified queue logic for all endpoints.

    Enqueues job to Redis, waits for worker to process it, handles client
    disconnection and timeouts. Returns result dict or raises HTTPException.
    """
    if not redis_queue:
        raise HTTPException(
            status_code=503,
            detail="Queue service not available"
        )

    # Increment total requests metric
    async with metrics_lock:
        metrics["total_requests"] += 1

    # Create future for result
    request_future = asyncio.Future()
    pending_futures[job_id] = request_future

    try:
        # Enqueue job
        await redis_queue.enqueue(
            job_id=job_id,
            model=model,
            request_data=request_data,
            job_type=job_type
        )
        logger.info(f"Enqueued {job_type} job {job_id} for model {model}")

        # Create disconnection checker task
        async def check_disconnection():
            """Periodically check if client disconnected."""
            while not request_future.done():
                if await request.is_disconnected():
                    logger.info(f"Client disconnected for job {job_id}")
                    if not request_future.done():
                        request_future.cancel()
                        logger.info(f"Job {job_id} was cancelled by client")
                    break
                # Check more frequently (0.1s) to catch n8n workflow cancellations faster
                await asyncio.sleep(0.1)

        disconnection_task = asyncio.create_task(check_disconnection())

        try:
            # Wait for result with timeout
            result = await asyncio.wait_for(
                request_future,
                timeout=config.request_timeout + 300  # Extra buffer for model switching
            )
            return result
        finally:
            # Cancel disconnection checker
            disconnection_task.cancel()
            try:
                await disconnection_task
            except asyncio.CancelledError:
                pass

    except asyncio.TimeoutError:
        logger.warning(f"Job {job_id} timed out")
        async with metrics_lock:
            metrics["failed_requests"] += 1
        # Clean up future (won't be processed by worker)
        pending_futures.pop(job_id, None)
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {config.request_timeout + 300}s"
        )

    except asyncio.CancelledError:
        # Already logged by disconnection checker
        async with metrics_lock:
            metrics["cancelled_requests"] += 1
        # DON'T delete future here - let worker detect cancellation and clean up
        raise HTTPException(
            status_code=499,
            detail="Request cancelled"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}")
        async with metrics_lock:
            metrics["failed_requests"] += 1
        # Clean up future if error happened before worker processed it
        pending_futures.pop(job_id, None)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


async def queue_worker(worker_id: int = 0):
    """Process requests from Redis queue."""
    if not redis_queue:
        logger.error("Redis queue not initialized")
        return

    logger.info(f"Queue worker {worker_id} started")
    consecutive_errors = 0

    while True:
        try:
            # Dequeue job with blocking wait (1s timeout per iteration)
            result = await redis_queue.dequeue_with_lock()

            if result is None:
                # Queue is empty, continue polling
                await asyncio.sleep(0.1)
                consecutive_errors = 0
                continue

            job_id, job_dict = result
            model = job_dict["model"]
            request_data = job_dict["request_data"]
            job_type = job_dict.get("job_type", "completion")  # Default to completion for backwards compatibility

            logger.info(f"[Worker {worker_id}] Processing {job_type} job {job_id} for model {model}")

            # Track active worker
            async with metrics_lock:
                metrics["active_workers"] += 1
                metrics["peak_concurrent_requests"] = max(
                    metrics["peak_concurrent_requests"],
                    metrics["active_workers"]
                )

            try:
                # Check if client cancelled before heavy work
                if job_id in pending_futures:
                    future = pending_futures[job_id]
                    if future.cancelled():
                        logger.info(f"[Worker {worker_id}] Job {job_id} was cancelled by client, skipping")
                        async with metrics_lock:
                            metrics["cancelled_requests"] += 1
                        await redis_queue.acknowledge_job(job_id)
                        continue

                # Switch to requested model (lock acquired inside if needed)
                await switch_model(model)

                # Check cancellation again after model switch
                if job_id in pending_futures:
                    future = pending_futures[job_id]
                    if future.cancelled():
                        logger.info(f"[Worker {worker_id}] Job {job_id} cancelled after model switch")
                        async with metrics_lock:
                            metrics["cancelled_requests"] += 1
                        await redis_queue.acknowledge_job(job_id)
                        continue

                # Check cancellation one more time before forwarding
                if job_id in pending_futures:
                    future = pending_futures[job_id]
                    if future.cancelled():
                        logger.info(f"[Worker {worker_id}] Job {job_id} cancelled before forwarding")
                        async with metrics_lock:
                            metrics["cancelled_requests"] += 1
                        await redis_queue.acknowledge_job(job_id)
                        continue

                # Forward request (no lock - llama-server handles concurrency)
                # Wrap in task so we can cancel it if client disconnects
                model_config = config.get_model(model)

                # Dispatch based on job type
                if job_type == "embedding":
                    forward_task = asyncio.create_task(
                        forward_embedding(model_config.backend_url, request_data)
                    )
                else:  # Default to completion
                    forward_task = asyncio.create_task(
                        forward_request(model_config.backend_url, request_data)
                    )

                # Poll for cancellation while waiting for backend (check every 0.1s for faster response)
                while not forward_task.done():
                    # Check if client cancelled
                    if job_id in pending_futures and pending_futures[job_id].cancelled():
                        logger.info(f"[Worker {worker_id}] Cancelling backend request for job {job_id}")
                        forward_task.cancel()
                        # Let CancelledError propagate from awaiting the task
                        await forward_task
                        # If we reach here without exception, something is wrong
                        raise RuntimeError("Task cancellation did not raise CancelledError")

                    # Check more frequently (0.1s instead of 0.5s)
                    await asyncio.sleep(0.1)

                # Get result from completed task
                result = await forward_task

                # Deliver result if future still pending
                if job_id in pending_futures:
                    future = pending_futures[job_id]
                    if not future.cancelled() and not future.done():
                        future.set_result(result)

                # Acknowledge successful processing
                await redis_queue.acknowledge_job(job_id)
                async with metrics_lock:
                    metrics["successful_requests"] += 1
                consecutive_errors = 0
                logger.info(f"[Worker {worker_id}] Job {job_id} completed successfully")

            except asyncio.CancelledError:
                logger.info(f"[Worker {worker_id}] Job {job_id} processing was cancelled")
                async with metrics_lock:
                    metrics["cancelled_requests"] += 1
                # Acknowledge cancelled job (don't retry - client disconnected intentionally)
                await redis_queue.acknowledge_job(job_id)
                # Don't re-raise - job is handled
                consecutive_errors = 0

            except Exception as e:
                logger.error(f"[Worker {worker_id}] Error processing job {job_id}: {e}")
                async with metrics_lock:
                    metrics["failed_requests"] += 1

                # Set exception on future if still pending
                if job_id in pending_futures:
                    future = pending_futures[job_id]
                    if not future.cancelled() and not future.done():
                        future.set_exception(e)

                # NACK job for retry
                await redis_queue.nack_job(job_id, str(e))

            finally:
                # Clean up future reference
                if job_id in pending_futures:
                    del pending_futures[job_id]

                # Decrement active workers
                async with metrics_lock:
                    metrics["active_workers"] -= 1

        except Exception as e:
            logger.error(f"Unexpected error in queue worker: {e}")
            consecutive_errors += 1
            if consecutive_errors > 5:
                logger.critical("Queue worker encountered too many errors, restarting...")
                await asyncio.sleep(5)
                consecutive_errors = 0
            else:
                await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # Initialize Redis
    await init_redis()

    # Start multiple queue workers
    worker_tasks = [
        asyncio.create_task(queue_worker(worker_id=i))
        for i in range(num_workers)
    ]
    logger.info(f"LLM Queue Proxy started with Redis backend and {num_workers} workers")

    yield

    # Shutdown all workers
    for task in worker_tasks:
        task.cancel()

    # Wait for all workers to finish
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    logger.info("All workers shut down")

    # Flush Langfuse traces before shutdown
    if langfuse_client:
        try:
            langfuse_client.flush()
            logger.info("Flushed Langfuse traces")
        except Exception as e:
            logger.error(f"Error flushing Langfuse traces: {e}")

    if redis_queue:
        await redis_queue.disconnect()
    logger.info("LLM Queue Proxy shutting down")


app = FastAPI(
    title="LLM Queue Proxy",
    description="OpenAI-compatible proxy with automatic model swapping, Redis queue, and LiteLLM integration",
    version="2.1.0-litellm",
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

    # Sort models alphabetically by id
    models.sort(key=lambda m: m.id)

    return ModelListResponse(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatCompletionRequest, request: Request):
    """Handle chat completion with queueing and model swapping."""
    model = chat_request.model

    # Extract metadata headers for batch/workflow tracking
    # Common headers: X-Request-ID, X-Correlation-ID, X-Workflow-ID, User-Agent
    batch_id = request.headers.get("x-batch-id") or request.headers.get("x-correlation-id") or request.headers.get("x-request-id")
    user_agent = request.headers.get("user-agent", "unknown")
    client_ip = request.client.host if request.client else "unknown"

    # Log metadata for debugging (only first time we see these headers)
    logger.info(f"Request metadata: batch_id={batch_id}, user_agent={user_agent}, client={client_ip}")

    # Validate model exists
    if model not in config.list_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {config.list_models()}"
        )

    # Validate model type is compatible with endpoint
    model_config = config.get_model(model)
    if model_config.type == "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is an embedding model. Use the /v1/embeddings endpoint instead."
        )

    if not redis_queue:
        raise HTTPException(
            status_code=503,
            detail="Queue service not available"
        )

    # Handle streaming requests directly
    if chat_request.stream:
        logger.info(f"Streaming request for model: {model}")

        # Switch to requested model (lock acquired inside if needed)
        await switch_model(model)

        # Get model config and stream response
        model_config = config.get_model(model)
        return StreamingResponse(
            stream_response(model_config.backend_url, chat_request.model_dump()),
            media_type="text/event-stream"
        )

    # Non-streaming: use Redis queue with unified enqueue_and_wait helper
    return await enqueue_and_wait(
        job_id=str(uuid.uuid4()),
        model=model,
        request_data=chat_request.model_dump(),
        request=request,
        job_type="completion"
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint via LiteLLM with queueing."""
    request_body = await request.json()
    model = request_body.get("model", "unknown")

    # Extract metadata headers for batch/workflow tracking
    batch_id = request.headers.get("x-batch-id") or request.headers.get("x-correlation-id") or request.headers.get("x-request-id")
    user_agent = request.headers.get("user-agent", "unknown")
    client_ip = request.client.host if request.client else "unknown"

    # Log metadata for debugging
    logger.info(f"Embedding request metadata: batch_id={batch_id}, user_agent={user_agent}, client={client_ip}")

    # Validate model exists
    if model not in config.list_models():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {config.list_models()}"
        )

    # Validate model type is compatible with endpoint
    model_config = config.get_model(model)
    if model_config.type != "embedding":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is a completion model. Use the /v1/chat/completions endpoint instead."
        )

    # Use Redis queue with unified enqueue_and_wait helper
    return await enqueue_and_wait(
        job_id=str(uuid.uuid4()),
        model=model,
        request_data=request_body,
        request=request,
        job_type="embedding"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    queue_stats = {}
    if redis_queue:
        queue_stats = await redis_queue.get_queue_stats()

    return HealthResponse(
        status="healthy",
        current_model=current_model,
        queue_size=queue_stats.get("pending", 0),
        queue_capacity=config.queue_size,
        message="Proxy is operational"
    )


@app.get("/metrics")
async def metrics_endpoint():
    """Get performance metrics."""
    if not redis_queue:
        raise HTTPException(status_code=503, detail="Queue not ready")

    queue_stats = await redis_queue.get_queue_stats()
    uptime = (datetime.utcnow() - datetime.fromisoformat(metrics["start_time"])).total_seconds()

    async with metrics_lock:
        return {
            "uptime_seconds": uptime,
            "total_requests": metrics["total_requests"],
            "successful_requests": metrics["successful_requests"],
            "failed_requests": metrics["failed_requests"],
            "cancelled_requests": metrics["cancelled_requests"],
            "active_workers": metrics["active_workers"],
            "peak_concurrent_requests": metrics["peak_concurrent_requests"],
            "num_workers": num_workers,
            "success_rate": (
                metrics["successful_requests"] / metrics["total_requests"]
                if metrics["total_requests"] > 0 else 0
            ),
            "queue_stats": {
                "pending_jobs": queue_stats.get("pending", 0),
                "processing_jobs": queue_stats.get("processing", 0),
                "dead_letter_queue": queue_stats.get("dead_letter_queue", 0),
                "total_jobs": queue_stats.get("total", 0),
            },
            "current_model": current_model,
            "pending_futures": len(pending_futures),
        }


@app.get("/queue/dlq")
async def get_dlq():
    """Get dead-letter queue jobs for inspection."""
    if not redis_queue:
        raise HTTPException(status_code=503, detail="Queue not ready")

    dlq_jobs = await redis_queue.get_dlq_jobs(limit=50)
    return {
        "count": len(dlq_jobs),
        "jobs": dlq_jobs
    }


@app.delete("/queue/dlq")
async def clear_dlq():
    """Clear dead-letter queue (admin operation)."""
    if not redis_queue:
        raise HTTPException(status_code=503, detail="Queue not ready")

    cleared = await redis_queue.clear_dlq()
    return {
        "message": f"Cleared {cleared} jobs from DLQ",
        "count": cleared
    }


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
