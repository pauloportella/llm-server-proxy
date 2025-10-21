#!/usr/bin/env python3
"""OpenAI-compatible vision server for Qwen3-VL-30B-A3B-Instruct"""

import json
import time
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Union, Optional
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread

app = FastAPI(title="Qwen3-VL Vision Server")

# Load model at startup from local path with 8-bit quantization
MODEL_PATH = "/models/qwen3-vl-30b-instruct"
print(f"Loading Qwen3-VL-30B-A3B-Instruct model from {MODEL_PATH} (8-bit quantization)...")

# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    attn_implementation="sdpa",  # PyTorch SDPA (no flash-attn needed)
    device_map="auto",
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
print("Model loaded successfully!")

class Message(BaseModel):
    model_config = ConfigDict(extra='ignore')

    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')

    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with vision support"""
    print(f"DEBUG: stream={request.stream}, type={type(request.stream)}")

    # Convert OpenAI format to Qwen3-VL format
    qwen_messages = []
    for msg in request.messages:
        qwen_msg = {"role": msg.role}

        # Handle content - can be string or list of content items
        if isinstance(msg.content, str):
            qwen_msg["content"] = [{"type": "text", "text": msg.content}]
        else:
            # Convert OpenAI image_url format to Qwen format
            content_items = []
            for item in msg.content:
                if item.get("type") == "text":
                    content_items.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    # Extract URL from OpenAI format
                    image_url = item["image_url"]
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url
                    content_items.append({"type": "image", "image": url})
            qwen_msg["content"] = content_items

        qwen_messages.append(qwen_msg)

    # Prepare inputs
    inputs = processor.apply_chat_template(
        qwen_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            stream_generator(inputs, request),
            media_type="text/event-stream"
        )

    # Non-streaming: Generate complete response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=request.temperature > 0
    )

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Return OpenAI-compatible format
    return {
        "id": "chatcmpl-qwen3vl",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(generated_ids_trimmed[0]),
            "total_tokens": inputs["input_ids"].shape[1] + len(generated_ids_trimmed[0])
        }
    }

def stream_generator(inputs, request):
    """Generate streaming response in SSE format"""
    # Create streamer
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Run generation in separate thread
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=request.temperature > 0,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream tokens as SSE events
    for text in streamer:
        chunk = {
            "id": "chatcmpl-qwen3vl",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk
    final_chunk = {
        "id": "chatcmpl-qwen3vl",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
