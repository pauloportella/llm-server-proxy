#!/usr/bin/env python3
"""Simple OpenAI-compatible embedding server for mxbai-embed-large-v1"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI(title="MXBAI Embedding Server")

# Load model at startup from HuggingFace cache
model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "mxbai-embed-large-v1"

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint"""
    # Handle both single string and list of strings
    texts = [request.input] if isinstance(request.input, str) else request.input

    # Generate embeddings
    embeddings = model.encode(texts, normalize_embeddings=True)

    # Convert to OpenAI format
    data = []
    for i, emb in enumerate(embeddings):
        data.append({
            "object": "embedding",
            "embedding": emb.tolist(),
            "index": i
        })

    return {
        "object": "list",
        "data": data,
        "model": "mxbai-embed-large-v1",
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts)
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
