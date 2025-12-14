#!/usr/bin/env python3
"""
vsEmbed: Simple embedding microservice.

- /embed_text: embed arbitrary text (batch).
- /embed_frames: (optional) embed video frames from vsVision-style output.
"""

from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import torch
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------

# Pick whatever local model you like; this is a solid default.
# You can swap this for a bigger one later without changing the API.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="vsEmbed",
    version="0.1.0",
    description="Embedding microservice for vs* ecosystem.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# Pydantic models
# -----------------------------

class EmbedTextRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings to embed.")
    normalize: bool = Field(
        True,
        description="If True, L2-normalize embeddings (recommended for cosine search).",
    )


class EmbedTextResponse(BaseModel):
    model: str
    dim: int
    embeddings: List[List[float]]


class FrameDetection(BaseModel):
    class_name: str
    confidence: float
    bbox: Optional[List[float]] = None


class FrameInput(BaseModel):
    frame_index: int
    timestamp_sec: float
    caption: str
    detections: List[FrameDetection] = []

class EmbedFramesRequest(BaseModel):
    """
    This matches the shape of your video analysis output (frames array),
    so you can pipe it directly here if you want.
    """
    frames: List[FrameInput]
    normalize: bool = True

class EmbedFramesResponseItem(BaseModel):
    frame_index: int
    timestamp_sec: float
    text: str
    embedding: List[float]


class EmbedFramesResponse(BaseModel):
    model: str
    dim: int
    frames: List[EmbedFramesResponseItem]


# -----------------------------
# Global model
# -----------------------------

embed_model: SentenceTransformer | None = None


@app.on_event("startup")
def load_model() -> None:
    global embed_model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embed_model.to(DEVICE)

# -----------------------------
# Helper functions
# -----------------------------

def compute_embeddings(texts: List[str], normalize: bool) -> List[List[float]]:
    assert embed_model is not None

    # sentence-transformers handles batching internally
    with torch.no_grad():
        embs = embed_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    if normalize:
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)

    return embs.cpu().tolist()


def frame_to_text(frame: FrameInput) -> str:
    """
    Convert a single FrameInput into a compact, embedding-friendly text string.
    You can tweak this template to taste.
    """
    parts: List[str] = []

    parts.append(f"[TIME {frame.timestamp_sec:.2f}s, FRAME {frame.frame_index}]")
    if frame.caption:
        parts.append(f"Caption: {frame.caption}")

    if frame.detections:
        det_strs = []
        for d in frame.detections:
            # Only include class + conf; bbox is usually not semantically important
            det_strs.append(f"{d.class_name} ({d.confidence:.2f})")
        parts.append("Detections: " + ", ".join(det_strs))

    return " | ".join(parts)

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": EMBED_MODEL_NAME,
        "device": DEVICE,
    }


@app.post("/embed_text", response_model=EmbedTextResponse)
async def embed_text(req: EmbedTextRequest) -> EmbedTextResponse:
    if not req.texts:
        return EmbedTextResponse(model=EMBED_MODEL_NAME, dim=0, embeddings=[])

    embeddings = compute_embeddings(req.texts, normalize=req.normalize)
    dim = len(embeddings[0]) if embeddings else 0

    return EmbedTextResponse(
        model=EMBED_MODEL_NAME,
        dim=dim,
        embeddings=embeddings,
    )

@app.post("/embed_frames", response_model=EmbedFramesResponse)
async def embed_frames(req: EmbedFramesRequest) -> EmbedFramesResponse:
    if not req.frames:
        return EmbedFramesResponse(model=EMBED_MODEL_NAME, dim=0, frames=[])

    # Convert each frame into a text doc
    texts: List[str] = []
    for f in req.frames:
        texts.append(frame_to_text(f))

    embeddings = compute_embeddings(texts, normalize=req.normalize)
    dim = len(embeddings[0]) if embeddings else 0

    items: List[EmbedFramesResponseItem] = []
    for frame, emb, text in zip(req.frames, embeddings, texts):
        items.append(
            EmbedFramesResponseItem(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                text=text,
                embedding=emb,
            )
        )

    return EmbedFramesResponse(
        model=EMBED_MODEL_NAME,
        dim=dim,
        frames=items,
    )

# -----------------------------
# Uvicorn entry point
# -----------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        reload=False,
    )
