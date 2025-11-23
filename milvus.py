#!/usr/bin/env python3
"""
vsMilvus API - simple FastAPI wrapper around Milvus Lite (non-Docker).

Endpoints:
- POST /createCollection
- POST /insertEntry
- POST /searchEntry
"""

import os
from pathlib import Path
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pymilvus import MilvusClient


# -----------------------------
# Config
# -----------------------------

# Path to Milvus Lite DB file (will be created if it doesn't exist)
MILVUS_DB_DIR = Path(os.getenv("MILVUS_DB_DIR", "/var/www/db")).resolve()
# Default DB name (used for /health and startup)
DEFAULT_DB_NAME = os.getenv("MILVUS_DB_NAME", "milvus")

# Convenience path string for the default DB (for info/debug only)
MILVUS_DB_PATH = str(MILVUS_DB_DIR / f"{DEFAULT_DB_NAME}.db")

# Default search limit
DEFAULT_TOP_K = 5


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="vsMilvus API",
    version="0.1.0",
    description="FastAPI wrapper around Milvus Lite (non-Docker) for quick testing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Global Milvus client
# -----------------------------

clients: Dict[str, MilvusClient] = {}
#client: MilvusClient | None = None


@app.on_event("startup")
def startup_event() -> None:
    # Milvus Lite: using a local file as URI
    get_client(DEFAULT_DB_NAME)  # default DB

@app.on_event("shutdown")
def shutdown_event() -> None:
    """
    Optional: cleanly close all clients.
    """
    for name, client in list(clients.items()):
        try:
            client.close()
        except Exception:
            pass
    clients.clear()

# -----------------------------
# Pydantic models
# -----------------------------

class CreateCollectionRequest(BaseModel):
    db: str = Field(DEFAULT_DB_NAME, description="Database Name")
    collection_name: str = Field(..., description="Name of the collection to create.")
    dimension: int = Field(..., gt=0, description="Vector dimension (e.g., 384 for MiniLM).")


class CreateCollectionResponse(BaseModel):
    collection_name: str
    dimension: int
    status: str


class InsertEntryRequest(BaseModel):
    db: str = Field(DEFAULT_DB_NAME, description="Database Name")
    collection_name: str = Field(..., description="Target collection name.")
    id: Optional[int] = Field(
        None,
        description="Optional primary key ID. If omitted, Milvus can auto-generate IDs (if auto_id enabled).",
    )
    vector: List[float] = Field(..., description="Embedding/vector to store.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional scalar fields to store with this entry.",
    )


class InsertEntryResponse(BaseModel):
    ids: List[Any]
    collection_name: str

class UpsertEntryRequest(BaseModel):
    db: str = Field(DEFAULT_DB_NAME, description="Database Name")
    collection_name: str
    id: int = Field(..., description="Primary key to upsert.")
    vector: List[float] = Field(..., description="Embedding/vector to store.")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional additional scalar fields to store with this entry.",
    )


class UpsertEntryResponse(BaseModel):
    upsert_count: int
    collection_name: str


class SearchEntryRequest(BaseModel):
    db: str = Field(DEFAULT_DB_NAME, description="Database Name")
    collection_name: str = Field(..., description="Collection to search in.")
    vector: List[float] = Field(..., description="Query vector.")
    top_k: int = Field(DEFAULT_TOP_K, gt=0, le=1000, description="Number of nearest neighbors to return.")
    filter: Optional[str] = Field(
        default=None,
        description="Optional filter expression (Milvus filter syntax).",
    )
    output_fields: Optional[List[str]] = Field(
        default=None,
        description="Optional list of scalar fields to return for each hit.",
    )


class SearchEntryResponse(BaseModel):
    collection_name: str
    top_k: int
    results: List[Dict[str, Any]]

class EntryExistsRequest(BaseModel):
    db: str = Field(DEFAULT_DB_NAME, description="Database Name")
    collection_name: str
    filter: str = Field(
        ..., 
        description="Milvus filter expression, e.g. \"id == 123\" or \"video_id == 'abc'\""
    )


class EntryExistsResponse(BaseModel):
    exists: bool
    count: int
    matching_ids: List[Any]
    collection_name: str

# -----------------------------
# Helpers
# -----------------------------

def get_client(name: str = "milvus") -> MilvusClient:
    """
    Lazily get/create a Milvus Lite client for a given DB name.

    - Uses /var/www/db/{name}.db by default.
    - Reuses the same MilvusClient instance per process.
    """
    # If already opened, just return it
    existing = clients.get(name)
    if existing is not None:
        return existing

    # Build file path: /var/www/db/{name}.db
    MILVUS_DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = MILVUS_DB_DIR / f"{name}.db"

    # Milvus Lite will create the file if it doesn't exist
    client = MilvusClient(str(db_path))
    clients[name] = client
    return client

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    """
    Basic health check. Uses the default DB.
    """
    c = get_client(DEFAULT_DB_NAME)
    return {
        "status": "ok",
        "default_db_name": DEFAULT_DB_NAME,
        "default_db_path": MILVUS_DB_PATH,
        "collections": c.list_collections(),
        "loaded_dbs": list(clients.keys()),
    }


@app.post("/createCollection", response_model=CreateCollectionResponse)
async def create_collection(req: CreateCollectionRequest) -> CreateCollectionResponse:
    c = get_client(name = req.db)
    if c.has_collection(collection_name=req.collection_name):
        # For testing, let's just be explicit instead of dropping.
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{req.collection_name}' already exists.",
        )

    # Basic collection with default schema & autoindex; you only need name + dimension.
    c.create_collection(
        collection_name=req.collection_name,
        dimension=req.dimension,
    )

    return CreateCollectionResponse(
        collection_name=req.collection_name,
        dimension=req.dimension,
        status="created",
    )


@app.post("/insertEntry", response_model=InsertEntryResponse)
async def insert_entry(req: InsertEntryRequest) -> InsertEntryResponse:
    c = get_client(name = req.db)

    if not c.has_collection(collection_name=req.collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection_name}' does not exist.",
        )

    entry: Dict[str, Any] = {
        "vector": req.vector,
    }

    # If you want to explicitly set an ID; otherwise Milvus can auto-id if the collection is created that way.
    if req.id is not None:
        entry["id"] = req.id

    # Attach metadata as dynamic fields (MilvusClient supports schema-less JSON-style fields)
    if req.metadata:
        entry.update(req.metadata)

    try:
        res = c.insert(
            collection_name=req.collection_name,
            data=[entry],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {e}")

    # MilvusClient.insert returns something like {"insert_count": ..., "ids": [...]}
    ids = res.get("ids", [])
    return InsertEntryResponse(
        ids=ids,
        collection_name=req.collection_name,
    )


@app.post("/searchEntry", response_model=SearchEntryResponse)
async def search_entry(req: SearchEntryRequest) -> SearchEntryResponse:
    c = get_client(name = req.db)

    if not c.has_collection(collection_name=req.collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection_name}' does not exist.",
        )

    try:
        results = c.search(
            collection_name=req.collection_name,
            data=[req.vector],          # single vector search
            limit=req.top_k,
            filter=req.filter,
            output_fields=req.output_fields,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    # MilvusClient.search returns a list of dicts, one per query vector
    # We passed only 1 query vector, so results[0] is the list of hits.
    hits = results[0] if results else []

    return SearchEntryResponse(
        collection_name=req.collection_name,
        top_k=req.top_k,
        results=hits,
    )


@app.post("/entryExists", response_model=EntryExistsResponse)
async def entry_exists(req: EntryExistsRequest) -> EntryExistsResponse:
    c = get_client(name = req.db)

    # Verify collection exists
    if not c.has_collection(req.collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection_name}' does not exist.",
        )

    try:
        # Query returns a list of dicts matching the filter
        results = c.query(
            collection_name=req.collection_name,
            filter=req.filter,
            output_fields=["id"],     # only fetch ID for existence check
            limit=1000                # safety cap; existence checks are tiny anyway
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    exists = len(results) > 0
    ids = [r["id"] for r in results] if exists else []

    return EntryExistsResponse(
        exists=exists,
        count=len(ids),
        matching_ids=ids,
        collection_name=req.collection_name,
    )

@app.post("/upsertEntry", response_model=UpsertEntryResponse)
async def upsert_entry(req: UpsertEntryRequest) -> UpsertEntryResponse:
    c = get_client(name = req.db)

    if not c.has_collection(collection_name=req.collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection_name}' does not exist.",
        )

    entry: Dict[str, Any] = {
        "id": req.id,
        "vector": req.vector,
    }

    if req.metadata:
        entry.update(req.metadata)

    try:
        res = c.upsert(
            collection_name=req.collection_name,
            data=[entry],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upsert failed: {e}")

    # MilvusClient.upsert returns something like {"upsert_count": N}
    upsert_count = res.get("upsert_count", 0)

    return UpsertEntryResponse(
        upsert_count=upsert_count,
        collection_name=req.collection_name,
    )


# -----------------------------
# Uvicorn entrypoint
# -----------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8300,
        reload=False,
    )
