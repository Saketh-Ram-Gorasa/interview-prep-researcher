"""Unit-length embeddings from the one local embedding model (Ollama /api/embed)."""

from typing import Literal

import httpx
import numpy as np

from backend import config

# nomic-embed-text expects these task prefixes; change them with EMBED_MODEL
PREFIX = {"document": "search_document: ", "query": "search_query: "}


def embed(texts: list[str], kind: Literal["document", "query"]) -> np.ndarray:
    """One row per text, normalized so a dot product is cosine similarity."""
    response = httpx.post(
        f"{config.LLM_BASE_URL}/api/embed",
        json={"model": config.EMBED_MODEL, "input": [PREFIX[kind] + t for t in texts]},
        timeout=config.RUN_BUDGET_S,
    )
    vectors = np.array(
        response.raise_for_status().json()["embeddings"], dtype=np.float32
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
