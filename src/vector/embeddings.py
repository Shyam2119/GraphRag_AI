"""Embedding function cascade: Gemini API → Sentence-Transformers → ONNX → Hash.

Selection logic (controlled by EMBEDDING_PROVIDER setting):
- "auto": tries each in order, uses the first that works
- "gemini": uses Gemini text-embedding-004 (requires API key)
- "sentence-transformers": uses all-MiniLM-L6-v2 locally (~90MB)
- "hash": deterministic hash embeddings (fast but low quality, demo-only)
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

import numpy as np

from config import get_settings


class HashEmbeddingFunction:
    """Deterministic bag-of-words hash embeddings — no ML dependencies.

    This is a last-resort fallback. Quality is low but it works everywhere
    and produces deterministic results for reproducible demos.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def name(self) -> str:
        return "hash-embedding"

    def __call__(self, input: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())

        # Use bigrams as well for slightly better semantic capture
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        all_tokens = tokens + bigrams

        for token in all_tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 128) % 2 == 0 else -1.0  # Random sign for better distribution
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()


class SentenceTransformerEmbedding:
    """Local sentence-transformers embeddings (all-MiniLM-L6-v2).

    High-quality semantic embeddings without any API key.
    Downloads ~90MB model on first use, then cached locally.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._name = model_name

    def name(self) -> str:
        return f"sentence-transformers/{self._name}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()


class GeminiEmbeddingFunction:
    """Gemini text-embedding-004 via Google GenAI SDK.

    Highest quality embeddings, requires API key.
    """

    def __init__(self, api_key: str) -> None:
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self._model = "text-embedding-004"

    def name(self) -> str:
        return f"gemini/{self._model}"

    def __call__(self, input: List[str]) -> List[List[float]]:
        from google.genai import types

        # Gemini embedding API supports batch requests
        results = []
        batch_size = 20  # API limit
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            response = self.client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            results.extend([e.values for e in response.embeddings])
        return results


def get_embedding_function():
    """Select the best available embedding function.

    Cascade: Gemini API → Sentence-Transformers → ChromaDB ONNX → Hash fallback.
    """
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    # Explicit provider selection
    if provider == "gemini":
        return _try_gemini(settings.gemini_api_key) or _fallback()
    if provider == "sentence-transformers":
        return _try_sentence_transformers() or _fallback()
    if provider == "hash":
        print("  [INFO] Using hash embeddings (demo-only quality)")
        return HashEmbeddingFunction()

    # Auto mode: try each in order
    if provider == "auto":
        # 1. Try Gemini
        if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key":
            ef = _try_gemini(settings.gemini_api_key)
            if ef:
                return ef

        # 2. Try Sentence-Transformers
        ef = _try_sentence_transformers()
        if ef:
            return ef

        # 3. Try ChromaDB ONNX
        ef = _try_chromadb_onnx()
        if ef:
            return ef

        # 4. Hash fallback
        return _fallback()

    # Unknown provider, try auto
    return get_embedding_function.__wrapped__() if hasattr(get_embedding_function, '__wrapped__') else _fallback()


def _try_gemini(api_key: str) -> Optional[GeminiEmbeddingFunction]:
    """Attempt to use Gemini embeddings."""
    try:
        ef = GeminiEmbeddingFunction(api_key)
        # Test with a short string
        result = ef(["test"])
        if result and len(result[0]) > 0:
            print(f"  [INFO] Using Gemini embeddings ({ef.name()})")
            return ef
    except Exception as e:
        print(f"  [WARN] Gemini embeddings unavailable: {e}")
    return None


def _try_sentence_transformers() -> Optional[SentenceTransformerEmbedding]:
    """Attempt to use sentence-transformers."""
    try:
        ef = SentenceTransformerEmbedding()
        result = ef(["test"])
        if result and len(result[0]) > 0:
            print(f"  [INFO] Using {ef.name()} embeddings (local, high quality)")
            return ef
    except Exception as e:
        print(f"  [WARN] Sentence-Transformers unavailable: {e}")
    return None


def _try_chromadb_onnx():
    """Attempt to use ChromaDB's built-in ONNX embeddings."""
    try:
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
        ef(["test"])
        print("  [INFO] Using ChromaDB ONNX embeddings")
        return ef
    except Exception:
        pass
    return None


def _fallback():
    """Last resort: hash embeddings."""
    print("  [WARN] All embedding providers unavailable — using hash embeddings (low quality)")
    print("         Install sentence-transformers for better results: pip install sentence-transformers")
    return HashEmbeddingFunction()
