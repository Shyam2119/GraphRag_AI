"""In-memory vector store fallback when ChromaDB is unavailable."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.models import RetrievalResult
from src.vector.embeddings import HashEmbeddingFunction

FALLBACK_VECTOR_PATH = Path("./data/fallback_vector.pkl")


class InMemoryVectorStore:
  def __init__(self) -> None:
    self._ef = HashEmbeddingFunction()
    self._docs: Dict[str, dict] = {}
    self._embeddings: Dict[str, np.ndarray] = {}

  def clear(self) -> None:
    self._docs.clear()
    self._embeddings.clear()
    if FALLBACK_VECTOR_PATH.exists():
      FALLBACK_VECTOR_PATH.unlink()

  def add_chunks(self, chunks: List[Dict]) -> None:
    for chunk in chunks:
      cid = chunk["id"]
      self._docs[cid] = {"text": chunk["text"], "metadata": chunk["metadata"]}
      self._embeddings[cid] = np.array(self._ef([chunk["text"]])[0], dtype=np.float32)

  def search(
    self,
    query: str,
    time_start: Optional[datetime] = None,
    time_end: Optional[datetime] = None,
    subreddits: Optional[List[str]] = None,
    author: Optional[str] = None,
    limit: int = 10,
  ) -> List[RetrievalResult]:
    if not self._docs:
      return []

    query_vec = np.array(self._ef([query])[0], dtype=np.float32)
    scored = []

    for doc_id, doc in self._docs.items():
      meta = doc["metadata"]
      if time_start and meta.get("created_ts", 0) < time_start.timestamp():
        continue
      if time_end and meta.get("created_ts", 0) > time_end.timestamp():
        continue
      if subreddits and meta.get("subreddit") not in subreddits:
        continue
      if author and meta.get("author") != author:
        continue

      vec = self._embeddings[doc_id]
      sim = float(np.dot(query_vec, vec))
      scored.append((sim, doc_id, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
      RetrievalResult(
        id=doc_id,
        text=doc["text"],
        score=score,
        source="vector",
        metadata=doc["metadata"],
      )
      for score, doc_id, doc in scored[:limit]
    ]

  def count(self) -> int:
    return len(self._docs)

  def save(self) -> None:
    FALLBACK_VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FALLBACK_VECTOR_PATH, "wb") as f:
      pickle.dump({
        "docs": self._docs,
        "embeddings": {k: v.tolist() for k, v in self._embeddings.items()},
      }, f)

  @classmethod
  def load(cls) -> "InMemoryVectorStore":
    store = cls()
    if FALLBACK_VECTOR_PATH.exists():
      with open(FALLBACK_VECTOR_PATH, "rb") as f:
        data = pickle.load(f)
      store._docs = data["docs"]
      store._embeddings = {k: np.array(v, dtype=np.float32) for k, v in data["embeddings"].items()}
    return store
