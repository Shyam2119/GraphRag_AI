"""ChromaDB vector index with metadata-filtered semantic search.

Justification: ChromaDB provides embedded persistence (no separate server),
rich metadata filtering for temporal/subreddit/author constraints, and
built-in ONNX embeddings (no PyTorch dependency).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import get_settings
from src.models import RetrievalResult
from src.vector.embeddings import get_embedding_function


class ChromaStore:
  COLLECTION = "reddit_content"

  def __init__(self) -> None:
    settings = get_settings()
    self.settings = settings
    self.client = chromadb.PersistentClient(
      path=settings.chroma_persist_dir,
      settings=ChromaSettings(anonymized_telemetry=False),
    )
    self._ef = get_embedding_function()
    self._collection = None

  @property
  def collection(self):
    if self._collection is None:
      self._collection = self.client.get_or_create_collection(
        name=self.COLLECTION,
        embedding_function=self._ef,
        metadata={"hnsw:space": "cosine"},
      )
    return self._collection

  def clear(self) -> None:
    try:
      self.client.delete_collection(self.COLLECTION)
    except Exception:
      pass
    self._collection = None

  def add_chunks(self, chunks: List[Dict]) -> None:
    if not chunks:
      return

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [self._sanitize_meta(c["metadata"]) for c in chunks]

    self.collection.add(
      ids=ids,
      documents=texts,
      metadatas=metadatas,
    )

  def search(
    self,
    query: str,
    time_start: Optional[datetime] = None,
    time_end: Optional[datetime] = None,
    subreddits: Optional[List[str]] = None,
    author: Optional[str] = None,
    limit: int = 10,
  ) -> List[RetrievalResult]:
    where = self._build_where(time_start, time_end, subreddits, author)

    kwargs: Dict[str, Any] = {
      "query_texts": [query],
      "n_results": limit,
      "include": ["documents", "metadatas", "distances"],
    }
    if where:
      kwargs["where"] = where

    results = self.collection.query(**kwargs)

    retrieval_results = []
    if not results["ids"] or not results["ids"][0]:
      return retrieval_results

    for i, doc_id in enumerate(results["ids"][0]):
      distance = results["distances"][0][i] if results["distances"] else 0
      score = 1.0 / (1.0 + distance)
      meta = results["metadatas"][0][i] if results["metadatas"] else {}
      retrieval_results.append(RetrievalResult(
        id=doc_id,
        text=results["documents"][0][i],
        score=score,
        source="vector",
        metadata=meta,
      ))

    return retrieval_results

  @staticmethod
  def _build_where(
    time_start: Optional[datetime],
    time_end: Optional[datetime],
    subreddits: Optional[List[str]],
    author: Optional[str],
  ) -> Optional[Dict]:
    conditions = []

    if time_start:
      conditions.append({"created_ts": {"$gte": time_start.timestamp()}})
    if time_end:
      conditions.append({"created_ts": {"$lte": time_end.timestamp()}})
    if subreddits:
      if len(subreddits) == 1:
        conditions.append({"subreddit": subreddits[0]})
      else:
        conditions.append({"subreddit": {"$in": subreddits}})
    if author:
      conditions.append({"author": author})

    if not conditions:
      return None
    if len(conditions) == 1:
      return conditions[0]
    return {"$and": conditions}

  @staticmethod
  def _sanitize_meta(meta: Dict) -> Dict:
    """ChromaDB only accepts str, int, float, bool metadata values."""
    clean = {}
    for k, v in meta.items():
      if isinstance(v, (str, int, float, bool)):
        clean[k] = v
      elif v is None:
        clean[k] = ""
      else:
        clean[k] = str(v)
    return clean

  def count(self) -> int:
    return self.collection.count()
