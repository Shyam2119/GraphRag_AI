"""Text chunking with metadata preservation."""

from __future__ import annotations

from typing import Dict, List

from src.models import EnrichedItem


def chunk_text(
  enriched: EnrichedItem,
  chunk_size: int = 500,
  overlap: int = 50,
) -> List[Dict]:
  """Split enriched content into chunks with full metadata tags."""
  item = enriched.item
  text = item.full_text
  if not text.strip():
    return []

  chunks = []
  if len(text) <= chunk_size:
    chunks.append(text)
  else:
    start = 0
    while start < len(text):
      end = start + chunk_size
      chunks.append(text[start:end])
      start = end - overlap

  base_meta = {
    "content_id": item.id,
    "content_type": item.content_type.value,
    "author": item.author,
    "subreddit": item.subreddit,
    "created_at": item.created_at.isoformat(),
    "created_ts": item.created_utc,
    "score": item.score,
    "url": item.permalink,
    "window": item.window_label,
    "sentiment": enriched.overall_sentiment,
    "topics": ",".join(enriched.topics),
  }

  result = []
  for i, chunk in enumerate(chunks):
    chunk_id = f"{item.id}_chunk_{i}"
    meta = {**base_meta, "chunk_index": i, "chunk_total": len(chunks)}
    result.append({"id": chunk_id, "text": chunk, "metadata": meta})

  return result
