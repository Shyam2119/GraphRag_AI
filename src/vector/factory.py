"""Factory for vector store — ChromaDB when enabled, in-memory fallback otherwise."""

from __future__ import annotations

from config import get_settings


def create_vector_store(*, reload: bool = False):
  settings = get_settings()
  use_chroma = getattr(settings, "use_chroma", False)

  if use_chroma:
    from src.vector.chroma_store import ChromaStore
    return ChromaStore()

  from src.vector.memory_store import InMemoryVectorStore
  return InMemoryVectorStore.load()
