"""Factory for graph store — Neo4j primary, in-memory fallback."""

from __future__ import annotations

from config import get_settings


def create_graph_store(*, reload: bool = False):
  settings = get_settings()
  use_fallback = getattr(settings, "use_fallback_graph", False)

  if not use_fallback:
    from src.graph.neo4j_store import Neo4jStore
    store = Neo4jStore()
    if store.verify_connection():
      return store
    store.close()
    print("  [WARN] Neo4j unavailable - using in-memory graph fallback")
    print("    For production: docker compose up -d")

  from src.graph.networkx_store import NetworkXGraphStore
  if reload:
    return NetworkXGraphStore.load()
  return NetworkXGraphStore.load()
