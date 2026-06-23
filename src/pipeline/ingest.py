"""End-to-end ingestion: Reddit → LLM enrichment → Graph + Vector stores."""

from __future__ import annotations

from config import get_settings
from src.graph.factory import create_graph_store
from src.ingestion.reddit_scraper import RedditScraper
from src.ingestion.llm_extractor import LLMExtractor
from src.ingestion.sample_data import generate_sample_data
from src.vector.factory import create_vector_store
from src.vector.chunker import chunk_text


def run_ingestion(clear_existing: bool = True) -> dict:
  settings = get_settings()
  print("=" * 60)
  print("GraphRAG Reddit - Ingestion Pipeline")
  print("=" * 60)

  graph = create_graph_store()
  vector = create_vector_store()

  if clear_existing:
    print("\n[1/5] Clearing existing data...")
    graph.clear()
    vector.clear()
    graph.init_schema()

  print("\n[2/5] Collecting Reddit data...")
  method = settings.effective_scrape_method
  print(f"  Scrape method: {method}")

  if method == "sample":
    print("  Using sample data (set SCRAPE_METHOD=web for live scraping)")
    items = generate_sample_data()
  elif method == "reddit_api":
    print("  Using Reddit API (PRAW)")
    scraper = RedditScraper()
    items = scraper.scrape_all_windows()
  elif method == "web":
    print(f"  Using web search + scraping via '{settings.web_scraper_backend}'")
    from src.ingestion.web_scraper import WebRedditScraper
    scraper = WebRedditScraper()
    items = scraper.scrape_all_windows()
    if not items:
      print("  [!] Web scraping returned no results, falling back to sample data")
      items = generate_sample_data()
  else:
    print(f"  Unknown scrape method '{method}', using sample data")
    items = generate_sample_data()

  print(f"  Collected {len(items)} posts/comments")

  print("\n[3/5] LLM entity extraction...")
  extractor = LLMExtractor()
  batch_size = getattr(settings, 'llm_batch_size', 5)
  enriched = extractor.enrich_batch(items, batch_size=batch_size)

  print("\n[4/5] Building knowledge graph...")
  for i, e in enumerate(enriched):
    graph.ingest_enriched(e)
    if (i + 1) % 20 == 0:
      print(f"  Graph: {i + 1}/{len(enriched)}")
  node_counts = graph.count_nodes()
  print(f"  Graph nodes: {node_counts}")

  print("\n[5/5] Building vector index...")
  all_chunks = []
  for e in enriched:
    all_chunks.extend(chunk_text(e))

  batch_size = 50
  for i in range(0, len(all_chunks), batch_size):
    vector.add_chunks(all_chunks[i : i + batch_size])
  print(f"  Vector chunks: {vector.count()}")

  graph.close()
  if hasattr(vector, "save"):
    vector.save()

  print("\n[OK] Ingestion complete!")
  return {
    "items": len(items),
    "enriched": len(enriched),
    "graph_nodes": node_counts,
    "vector_chunks": vector.count(),
  }


if __name__ == "__main__":
  run_ingestion()
