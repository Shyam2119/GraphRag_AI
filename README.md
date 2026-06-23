# GraphRag_AI

Hybrid GraphRAG project for time-series Reddit intelligence.

The system ingests Reddit discussions across multiple time windows, builds:
- a temporal knowledge graph
- a vector index

It then answers queries using hybrid retrieval (graph + vector fusion) with citations.

## Features

- Temporal ingestion over multiple windows
- LLM-based extraction with fallback-safe keyword extraction
- Graph retrieval for influence/community relationships
- Vector retrieval for semantic relevance
- Weighted reciprocal rank fusion (RRF)
- Temporal comparison support
- Works with safe defaults without external credentials

## Quick Start

```bash
git clone <repo-url>
cd GraphRAG_assignment
python -m venv venv
source venv/bin/activate   # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demo.py
```

Output is saved to:
- `demo_results.json`

## Live Data Mode (Web Scraping)

```bash
copy .env.example .env     # macOS/Linux: cp .env.example .env
# Set in .env:
# USE_SAMPLE_DATA=false
# SCRAPE_METHOD=web
# WEB_SCRAPER_BACKEND=ddgs
python -m src.pipeline.ingest
python demo.py --skip-ingest
```

See:
- `WEB_SCRAPING_ALTERNATIVES.md`
- `SETUP.md`

## Demo Query Set

1. Semantic query  
2. Graph-dominant query  
3. Hybrid query  
4. Temporal comparison query

For each query, demo prints:
- graph-only results
- vector-only results
- fused results
- final answer with citations

## Project Structure

```text
demo.py
config.py
requirements.txt
.env.example
SETUP.md
WEB_SCRAPING_ALTERNATIVES.md
demo_results.json
src/
tests/
```

## Validate and Test

```bash
python validate.py
python -m pytest tests/test_query_parser.py tests/test_fusion.py tests/test_chunker.py tests/test_llm_fallback.py -q
```

## License

MIT