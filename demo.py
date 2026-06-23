#!/usr/bin/env python3
"""GraphRAG Reddit — Demo Script

Runs 4 representative queries demonstrating:
  1. Semantic (vector-dominant)
  2. Graph traversal (graph-dominant)
  3. Hybrid (both retrievers)
  4. Temporal comparison (time-series)

For each query, shows graph-only, vector-only, and fused results
with retrieval quality metrics demonstrating complementary coverage.

Usage:
  python demo.py                    # Full demo (ingest + 4 queries)
  python demo.py --query "question" # Single interactive query
  python demo.py --skip-ingest      # Skip ingestion, use existing data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for Rich's Unicode symbols
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.columns import Columns
from rich import box

from config import get_settings
from src.pipeline.ingest import run_ingestion
from src.pipeline.query import QueryEngine
from src.models import QueryResponse, RetrievalResult

console = Console()

DEMO_QUERIES = [
    {
        "type": "semantic",
        "label": "1 │ Semantic Query (vector-dominant)",
        "question": "What are the main challenges people face when building RAG pipelines?",
    },
    {
        "type": "graph",
        "label": "2 │ Graph Traversal Query (graph-dominant)",
        "question": "Who are the most influential voices in discussions about AI regulation, and what are they saying?",
    },
    {
        "type": "hybrid",
        "label": "3 │ Hybrid Query (graph + vector)",
        "question": "Which communities are leading the conversation on open-source LLMs, and what priorities distinguish them?",
    },
    {
        "type": "temporal",
        "label": "4 │ Temporal Comparison Query",
        "question": "What emerging concerns about AI safety appeared in Q1 2026 that weren't discussed in Q4 2025?",
    },
]


def print_system_info() -> None:
    """Print system configuration at startup."""
    settings = get_settings()

    info_table = Table(
        title="System Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    info_table.add_column("Component", style="bold")
    info_table.add_column("Value", style="green")

    info_table.add_row("Python", sys.version.split()[0])
    info_table.add_row("LLM Provider", settings.llm_provider)
    info_table.add_row("LLM API Key", "✓ configured" if _has_llm_key(settings) else "✗ using fallback")
    info_table.add_row("Embedding", settings.embedding_provider)
    info_table.add_row("Graph Store", "Neo4j" if not settings.use_fallback_graph else "In-Memory (NetworkX)")
    info_table.add_row("Vector Store", "ChromaDB" if settings.use_chroma else "In-Memory")
    data_mode = "Sample data" if settings.effective_scrape_method == "sample" else settings.effective_scrape_method
    if data_mode == "web":
        data_mode = f"Web scraping ({settings.web_scraper_backend})"
    elif data_mode == "reddit_api":
        data_mode = "Live Reddit API"
    info_table.add_row("Data Mode", data_mode)
    info_table.add_row("Time Windows", " → ".join(w[0] for w in settings.time_windows))

    console.print(info_table)


def _has_llm_key(settings) -> bool:
    """Check if a real API key is configured."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return bool(settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key")
    if provider == "openai":
        return bool(settings.openai_api_key and settings.openai_api_key != "your_openai_api_key")
    if provider == "groq":
        return bool(settings.groq_api_key and settings.groq_api_key != "your_groq_api_key")
    return False


def print_results_table(title: str, results: list[RetrievalResult], compact: bool = False) -> None:
    """Display retrieval results in a formatted table."""
    table = Table(title=title, show_lines=True, expand=True, box=box.SIMPLE_HEAVY)
    table.add_column("Rank", style="cyan", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Source", style="yellow", width=8)
    table.add_column("Author", style="magenta", width=16)
    table.add_column("Content", style="white", ratio=1)

    max_rows = 3 if compact else 5
    for i, r in enumerate(results[:max_rows], 1):
        preview = r.text[:150].replace("\n", " ")
        if len(r.text) > 150:
            preview += "…"
        author = r.metadata.get("author", r.metadata.get("type", "—"))
        table.add_row(str(i), f"{r.score:.4f}", r.source, str(author)[:15], preview)

    if not results:
        table.add_row("—", "—", "—", "—", "[dim]No results[/dim]")

    console.print(table)


def print_retrieval_metrics(response: QueryResponse) -> None:
    """Print retrieval quality metrics showing why fusion helps."""
    g_ids = {r.id.rsplit("_chunk_", 1)[0] for r in response.graph_results}
    v_ids = {r.id.rsplit("_chunk_", 1)[0] for r in response.vector_results}
    f_ids = {r.id.rsplit("_chunk_", 1)[0] for r in response.fused_results}

    overlap = g_ids & v_ids
    unique_graph = g_ids - v_ids
    unique_vector = v_ids - g_ids

    metrics = Table(
        title="Retrieval Quality Metrics",
        box=box.ROUNDED,
        show_header=False,
        pad_edge=False,
    )
    metrics.add_column("Metric", style="bold", width=30)
    metrics.add_column("Value", style="cyan", width=12)
    metrics.add_column("Metric", style="bold", width=30)
    metrics.add_column("Value", style="cyan", width=12)

    metrics.add_row(
        "Graph results", str(len(response.graph_results)),
        "Vector results", str(len(response.vector_results)),
    )
    metrics.add_row(
        "Fused results", str(len(response.fused_results)),
        "Overlap (shared IDs)", str(len(overlap)),
    )
    metrics.add_row(
        "Graph-unique results", str(len(unique_graph)),
        "Vector-unique results", str(len(unique_vector)),
    )

    # Show diversity
    fused_authors = set(r.metadata.get("author", "") for r in response.fused_results)
    fused_subs = set(r.metadata.get("subreddit", "") for r in response.fused_results)
    metrics.add_row(
        "Fused unique authors", str(len(fused_authors)),
        "Fused unique subreddits", str(len(fused_subs)),
    )

    console.print(metrics)

    # Highlight fusion benefit
    if unique_graph and unique_vector:
        console.print(
            f"  [bold green]✓ Fusion benefit:[/bold green] Combined {len(unique_graph)} "
            f"graph-only + {len(unique_vector)} vector-only + {len(overlap)} shared results"
        )


def run_demo_query(engine: QueryEngine, spec: dict) -> QueryResponse:
    """Execute a single demo query with full output."""
    console.print()
    console.print(Panel(
        f"[bold]{spec['label']}[/bold]\n\n[italic]{spec['question']}[/italic]",
        border_style="blue",
        padding=(1, 2),
    ))

    start_time = time.time()
    response = engine.query(spec["question"])
    elapsed = time.time() - start_time

    # Query parsing info
    console.print(f"\n  [dim]Parsed type: {response.parsed_query.query_type} │ "
                  f"Entities: {response.parsed_query.graph_entities or '(auto-detected)'} │ "
                  f"Elapsed: {elapsed:.1f}s[/dim]")
    if response.parsed_query.time_start:
        console.print(
            f"  [dim]Time filter: {response.parsed_query.time_start.strftime('%Y-%m-%d')} → "
            f"{response.parsed_query.time_end.strftime('%Y-%m-%d') if response.parsed_query.time_end else 'now'}[/dim]"
        )

    # Results tables
    print_results_table("Graph-Only Results", response.graph_results)
    print_results_table("Vector-Only Results", response.vector_results)
    print_results_table("Fused Results (Weighted RRF)", response.fused_results)

    # Metrics
    print_retrieval_metrics(response)

    # Temporal comparison
    if response.period_comparison:
        console.print("\n[bold]Temporal Comparison:[/bold]")
        for key in ("period_a", "period_b"):
            period = response.period_comparison[key]
            console.print(
                f"  {period['label']}: {period['start']} → {period['end']} "
                f"({len(period['graph'])} graph, {len(period['vector'])} vector hits)"
            )
            sentiment = period.get("sentiment", {})
            if sentiment.get("periods"):
                for s in sentiment["periods"][:3]:
                    console.print(f"    └─ {s.get('window', '?')}: {s.get('sentiment', '?')} "
                                  f"(n={s.get('cnt', 0)})")

    # Answer
    console.print(Panel(
        Markdown(response.answer),
        title="[bold green]Generated Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    return response


def save_results(results: list, output_path: str = "demo_results.json") -> None:
    """Save structured demo results to JSON for reproducibility."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for resp in results:
        serializable.append({
            "question": resp.question,
            "query_type": resp.parsed_query.query_type,
            "entities": resp.parsed_query.graph_entities,
            "time_start": resp.parsed_query.time_start.isoformat() if resp.parsed_query.time_start else None,
            "time_end": resp.parsed_query.time_end.isoformat() if resp.parsed_query.time_end else None,
            "graph_result_count": len(resp.graph_results),
            "vector_result_count": len(resp.vector_results),
            "fused_result_count": len(resp.fused_results),
            "answer_length": len(resp.answer),
            "answer": resp.answer,
            "fused_results": [
                {"id": r.id, "score": r.score, "source": r.source, "text": r.text[:300]}
                for r in resp.fused_results
            ],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    console.print(f"\n  [dim]Results saved to {output_path}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphRAG Reddit Intelligence — Demo & Query Tool"
    )
    parser.add_argument("--query", "-q", type=str, help="Run a single query instead of the full demo")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion (use existing data)")
    args = parser.parse_args()

    settings = get_settings()

    # ── Header ─────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold magenta]GraphRAG for Time-Series Reddit Intelligence[/bold magenta]\n"
        "[dim]Hybrid retrieval system fusing graph traversal + semantic vector search[/dim]\n"
        "[dim]Jupiter Meta Labs — GenAI Engineer Assignment[/dim]",
        border_style="magenta",
        padding=(1, 2),
    ))

    print_system_info()

    # ── Ingestion ──────────────────────────────────────────────────────────
    if not args.skip_ingest:
        console.print("\n[bold]Step 1: Data Ingestion[/bold]")
        start = time.time()
        stats = run_ingestion(clear_existing=True)
        elapsed = time.time() - start
        console.print(
            f"  [green]✓[/green] Ingested {stats['items']} items → "
            f"{stats['vector_chunks']} vector chunks, "
            f"{sum(stats['graph_nodes'].values())} graph nodes "
            f"[dim]({elapsed:.1f}s)[/dim]"
        )
        # Print node breakdown
        for label, count in sorted(stats['graph_nodes'].items()):
            console.print(f"    └─ {label}: {count}")
    else:
        console.print("\n[bold]Step 1: Skipping ingestion (--skip-ingest)[/bold]")

    # ── Queries ────────────────────────────────────────────────────────────
    console.print("\n[bold]Step 2: Running Queries[/bold]")
    engine = QueryEngine()

    try:
        if args.query:
            # Single interactive query
            spec = {
                "type": "interactive",
                "label": "Interactive Query",
                "question": args.query,
            }
            run_demo_query(engine, spec)
        else:
            # Full demo with all 4 queries
            responses = []
            for spec in DEMO_QUERIES:
                response = run_demo_query(engine, spec)
                responses.append(response)
                console.print("\n" + "━" * 80)

            # Save results
            save_results(responses)
    finally:
        engine.close()

    # ── Summary ────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold green]✓ Demo complete![/bold green]\n\n"
        "This demo showed:\n"
        "  1. [cyan]Semantic retrieval[/cyan] — vector search for conceptual questions\n"
        "  2. [cyan]Graph traversal[/cyan] — entity/relationship queries via knowledge graph\n"
        "  3. [cyan]Hybrid fusion[/cyan] — weighted RRF combining both retrievers\n"
        "  4. [cyan]Temporal comparison[/cyan] — side-by-side period analysis\n\n"
        "[dim]For each query, fused results combine graph-unique and vector-unique\n"
        "insights that neither retriever could provide alone.[/dim]",
        border_style="green",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    main()
