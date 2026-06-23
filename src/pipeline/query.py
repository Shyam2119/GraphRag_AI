"""Query engine: parse → retrieve (graph + vector + fuse) → LLM answer.

Orchestrates the full query pipeline with query-type-aware context building
and answer generation. Produces structured responses with graph-only,
vector-only, and fused results for comparison.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.graph.factory import create_graph_store
from src.vector.factory import create_vector_store
from src.llm.client import LLMClient
from src.llm.query_parser import QueryParser
from src.retrieval.hybrid_pipeline import HybridRetriever
from src.models import ParsedQuery, QueryResponse, RetrievalResult

ANSWER_SYSTEM = """You are an AI research analyst synthesizing Reddit intelligence.
Answer the user's question based ONLY on the provided context.
Include specific citations like [author in r/subreddit, date].
Be concise but thorough. If comparing time periods, structure your answer with clear sections.
Highlight patterns, trends, and notable viewpoints.
Acknowledge limitations if context is insufficient.
If the context contains a 'Novel signals' section, explicitly call out what is new in the later period."""


class QueryEngine:
    """Full pipeline: parse question → retrieve from graph+vector → generate answer."""

    def __init__(self) -> None:
        self.graph = create_graph_store()
        self.vector = create_vector_store()
        self.retriever = HybridRetriever(self.graph, self.vector)
        self.parser = QueryParser()
        self.llm = LLMClient()

    def close(self) -> None:
        self.graph.close()

    def query(self, question: str, top_k: int = 8) -> QueryResponse:
        """Execute the full query pipeline."""
        parsed = self.parser.parse(question)

        graph_results = self.retriever.retrieve_graph(parsed, limit=top_k)
        vector_results = self.retriever.retrieve_vector(parsed, limit=top_k)
        fused_results = self.retriever.retrieve_fused(parsed, limit=top_k)

        period_comparison = None
        if parsed.query_type == "temporal" or parsed.compare_start:
            period_comparison = self.retriever.temporal_comparison(parsed)

        context = self._build_context(fused_results, parsed, period_comparison)
        answer = self._generate_answer(question, context, parsed)

        return QueryResponse(
            question=question,
            answer=answer,
            graph_results=graph_results,
            vector_results=vector_results,
            fused_results=fused_results,
            parsed_query=parsed,
            period_comparison=period_comparison,
        )

    def _build_context(
        self,
        results: List[RetrievalResult],
        parsed: ParsedQuery,
        period_comparison: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build query-type-aware context for LLM answer generation."""
        parts = ["## Retrieved Context\n"]

        if parsed.query_type == "graph":
            parts.append("### Graph Traversal Results\n")
            for i, r in enumerate(results, 1):
                parts.append(f"**Source {i}** ({r.source}, score={r.score:.3f})")
                parts.append(r.text[:600])
                parts.append(f"Citation: {r.citation}\n")

        elif parsed.query_type == "temporal":
            # For temporal queries, structure by time period
            parts.append("### Temporal Analysis Results\n")
            for i, r in enumerate(results, 1):
                window = r.metadata.get("window", "unknown")
                sentiment = r.metadata.get("sentiment", "unknown")
                parts.append(f"**Source {i}** (window={window}, sentiment={sentiment}, score={r.score:.3f})")
                parts.append(r.text[:600])
                parts.append(f"Citation: {r.citation}\n")

        else:
            # Standard context (semantic/hybrid)
            for i, r in enumerate(results, 1):
                contributing = r.metadata.get("contributing_sources", {})
                source_info = ", ".join(f"{s}={c}" for s, c in contributing.items()) if contributing else r.source
                parts.append(f"### Source {i} ({source_info}, score={r.score:.3f})")
                parts.append(r.text[:600])
                parts.append(f"Citation: {r.citation}\n")

        # Add temporal comparison context
        if period_comparison:
            parts.append("\n## Temporal Comparison\n")
            for key in ("period_a", "period_b"):
                period = period_comparison[key]
                parts.append(f"### {period['label']} ({period['start']} → {period['end']})")
                sentiment = period.get("sentiment", {})
                if sentiment.get("periods"):
                    parts.append(f"Sentiment breakdown: {sentiment['periods']}")
                # Add top results from each period
                for r in period.get("graph", [])[:3]:
                    parts.append(f"  - [Graph] {r.text[:200]}")
                for r in period.get("vector", [])[:3]:
                    parts.append(f"  - [Vector] {r.text[:200]}")
                parts.append("")
            if period_comparison.get("novel_signals"):
                parts.append("### Novel signals in the later period")
                for signal in period_comparison["novel_signals"][:8]:
                    parts.append(f"- {signal}")
                parts.append("")

        # Add retrieval quality metadata
        parts.append("\n## Retrieval Quality")
        parts.append(f"- Total fused results: {len(results)}")
        unique_authors = len(set(r.metadata.get("author", "") for r in results))
        unique_subs = len(set(r.metadata.get("subreddit", "") for r in results))
        parts.append(f"- Unique authors: {unique_authors}")
        parts.append(f"- Unique subreddits: {unique_subs}")

        return "\n".join(parts)

    def _generate_answer(
        self, question: str, context: str, parsed: ParsedQuery
    ) -> str:
        """Generate an answer using the LLM with query-type-aware prompting."""
        type_instruction = {
            "semantic": "Focus on explaining concepts and practices. Cite specific examples from the community.",
            "graph": "Focus on relationships, influence patterns, and community dynamics. Name specific authors and subreddits.",
            "temporal": "Structure your answer by time period. Highlight what changed, what emerged, and what declined.",
            "hybrid": "Balance factual content with structural insights. Show how different communities and authors contribute.",
        }.get(parsed.query_type, "")

        prompt = f"""Question: {question}

Query type: {parsed.query_type}
Time range: {parsed.time_start} to {parsed.time_end}

{type_instruction}

{context}

Synthesize a comprehensive answer with citations."""

        return self.llm.complete(prompt, system=ANSWER_SYSTEM, temperature=0.3)
