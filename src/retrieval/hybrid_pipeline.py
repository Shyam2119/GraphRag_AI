"""Hybrid retrieval pipeline combining graph traversal and vector search.

Routes queries to appropriate graph operations based on query type,
applies query-type-aware fusion weights, and supports temporal
side-by-side comparison for time-series queries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.graph.factory import create_graph_store
from src.vector.factory import create_vector_store
from src.models import ParsedQuery, RetrievalResult
from src.retrieval.fusion import reciprocal_rank_fusion, get_query_weights


class HybridRetriever:
    """Orchestrates graph + vector retrieval with intelligent routing."""

    def __init__(
        self,
        graph=None,
        vector=None,
    ) -> None:
        self.graph = graph or create_graph_store()
        self.vector = vector or create_vector_store()

    def retrieve_graph(self, parsed: ParsedQuery, limit: int = 10) -> List[RetrievalResult]:
        """Route to the appropriate graph query based on query intent."""
        q_lower = parsed.original.lower()

        # Influence/author-focused queries
        if any(w in q_lower for w in ("influential", "who are", "key voices", "top contributors", "most active")):
            if any(w in q_lower for w in ("what are they saying", "voices", "people", "leaders")):
                entity_results = self.graph.find_influential_entities(
                    parsed.graph_entities,
                    parsed.time_start,
                    parsed.time_end,
                    limit,
                    entity_type="person",
                )
                if entity_results:
                    return entity_results
            user_results = self.graph.find_influential_users(
                parsed.graph_entities,
                parsed.time_start,
                parsed.time_end,
                limit,
            )
            # Fallback for sparse extraction on live data.
            if user_results:
                return user_results
            return self.graph.find_influential_users(
                [],
                parsed.time_start,
                parsed.time_end,
                limit,
            )

        # Community/subreddit-focused queries
        if any(w in q_lower for w in ("communities", "leading", "subreddits", "which communities")):
            community_results = self.graph.community_leadership(
                parsed.graph_entities,
                parsed.time_start,
                parsed.time_end,
                limit,
            )
            if community_results:
                return community_results
            return self.graph.community_leadership(
                [],
                parsed.time_start,
                parsed.time_end,
                limit,
            )

        # Co-occurrence queries
        if any(w in q_lower for w in ("related to", "associated with", "connected to", "co-occur")):
            if parsed.graph_entities:
                return self.graph.entity_co_occurrence(
                    parsed.graph_entities[0],
                    parsed.time_start,
                    parsed.time_end,
                    limit,
                )

        # Default: entity-based search
        results = self.graph.search(
            parsed.graph_entities,
            parsed.time_start,
            parsed.time_end,
            parsed.subreddits or None,
            limit,
        )
        if results:
            return results
        # For graph/temporal questions, return top time-filtered graph records
        # even when no entities were extracted.
        if parsed.query_type in {"graph", "temporal"}:
            return self.graph.search(
                [],
                parsed.time_start,
                parsed.time_end,
                parsed.subreddits or None,
                limit,
            )
        return results

    def retrieve_vector(self, parsed: ParsedQuery, limit: int = 10) -> List[RetrievalResult]:
        """Semantic search over vector index with metadata filtering."""
        return self.vector.search(
            parsed.semantic_query,
            parsed.time_start,
            parsed.time_end,
            parsed.subreddits or None,
            limit=limit,
        )

    def retrieve_fused(
        self, parsed: ParsedQuery, limit: int = 10
    ) -> List[RetrievalResult]:
        """Fuse graph and vector results using query-type-aware weighted RRF."""
        graph_results = self.retrieve_graph(parsed, limit=limit)
        vector_results = self.retrieve_vector(parsed, limit=limit)
        weights = get_query_weights(parsed.query_type)
        return reciprocal_rank_fusion(
            [graph_results, vector_results],
            weights=weights,
            top_n=limit,
        )

    def temporal_comparison(self, parsed: ParsedQuery) -> Dict[str, Any]:
        """Side-by-side analysis for two time periods."""
        # Determine the two periods
        period_a_start = parsed.compare_start or parsed.time_start
        period_a_end = parsed.compare_end or parsed.time_end
        period_b_start = parsed.time_start
        period_b_end = parsed.time_end

        if parsed.compare_start and parsed.time_start:
            period_a_start, period_a_end = parsed.compare_start, parsed.compare_end
            period_b_start, period_b_end = parsed.time_start, parsed.time_end

        entities = parsed.graph_entities or ["AI safety"]

        # Period A retrieval
        period_a_graph = self.graph.search(entities, period_a_start, period_a_end, limit=5)
        period_a_vector = self.vector.search(
            parsed.semantic_query, period_a_start, period_a_end, limit=5
        )
        period_a_sentiment = self.graph.sentiment_over_time(
            entities[0], period_a_start, period_a_end
        )

        # Period B retrieval
        period_b_graph = self.graph.search(entities, period_b_start, period_b_end, limit=5)
        period_b_vector = self.vector.search(
            parsed.semantic_query, period_b_start, period_b_end, limit=5
        )
        period_b_sentiment = self.graph.sentiment_over_time(
            entities[0], period_b_start, period_b_end
        )

        period_a = {
            "label": "Period A",
            "start": period_a_start.isoformat() if period_a_start else None,
            "end": period_a_end.isoformat() if period_a_end else None,
            "graph": period_a_graph,
            "vector": period_a_vector,
            "sentiment": period_a_sentiment,
        }
        period_b = {
            "label": "Period B",
            "start": period_b_start.isoformat() if period_b_start else None,
            "end": period_b_end.isoformat() if period_b_end else None,
            "graph": period_b_graph,
            "vector": period_b_vector,
            "sentiment": period_b_sentiment,
        }

        novel_signals = self._extract_novel_signals(period_a_graph, period_a_vector, period_b_graph, period_b_vector)

        return {"period_a": period_a, "period_b": period_b, "novel_signals": novel_signals}

    @staticmethod
    def _extract_novel_signals(
        earlier_graph: List[RetrievalResult],
        earlier_vector: List[RetrievalResult],
        later_graph: List[RetrievalResult],
        later_vector: List[RetrievalResult],
    ) -> List[str]:
        """Surface concepts that are present in the later period but not the earlier one."""
        stop_words = {
            "the", "and", "for", "with", "that", "this", "from", "into", "about", "they",
            "their", "have", "were", "what", "when", "where", "which", "them", "than",
            "more", "less", "over", "under", "into", "onto", "across", "being", "been",
            "make", "made", "using", "used", "people", "community", "discussion", "reddit",
            "post", "comment", "like", "just", "still", "only", "much", "many",
        }

        def tokenize(results: List[RetrievalResult]) -> set[str]:
            tokens: set[str] = set()
            for result in results:
                for token in result.text.lower().split():
                    cleaned = "".join(ch for ch in token if ch.isalnum() or ch in {"-", "/"})
                    if len(cleaned) < 4 or cleaned in stop_words or cleaned.isdigit():
                        continue
                    tokens.add(cleaned)
                matched_entity = result.metadata.get("matched_entity")
                if matched_entity:
                    tokens.add(str(matched_entity).lower())
            return tokens

        earlier_tokens = tokenize(earlier_graph + earlier_vector)
        later_tokens = tokenize(later_graph + later_vector)
        novelty = sorted(token for token in later_tokens - earlier_tokens if token not in stop_words)
        return novelty[:12]
