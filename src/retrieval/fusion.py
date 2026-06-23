"""Reciprocal Rank Fusion for combining graph and vector retrieval results.

Supports:
- Weighted RRF for query-type-aware fusion
- Score normalization before fusion
- Diversity-aware reranking (MMR-style) to avoid result homogeneity
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from src.models import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: List[List[RetrievalResult]],
    k: int = 60,
    top_n: int = 10,
    weights: Optional[List[float]] = None,
) -> List[RetrievalResult]:
    """Fuse multiple ranked lists using weighted Reciprocal Rank Fusion.

    RRF score = sum(weight_i / (k + rank_i)) across retrievers.
    Deduplicates by content_id (strips chunk suffix).

    Args:
        result_lists: List of ranked result lists from different retrievers
        k: RRF constant (higher = more emphasis on overall presence vs rank)
        top_n: Number of results to return
        weights: Optional per-retriever weights (default: equal weighting)
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: Dict[str, float] = {}
    best_result: Dict[str, RetrievalResult] = {}
    source_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for list_idx, results in enumerate(result_lists):
        w = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, result in enumerate(results):
            dedup_key = _dedup_key(result.id)
            rrf_score = w / (k + rank + 1)
            scores[dedup_key] = scores.get(dedup_key, 0) + rrf_score

            # Track which sources contributed
            source_counts[dedup_key][result.source] += 1

            existing = best_result.get(dedup_key)
            if existing is None or result.score > existing.score:
                best_result[dedup_key] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n * 2]

    fused = []
    for dedup_key, rrf_score in ranked:
        result = best_result[dedup_key]
        sources = source_counts[dedup_key]
        fused.append(RetrievalResult(
            id=result.id,
            text=result.text,
            score=rrf_score,
            source="fused",
            metadata={
                **result.metadata,
                "rrf_score": rrf_score,
                "contributing_sources": dict(sources),
            },
        ))

    # Apply diversity reranking
    return diversity_rerank(fused, top_n=top_n)


def diversity_rerank(
    results: List[RetrievalResult],
    top_n: int = 10,
    author_penalty: float = 0.7,
    subreddit_penalty: float = 0.9,
) -> List[RetrievalResult]:
    """MMR-style reranking to promote diversity in results.

    Penalizes results from authors/subreddits already represented,
    ensuring the final list covers more perspectives.
    """
    if len(results) <= top_n:
        return results

    selected = []
    remaining = list(results)
    seen_authors: Dict[str, int] = defaultdict(int)
    seen_subreddits: Dict[str, int] = defaultdict(int)

    while len(selected) < top_n and remaining:
        best_idx = 0
        best_adjusted = -1.0

        for i, r in enumerate(remaining):
            author = r.metadata.get("author", "")
            subreddit = r.metadata.get("subreddit", "")

            # Penalize if author/subreddit already seen
            author_factor = author_penalty ** seen_authors.get(author, 0)
            subreddit_factor = subreddit_penalty ** seen_subreddits.get(subreddit, 0)

            adjusted = r.score * author_factor * subreddit_factor
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = i

        chosen = remaining.pop(best_idx)
        selected.append(chosen)

        author = chosen.metadata.get("author", "")
        subreddit = chosen.metadata.get("subreddit", "")
        if author:
            seen_authors[author] += 1
        if subreddit:
            seen_subreddits[subreddit] += 1

    return selected


def get_query_weights(query_type: str) -> List[float]:
    """Return fusion weights [graph_weight, vector_weight] based on query type.

    Semantic queries favor vector results, graph queries favor graph results,
    temporal and hybrid queries use balanced weights.
    """
    return {
        "semantic": [0.4, 1.0],   # vector-dominant
        "graph": [1.0, 0.4],      # graph-dominant
        "hybrid": [0.8, 0.8],     # balanced
        "temporal": [0.7, 0.7],   # balanced with slight graph preference
    }.get(query_type, [0.8, 0.8])


def _dedup_key(result_id: str) -> str:
    if "_chunk_" in result_id:
        return result_id.rsplit("_chunk_", 1)[0]
    return result_id
