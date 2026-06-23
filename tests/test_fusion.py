"""Tests for Reciprocal Rank Fusion and diversity reranking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.models import RetrievalResult
from src.retrieval.fusion import (
    reciprocal_rank_fusion,
    diversity_rerank,
    get_query_weights,
    _dedup_key,
)


def _make_result(id: str, score: float, source: str, author: str = "user1", subreddit: str = "test") -> RetrievalResult:
    return RetrievalResult(
        id=id,
        text=f"Content for {id}",
        score=score,
        source=source,
        metadata={"author": author, "subreddit": subreddit},
    )


class TestReciprocalRankFusion:
    """Test RRF fusion logic."""

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([[], []])
        assert result == []

    def test_single_list(self):
        results = [_make_result("a", 0.9, "graph")]
        fused = reciprocal_rank_fusion([results])
        assert len(fused) == 1
        assert fused[0].source == "fused"

    def test_overlapping_results_get_higher_score(self):
        """Results appearing in both lists should rank higher than single-list results."""
        graph = [_make_result("a", 0.9, "graph"), _make_result("b", 0.8, "graph")]
        vector = [_make_result("a", 0.95, "vector"), _make_result("c", 0.85, "vector")]

        fused = reciprocal_rank_fusion([graph, vector])

        # "a" appears in both, should be ranked first
        assert fused[0].id == "a"
        assert fused[0].score > fused[1].score

    def test_deduplication_across_chunks(self):
        """Chunks from the same document should be deduplicated."""
        graph = [_make_result("doc1_chunk_0", 0.9, "graph")]
        vector = [_make_result("doc1_chunk_1", 0.85, "vector")]

        fused = reciprocal_rank_fusion([graph, vector])

        # Both chunks map to "doc1", should be merged
        assert len(fused) == 1
        assert fused[0].id in ("doc1_chunk_0", "doc1_chunk_1")

    def test_weighted_rrf_favors_weighted_list(self):
        """Higher-weighted list should contribute more to final ranking."""
        graph = [_make_result("a", 0.9, "graph")]
        vector = [_make_result("b", 0.9, "vector")]

        # Weight graph 2x
        fused_graph_heavy = reciprocal_rank_fusion([graph, vector], weights=[2.0, 1.0])
        # Weight vector 2x
        fused_vector_heavy = reciprocal_rank_fusion([graph, vector], weights=[1.0, 2.0])

        assert fused_graph_heavy[0].id == "a"
        assert fused_vector_heavy[0].id == "b"

    def test_fused_source_label(self):
        """All fused results should have source='fused'."""
        graph = [_make_result("a", 0.9, "graph")]
        vector = [_make_result("b", 0.8, "vector")]
        fused = reciprocal_rank_fusion([graph, vector])
        for r in fused:
            assert r.source == "fused"

    def test_rrf_score_decreases_with_rank(self):
        """Later-ranked items should have lower RRF scores."""
        results = [_make_result(f"item_{i}", 1.0 - i * 0.1, "graph") for i in range(5)]
        fused = reciprocal_rank_fusion([results])
        for i in range(len(fused) - 1):
            assert fused[i].score >= fused[i + 1].score

    def test_top_n_limits_output(self):
        """Output should be limited to top_n results."""
        results = [_make_result(f"item_{i}", 1.0, "graph", author=f"user_{i}") for i in range(20)]
        fused = reciprocal_rank_fusion([results], top_n=5)
        assert len(fused) <= 5


class TestDiversityRerank:
    """Test MMR-style diversity reranking."""

    def test_penalizes_same_author(self):
        """Results from the same author should be penalized in ranking."""
        results = [
            _make_result("a", 0.9, "fused", author="alice", subreddit="ml"),
            _make_result("b", 0.85, "fused", author="alice", subreddit="ai"),
            _make_result("c", 0.8, "fused", author="bob", subreddit="llm"),
        ]
        # top_n=2 with 3 candidates forces actual reranking
        reranked = diversity_rerank(results, top_n=2, author_penalty=0.5)

        # After alice's first result, bob (0.8) should rank above
        # penalized alice (0.85 * 0.5 = 0.425)
        authors = [r.metadata["author"] for r in reranked]
        assert len(reranked) == 2
        assert authors[0] == "alice"  # First is still alice (highest score)
        assert authors[1] == "bob"    # Bob promoted over penalized alice

    def test_preserves_count(self):
        """Reranking shouldn't change result count when within top_n."""
        results = [_make_result(f"item_{i}", 1.0 - i * 0.1, "fused", author=f"user_{i}") for i in range(5)]
        reranked = diversity_rerank(results, top_n=5)
        assert len(reranked) == 5

    def test_no_change_for_diverse_results(self):
        """Already-diverse results should maintain their order."""
        results = [
            _make_result("a", 0.9, "fused", author="alice", subreddit="ml"),
            _make_result("b", 0.8, "fused", author="bob", subreddit="llama"),
            _make_result("c", 0.7, "fused", author="charlie", subreddit="ai"),
        ]
        reranked = diversity_rerank(results, top_n=3)
        assert [r.id for r in reranked] == ["a", "b", "c"]


class TestQueryWeights:
    """Test query-type-aware weight selection."""

    def test_semantic_favors_vector(self):
        weights = get_query_weights("semantic")
        assert weights[1] > weights[0]  # vector > graph

    def test_graph_favors_graph(self):
        weights = get_query_weights("graph")
        assert weights[0] > weights[1]  # graph > vector

    def test_hybrid_is_balanced(self):
        weights = get_query_weights("hybrid")
        assert weights[0] == weights[1]

    def test_unknown_type_returns_balanced(self):
        weights = get_query_weights("unknown_type")
        assert len(weights) == 2


class TestDedupKey:
    """Test chunk deduplication key extraction."""

    def test_strips_chunk_suffix(self):
        assert _dedup_key("post_123_chunk_0") == "post_123"
        assert _dedup_key("post_123_chunk_5") == "post_123"

    def test_preserves_non_chunked_id(self):
        assert _dedup_key("post_123") == "post_123"
        assert _dedup_key("user_alice") == "user_alice"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
