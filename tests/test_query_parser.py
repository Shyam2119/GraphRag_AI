"""Tests for query parser temporal expression resolution and intent classification."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime
from unittest.mock import patch

from src.llm.query_parser import QueryParser


@pytest.fixture
def parser():
    """Create a QueryParser with a fixed reference date for deterministic tests."""
    p = QueryParser()
    p._reference_date = datetime(2026, 6, 15)  # Fixed reference
    return p


class TestQueryTypeClassification:
    """Test heuristic query type detection."""

    def test_semantic_query(self, parser):
        result = parser._heuristic_parse("What are the main challenges of building RAG?")
        assert result["query_type"] == "semantic"

    def test_graph_query_influential(self, parser):
        result = parser._heuristic_parse("Who are the most influential voices in AI?")
        assert result["query_type"] == "graph"

    def test_graph_query_communities(self, parser):
        result = parser._heuristic_parse("Which communities lead the discussion on LLMs?")
        assert result["query_type"] == "graph"

    def test_temporal_query_changed_over(self, parser):
        result = parser._heuristic_parse("How has sentiment changed over the last 6 months?")
        assert result["query_type"] == "temporal"

    def test_temporal_query_quarter_comparison(self, parser):
        result = parser._heuristic_parse("What emerged in Q1 2026 that wasn't discussed in Q4 2025?")
        assert result["query_type"] == "temporal"

    def test_hybrid_default(self, parser):
        result = parser._heuristic_parse("Tell me about RAG and knowledge graphs")
        assert result["query_type"] == "hybrid"

    def test_hybrid_query_not_misclassified_as_temporal(self, parser):
        result = parser._heuristic_parse(
            "Which communities are leading the conversation on open-source LLMs, and what priorities distinguish them?"
        )
        assert result["query_type"] == "hybrid"


class TestTimeExpressionResolution:
    """Test relative and absolute time expression parsing."""

    def test_absolute_quarter_q4_2025(self, parser):
        result = parser._heuristic_parse("What happened in Q4 2025?")
        assert result["time_start"] == "2025-10-01T00:00:00"
        assert "2025-12-31" in result["time_end"]

    def test_absolute_quarter_q1_2026(self, parser):
        result = parser._heuristic_parse("Trends in Q1 2026")
        assert result["time_start"] == "2026-01-01T00:00:00"
        assert "2026-03-31" in result["time_end"]

    def test_last_6_months(self, parser):
        result = parser._heuristic_parse("What happened in the last 6 months?")
        assert result["time_start"] is not None
        assert result["time_end"] is not None
        start = datetime.fromisoformat(result["time_start"])
        end = datetime.fromisoformat(result["time_end"])
        assert (end - start).days >= 170  # ~6 months

    def test_last_3_months(self, parser):
        result = parser._heuristic_parse("Developments in the past 3 months")
        assert result["time_start"] is not None

    def test_since_month(self, parser):
        result = parser._heuristic_parse("What's happened since January?")
        assert result["time_start"] is not None
        start = datetime.fromisoformat(result["time_start"])
        assert start.month == 1

    def test_since_month_with_year(self, parser):
        result = parser._heuristic_parse("Changes since March 2025")
        assert result["time_start"] is not None
        start = datetime.fromisoformat(result["time_start"])
        assert start.month == 3
        assert start.year == 2025

    def test_no_time_expression(self, parser):
        result = parser._heuristic_parse("What is RAG?")
        assert result["time_start"] is None
        assert result["time_end"] is None


class TestComparisonPeriods:
    """Test comparison period resolution for temporal queries."""

    def test_two_quarters_comparison(self, parser):
        result = parser._heuristic_parse("What emerged in Q1 2026 that wasn't discussed in Q4 2025?")
        # The "weren't discussed in Q4 2025" pattern captures Q4 2025 as comparison
        assert result["compare_start"] is not None
        compare_start = datetime.fromisoformat(result["compare_start"])
        # Q4 2025 starts October
        assert compare_start.year == 2025
        assert compare_start.month == 10
        assert compare_start.year == 2025

    def test_werent_discussed_pattern(self, parser):
        result = parser._heuristic_parse("New concerns that weren't discussed in Q4 2025")
        assert result["compare_start"] is not None


class TestEntityExtraction:
    """Test entity extraction from queries."""

    def test_extracts_rag(self, parser):
        result = parser._heuristic_parse("How has RAG evolved?")
        assert "RAG" in result["graph_entities"]

    def test_extracts_ai_safety(self, parser):
        result = parser._heuristic_parse("What are the AI safety concerns?")
        assert "AI safety" in result["graph_entities"]

    def test_extracts_multiple(self, parser):
        result = parser._heuristic_parse("RAG and AI safety developments")
        assert "RAG" in result["graph_entities"]
        assert "AI safety" in result["graph_entities"]

    def test_no_entities_for_generic_query(self, parser):
        result = parser._heuristic_parse("What is the weather?")
        assert result["graph_entities"] == []

    def test_extracts_person_entity(self, parser):
        result = parser._heuristic_parse("What is Yoshua Bengio saying about AI regulation?")
        assert "Yoshua Bengio" in result["graph_entities"]


class TestParsedQuery:
    """Test the full parse() method output."""

    def test_parse_returns_parsed_query(self, parser):
        result = parser.parse("What are RAG challenges?")
        assert result.original == "What are RAG challenges?"
        assert result.query_type in ("semantic", "hybrid", "graph", "temporal")

    def test_parse_temporal_has_time_range(self, parser):
        result = parser.parse("How has RAG changed over the last 6 months?")
        assert result.time_start is not None
        assert result.time_end is not None

    def test_parse_comparison_has_two_periods(self, parser):
        result = parser.parse("What emerged in Q1 2026 that wasn't discussed in Q4 2025?")
        assert result.time_start is not None
        assert result.compare_start is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
