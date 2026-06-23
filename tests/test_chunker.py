"""Tests for text chunking with metadata preservation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from datetime import datetime

from src.models import ContentType, RedditItem, EnrichedItem, ExtractedEntity
from src.vector.chunker import chunk_text


@pytest.fixture
def sample_enriched():
    """Create a sample enriched item for testing."""
    item = RedditItem(
        id="post_test1",
        content_type=ContentType.POST,
        title="Test Post Title",
        body="This is the body of a test post about RAG systems.",
        author="test_user",
        subreddit="MachineLearning",
        created_utc=datetime(2025, 10, 15, 12, 0, 0).timestamp(),
        score=42,
        url="/r/MachineLearning/comments/test1/",
        window_label="Q4_2025",
    )
    return EnrichedItem(
        item=item,
        entities=[
            ExtractedEntity(name="RAG", entity_type="technology", sentiment="positive"),
        ],
        overall_sentiment="positive",
        topics=["RAG", "AI"],
        summary="A test post about RAG systems.",
    )


@pytest.fixture
def long_enriched():
    """Create an enriched item with content longer than chunk size."""
    body = "This is a sentence about AI and RAG systems. " * 50  # ~2500 chars
    item = RedditItem(
        id="post_long1",
        content_type=ContentType.POST,
        title="Long Post",
        body=body,
        author="test_user",
        subreddit="MachineLearning",
        created_utc=datetime(2025, 10, 15, 12, 0, 0).timestamp(),
        score=100,
        url="/r/MachineLearning/comments/long1/",
        window_label="Q4_2025",
    )
    return EnrichedItem(item=item, overall_sentiment="positive", topics=["AI"])


class TestChunking:
    """Test text chunking logic."""

    def test_short_text_single_chunk(self, sample_enriched):
        chunks = chunk_text(sample_enriched, chunk_size=500)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, long_enriched):
        chunks = chunk_text(long_enriched, chunk_size=500)
        assert len(chunks) > 1

    def test_empty_text_no_chunks(self):
        item = RedditItem(
            id="post_empty", content_type=ContentType.POST,
            title="", body="", author="u", subreddit="s",
            created_utc=0, score=0, url="/",
        )
        chunks = chunk_text(EnrichedItem(item=item))
        assert chunks == []

    def test_chunk_overlap(self, long_enriched):
        chunks = chunk_text(long_enriched, chunk_size=200, overlap=50)
        if len(chunks) >= 2:
            # The end of chunk 0 should overlap with the start of chunk 1
            chunk0_end = chunks[0]["text"][-50:]
            chunk1_start = chunks[1]["text"][:50]
            assert chunk0_end == chunk1_start


class TestChunkMetadata:
    """Test that metadata is correctly preserved in chunks."""

    def test_metadata_has_required_fields(self, sample_enriched):
        chunks = chunk_text(sample_enriched)
        assert len(chunks) == 1
        meta = chunks[0]["metadata"]

        assert meta["content_id"] == "post_test1"
        assert meta["content_type"] == "post"
        assert meta["author"] == "test_user"
        assert meta["subreddit"] == "MachineLearning"
        assert meta["window"] == "Q4_2025"
        assert meta["sentiment"] == "positive"

    def test_metadata_has_timestamp(self, sample_enriched):
        chunks = chunk_text(sample_enriched)
        meta = chunks[0]["metadata"]
        assert "created_ts" in meta
        assert isinstance(meta["created_ts"], float)
        assert "created_at" in meta

    def test_metadata_has_topics(self, sample_enriched):
        chunks = chunk_text(sample_enriched)
        meta = chunks[0]["metadata"]
        assert "topics" in meta
        assert "RAG" in meta["topics"]

    def test_chunk_index_tracking(self, long_enriched):
        chunks = chunk_text(long_enriched, chunk_size=500)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["chunk_total"] == len(chunks)

    def test_chunk_id_format(self, sample_enriched):
        chunks = chunk_text(sample_enriched)
        assert chunks[0]["id"] == "post_test1_chunk_0"

    def test_multi_chunk_ids_are_unique(self, long_enriched):
        chunks = chunk_text(long_enriched, chunk_size=200)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
