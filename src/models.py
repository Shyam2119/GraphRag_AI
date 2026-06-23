"""Shared data models for ingestion, graph, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ContentType(str, Enum):
    POST = "post"
    COMMENT = "comment"


@dataclass
class RedditItem:
    id: str
    content_type: ContentType
    title: str
    body: str
    author: str
    subreddit: str
    created_utc: float
    score: int
    url: str
    parent_id: Optional[str] = None
    post_id: Optional[str] = None
    window_label: str = ""

    @property
    def created_at(self) -> datetime:
        return datetime.utcfromtimestamp(self.created_utc)

    @property
    def full_text(self) -> str:
        if self.content_type == ContentType.POST:
            return f"{self.title}\n\n{self.body}".strip()
        return self.body.strip()

    @property
    def permalink(self) -> str:
        return f"https://reddit.com{self.url}" if self.url.startswith("/") else self.url


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str  # topic, technology, person, concern, organization
    sentiment: str  # positive, negative, neutral, mixed
    confidence: float = 0.8


@dataclass
class EnrichedItem:
    item: RedditItem
    entities: List[ExtractedEntity] = field(default_factory=list)
    overall_sentiment: str = "neutral"
    topics: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class RetrievalResult:
    id: str
    text: str
    score: float
    source: str  # graph | vector | fused
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        author = self.metadata.get("author", "unknown")
        sub = self.metadata.get("subreddit", "")
        ts = self.metadata.get("created_at", "")
        url = self.metadata.get("url", self.id)
        return f"[{author} in r/{sub}, {ts}] {url}"


@dataclass
class ParsedQuery:
    original: str
    semantic_query: str
    graph_entities: List[str] = field(default_factory=list)
    graph_relationships: List[str] = field(default_factory=list)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    compare_start: Optional[datetime] = None
    compare_end: Optional[datetime] = None
    query_type: str = "hybrid"  # semantic | graph | hybrid | temporal
    subreddits: List[str] = field(default_factory=list)


@dataclass
class QueryResponse:
    question: str
    answer: str
    graph_results: List[RetrievalResult]
    vector_results: List[RetrievalResult]
    fused_results: List[RetrievalResult]
    parsed_query: ParsedQuery
    period_comparison: Optional[Dict[str, Any]] = None
