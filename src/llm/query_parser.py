"""LLM-powered query understanding with temporal expression resolution.

Handles relative time expressions, query intent classification, and
entity extraction to route queries to the optimal retrieval strategy.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

from dateutil.relativedelta import relativedelta

from config import get_settings
from src.models import ParsedQuery
from src.llm.client import LLMClient

SYSTEM = """You are a query parser for a Reddit GraphRAG system.
Parse user questions into structured retrieval parameters.
Return ONLY valid JSON with these fields:
- semantic_query: reformulated query for vector search
- graph_entities: list of entity/topic names to traverse in the knowledge graph
- graph_relationships: relationship types to follow (MENTIONS, AUTHORED, POSTED_IN, HAS_SENTIMENT, DISCUSSES)
- query_type: one of "semantic", "graph", "hybrid", "temporal"
- time_start: ISO datetime or null
- time_end: ISO datetime or null
- compare_start: ISO datetime or null (for period comparison queries)
- compare_end: ISO datetime or null
- subreddits: list of subreddit names or empty list

Classify query_type:
- semantic: definitional/explanatory ("what is", "how does", "main challenges")
- graph: relationship/influence ("who are influential", "which communities lead")
- hybrid: needs both graph structure and semantic content
- temporal: compares time periods or asks about trends over time"""


class QueryParser:
    """Parse natural language queries into structured retrieval parameters."""

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.llm = llm or LLMClient()
        self.settings = get_settings()
        # Use current time as reference for relative expressions
        self._reference_date = datetime.now()

    def parse(self, question: str) -> ParsedQuery:
        """Parse a natural language question into retrieval parameters."""
        if not self.llm.has_api_key():
            data = self._heuristic_parse(question)
        else:
            prompt = f"""Parse this user question for Reddit GraphRAG retrieval.
Reference date for this query: {self._reference_date.isoformat()}

Question: {question}

Resolve relative times like "last 6 months", "Q4 2025", "past quarter" to ISO datetimes.
Return JSON only."""

            try:
                data = self.llm.complete_json(prompt, system=SYSTEM)
                if (
                    not self._looks_like_parse_result(data)
                    or str(data.get("semantic_query", "")).startswith("Parse this user question")
                ):
                    data = self._heuristic_parse(question)
            except Exception:
                data = self._heuristic_parse(question)

        return ParsedQuery(
            original=question,
            semantic_query=data.get("semantic_query", question),
            graph_entities=data.get("graph_entities", []),
            graph_relationships=data.get("graph_relationships", []),
            time_start=self._parse_dt(data.get("time_start")),
            time_end=self._parse_dt(data.get("time_end")),
            compare_start=self._parse_dt(data.get("compare_start")),
            compare_end=self._parse_dt(data.get("compare_end")),
            query_type=data.get("query_type", "hybrid"),
            subreddits=data.get("subreddits", []),
        )

    def _heuristic_parse(self, question: str) -> dict:
        """Parse query using heuristic rules when no LLM is available."""
        q = question.lower()

        # ── Query type classification ──────────────────────────────────────
        graph_signals = any(w in q for w in (
            "who are", "influential", "communities lead", "leading the conversation",
            "most active", "top contributors", "key voices",
        ))
        semantic_signals = (
            q.startswith(("what are", "what is", "how does", "explain", "define"))
            or any(w in q for w in ("main challenges", "best practices", "guide to", "priorities", "distinguish"))
        )
        temporal_signals = any(w in q for w in (
            "changed over", "emerging", "weren't discussed",
            "trend", "evolution", "shift", "compared to",
            "last 6 months", "past quarter", "over time",
            "q1 2025", "q2 2025", "q3 2025", "q4 2025",
            "q1 2026", "q2 2026",
        ))

        query_type = "hybrid"
        if graph_signals and not semantic_signals:
            query_type = "graph"
        elif semantic_signals and not graph_signals:
            query_type = "semantic"
        elif graph_signals and semantic_signals:
            query_type = "hybrid"

        # Temporal indicators override non-hybrid types only when the
        # question is clearly about change over time rather than structure.
        if temporal_signals and not (graph_signals and semantic_signals):
            query_type = "temporal"

        # ── Time expression resolution ─────────────────────────────────────
        time_start, time_end = self._resolve_time_expressions(q)
        compare_start, compare_end = self._resolve_comparison_periods(q)

        # ── Entity extraction ──────────────────────────────────────────────
        entities = self._extract_entities(q)

        # ── Semantic query reformulation ───────────────────────────────────
        semantic_query = question
        # Strip temporal modifiers for cleaner vector search
        for pattern in (
            r"in q[1-4] 20\d{2}", r"over the last \d+ months?",
            r"in the past \d+ months?", r"since \w+ 20\d{2}",
            r"that weren'?t discussed in .*$",
        ):
            semantic_query = re.sub(pattern, "", semantic_query, flags=re.IGNORECASE).strip()

        return {
            "semantic_query": semantic_query,
            "graph_entities": entities,
            "graph_relationships": ["MENTIONS", "HAS_SENTIMENT"],
            "query_type": query_type,
            "time_start": time_start.isoformat() if time_start else None,
            "time_end": time_end.isoformat() if time_end else None,
            "compare_start": compare_start.isoformat() if compare_start else None,
            "compare_end": compare_end.isoformat() if compare_end else None,
            "subreddits": [],
        }

    def _resolve_time_expressions(self, q: str) -> tuple:
        """Resolve relative and absolute time expressions."""
        ref = self._reference_date
        time_start, time_end = None, None

        # Absolute quarters: Q1 2026, Q4 2025, etc.
        quarter_match = re.search(r"q([1-4])\s*20(\d{2})", q)
        if quarter_match:
            qnum = int(quarter_match.group(1))
            year = int(f"20{quarter_match.group(2)}")
            month_start = (qnum - 1) * 3 + 1
            time_start = datetime(year, month_start, 1)
            month_end = month_start + 2
            if month_end == 12:
                time_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                time_end = datetime(year, month_end + 1, 1) - timedelta(seconds=1)

        # Relative: "last N months", "past N months"
        months_match = re.search(r"(?:last|past)\s+(\d+)\s+months?", q)
        if months_match:
            n = int(months_match.group(1))
            time_end = ref
            time_start = ref - relativedelta(months=n)

        # Relative: "last quarter", "past quarter"
        if "last quarter" in q or "past quarter" in q:
            current_quarter = (ref.month - 1) // 3
            if current_quarter == 0:
                time_start = datetime(ref.year - 1, 10, 1)
                time_end = datetime(ref.year - 1, 12, 31, 23, 59, 59)
            else:
                month_start = (current_quarter - 1) * 3 + 1
                month_end = month_start + 2
                time_start = datetime(ref.year, month_start, 1)
                if month_end == 12:
                    time_end = datetime(ref.year, 12, 31, 23, 59, 59)
                else:
                    time_end = datetime(ref.year, month_end + 1, 1) - timedelta(seconds=1)

        # Relative: "last year", "past year"
        if "last year" in q or "past year" in q:
            time_end = ref
            time_start = ref - relativedelta(years=1)

        # Relative: "this month", "this quarter"
        if "this month" in q:
            time_start = datetime(ref.year, ref.month, 1)
            time_end = ref

        if "this quarter" in q:
            current_quarter = (ref.month - 1) // 3
            month_start = current_quarter * 3 + 1
            time_start = datetime(ref.year, month_start, 1)
            time_end = ref

        # "since January", "since March 2025"
        since_match = re.search(r"since\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?", q)
        if since_match:
            months = {"january": 1, "february": 2, "march": 3, "april": 4,
                       "may": 5, "june": 6, "july": 7, "august": 8,
                       "september": 9, "october": 10, "november": 11, "december": 12}
            month = months[since_match.group(1)]
            year = int(since_match.group(2)) if since_match.group(2) else ref.year
            time_start = datetime(year, month, 1)
            time_end = ref

        # "6 months ago", "3 months ago"
        ago_match = re.search(r"(\d+)\s+months?\s+ago", q)
        if ago_match and not time_start:
            n = int(ago_match.group(1))
            time_start = ref - relativedelta(months=n)
            time_end = ref

        return time_start, time_end

    def _resolve_comparison_periods(self, q: str) -> tuple:
        """Resolve comparison period (the 'other' period in temporal comparisons)."""
        compare_start, compare_end = None, None

        # Handle "X that weren't discussed in Y" patterns first so the
        # comparison period is the earlier reference, not the first quarter mentioned.
        if "weren't discussed" in q or "wasn't discussed" in q or "weren't mentioned" in q or "not discussed" in q:
            weren_match = re.search(r"(?:weren't|wasn't|were not|was not|not)\s+discussed\s+in\s+q([1-4])\s*20(\d{2})", q)
            if weren_match:
                qnum = int(weren_match.group(1))
                year = int(f"20{weren_match.group(2)}")
                month_start = (qnum - 1) * 3 + 1
                compare_start = datetime(year, month_start, 1)
                month_end = month_start + 2
                if month_end == 12:
                    compare_end = datetime(year, 12, 31, 23, 59, 59)
                else:
                    compare_end = datetime(year, month_end + 1, 1) - timedelta(seconds=1)
                return compare_start, compare_end

        # Look for two quarter references for comparison
        quarters = re.findall(r"q([1-4])\s*20(\d{2})", q)
        if len(quarters) >= 2:
            # First mentioned quarter becomes the comparison period
            qnum = int(quarters[0][0])
            year = int(f"20{quarters[0][1]}")
            month_start = (qnum - 1) * 3 + 1
            compare_start = datetime(year, month_start, 1)
            month_end = month_start + 2
            if month_end == 12:
                compare_end = datetime(year, 12, 31, 23, 59, 59)
            else:
                compare_end = datetime(year, month_end + 1, 1) - timedelta(seconds=1)

        # "compared to last quarter", "vs last month"
        if ("compared to" in q or "vs" in q) and "last quarter" in q:
            ref = self._reference_date
            current_quarter = (ref.month - 1) // 3
            if current_quarter == 0:
                compare_start = datetime(ref.year - 1, 10, 1)
                compare_end = datetime(ref.year - 1, 12, 31, 23, 59, 59)
            else:
                month_start = (current_quarter - 1) * 3 + 1
                month_end = month_start + 2
                compare_start = datetime(ref.year, month_start, 1)
                if month_end == 12:
                    compare_end = datetime(ref.year, 12, 31, 23, 59, 59)
                else:
                    compare_end = datetime(ref.year, month_end + 1, 1) - timedelta(seconds=1)

        return compare_start, compare_end

    @staticmethod
    def _extract_entities(q: str) -> list:
        """Extract known entities and topics from the query."""
        entities = []
        # Ordered by specificity (longer matches first)
        known_terms = [
            "retrieval augmented generation", "open-source LLM", "open source LLM",
            "AI regulation", "AI safety", "agentic AI", "AI agents",
            "knowledge graph", "vector search",
            "GraphRAG", "RAG", "LLM",
        ]
        for term in known_terms:
            if term.lower() in q:
                entities.append(term)

        person_terms = [
            "Yoshua Bengio", "Stuart Russell", "Timnit Gebru",
            "Sam Altman", "Yann LeCun", "Andrew Ng", "Demis Hassabis",
        ]
        for term in person_terms:
            if term.lower() in q:
                entities.append(term)

        return entities

    @staticmethod
    def _parse_dt(value) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00").split("+")[0])
        except ValueError:
            return None

    @staticmethod
    def _looks_like_parse_result(data: object) -> bool:
        if not isinstance(data, dict):
            return False
        return all(
            key in data
            for key in ("semantic_query", "graph_entities", "query_type")
        )
