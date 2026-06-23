"""In-memory graph fallback using dictionary-based storage when Neo4j is unavailable.

Provides the same API as Neo4jStore but stores everything in-memory with
pickle persistence. Suitable for demos and environments without Docker.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_settings
from src.models import ContentType, EnrichedItem, RetrievalResult

FALLBACK_GRAPH_PATH = Path("./data/fallback_graph.pkl")


class NetworkXGraphStore:
    """Lightweight in-memory graph for demo without Docker/Neo4j."""

    def __init__(self) -> None:
        self.posts: Dict[str, dict] = {}
        self.comments: Dict[str, dict] = {}
        self.users: Dict[str, dict] = {}
        self.entities: Dict[str, dict] = {}
        self.topics: Dict[str, dict] = {}
        self.mentions: List[dict] = []

    def close(self) -> None:
        self.save()

    def save(self) -> None:
        FALLBACK_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FALLBACK_GRAPH_PATH, "wb") as f:
            pickle.dump({
                "posts": self.posts,
                "comments": self.comments,
                "users": self.users,
                "entities": self.entities,
                "topics": self.topics,
                "mentions": self.mentions,
            }, f)

    @classmethod
    def load(cls) -> "NetworkXGraphStore":
        store = cls()
        if FALLBACK_GRAPH_PATH.exists():
            with open(FALLBACK_GRAPH_PATH, "rb") as f:
                data = pickle.load(f)
            store.posts = data.get("posts", {})
            store.comments = data.get("comments", {})
            store.users = data.get("users", {})
            store.entities = data.get("entities", {})
            store.topics = data.get("topics", {})
            store.mentions = data.get("mentions", [])
        return store

    def verify_connection(self) -> bool:
        return True

    def init_schema(self) -> None:
        pass

    def clear(self) -> None:
        self.posts = {}
        self.comments = {}
        self.users = {}
        self.entities = {}
        self.topics = {}
        self.mentions = []
        if FALLBACK_GRAPH_PATH.exists():
            FALLBACK_GRAPH_PATH.unlink()

    def ingest_enriched(self, enriched: EnrichedItem) -> None:
        item = enriched.item
        created_iso = item.created_at.isoformat()

        self.users[item.author] = {
            "name": item.author,
            "last_seen": created_iso,
            "total_posts": self.users.get(item.author, {}).get("total_posts", 0) + 1,
        }

        record = {
            "id": item.id,
            "title": item.title,
            "body": item.body,
            "author": item.author,
            "subreddit": item.subreddit,
            "created_at": created_iso,
            "score": item.score,
            "url": item.permalink,
            "window": item.window_label,
            "summary": enriched.summary,
            "sentiment": enriched.overall_sentiment,
        }

        if item.content_type == ContentType.POST:
            self.posts[item.id] = record
        else:
            self.comments[item.id] = record

        for topic in enriched.topics:
            self.topics[topic] = {"name": topic, "created_at": created_iso}
            self.mentions.append({
                "content_id": item.id,
                "entity": topic,
                "entity_type": "topic",
                "sentiment": enriched.overall_sentiment,
            })

        for entity in enriched.entities:
            self.entities[entity.name] = {
                "name": entity.name,
                "type": entity.entity_type,
                "created_at": created_iso,
            }
            self.mentions.append({
                "content_id": item.id,
                "entity": entity.name,
                "entity_type": "entity",
                "sentiment": entity.sentiment,
                "confidence": entity.confidence,
            })

    def _all_content(self) -> List[dict]:
        return list(self.posts.values()) + list(self.comments.values())

    def _text(self, record: dict) -> str:
        return f"{record.get('title', '')} {record.get('body', '')}".strip()

    def _entity_matches(self, candidate: str, entities: List[str]) -> bool:
        if not entities:
            return True
        candidate_lower = candidate.lower()
        return any(
            ent.lower() in candidate_lower or candidate_lower in ent.lower()
            for ent in entities
        )

    def _filter_time(
        self,
        records: List[dict],
        time_start: Optional[datetime],
        time_end: Optional[datetime],
    ) -> List[dict]:
        result = []
        for r in records:
            ts = r.get("created_at", "")
            if time_start and ts < time_start.isoformat():
                continue
            if time_end and ts > time_end.isoformat():
                continue
            result.append(r)
        return result

    # ── Search Queries ─────────────────────────────────────────────────────

    def search(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        subreddits: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        records = self._filter_time(self._all_content(), time_start, time_end)

        if subreddits:
            records = [r for r in records if r["subreddit"] in subreddits]

        scored = []
        for r in records:
            text = self._text(r)
            mention_count = 0
            matched = ""
            for m in self.mentions:
                if m["content_id"] != r["id"]:
                    continue
                for ent in entities:
                    if ent.lower() in m["entity"].lower():
                        mention_count += 1
                        matched = m["entity"]

            if entities and mention_count == 0:
                continue

            score = (mention_count or 1) * r.get("score", 1)
            scored.append((score, r, matched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalResult(
                id=r["id"],
                text=self._text(r),
                score=float(score),
                source="graph",
                metadata={
                    "author": r["author"],
                    "subreddit": r["subreddit"],
                    "created_at": r["created_at"],
                    "url": r["url"],
                    "window": r.get("window", ""),
                    "sentiment": r.get("sentiment", "neutral"),
                    "matched_entity": matched,
                },
            )
            for score, r, matched in scored[:limit]
        ]

    def find_influential_users(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        results = self.search(entities, time_start, time_end, limit=100)
        user_data: Dict[str, dict] = defaultdict(lambda: {"score": 0, "posts": [], "count": 0})

        for r in results:
            author = r.metadata["author"]
            user_data[author]["score"] += r.score
            user_data[author]["count"] += 1
            user_data[author]["posts"].append(r.text[:300])

        ranked = sorted(user_data.items(), key=lambda x: x[1]["score"], reverse=True)[:limit]
        return [
            RetrievalResult(
                id=f"user_{author}",
                text=f"Author: {author} (score={data['score']}, posts={data['count']})\n"
                     + "\n---\n".join(data["posts"][:3]),
                score=float(data["score"]),
                source="graph",
                metadata={"author": author, "post_count": data["count"], "type": "influential_user"},
            )
            for author, data in ranked
        ]

    def find_influential_entities(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
        entity_type: str = "person",
    ) -> List[RetrievalResult]:
        """Find influential named entities mentioned in relevant content."""
        content_by_id = {r["id"]: r for r in self._filter_time(self._all_content(), time_start, time_end)}
        entity_data: Dict[str, dict] = defaultdict(
            lambda: {"score": 0.0, "count": 0, "authors": set(), "subreddits": set(), "samples": []}
        )

        for mention in self.mentions:
            content = content_by_id.get(mention["content_id"])
            if not content:
                continue
            entity_name = mention["entity"]
            entity_info = self.entities.get(entity_name, {})
            if entity_type and entity_info.get("type") != entity_type:
                continue
            if entities and not any(
                self._entity_matches(linked["entity"], entities)
                for linked in self.mentions
                if linked["content_id"] == mention["content_id"]
            ):
                continue

            entity_data[entity_name]["score"] += max(1.0, float(content.get("score", 0)))
            entity_data[entity_name]["count"] += 1
            entity_data[entity_name]["authors"].add(content.get("author", "unknown"))
            entity_data[entity_name]["subreddits"].add(content.get("subreddit", ""))
            if len(entity_data[entity_name]["samples"]) < 3:
                entity_data[entity_name]["samples"].append(self._text(content)[:240])

        ranked = sorted(entity_data.items(), key=lambda item: (item[1]["score"], item[1]["count"]), reverse=True)[:limit]
        return [
            RetrievalResult(
                id=f"entity_{name}",
                text=(
                    f"{name}: mentioned {data['count']} times across {len(data['authors'])} authors "
                    f"in {', '.join(sorted(s for s in data['subreddits'] if s)[:3]) or 'multiple communities'}.\n"
                    + "\n---\n".join(data["samples"])
                ),
                score=float(data["score"]),
                source="graph",
                metadata={
                    "entity": name,
                    "entity_type": entity_type,
                    "author": name,
                    "subreddit": ", ".join(sorted(data["subreddits"])[:3]),
                    "type": "influential_entity",
                },
            )
            for name, data in ranked
        ]

    def community_leadership(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        results = self.search(entities, time_start, time_end, limit=100)
        sub_data: Dict[str, dict] = defaultdict(lambda: {"count": 0, "score": 0})

        for r in results:
            sub = r.metadata["subreddit"]
            sub_data[sub]["count"] += 1
            sub_data[sub]["score"] += r.score

        ranked = sorted(sub_data.items(), key=lambda x: x[1]["count"], reverse=True)[:limit]
        return [
            RetrievalResult(
                id=f"subreddit_{sub}",
                text=f"r/{sub}: {data['count']} discussions, total score {data['score']}",
                score=float(data["count"]),
                source="graph",
                metadata={"subreddit": sub, "discussion_count": data["count"], "type": "community_leadership"},
            )
            for sub, data in ranked
        ]

    def entity_co_occurrence(
        self,
        entity: str,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Find entities that frequently co-occur with the given entity."""
        # Find all content IDs that mention the target entity
        matching_ids = set()
        for m in self.mentions:
            if entity.lower() in m["entity"].lower():
                matching_ids.add(m["content_id"])

        # Apply time filter
        all_content = {r["id"]: r for r in self._all_content()}
        if time_start or time_end:
            filtered = self._filter_time(list(all_content.values()), time_start, time_end)
            valid_ids = {r["id"] for r in filtered}
            matching_ids &= valid_ids

        # Find co-occurring entities
        co_counts: Dict[str, int] = defaultdict(int)
        co_posts: Dict[str, set] = defaultdict(set)

        for m in self.mentions:
            if m["content_id"] in matching_ids and entity.lower() not in m["entity"].lower():
                co_counts[m["entity"]] += 1
                co_posts[m["entity"]].add(m["content_id"])

        ranked = sorted(co_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            RetrievalResult(
                id=f"cooccur_{co_ent}",
                text=f"{entity} ↔ {co_ent}: co-occurs in {count} mentions across {len(co_posts[co_ent])} posts",
                score=float(count),
                source="graph",
                metadata={
                    "co_entity": co_ent,
                    "co_count": count,
                    "type": "co_occurrence",
                },
            )
            for co_ent, count in ranked
        ]

    def topic_evolution(
        self,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Track how topic mention frequency changes across windows."""
        all_content = {r["id"]: r for r in self._all_content()}

        if time_start or time_end:
            filtered = self._filter_time(list(all_content.values()), time_start, time_end)
            valid_ids = {r["id"] for r in filtered}
        else:
            valid_ids = set(all_content.keys())

        # Count topic mentions per window
        topic_window_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for m in self.mentions:
            if m["content_id"] in valid_ids and m["content_id"] in all_content:
                window = all_content[m["content_id"]].get("window", "unknown")
                topic_window_counts[m["entity"]][window] += 1

        results = []
        for topic, windows in sorted(topic_window_counts.items()):
            for window, count in sorted(windows.items()):
                results.append({
                    "topic": topic,
                    "window": window,
                    "mention_count": count,
                })

        return results[:limit * 3]  # Multiple entries per topic

    def sentiment_over_time(
        self,
        entity: str,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Track sentiment for an entity across time windows."""
        all_content = {r["id"]: r for r in self._all_content()}

        # Find content mentioning this entity
        matching_ids = set()
        for m in self.mentions:
            if entity.lower() in m["entity"].lower():
                matching_ids.add(m["content_id"])

        # Apply time filter
        if time_start or time_end:
            filtered = self._filter_time(list(all_content.values()), time_start, time_end)
            valid_ids = {r["id"] for r in filtered}
            matching_ids &= valid_ids

        # Aggregate by window
        periods: Dict[str, dict] = defaultdict(lambda: {"cnt": 0, "sentiments": [], "scores": []})
        for cid in matching_ids:
            if cid in all_content:
                record = all_content[cid]
                window = record.get("window", "unknown")
                periods[window]["cnt"] += 1
                periods[window]["sentiments"].append(record.get("sentiment", "neutral"))
                periods[window]["scores"].append(record.get("score", 0))

        return {
            "entity": entity,
            "periods": [
                {
                    "window": w,
                    "cnt": d["cnt"],
                    "sentiment": max(set(d["sentiments"]), key=d["sentiments"].count) if d["sentiments"] else "neutral",
                    "avg_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0,
                }
                for w, d in sorted(periods.items())
            ],
        }

    def count_nodes(self) -> Dict[str, int]:
        return {
            "User": len(self.users),
            "Post": len(self.posts),
            "Comment": len(self.comments),
            "Topic": len(self.topics),
            "Entity": len(self.entities),
        }
