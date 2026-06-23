"""Neo4j knowledge graph schema and temporal entity-relationship model.

Graph Design:
  Nodes: User, Post, Comment, Subreddit, Topic, Entity, Sentiment
  Relationships (all carry created_at):
    (User)-[:AUTHORED]->(Post|Comment)
    (Post)-[:POSTED_IN]->(Subreddit)
    (Comment)-[:ON_POST]->(Post)
    (Comment)-[:REPLIED_TO]->(Comment)
    (Post|Comment)-[:MENTIONS]->(Entity|Topic)
    (Post|Comment)-[:HAS_SENTIMENT]->(Sentiment)
    (User)-[:ACTIVE_IN]->(Subreddit)

Justification: Neo4j excels at multi-hop traversals (influence chains,
community leadership, entity co-occurrence) and native temporal filtering
via relationship/node properties. Cypher queries express graph patterns
naturally — something vector databases cannot do.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from config import get_settings
from src.models import ContentType, EnrichedItem, RetrievalResult


class Neo4jStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def init_schema(self) -> None:
        constraints = [
            "CREATE CONSTRAINT user_name IF NOT EXISTS FOR (u:User) REQUIRE u.name IS UNIQUE",
            "CREATE CONSTRAINT post_id IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT subreddit_name IF NOT EXISTS FOR (s:Subreddit) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX post_created IF NOT EXISTS FOR (p:Post) ON (p.created_at)",
            "CREATE INDEX comment_created IF NOT EXISTS FOR (c:Comment) ON (c.created_at)",
            "CREATE INDEX post_window IF NOT EXISTS FOR (p:Post) ON (p.window)",
            "CREATE INDEX comment_window IF NOT EXISTS FOR (c:Comment) ON (c.window)",
        ]
        with self.driver.session() as session:
            for cypher in constraints:
                session.run(cypher)

    def clear(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def ingest_enriched(self, enriched: EnrichedItem) -> None:
        item = enriched.item
        with self.driver.session() as session:
            session.execute_write(self._write_item, enriched)

    @staticmethod
    def _write_item(tx, enriched: EnrichedItem) -> None:
        item = enriched.item
        created_iso = item.created_at.isoformat()

        tx.run(
            "MERGE (u:User {name: $author}) "
            "ON CREATE SET u.first_seen = $created_at "
            "SET u.last_seen = $created_at, u.total_posts = coalesce(u.total_posts, 0) + 1",
            author=item.author, created_at=created_iso,
        )
        tx.run(
            "MERGE (s:Subreddit {name: $subreddit}) "
            "ON CREATE SET s.created_at = $created_at",
            subreddit=item.subreddit, created_at=created_iso,
        )
        tx.run(
            "MATCH (u:User {name: $author}), (s:Subreddit {name: $subreddit}) "
            "MERGE (u)-[r:ACTIVE_IN]->(s) "
            "ON CREATE SET r.created_at = $created_at, r.count = 1 "
            "ON MATCH SET r.count = r.count + 1, r.last_active = $created_at",
            author=item.author, subreddit=item.subreddit, created_at=created_iso,
        )

        if item.content_type == ContentType.POST:
            tx.run(
                "MERGE (p:Post {id: $id}) "
                "SET p.title = $title, p.body = $body, p.author = $author, "
                "p.subreddit = $subreddit, p.created_at = $created_at, "
                "p.score = $score, p.url = $url, p.window = $window, "
                "p.summary = $summary, p.sentiment = $sentiment",
                id=item.id, title=item.title, body=item.body,
                author=item.author, subreddit=item.subreddit,
                created_at=created_iso, score=item.score,
                url=item.permalink, window=item.window_label,
                summary=enriched.summary, sentiment=enriched.overall_sentiment,
            )
            tx.run(
                "MATCH (u:User {name: $author}), (p:Post {id: $id}) "
                "MERGE (u)-[r:AUTHORED]->(p) SET r.created_at = $created_at",
                author=item.author, id=item.id, created_at=created_iso,
            )
            tx.run(
                "MATCH (p:Post {id: $id}), (s:Subreddit {name: $subreddit}) "
                "MERGE (p)-[r:POSTED_IN]->(s) SET r.created_at = $created_at",
                id=item.id, subreddit=item.subreddit, created_at=created_iso,
            )
            content_node = "Post"
            content_id = item.id
        else:
            tx.run(
                "MERGE (c:Comment {id: $id}) "
                "SET c.body = $body, c.author = $author, c.subreddit = $subreddit, "
                "c.created_at = $created_at, c.score = $score, c.url = $url, "
                "c.window = $window, c.summary = $summary, c.sentiment = $sentiment",
                id=item.id, body=item.body, author=item.author,
                subreddit=item.subreddit, created_at=created_iso,
                score=item.score, url=item.permalink, window=item.window_label,
                summary=enriched.summary, sentiment=enriched.overall_sentiment,
            )
            tx.run(
                "MATCH (u:User {name: $author}), (c:Comment {id: $id}) "
                "MERGE (u)-[r:AUTHORED]->(c) SET r.created_at = $created_at",
                author=item.author, id=item.id, created_at=created_iso,
            )
            if item.post_id:
                tx.run(
                    "MATCH (c:Comment {id: $cid}), (p:Post {id: $pid}) "
                    "MERGE (c)-[r:ON_POST]->(p) SET r.created_at = $created_at",
                    cid=item.id, pid=f"post_{item.post_id}", created_at=created_iso,
                )
            content_node = "Comment"
            content_id = item.id

        for topic in enriched.topics:
            tx.run(
                f"MERGE (t:Topic {{name: $name}}) "
                f"ON CREATE SET t.created_at = $created_at "
                f"WITH t "
                f"MATCH (n:{content_node} {{id: $content_id}}) "
                f"MERGE (n)-[r:MENTIONS]->(t) "
                f"SET r.created_at = $created_at, r.sentiment = $sentiment",
                name=topic, created_at=created_iso, content_id=content_id,
                sentiment=enriched.overall_sentiment,
            )

        for entity in enriched.entities:
            tx.run(
                f"MERGE (e:Entity {{name: $name}}) "
                f"ON CREATE SET e.type = $etype, e.created_at = $created_at "
                f"WITH e "
                f"MATCH (n:{content_node} {{id: $content_id}}) "
                f"MERGE (n)-[r:MENTIONS]->(e) "
                f"SET r.created_at = $created_at, r.sentiment = $sentiment, "
                f"r.confidence = $confidence",
                name=entity.name, etype=entity.entity_type,
                created_at=created_iso, content_id=content_id,
                sentiment=entity.sentiment, confidence=entity.confidence,
            )

        tx.run(
            f"MERGE (sent:Sentiment {{label: $label}}) "
            f"WITH sent "
            f"MATCH (n:{content_node} {{id: $content_id}}) "
            f"MERGE (n)-[r:HAS_SENTIMENT]->(sent) "
            f"SET r.created_at = $created_at",
            label=enriched.overall_sentiment, content_id=content_id,
            created_at=created_iso,
        )

    # ── Search Queries ─────────────────────────────────────────────────────

    def search(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        subreddits: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        time_filter = ""
        params: Dict[str, Any] = {"entities": entities, "limit": limit}

        if time_start:
            time_filter += " AND n.created_at >= $time_start"
            params["time_start"] = time_start.isoformat()
        if time_end:
            time_filter += " AND n.created_at <= $time_end"
            params["time_end"] = time_end.isoformat()
        if subreddits:
            time_filter += " AND n.subreddit IN $subreddits"
            params["subreddits"] = subreddits

        if entities:
            cypher = f"""
            UNWIND $entities AS entity_name
            MATCH (e) WHERE (e:Entity OR e:Topic) AND toLower(e.name) CONTAINS toLower(entity_name)
            MATCH (n)-[r:MENTIONS]->(e)
            WHERE (n:Post OR n:Comment){time_filter}
            WITH n, e, r, count(r) AS mention_count
            RETURN n.id AS id,
                   coalesce(n.title, '') + ' ' + coalesce(n.body, '') AS text,
                   mention_count * coalesce(n.score, 1) AS score,
                   n.author AS author, n.subreddit AS subreddit,
                   n.created_at AS created_at, n.url AS url,
                   n.window AS window, n.sentiment AS sentiment,
                   e.name AS matched_entity
            ORDER BY score DESC
            LIMIT $limit
            """
        else:
            cypher = f"""
            MATCH (n) WHERE (n:Post OR n:Comment){time_filter.replace('n.', 'n.')}
            RETURN n.id AS id,
                   coalesce(n.title, '') + ' ' + coalesce(n.body, '') AS text,
                   coalesce(n.score, 1) AS score,
                   n.author AS author, n.subreddit AS subreddit,
                   n.created_at AS created_at, n.url AS url,
                   n.window AS window, n.sentiment AS sentiment,
                   '' AS matched_entity
            ORDER BY score DESC
            LIMIT $limit
            """

        with self.driver.session() as session:
            records = session.run(cypher, **params)
            return [
                RetrievalResult(
                    id=rec["id"],
                    text=rec["text"].strip(),
                    score=float(rec["score"]),
                    source="graph",
                    metadata={
                        "author": rec["author"],
                        "subreddit": rec["subreddit"],
                        "created_at": rec["created_at"],
                        "url": rec["url"],
                        "window": rec["window"],
                        "sentiment": rec["sentiment"],
                        "matched_entity": rec["matched_entity"],
                    },
                )
                for rec in records
            ]

    def find_influential_users(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        params: Dict[str, Any] = {"entities": entities or [""], "limit": limit}
        time_filter = ""
        if time_start:
            time_filter += " AND n.created_at >= $time_start"
            params["time_start"] = time_start.isoformat()
        if time_end:
            time_filter += " AND n.created_at <= $time_end"
            params["time_end"] = time_end.isoformat()

        cypher = f"""
        UNWIND $entities AS entity_name
        MATCH (e) WHERE (e:Entity OR e:Topic)
          AND (entity_name = '' OR toLower(e.name) CONTAINS toLower(entity_name))
        MATCH (n)-[:MENTIONS]->(e)
        WHERE (n:Post OR n:Comment){time_filter}
        MATCH (u:User)-[:AUTHORED]->(n)
        WITH u, collect(DISTINCT n) AS posts,
             sum(coalesce(n.score, 0)) AS total_score,
             count(n) AS post_count
        RETURN u.name AS author, total_score, post_count,
               [p IN posts | coalesce(p.title, '') + ' ' + coalesce(p.body, '')][..3] AS sample_texts
        ORDER BY total_score DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            records = session.run(cypher, **params)
            results = []
            for rec in records:
                texts = rec["sample_texts"] or []
                combined = f"Author: {rec['author']} (score={rec['total_score']}, posts={rec['post_count']})\n"
                combined += "\n---\n".join(t[:300] for t in texts)
                results.append(RetrievalResult(
                    id=f"user_{rec['author']}",
                    text=combined,
                    score=float(rec["total_score"]),
                    source="graph",
                    metadata={
                        "author": rec["author"],
                        "post_count": rec["post_count"],
                        "type": "influential_user",
                    },
                ))
            return results

    def find_influential_entities(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
        entity_type: str = "person",
    ) -> List[RetrievalResult]:
        params: Dict[str, Any] = {
            "entities": entities or [""],
            "limit": limit,
            "entity_type": entity_type,
        }
        time_filter = ""
        if time_start:
            time_filter += " AND n.created_at >= $time_start"
            params["time_start"] = time_start.isoformat()
        if time_end:
            time_filter += " AND n.created_at <= $time_end"
            params["time_end"] = time_end.isoformat()

        cypher = f"""
        UNWIND $entities AS topic_name
        MATCH (target) WHERE (target:Entity OR target:Topic)
          AND (topic_name = '' OR toLower(target.name) CONTAINS toLower(topic_name))
        MATCH (n)-[:MENTIONS]->(target)
        WHERE (n:Post OR n:Comment){time_filter}
        MATCH (n)-[:MENTIONS]->(voice:Entity {{type: $entity_type}})
        WITH voice, collect(DISTINCT n) AS posts,
             sum(coalesce(n.score, 0)) AS total_score,
             count(DISTINCT n) AS mention_count
        RETURN voice.name AS entity_name,
               total_score,
               mention_count,
               [p IN posts | coalesce(p.title, '') + ' ' + coalesce(p.body, '')][..3] AS sample_texts,
               [p IN posts | p.subreddit][..3] AS subreddits
        ORDER BY total_score DESC, mention_count DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            records = session.run(cypher, **params)
            return [
                RetrievalResult(
                    id=f"entity_{rec['entity_name']}",
                    text=(
                        f"{rec['entity_name']}: mentioned in {rec['mention_count']} relevant discussions "
                        f"across {', '.join(sorted(set(rec['subreddits'] or []))[:3]) or 'multiple communities'}.\n"
                        + "\n---\n".join(t[:300] for t in (rec["sample_texts"] or []))
                    ),
                    score=float(rec["total_score"]),
                    source="graph",
                    metadata={
                        "entity": rec["entity_name"],
                        "entity_type": entity_type,
                        "author": rec["entity_name"],
                        "subreddit": ", ".join(sorted(set(rec["subreddits"] or []))[:3]),
                        "type": "influential_entity",
                    },
                )
                for rec in records
            ]

    def community_leadership(
        self,
        entities: List[str],
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        params: Dict[str, Any] = {"entities": entities or [""], "limit": limit}
        time_filter = ""
        if time_start:
            time_filter += " AND n.created_at >= $time_start"
            params["time_start"] = time_start.isoformat()
        if time_end:
            time_filter += " AND n.created_at <= $time_end"
            params["time_end"] = time_end.isoformat()

        cypher = f"""
        UNWIND $entities AS entity_name
        MATCH (e) WHERE (e:Entity OR e:Topic)
          AND (entity_name = '' OR toLower(e.name) CONTAINS toLower(entity_name))
        MATCH (n)-[:MENTIONS]->(e)
        WHERE (n:Post OR n:Comment){time_filter}
        WITH n.subreddit AS sub_name, n
        WITH sub_name, count(n) AS discussion_count, sum(coalesce(n.score, 0)) AS total_score
        RETURN sub_name AS subreddit, discussion_count, total_score
        ORDER BY discussion_count DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            records = session.run(cypher, **params)
            return [
                RetrievalResult(
                    id=f"subreddit_{rec['subreddit']}",
                    text=f"r/{rec['subreddit']}: {rec['discussion_count']} discussions, total score {rec['total_score']}",
                    score=float(rec["discussion_count"]),
                    source="graph",
                    metadata={
                        "subreddit": rec["subreddit"],
                        "discussion_count": rec["discussion_count"],
                        "type": "community_leadership",
                    },
                )
                for rec in records
            ]

    def entity_co_occurrence(
        self,
        entity: str,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Find entities that frequently co-occur with the given entity."""
        params: Dict[str, Any] = {"entity": entity, "limit": limit}
        time_filter = ""
        if time_start:
            time_filter += " AND n.created_at >= $time_start"
            params["time_start"] = time_start.isoformat()
        if time_end:
            time_filter += " AND n.created_at <= $time_end"
            params["time_end"] = time_end.isoformat()

        cypher = f"""
        MATCH (e1) WHERE (e1:Entity OR e1:Topic) AND toLower(e1.name) CONTAINS toLower($entity)
        MATCH (n)-[:MENTIONS]->(e1)
        WHERE (n:Post OR n:Comment){time_filter}
        MATCH (n)-[:MENTIONS]->(e2) WHERE e2 <> e1
        WITH e2.name AS co_entity, count(n) AS co_count, collect(DISTINCT n.id) AS shared_posts
        RETURN co_entity, co_count, size(shared_posts) AS shared_post_count
        ORDER BY co_count DESC
        LIMIT $limit
        """

        with self.driver.session() as session:
            records = session.run(cypher, **params)
            return [
                RetrievalResult(
                    id=f"cooccur_{rec['co_entity']}",
                    text=f"{entity} ↔ {rec['co_entity']}: co-occurs in {rec['co_count']} mentions across {rec['shared_post_count']} posts",
                    score=float(rec["co_count"]),
                    source="graph",
                    metadata={
                        "co_entity": rec["co_entity"],
                        "co_count": rec["co_count"],
                        "type": "co_occurrence",
                    },
                )
                for rec in records
            ]

    def topic_evolution(
        self,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Track how topic mention frequency changes across windows."""
        params: Dict[str, Any] = {"limit": limit}
        time_filter = ""
        if time_start:
            time_filter += " AND n.created_at >= $time_start"
            params["time_start"] = time_start.isoformat()
        if time_end:
            time_filter += " AND n.created_at <= $time_end"
            params["time_end"] = time_end.isoformat()

        cypher = f"""
        MATCH (n)-[:MENTIONS]->(t:Topic)
        WHERE (n:Post OR n:Comment){time_filter}
        WITH t.name AS topic, n.window AS window, count(n) AS mention_count
        RETURN topic, window, mention_count
        ORDER BY topic, window
        """

        with self.driver.session() as session:
            records = session.run(cypher, **params)
            return [dict(rec) for rec in records]

    def sentiment_over_time(
        self,
        entity: str,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"entity": entity}
        time_filter = ""
        if time_start:
            time_filter += " AND n.created_at >= $time_start"
            params["time_start"] = time_start.isoformat()
        if time_end:
            time_filter += " AND n.created_at <= $time_end"
            params["time_end"] = time_end.isoformat()

        cypher = f"""
        MATCH (e) WHERE (e:Entity OR e:Topic) AND toLower(e.name) CONTAINS toLower($entity)
        MATCH (n)-[r:MENTIONS]->(e)
        WHERE (n:Post OR n:Comment){time_filter}
        RETURN n.sentiment AS sentiment, n.window AS window,
               count(n) AS cnt, avg(n.score) AS avg_score
        ORDER BY n.window
        """

        with self.driver.session() as session:
            records = session.run(cypher, **params)
            return {
                "entity": entity,
                "periods": [dict(rec) for rec in records],
            }

    def count_nodes(self) -> Dict[str, int]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt"
            )
            return {rec["label"]: rec["cnt"] for rec in result}
