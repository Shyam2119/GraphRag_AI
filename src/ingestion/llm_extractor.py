"""LLM-based entity, topic, and sentiment extraction during ingestion."""

from __future__ import annotations

import logging
from typing import List

from src.models import EnrichedItem, ExtractedEntity, RedditItem
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You extract structured knowledge from Reddit posts/comments.
Return ONLY valid JSON with:
- overall_sentiment: positive|negative|neutral|mixed
- topics: list of 1-5 topic strings
- summary: one sentence summary
- entities: list of {name, type, sentiment, confidence}
  where type is one of: topic, technology, person, concern, organization, product"""


class LLMExtractor:
  def __init__(self, llm: LLMClient | None = None) -> None:
    self.llm = llm or LLMClient()
    self._llm_disabled = False

  def enrich(self, item: RedditItem) -> EnrichedItem:
    text = item.full_text
    if len(text) < 10:
      return EnrichedItem(
        item=item,
        overall_sentiment="neutral",
        topics=[],
        summary=text[:100],
      )

    if not self._can_use_llm():
      data = self._keyword_fallback(text)
    else:
      prompt = f"""Extract entities, topics, and sentiment from this Reddit {item.content_type.value}.

Subreddit: r/{item.subreddit}
Author: {item.author}
Date: {item.created_at.isoformat()}

Content:
{text[:3000]}

Return JSON only."""

      try:
        data = self.llm.complete_json(prompt, system=EXTRACTION_SYSTEM)
      except Exception as exc:
        self._llm_disabled = True
        logger.warning("LLM extraction failed; using keyword fallback for this run: %s", exc)
        data = self._keyword_fallback(text)

    return self._to_enriched(item, data)

  def enrich_batch(self, items: List[RedditItem], show_progress: bool = True, batch_size: int = 5) -> List[EnrichedItem]:
    """Process items in batches for ~10x speedup.
    
    Instead of 1814 individual LLM calls, groups items and processes ~360 batches.
    """
    results = []
    
    # If no API key, fall back to keyword extraction (fast)
    if not self._can_use_llm():
      for i, item in enumerate(items):
        if show_progress and i % 50 == 0:
          print(f"    Enriching {i + 1}/{len(items)}...")
        results.append(self._enrich_keyword(item))
      return results
    
    # Process in batches
    for batch_start in range(0, len(items), batch_size):
      batch_end = min(batch_start + batch_size, len(items))
      batch = items[batch_start:batch_end]
      
      if show_progress:
        print(f"    Enriching {batch_end}/{len(items)}...")

      if not self._can_use_llm():
        results.extend(self._enrich_keyword(item) for item in items[batch_start:])
        break
      
      # Try batch extraction, fall back to individual if it fails
      try:
        batch_results = self._enrich_batch_llm(batch)
        results.extend(batch_results)
      except Exception as e:
        self._llm_disabled = True
        logger.warning("Batch processing failed; using keyword fallback for remaining items: %s", e)
        results.extend(self._enrich_keyword(item) for item in items[batch_start:])
        break
    
    return results

  def _enrich_batch_llm(self, items: List[RedditItem]) -> List[EnrichedItem]:
    """Process a batch of items with a single LLM call."""
    # Build batch prompt
    items_text = "\n\n---ITEM SEPARATOR---\n\n".join([
      f"Content #{i+1}:\n{item.full_text[:1200]}"
      for i, item in enumerate(items)
    ])
    
    prompt = f"""Extract entities, topics, and sentiment from these {len(items)} Reddit items.

{items_text}

Return a JSON array with {len(items)} objects, each with:
- overall_sentiment: positive|negative|neutral|mixed
- topics: list of 1-5 strings
- summary: one sentence
- entities: list of {{name, type, sentiment, confidence}}
  (type: topic, technology, person, concern, organization, product)

Return ONLY the JSON array, no other text."""

    try:
      # Get batch response
      data = self.llm.complete_json(prompt, system=EXTRACTION_SYSTEM)
      
      # Handle if response is a list
      if isinstance(data, list):
        results_list = data
      elif isinstance(data, dict) and "results" in data:
        results_list = data["results"]
      else:
        # Single response, wrap it
        results_list = [data]
      
      # Convert to EnrichedItems
      enriched = []
      for i, item in enumerate(items[:len(results_list)]):
        item_data = results_list[i] if i < len(results_list) else {}
        enriched.append(self._to_enriched(item, item_data))
      
      # For remaining items in batch, use fast keyword extraction.
      for i in range(len(results_list), len(items)):
        enriched.append(self._enrich_keyword(items[i]))
      
      return enriched
    
    except Exception as e:
      self._llm_disabled = True
      logger.debug("Batch LLM call failed: %s, using keyword fallback", e)
      return [self._enrich_keyword(item) for item in items]

  def _can_use_llm(self) -> bool:
    return not self._llm_disabled and self.llm.has_api_key()

  def _enrich_keyword(self, item: RedditItem) -> EnrichedItem:
    return self._to_enriched(item, self._keyword_fallback(item.full_text))

  @staticmethod
  def _to_enriched(item: RedditItem, data: dict) -> EnrichedItem:
    if not isinstance(data, dict):
      data = {}

    entities = [
      ExtractedEntity(
        name=e["name"],
        entity_type=e.get("type", e.get("entity_type", "topic")),
        sentiment=e.get("sentiment", "neutral"),
        confidence=float(e.get("confidence", 0.7)),
      )
      for e in data.get("entities", [])
      if e.get("name")
    ]

    return EnrichedItem(
      item=item,
      entities=entities,
      overall_sentiment=data.get("overall_sentiment", "neutral"),
      topics=data.get("topics", []),
      summary=data.get("summary", ""),
    )

  @staticmethod
  def _keyword_fallback(text: str) -> dict:
    lower = text.lower()
    entities = []
    keywords = {
      "RAG": ("technology", "positive"),
      "retrieval augmented": ("technology", "positive"),
      "GraphRAG": ("technology", "positive"),
      "AI safety": ("concern", "mixed"),
      "open source": ("topic", "positive"),
      "LLM": ("technology", "neutral"),
      "regulation": ("concern", "negative"),
      "Gemini": ("product", "neutral"),
      "GPT": ("product", "positive"),
      "Yoshua Bengio": ("person", "neutral"),
      "Stuart Russell": ("person", "neutral"),
      "Timnit Gebru": ("person", "neutral"),
      "Sam Altman": ("person", "neutral"),
      "Yann LeCun": ("person", "neutral"),
      "Andrew Ng": ("person", "neutral"),
      "Demis Hassabis": ("person", "neutral"),
      "EU AI Act": ("organization", "mixed"),
      "G7": ("organization", "neutral"),
      "OWASP": ("organization", "neutral"),
      "Llama": ("product", "positive"),
      "Qwen": ("product", "positive"),
      "DeepSeek": ("product", "positive"),
    }
    topics = []
    for kw, (etype, sentiment) in keywords.items():
      if kw.lower() in lower:
        entities.append({
          "name": kw,
          "type": etype,
          "sentiment": sentiment,
          "confidence": 0.6,
        })
        topics.append(kw)

    sentiment = "neutral"
    if any(w in lower for w in ("great", "amazing", "love", "excellent")):
      sentiment = "positive"
    elif any(w in lower for w in ("worried", "concern", "dangerous", "bad")):
      sentiment = "negative"

    return {
      "overall_sentiment": sentiment,
      "topics": topics or ["AI", "machine learning"],
      "summary": text[:150],
      "entities": entities,
    }
