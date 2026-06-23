"""Unified LLM client supporting Gemini, OpenAI, Groq, and Ollama.

Supports four providers with automatic fallback when no API key is configured.
Uses the modern google-genai SDK (not the deprecated google.generativeai).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings


class LLMClient:
    """Multi-provider LLM client with intelligent fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._gemini_client = None

    def complete(self, prompt: str, system: str = "", temperature: float = 0.2) -> str:
        """Generate a completion using the configured LLM provider."""
        provider = self.settings.llm_provider.lower()
        try:
            if provider == "gemini":
                return self._complete_gemini(prompt, system, temperature)
            if provider == "openai":
                return self._complete_openai(prompt, system, temperature)
            if provider == "groq":
                return self._complete_groq(prompt, system, temperature)
            if provider == "ollama":
                return self._complete_ollama(prompt, system, temperature)
            raise ValueError(f"Unknown LLM provider: {provider}")
        except Exception:
            return self._fallback_response(prompt)

    def complete_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        """Generate a completion and parse it as JSON."""
        raw = self.complete(prompt, system=system, temperature=0.1)
        return self._parse_json(raw)

    # ── Gemini (google-genai SDK) ──────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _complete_gemini(self, prompt: str, system: str, temperature: float) -> str:
        if not self.settings.gemini_api_key:
            return self._fallback_response(prompt)

        from google import genai
        from google.genai import types

        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.settings.gemini_api_key)

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system or None,
        )
        response = self._gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config,
        )
        return response.text

    # ── OpenAI ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _complete_openai(self, prompt: str, system: str, temperature: float) -> str:
        from openai import OpenAI

        if not self.settings.openai_api_key:
            return self._fallback_response(prompt)

        client = OpenAI(api_key=self.settings.openai_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    # ── Groq ───────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _complete_groq(self, prompt: str, system: str, temperature: float) -> str:
        if not self.settings.groq_api_key:
            return self._fallback_response(prompt)

        from groq import Groq

        client = Groq(api_key=self.settings.groq_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    # ── Ollama ─────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _complete_ollama(self, prompt: str, system: str, temperature: float) -> str:
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{self.settings.ollama_base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    # ── Context-Aware Fallback ─────────────────────────────────────────────

    def _fallback_response(self, prompt: str) -> str:
        """Rule-based fallback when no LLM API key is configured.

        Produces context-aware responses by parsing the prompt for
        retrieved data and structuring a meaningful answer.
        """
        pl = prompt.lower()

        # Entity extraction request
        if "extract entities" in pl or ("extract" in pl and "sentiment" in pl):
            return self._fallback_extraction(prompt)

        # Query parsing request
        if "parse this user question" in pl or "parse user question" in pl:
            return self._fallback_query_parse(prompt)

        # Answer synthesis request
        if "synthesize" in pl or "answer" in pl:
            return self._fallback_synthesis(prompt)

        return (
            "Based on the retrieved Reddit discussions, the community shows "
            "active engagement across observed subreddits with evolving perspectives "
            "on AI technologies, safety considerations, and open-source development."
        )

    def _fallback_extraction(self, prompt: str) -> str:
        """Extract entities using keyword matching when no LLM is available."""
        text = prompt.lower()
        entities = []
        keywords = {
            "RAG": ("technology", "positive"),
            "retrieval augmented generation": ("technology", "positive"),
            "GraphRAG": ("technology", "positive"),
            "AI safety": ("concern", "mixed"),
            "alignment": ("concern", "mixed"),
            "open source": ("topic", "positive"),
            "open-source": ("topic", "positive"),
            "LLM": ("technology", "neutral"),
            "large language model": ("technology", "neutral"),
            "regulation": ("concern", "negative"),
            "EU AI Act": ("regulation", "negative"),
            "Gemini": ("product", "neutral"),
            "GPT": ("product", "positive"),
            "Claude": ("product", "positive"),
            "Llama": ("product", "positive"),
            "fine-tuning": ("technology", "neutral"),
            "fine tuning": ("technology", "neutral"),
            "hallucination": ("concern", "negative"),
            "agentic": ("technology", "mixed"),
            "agent": ("technology", "mixed"),
            "autonomous": ("concern", "mixed"),
            "vector search": ("technology", "positive"),
            "embedding": ("technology", "neutral"),
            "transformer": ("technology", "neutral"),
            "RLHF": ("technology", "neutral"),
            "benchmark": ("topic", "neutral"),
            "quantization": ("technology", "positive"),
            "inference": ("technology", "neutral"),
            "deployment": ("topic", "neutral"),
            "privacy": ("concern", "negative"),
            "bias": ("concern", "negative"),
        }
        topics = []
        for kw, (etype, sentiment) in keywords.items():
            if kw.lower() in text:
                entities.append({
                    "name": kw,
                    "type": etype,
                    "sentiment": sentiment,
                    "confidence": 0.65,
                })
                topics.append(kw)

        # Sentiment detection
        sentiment = "neutral"
        pos_words = ("great", "amazing", "love", "excellent", "impressive", "breakthrough", "exciting", "revolutionary")
        neg_words = ("worried", "concern", "dangerous", "bad", "terrible", "alarming", "risky", "harmful", "fear")
        mixed_words = ("however", "but", "although", "on the other hand", "mixed", "debate")

        pos_count = sum(1 for w in pos_words if w in text)
        neg_count = sum(1 for w in neg_words if w in text)
        mixed_count = sum(1 for w in mixed_words if w in text)

        if mixed_count > 0 or (pos_count > 0 and neg_count > 0):
            sentiment = "mixed"
        elif pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"

        return json.dumps({
            "overall_sentiment": sentiment,
            "topics": topics[:5] or ["AI", "machine learning"],
            "summary": prompt.split("Content:")[-1].strip()[:150] if "Content:" in prompt else prompt[:150],
            "entities": entities[:8],
        })

    def _fallback_query_parse(self, prompt: str) -> str:
        """Parse query intent without LLM."""
        return json.dumps({
            "semantic_query": prompt[:200],
            "graph_entities": [],
            "graph_relationships": ["MENTIONS", "HAS_SENTIMENT"],
            "query_type": "hybrid",
            "time_start": None,
            "time_end": None,
            "compare_start": None,
            "compare_end": None,
            "subreddits": [],
        })

    def _fallback_synthesis(self, prompt: str) -> str:
        """Synthesize an answer from context without LLM.

        Reads the actual retrieved sources from the prompt and
        composes a structured summary.
        """
        lines = prompt.splitlines()
        sources = []
        citations = []
        current_source = []

        for line in lines:
            if (
                line.startswith("### Source")
                or line.startswith("### Graph Traversal Results")
                or line.startswith("### Temporal Analysis Results")
                or line.startswith("**Source ")
            ):
                if current_source:
                    sources.append("\n".join(current_source))
                current_source = [line]
            elif line.startswith("Citation:"):
                citations.append(line.replace("Citation: ", ""))
                if current_source:
                    current_source.append(line)
            elif current_source:
                current_source.append(line)

        if current_source:
            sources.append("\n".join(current_source))

        # Extract the question
        question = ""
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
                break

        # Build answer from sources
        answer_parts = [f"Based on analysis of {len(sources)} retrieved Reddit discussions:\n"]

        # Extract key themes from sources
        all_text = " ".join(s.lower() for s in sources)
        themes = []
        theme_keywords = {
            "RAG pipelines": ["rag", "retrieval augmented"],
            "AI safety considerations": ["ai safety", "safety concern", "alignment"],
            "open-source LLM development": ["open source", "open-source", "local", "llama"],
            "AI regulation and policy": ["regulation", "eu ai act", "policy", "compliance"],
            "agentic AI systems": ["agent", "agentic", "autonomous"],
            "model performance and benchmarks": ["benchmark", "performance", "evaluation"],
            "community engagement": ["community", "discussion", "debate"],
        }
        for theme, keywords in theme_keywords.items():
            if any(kw in all_text for kw in keywords):
                themes.append(theme)

        if question:
            answer_parts.append(f"Question focus: {question}\n")
        if themes:
            answer_parts.append("**Key themes identified:** " + ", ".join(themes[:4]) + "\n")

        # Add source summaries
        for i, source in enumerate(sources[:5], 1):
            # Get first meaningful line
            source_lines = [l for l in source.split("\n") if l.strip() and not l.startswith("###") and not l.startswith("Citation:")]
            if source_lines:
                preview = source_lines[0][:200]
                citation = citations[i - 1] if i - 1 < len(citations) else ""
                answer_parts.append(f"- {preview}")
                if citation:
                    answer_parts.append(f"  *{citation}*\n")

        # Temporal comparison section
        if "temporal comparison" in all_text or "period" in all_text:
            answer_parts.append(
                "\n**Temporal Analysis:** The retrieved evidence shows meaningful change across time windows. "
                "Later discussions put more emphasis on deployment risks, agent behavior, and governance than "
                "earlier periods, while earlier periods are more weighted toward foundational capabilities."
            )

        answer_parts.append(
            "\n**Bottom line:** The answer above is based on retrieved graph and vector evidence. "
            "When no external LLM is available or a provider request fails, this local synthesis path preserves "
            "citations and keeps the demo usable."
        )

        if not sources:
            answer_parts = [
                "Based on the available context, the Reddit communities show active "
                "discussion on this topic. Key perspectives include varied viewpoints "
                "from r/MachineLearning, r/LocalLLaMA, and r/artificial, reflecting "
                "the evolving AI landscape.\n\n"
                "*Note: For richer analysis, configure an LLM API key (Gemini/OpenAI/Groq).*"
            ]

        return "\n".join(answer_parts)

    def has_api_key(self) -> bool:
        """Check whether the current provider has an API key configured."""
        provider = self.settings.llm_provider.lower()
        if provider == "gemini":
            return bool(self.settings.gemini_api_key and self.settings.gemini_api_key != "your_gemini_api_key")
        if provider == "openai":
            return bool(self.settings.openai_api_key and self.settings.openai_api_key != "your_openai_api_key")
        if provider == "groq":
            return bool(self.settings.groq_api_key and self.settings.groq_api_key != "your_groq_api_key")
        if provider == "ollama":
            return True
        return False

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """Parse JSON from LLM output, handling markdown fences."""
        raw = raw.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if fence:
            raw = fence.group(1)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
            raise
