"""Tests for provider failure fallback behavior."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.client import LLMClient


def test_complete_falls_back_when_provider_raises(monkeypatch):
    client = LLMClient()

    def raise_error(prompt: str, system: str, temperature: float) -> str:
        raise RuntimeError("provider failed")

    monkeypatch.setattr(client, "_complete_gemini", raise_error)
    client.settings.llm_provider = "gemini"

    result = client.complete(
        "Question: What changed?\n\n## Retrieved Context\n### Source 1\nSome context\nCitation: [user in r/test, 2026-01-01] url\n\nSynthesize a comprehensive answer with citations."
    )

    assert "provider failed" not in result
    assert "Based on analysis" in result or "Bottom line" in result
