"""Central configuration loaded from environment variables."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Reddit
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "GraphRAG/1.0"
    reddit_subreddits: str = "MachineLearning,LocalLLaMA,artificial"
    reddit_search_query: str = 'RAG OR "retrieval augmented" OR "AI safety"'
    reddit_post_limit: int = 50
    reddit_comment_limit: int = 20
    
    # LLM batch processing (for speed optimization)
    llm_batch_size: int = 5  # Process 5 items per LLM call (faster)

    # Scraping method: auto | web | reddit_api | sample
    scrape_method: str = "auto"
    scrape_delay: float = 1.5  # seconds between HTTP requests
    scrape_concurrency: int = 4
    scrape_timeout: float = 10.0
    
    # Web scraping backend: ddgs|duckduckgo | crawl4ai | firecrawl | brave | tavily | trivial
    web_scraper_backend: str = "ddgs"  # Default to DuckDuckGo (no API key needed)
    web_scraper_fallback_backends: str = "firecrawl,trivial,brave,duckduckgo"
    firecrawl_api_key: str = ""  # For Firecrawl backend
    brave_search_api_key: str = ""  # For Brave Search API
    tavily_api_key: str = ""  # For Tavily search API

    # LLM
    llm_provider: str = "gemini"  # gemini | openai | groq | ollama
    gemini_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Embeddings
    embedding_provider: str = "auto"  # auto | gemini | sentence-transformers | hash

    # Graph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag123"
    use_fallback_graph: bool = True

    # Vector
    chroma_persist_dir: str = "./data/chroma"
    use_chroma: bool = False

    # Time windows (optional overrides)
    window_1_start: Optional[str] = None
    window_1_end: Optional[str] = None
    window_2_start: Optional[str] = None
    window_2_end: Optional[str] = None
    window_3_start: Optional[str] = None
    window_3_end: Optional[str] = None

    use_sample_data: bool = True

    # Logging
    log_level: str = "INFO"

    @property
    def subreddit_list(self) -> List[str]:
        return [s.strip() for s in self.reddit_subreddits.split(",") if s.strip()]

    @property
    def time_windows(self) -> List[Tuple[str, str, str]]:
        """Return (label, start_iso, end_iso) for each scraping window."""
        defaults = [
            ("Q3_2025", "2025-07-01", "2025-09-30"),
            ("Q4_2025", "2025-10-01", "2025-12-31"),
            ("Q1_2026", "2026-01-01", "2026-03-31"),
        ]
        overrides = [
            (self.window_1_start, self.window_1_end),
            (self.window_2_start, self.window_2_end),
            (self.window_3_start, self.window_3_end),
        ]
        result = []
        for i, (label, d_start, d_end) in enumerate(defaults):
            start = overrides[i][0] or d_start
            end = overrides[i][1] or d_end
            result.append((label, start, end))
        return result

    def reddit_configured(self) -> bool:
        return bool(self.reddit_client_id and self.reddit_client_secret
                     and self.reddit_client_id != "your_client_id")

    @property
    def effective_scrape_method(self) -> str:
        """Resolve 'auto' to the best available scrape method."""
        method = self.scrape_method.lower()
        if method != "auto":
            return method
        if self.use_sample_data:
            return "sample"
        if self.reddit_configured():
            return "reddit_api"
        return "web"


@lru_cache
def get_settings() -> Settings:
    return Settings()
