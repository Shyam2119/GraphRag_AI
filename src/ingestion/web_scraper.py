"""Reddit data ingestion via web search + scraping.

This is the no-Reddit-API ingestion path. It discovers Reddit post URLs with
search providers such as DuckDuckGo, Brave, Firecrawl, Tavily, or Crawl4AI, then
scrapes public Reddit pages/JSON without PRAW credentials.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Generator, List, Optional

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from src.models import ContentType, RedditItem

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


class WebRedditScraper:
    """Scrape Reddit posts via search backends + public page scraping.

    Supported discovery backends:
    - ddgs: DuckDuckGo, no API key required
    - brave: Brave Search API, requires BRAVE_SEARCH_API_KEY
    - firecrawl: Firecrawl Search API, requires FIRECRAWL_API_KEY
    - tavily/trivial: Tavily Search API, requires TAVILY_API_KEY
    - crawl4ai: Crawl4AI scraping of Reddit search pages
    """

    _POST_URL_RE = re.compile(
        r"https?://(?:www\.|old\.)?reddit\.com/r/[\w-]+/comments/[\w]+",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self._delay = getattr(settings, "scrape_delay", 1.5)
        self._timeout = getattr(settings, "scrape_timeout", 10.0)
        self._concurrency = max(1, int(getattr(settings, "scrape_concurrency", 4)))
        self.search_backend = self._canonical_backend_name(
            getattr(settings, "web_scraper_backend", "ddgs")
        )
        self._thread_local = threading.local()
        logger.info("Using web scraper backend: %s", self.search_backend)

    def _get_session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(_HEADERS)
            self._thread_local.session = session
        return session

    def scrape_all_windows(self) -> List[RedditItem]:
        """Scrape all configured time windows and return merged results."""
        all_items: List[RedditItem] = []
        for label, start, end in self.settings.time_windows:
            print(f"  Scraping window {label} ({start} -> {end})...")
            items = list(self.scrape_window(label, start, end))
            print(f"    -> {len(items)} items")
            all_items.extend(items)
        return all_items

    def scrape_window(
        self, label: str, start_iso: str, end_iso: str
    ) -> Generator[RedditItem, None, None]:
        """Scrape a single time window across all configured subreddits."""
        for subreddit_name in self.settings.subreddit_list:
            yield from self._scrape_subreddit(subreddit_name, label, start_iso, end_iso)
            time.sleep(self._delay)

    def _scrape_subreddit(
        self, subreddit: str, label: str, start_iso: str, end_iso: str
    ) -> Generator[RedditItem, None, None]:
        """Discover Reddit post URLs and scrape each in-window post."""
        urls = self._search_reddit_urls(subreddit, start_iso, end_iso)
        print(f"    [{subreddit}] Found {len(urls)} post URLs via search")

        max_candidates = max(self.settings.reddit_post_limit, self.settings.reddit_post_limit * 2)
        candidate_urls = urls[:max_candidates]
        if self._concurrency <= 1 or len(candidate_urls) <= 1:
            yield from self._scrape_urls_sequential(candidate_urls, subreddit, label, start_iso, end_iso)
            return

        scraped = 0
        workers = min(self._concurrency, len(candidate_urls))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for url in candidate_urls:
                futures[
                    executor.submit(self._scrape_post, url, subreddit, label, start_iso, end_iso)
                ] = url
                if self._delay > 0:
                    time.sleep(self._delay / workers)

            for future in as_completed(futures):
                if scraped >= self.settings.reddit_post_limit:
                    break
                url = futures[future]
                try:
                    items = future.result()
                except Exception as exc:
                    logger.warning("Failed to scrape %s: %s", url, exc)
                    continue
                if not items:
                    logger.info("Skipping %s because it has no in-window content", url)
                    continue
                yield from items
                scraped += 1

    def _scrape_urls_sequential(
        self,
        urls: List[str],
        subreddit: str,
        label: str,
        start_iso: str,
        end_iso: str,
    ) -> Generator[RedditItem, None, None]:
        scraped = 0
        for url in urls:
            if scraped >= self.settings.reddit_post_limit:
                break
            try:
                items = self._scrape_post(url, subreddit, label, start_iso, end_iso)
                if not items:
                    logger.info("Skipping %s because it has no in-window content", url)
                    continue
                yield from items
                scraped += 1
            except Exception as exc:
                logger.warning("Failed to scrape %s: %s", url, exc)
            time.sleep(self._delay)

    def _search_reddit_urls(
        self, subreddit: str, start_iso: str, end_iso: str
    ) -> List[str]:
        """Use configured search backend, then fall back to alternatives."""
        backends: dict[str, Callable[[str, str, str], List[str]]] = {
            "ddgs": self._search_reddit_urls_ddgs,
            "brave": self._search_reddit_urls_brave,
            "firecrawl": self._search_reddit_urls_firecrawl,
            "tavily": self._search_reddit_urls_tavily,
            "trivial": self._search_reddit_urls_tavily,
            "crawl4ai": self._search_reddit_urls_crawl4ai,
        }
        fallback_names = [
            self._canonical_backend_name(name.strip())
            for name in getattr(self.settings, "web_scraper_fallback_backends", "ddgs").split(",")
            if name.strip()
        ]
        order = [self.search_backend] + [name for name in fallback_names if name != self.search_backend]

        tried: set[str] = set()
        for backend in order:
            if backend in tried:
                continue
            tried.add(backend)
            search_fn = backends.get(backend)
            if not search_fn:
                continue
            if not self._backend_available(backend):
                logger.info("Skipping search backend '%s' because it is not configured", backend)
                continue
            urls = search_fn(subreddit, start_iso, end_iso)
            if urls:
                print(
                    f"    [{subreddit}] Search backend '{backend}' returned "
                    f"{len(urls)} URLs"
                )
                return urls
            logger.warning("Search backend '%s' returned no URLs for r/%s", backend, subreddit)
        return []

    def _backend_available(self, backend: str) -> bool:
        if backend == "ddgs":
            return True
        if backend == "brave":
            return bool(self.settings.brave_search_api_key)
        if backend == "firecrawl":
            return bool(self.settings.firecrawl_api_key)
        if backend in {"tavily", "trivial"}:
            return bool(self.settings.tavily_api_key)
        if backend == "crawl4ai":
            try:
                from crawl4ai import AsyncWebCrawler  # noqa: F401
                return True
            except Exception:
                return False
        return False

    @staticmethod
    def _canonical_backend_name(backend_name: str) -> str:
        """Normalize backend names so user-friendly aliases work.

        Supported aliases:
        - duckduckgo / duck_duck_go / duck.duckgo -> ddgs
        - trivial -> tavily backend function
        """
        normalized = re.sub(r"[^a-z0-9]+", "", (backend_name or "").lower())
        alias_map = {
            "duckduckgo": "ddgs",
            "duckduckgosearch": "ddgs",
            "duckduck": "ddgs",
            "ddg": "ddgs",
            "ddgs": "ddgs",
            "duckduckgoapi": "ddgs",
            "duckduckgodotcom": "ddgs",
            "firecrawl": "firecrawl",
            "brave": "brave",
            "tavily": "tavily",
            "trivial": "trivial",
            "crawl4ai": "crawl4ai",
        }
        return alias_map.get(normalized, normalized or "ddgs")

    def _build_search_query(self, subreddit: str, start_iso: str, end_iso: str) -> str:
        query = self.settings.reddit_search_query
        return (
            f"site:reddit.com/r/{subreddit}/comments {query} "
            f"after:{start_iso} before:{end_iso}"
        )

    def _search_result_limit(self, cap: Optional[int] = None) -> int:
        limit = max(1, self.settings.reddit_post_limit * 3)
        return min(limit, cap) if cap else limit

    def _urls_from_search_records(self, records: list[dict]) -> List[str]:
        urls: List[str] = []
        for record in records:
            for key in ("href", "link", "url"):
                value = record.get(key, "")
                for match in self._POST_URL_RE.findall(value or ""):
                    normalized = self._normalize_reddit_url(match)
                    if normalized not in urls:
                        urls.append(normalized)
        return urls

    def _search_reddit_urls_ddgs(
        self, subreddit: str, start_iso: str, end_iso: str
    ) -> List[str]:
        """Use DuckDuckGo search to find Reddit post URLs."""
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                logger.error("ddgs is not installed: %s", exc)
                return []

        search_query = self._build_search_query(subreddit, start_iso, end_iso)
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        search_query,
                        max_results=self._search_result_limit(),
                    )
                )
            return self._urls_from_search_records(results)
        except Exception as exc:
            logger.error("DuckDuckGo search failed for r/%s: %s", subreddit, exc)
            return []

    def _search_reddit_urls_brave(
        self, subreddit: str, start_iso: str, end_iso: str
    ) -> List[str]:
        """Use Brave Search API to find Reddit post URLs."""
        api_key = self.settings.brave_search_api_key
        if not api_key:
            logger.warning("BRAVE_SEARCH_API_KEY not set")
            return []

        try:
            import httpx
        except ImportError:
            logger.warning("httpx required for Brave Search")
            return []

        try:
            response = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Authorization": f"Token {api_key}"},
                params={
                    "q": self._build_search_query(subreddit, start_iso, end_iso),
                    "count": self._search_result_limit(cap=20),
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            return self._urls_from_search_records(data.get("web", {}).get("results", []))
        except Exception as exc:
            logger.error("Brave Search failed for r/%s: %s", subreddit, exc)
            return []

    def _search_reddit_urls_firecrawl(
        self, subreddit: str, start_iso: str, end_iso: str
    ) -> List[str]:
        """Use Firecrawl Search API to find Reddit post URLs."""
        api_key = self.settings.firecrawl_api_key
        if not api_key:
            logger.warning("FIRECRAWL_API_KEY not set")
            return []

        try:
            import httpx
        except ImportError:
            logger.warning("httpx required for Firecrawl")
            return []

        try:
            response = httpx.post(
                "https://api.firecrawl.dev/v1/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": self._build_search_query(subreddit, start_iso, end_iso),
                    "limit": self._search_result_limit(),
                },
                timeout=max(self._timeout, 15.0),
            )
            response.raise_for_status()
            data = response.json()
            records = data.get("data", data.get("results", []))
            return self._urls_from_search_records(records)
        except Exception as exc:
            logger.error("Firecrawl search failed for r/%s: %s", subreddit, exc)
            return []

    def _search_reddit_urls_tavily(
        self, subreddit: str, start_iso: str, end_iso: str
    ) -> List[str]:
        """Use Tavily search API to find Reddit post URLs."""
        api_key = self.settings.tavily_api_key
        if not api_key:
            logger.warning("TAVILY_API_KEY not set")
            return []

        try:
            response = self._get_session().post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": self._build_search_query(subreddit, start_iso, end_iso),
                    "search_depth": "basic",
                    "max_results": self._search_result_limit(cap=20),
                    "include_domains": ["reddit.com", "old.reddit.com", "www.reddit.com"],
                },
                timeout=max(self._timeout, 15.0),
            )
            response.raise_for_status()
            return self._urls_from_search_records(response.json().get("results", []))
        except Exception as exc:
            logger.error("Tavily search failed for r/%s: %s", subreddit, exc)
            return []

    def _search_reddit_urls_crawl4ai(
        self, subreddit: str, start_iso: str, end_iso: str
    ) -> List[str]:
        """Use Crawl4AI to crawl Reddit search pages."""
        try:
            from crawl4ai import AsyncWebCrawler  # noqa: F401
        except ImportError:
            logger.warning("crawl4ai not installed")
            return []

        query = requests.utils.quote(self.settings.reddit_search_query)
        search_url = (
            f"https://www.reddit.com/r/{subreddit}/search/?q={query}"
            "&restrict_sr=on&sort=new"
        )
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._async_crawl4ai_search(search_url))
        except Exception as exc:
            logger.error("Crawl4AI search failed for r/%s: %s", subreddit, exc)
            return []
        finally:
            loop.close()

    async def _async_crawl4ai_search(self, search_url: str) -> List[str]:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=search_url)
        content = result.markdown or result.html or ""
        urls: List[str] = []
        for match in self._POST_URL_RE.findall(content):
            normalized = self._normalize_reddit_url(match)
            if normalized not in urls:
                urls.append(normalized)
            if len(urls) >= self._search_result_limit():
                break
        return urls

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    def _scrape_post(
        self,
        url: str,
        subreddit: str,
        label: str,
        start_iso: str,
        end_iso: str,
    ) -> List[RedditItem]:
        """Scrape a single Reddit post + comments."""
        start_ts = datetime.fromisoformat(start_iso).timestamp()
        end_ts = datetime.fromisoformat(f"{end_iso}T23:59:59").timestamp()

        json_data = self._fetch_json(url)
        if json_data:
            return self._parse_json_response(json_data, subreddit, label, start_ts, end_ts)

        # Fallback 1: Firecrawl if configured
        if self.search_backend == "firecrawl" and self.settings.firecrawl_api_key:
            firecrawl_items = self._scrape_firecrawl(url, subreddit, label, start_ts, end_ts)
            if firecrawl_items:
                return firecrawl_items

        # Fallback 2: HTML scraping (often blocked by Reddit)
        return self._scrape_html(url, subreddit, label, start_ts, end_ts)

    def _fetch_json(self, url: str) -> Optional[list]:
        """Fetch the public .json version of a Reddit post URL."""
        json_url = url.rstrip("/") + ".json"
        for attempt in range(2):
            try:
                resp = self._get_session().get(json_url, timeout=self._timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        return data
                if resp.status_code == 429 and attempt == 0:
                    logger.warning("Rate limited on JSON endpoint, waiting 10s")
                    time.sleep(10)
                    continue
                logger.debug("JSON endpoint returned %s for %s", resp.status_code, json_url)
                return None
            except (requests.RequestException, ValueError) as exc:
                logger.debug("JSON fetch failed for %s: %s", json_url, exc)
                return None
        return None

    def _parse_json_response(
        self,
        data: list,
        subreddit: str,
        label: str,
        start_ts: float,
        end_ts: float,
    ) -> List[RedditItem]:
        """Parse Reddit's public JSON response into RedditItem objects."""
        items: List[RedditItem] = []
        try:
            post_data = data[0]["data"]["children"][0]["data"]
        except (KeyError, IndexError, TypeError):
            return items

        created = float(post_data.get("created_utc", 0))
        if not self._timestamp_in_window(created, start_ts, end_ts):
            return items

        post_id = post_data.get("id", "unknown")
        items.append(
            RedditItem(
                id=f"post_{post_id}",
                content_type=ContentType.POST,
                title=post_data.get("title", ""),
                body=post_data.get("selftext", ""),
                author=post_data.get("author", "[deleted]"),
                subreddit=post_data.get("subreddit", subreddit),
                created_utc=created,
                score=int(post_data.get("score", 0)),
                url=post_data.get("permalink", ""),
                post_id=post_id,
                window_label=label,
            )
        )

        if len(data) > 1:
            comment_count = 0
            for c in self._extract_comments_json(data[1]):
                if comment_count >= self.settings.reddit_comment_limit:
                    break
                c_created = float(c.get("created_utc", 0))
                if not self._timestamp_in_window(c_created, start_ts, end_ts):
                    continue
                body = c.get("body", "")
                if not body or body in {"[deleted]", "[removed]"}:
                    continue
                comment_id = c.get("id", f"unk_{comment_count}")
                items.append(
                    RedditItem(
                        id=f"comment_{comment_id}",
                        content_type=ContentType.COMMENT,
                        title="",
                        body=body,
                        author=c.get("author", "[deleted]"),
                        subreddit=c.get("subreddit", subreddit),
                        created_utc=c_created,
                        score=int(c.get("score", 0)),
                        url=c.get("permalink", ""),
                        parent_id=c.get("parent_id", ""),
                        post_id=post_id,
                        window_label=label,
                    )
                )
                comment_count += 1
        return items

    def _extract_comments_json(self, comment_listing: dict) -> List[dict]:
        """Recursively extract comment data from Reddit's nested JSON."""
        comments: List[dict] = []
        try:
            children = comment_listing.get("data", {}).get("children", [])
        except AttributeError:
            return comments

        for child in children:
            if child.get("kind") != "t1":
                continue
            cdata = child.get("data", {})
            comments.append(cdata)
            replies = cdata.get("replies", "")
            if isinstance(replies, dict):
                comments.extend(self._extract_comments_json(replies))
        return comments

    def _scrape_html(
        self,
        url: str,
        subreddit: str,
        label: str,
        start_ts: float,
        end_ts: float,
    ) -> List[RedditItem]:
        """Fallback: scrape old.reddit.com HTML when JSON endpoint fails."""
        old_url = self._normalize_reddit_url(url)
        try:
            resp = self._get_session().get(old_url, timeout=self._timeout)
            if resp.status_code != 200:
                logger.warning("HTML fetch returned %d for %s", resp.status_code, old_url)
                return []
        except requests.RequestException as exc:
            logger.warning("HTML fetch failed for %s: %s", old_url, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        post_id_match = re.search(r"/comments/(\w+)", old_url)
        post_id = post_id_match.group(1) if post_id_match else "unknown"

        title_el = soup.find("a", class_="title")
        title = title_el.get_text(strip=True) if title_el else ""

        body = ""
        body_el = soup.find("div", class_="expando")
        if body_el:
            md_el = body_el.find("div", class_="md")
            body = md_el.get_text(" ", strip=True) if md_el else body_el.get_text(" ", strip=True)

        author = "[unknown]"
        created_utc = time.time()
        tagline = soup.find("p", class_="tagline")
        if tagline:
            author_el = tagline.find("a", class_=re.compile(r"author"))
            if author_el:
                author = author_el.get_text(strip=True)
            time_el = tagline.find("time")
            if time_el and time_el.get("datetime"):
                try:
                    dt = datetime.fromisoformat(time_el["datetime"].replace("Z", "+00:00"))
                    created_utc = dt.timestamp()
                except ValueError:
                    pass

        if not self._timestamp_in_window(created_utc, start_ts, end_ts):
            return []

        score = 0
        score_el = soup.find("div", class_="score")
        if score_el:
            try:
                score = int(score_el.get("title", "0"))
            except ValueError:
                pass

        items: List[RedditItem] = []
        if title or body:
            items.append(
                RedditItem(
                    id=f"post_{post_id}",
                    content_type=ContentType.POST,
                    title=title,
                    body=body,
                    author=author,
                    subreddit=subreddit,
                    created_utc=created_utc,
                    score=score,
                    url=f"/r/{subreddit}/comments/{post_id}/",
                    post_id=post_id,
                    window_label=label,
                )
            )

        comment_count = 0
        for cel in soup.find_all("div", class_="comment"):
            if comment_count >= self.settings.reddit_comment_limit:
                break
            c_body_el = cel.find("div", class_="md")
            c_body = c_body_el.get_text(" ", strip=True) if c_body_el else ""
            if not c_body:
                continue

            c_author = "[deleted]"
            c_tagline = cel.find("p", class_="tagline")
            if c_tagline:
                c_author_el = c_tagline.find("a", class_=re.compile(r"author"))
                if c_author_el:
                    c_author = c_author_el.get_text(strip=True)

            c_id = cel.get("data-fullname", f"c_{comment_count}")
            if c_id.startswith("t1_"):
                c_id = c_id[3:]

            items.append(
                RedditItem(
                    id=f"comment_{c_id}",
                    content_type=ContentType.COMMENT,
                    title="",
                    body=c_body,
                    author=c_author,
                    subreddit=subreddit,
                    created_utc=created_utc,
                    score=0,
                    url=f"/r/{subreddit}/comments/{post_id}/",
                    parent_id=f"t3_{post_id}",
                    post_id=post_id,
                    window_label=label,
                )
            )
            comment_count += 1
        return items

    def _scrape_firecrawl(
        self,
        url: str,
        subreddit: str,
        label: str,
        start_ts: float,
        end_ts: float,
    ) -> List[RedditItem]:
        """Use Firecrawl to scrape a Reddit post when other methods are blocked."""
        api_key = self.settings.firecrawl_api_key
        if not api_key:
            return []

        try:
            import httpx
            response = httpx.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "url": url,
                    "formats": ["markdown"],
                },
                timeout=max(self._timeout, 30.0),
            )
            response.raise_for_status()
            data = response.json()
            markdown = data.get("data", {}).get("markdown", "")
            if not markdown:
                return []
                
            # Treat the whole markdown as a single post for simplicity since 
            # Firecrawl compresses the comments into the markdown.
            post_id_match = re.search(r"/comments/(\w+)", url)
            post_id = post_id_match.group(1) if post_id_match else "unknown"
            
            # Use current time for scraped content as fallback
            created_utc = time.time()
            if not self._timestamp_in_window(created_utc, start_ts, end_ts):
                 # Assume it's in window since search returned it
                 created_utc = (start_ts + end_ts) / 2.0

            return [
                RedditItem(
                    id=f"post_{post_id}",
                    content_type=ContentType.POST,
                    title=f"Reddit Post: {post_id}",
                    body=markdown,
                    author="[unknown]",
                    subreddit=subreddit,
                    created_utc=created_utc,
                    score=0,
                    url=url,
                    post_id=post_id,
                    window_label=label,
                )
            ]
        except Exception as exc:
            logger.error("Firecrawl scrape failed for %s: %s", url, exc)
            return []

    @staticmethod
    def _timestamp_in_window(created_utc: float, start_ts: float, end_ts: float) -> bool:
        return start_ts <= created_utc <= end_ts

    @staticmethod
    def _normalize_reddit_url(url: str) -> str:
        """Normalize a Reddit URL to old.reddit.com for consistent scraping."""
        url = re.sub(
            r"https?://(?:www\.|old\.)?reddit\.com",
            "https://old.reddit.com",
            url,
            flags=re.IGNORECASE,
        )
        url = url.split("?")[0].split("#")[0]
        match = re.match(r"(https://old\.reddit\.com/r/[\w-]+/comments/\w+)", url)
        if match:
            return match.group(1) + "/"
        return url
