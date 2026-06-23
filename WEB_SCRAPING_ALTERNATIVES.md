# Web Scraping Alternatives for Reddit Data

Since Reddit API credentials are difficult to obtain, this project now supports multiple web scraping backends to collect Reddit data without requiring official API access.

## Quick Start

The default configuration uses **DuckDuckGo Search** (no API key required) to search for Reddit posts and scrapes them using Reddit's public JSON endpoint.

To enable web scraping instead of Reddit API:

```bash
# Set scrape method to web
export SCRAPE_METHOD=web

# Run ingestion
python -m src.pipeline.ingest
```

## Available Backends

### 1. **DuckDuckGo (Recommended - Default)**

✅ **Pros:**
- No API key required
- Free tier available
- Good rate limiting
- Reliable search results

❌ **Cons:**
- Slightly slower than other methods
- Rate limited to reasonable levels

**Setup:**
```bash
# Install
pip install ddgs

# Set in .env or environment
SCRAPE_METHOD=web
WEB_SCRAPER_BACKEND=ddgs
```

---

### 2. **Brave Search API**

✅ **Pros:**
- Privacy-focused
- Good search quality
- Affordable API plans
- No personal data collection

❌ **Cons:**
- Requires API key
- Not free

**Setup:**
1. Get API key from https://api.search.brave.com/
2. Add to `.env`:
   ```bash
   SCRAPE_METHOD=web
   WEB_SCRAPER_BACKEND=brave
   BRAVE_SEARCH_API_KEY=your_api_key_here
   ```

**Pricing:** Free tier available (100 requests/day)

---

### 3. **Crawl4AI**

✅ **Pros:**
- Advanced web scraping capabilities
- Can handle JavaScript-heavy sites
- Open-source friendly
- Good for complex scraping

❌ **Cons:**
- Slower than simple search APIs
- Requires additional dependencies

**Setup:**
1. Install:
   ```bash
   pip install crawl4ai
   ```

2. Add to `.env`:
   ```bash
   SCRAPE_METHOD=web
   WEB_SCRAPER_BACKEND=crawl4ai
   ```

**Note:** Crawl4AI works best for scraping the content of pages once URLs are discovered.

---

### 4. **Firecrawl API**

✅ **Pros:**
- Excellent for complex scraping scenarios
- Handles JavaScript rendering
- Good documentation
- Reliable API

❌ **Cons:**
- Requires paid subscription (free trial available)
- API key needed
- Most expensive option

**Setup:**
1. Get API key from https://www.firecrawl.dev/
2. Add to `.env`:
   ```bash
   SCRAPE_METHOD=web
   WEB_SCRAPER_BACKEND=firecrawl
   FIRECRAWL_API_KEY=your_api_key_here
   ```

**Pricing:** Free trial available, then paid plans start at $99/month

---

## Configuration

### Example `.env` file for web scraping:

```bash
# Enable web scraping
SCRAPE_METHOD=web

# Choose backend
WEB_SCRAPER_BACKEND=ddgs  # or brave, crawl4ai, firecrawl

# Optional: API keys for premium backends
BRAVE_SEARCH_API_KEY=your_brave_key_here
FIRECRAWL_API_KEY=your_firecrawl_key_here

# Scraping settings
SCRAPE_DELAY=1.5  # Delay between requests (seconds)
REDDIT_POST_LIMIT=50
REDDIT_COMMENT_LIMIT=20
REDDIT_SEARCH_QUERY='RAG OR "retrieval augmented" OR "AI safety"'
```

### Multi-Backend Fallback

The scraper automatically falls back to DuckDuckGo if:
- The selected backend fails
- API key is missing for premium backends
- Required packages are not installed

## Installation Guide

### Option 1: Minimal Setup (DuckDuckGo only)

```bash
pip install ddgs beautifulsoup4
```

### Option 2: With Crawl4AI Support

```bash
pip install ddgs beautifulsoup4 crawl4ai
```

### Option 3: Full Setup (All backends)

```bash
pip install -r requirements.txt
```

## Troubleshooting

### "ddgs is required but not installed"
```bash
pip install ddgs
```

### "httpx required for Brave Search"
```bash
pip install httpx
```

### "Brave Search API key not set"
- Get key from https://api.search.brave.com/
- Add to `.env`: `BRAVE_SEARCH_API_KEY=your_key`

### "Rate limited by search backend"
- Increase `SCRAPE_DELAY` in `.env`
- Switch to a different backend
- Check rate limit policies of your chosen backend

### "Crawl4AI async errors"
- Ensure all dependencies: `pip install crawl4ai[all]`
- Check Python version (3.7+)

## How It Works

The web scraping pipeline:

1. **Search Phase:** Uses configured backend (DDGS/Brave/Crawl4AI/Firecrawl) to find Reddit post URLs
2. **Discovery Phase:** Filters results to actual Reddit post URLs (format: `/r/*/comments/*`)
3. **Scraping Phase:** Fetches full post + comment data via:
   - Primary: Reddit's public `.json` endpoint (most reliable)
   - Fallback: HTML scraping of old.reddit.com

## Performance Comparison

| Backend | Speed | Accuracy | Cost | API Key | Fallback |
|---------|-------|----------|------|---------|----------|
| DuckDuckGo | Medium | High | Free | No | - |
| Brave | Fast | High | $0-9/mo | Yes | DDGS |
| Crawl4AI | Slow | Very High | Free | No | DDGS |
| Firecrawl | Fast | Excellent | $99+/mo | Yes | DDGS |

## Rate Limiting & Ethics

When using web scraping:

- **Respect robots.txt** - All backends automatically respect Reddit's robots.txt
- **Use reasonable delays** - Default SCRAPE_DELAY=1.5 seconds
- **Don't overload** - Limit requests with `REDDIT_POST_LIMIT`
- **Reddit's Terms** - Ensure compliance with Reddit's Terms of Service
- **Be respectful** - Don't scrape more than necessary

## Next Steps

1. Choose a backend based on your needs
2. Install required packages: `pip install -r requirements.txt`
3. Configure `.env` with your chosen backend
4. Run: `python -m src.pipeline.ingest`
5. Check `demo_results.json` for the saved demo output

## Support

If issues persist:

1. Check logs for specific error messages
2. Try fallback to DuckDuckGo: `WEB_SCRAPER_BACKEND=ddgs`
3. Verify all required packages are installed
4. Check API rate limits for premium backends
5. Increase SCRAPE_DELAY if rate limited

---

**Note:** This solution respects Reddit's public data and terms of service by using public endpoints and search APIs rather than circumventing authentication systems.
