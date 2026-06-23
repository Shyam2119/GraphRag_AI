# 🚀 Quick Start Setup Guide

This guide will get you running the GraphRAG demo in **under 5 minutes**.

## Option 1: Zero Config (Fastest) ⚡

Works with no API keys, no Docker, and no `.env`:

```bash
# 1. Clone repository
git clone <your-repo-url>
cd GraphRAG_assignment

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# .\venv\Scripts\activate       # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run demo with built-in safe defaults
python demo.py
```

**What happens:**
- Uses synthetic Reddit data (40+ realistic samples)
- Uses hash embeddings (no ML, instant)
- Uses in-memory graph and vector stores
- Rule-based fallback for entity extraction
- No API keys needed ✓

**Output:**
- 4 demo queries with full retrieval metrics
- Fused results showing hybrid retrieval benefits
- Generated answers with citations

---

## Option 2: Best Free Quality (Gemini) 💡

Upgrade to real LLM and embeddings, still free:

```bash
# 1-3. Same as Option 1 (clone, venv, pip install)

# 4. Get free Gemini API key
#    → Go to https://aistudio.google.com/app/apikeys
#    → Click "Create API Key"
#    → Copy the key

# 5. Create .env file
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux

# 6. Edit .env with your key
#    Windows: notepad .env
#    macOS/Linux: nano .env
#    
#    Add these lines:
#    GEMINI_API_KEY=your_key_here
#    EMBEDDING_PROVIDER=auto
#    USE_SAMPLE_DATA=true

# 7. Run demo
python demo.py
```

**What happens:**
- Uses Gemini 2.0 Flash for entity extraction and answering
- Uses sentence-transformers embeddings (high quality, local)
- Still uses sample data and in-memory storage
- Significantly better answer quality ↑

**Gemini free tier limits:**
- 60 requests/minute (plenty for demos)
- 32K tokens per request (sufficient)
- Always free tier available

---

## Option 3: Live Reddit Data 🌐

Scrape real Reddit data across 3 time windows:

```bash
# 1-5. Same as Option 2 (get Gemini key, create .env)

# 6. Get Reddit API credentials
#    → Go to https://www.reddit.com/prefs/apps
#    → Click "Create App" → select "script" type
#    → Copy client_id and client_secret

# 7. Add to .env:
#    REDDIT_CLIENT_ID=your_id_here
#    REDDIT_CLIENT_SECRET=your_secret_here
#    USE_SAMPLE_DATA=false

# 8. Run ingestion + demo
#    (First run takes 2-3 minutes for Reddit scraping)
python demo.py
```

**What happens:**
- Scrapes real Reddit posts/comments from MachineLearning, LocalLLaMA, artificial subreddits
- Covers Q3 2025, Q4 2025, Q1 2026 time windows
- Extracts real entities, topics, sentiment
- Builds temporal knowledge graph
- All 4 demo queries work on real data

---

## Option 4: Production Setup 🏢

Full power: Neo4j, ChromaDB, premium LLM:

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (optional, use Gemini instead)

### Setup

```bash
# 1-5. Same as Option 2

# 6. Start Neo4j database
docker compose up -d

# 7. Add to .env:
#    NEO4J_URI=bolt://localhost:7687
#    NEO4J_USER=neo4j
#    NEO4J_PASSWORD=graphrag123
#    USE_FALLBACK_GRAPH=false
#    CHROMA_PERSIST_DIR=./data/chroma
#    USE_CHROMA=true

# 8. Run with real data
#    (Add Reddit credentials as in Option 3)
python demo.py

# 9. Optional: view Neo4j dashboard
#    → Open http://localhost:7474 in browser
#    → Username: neo4j
#    → Password: graphrag123
```

**What happens:**
- Persistent Neo4j graph database
- Persistent ChromaDB vector store
- Real Reddit data ingestion
- Full temporal query support
- Production-grade persistence

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'pydantic_settings'"
```bash
# Dependencies not installed
pip install -r requirements.txt
```

### "RuntimeError: Neo4j connection failed"
```bash
# Neo4j not running (if USE_FALLBACK_GRAPH=false)
# Either:
# A) Start Docker: docker compose up -d
# B) Or use fallback: set USE_FALLBACK_GRAPH=true
```

### "REDDIT_CLIENT_ID... not configured"
```bash
# Using real Reddit data but credentials not set
# Either:
# A) Add credentials to .env (see Option 3)
# B) Or use sample data: set USE_SAMPLE_DATA=true
```

### "Gemini API key is invalid"
```bash
# API key not recognized or invalid
# 1) Verify key in .env has no extra spaces
# 2) Check key works: https://aistudio.google.com/
# 3) Keys expire after 30 days of inactivity — regenerate if old
```

### Script hangs on "Collecting Reddit data..."
```bash
# Reddit API rate limiting
# Normal — just wait or reduce REDDIT_POST_LIMIT in .env
```

---

## What to Expect

### Demo Output Structure

For each of the 4 demo queries, you'll see:

1. **Query Details**
   - Original question
   - Parsed query type (semantic/graph/hybrid/temporal)
   - Entities extracted
   - Time range applied

2. **Graph-Only Results** 
   - Pure knowledge graph traversal
   - Shows what graph alone can retrieve

3. **Vector-Only Results**
   - Pure semantic search
   - Shows what embeddings alone can retrieve

4. **Fused Results (Weighted RRF)**
   - Combined and reranked
   - Unified list with better coverage

5. **Retrieval Metrics**
   - Overlap between retrievers
   - Unique contributions from each
   - Author and subreddit diversity
   - Why fusion is better than either alone

6. **Generated Answer**
   - LLM-synthesized response
   - Citations to sources
   - Query-type-aware structure

### Example Query Results

```
1 │ Semantic Query (vector-dominant)
"What are the main challenges people face when building RAG pipelines?"

Graph-Only Results: 3 results, score avg 0.45
  - Often misses semantic nuance
  - Good for entity-based filtering only

Vector-Only Results: 8 results, score avg 0.78
  - Excellent semantic matching
  - But misses some structured relationships

Fused Results (RRF): 10 unique results, avg score 0.68
  - Combines best of both
  - Better coverage of perspectives
  - Diversity across authors/subreddits
```

---

## Next Steps

- **Explore Custom Queries**: `python demo.py --query "Your question here"`
- **Tweak Configuration**: Edit `.env` for different settings
- **View Results**: Check `demo_results.json` for full structured output
- **Neo4j Browser**: Browse graph at http://localhost:7474 (if running Docker)
- **Check Logs**: Look for detailed execution traces

---

## Common Configuration Scenarios

### "I want the fastest demo possible"
```env
USE_SAMPLE_DATA=true
EMBEDDING_PROVIDER=hash
USE_FALLBACK_GRAPH=true
```
→ No downloads, instant run

### "I want best quality with free services"
```env
USE_SAMPLE_DATA=true
GEMINI_API_KEY=your_key
EMBEDDING_PROVIDER=auto
```
→ Good quality, still free, no Docker needed

### "I want to showcase the hybrid retrieval"
```env
USE_SAMPLE_DATA=true
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
GEMINI_API_KEY=your_key
```
→ Real data, real entities, real graph

### "I want production-grade everything"
```env
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
OPENAI_API_KEY=your_key
USE_FALLBACK_GRAPH=false
USE_CHROMA=true
```
→ Best quality, persistent storage (costs ~$0.50-1.00 per demo run)

---

## Support

For issues:
1. Check `.env` configuration
2. Run: `python validate.py` (checks project setup)
3. Run: `python test_integration.py` (tests all components)
4. Check the troubleshooting section above
