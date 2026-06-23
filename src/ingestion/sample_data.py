"""Synthetic Reddit data spanning 3 time windows for demo without API keys.

Generates realistic, diverse sample data with temporal evolution:
- Q3 2025: Early RAG excitement, initial LLM scaling discussions
- Q4 2025: RAG scaling pain points, early AI safety concerns, regulation talk
- Q1 2026: Agentic AI safety fears, GraphRAG emergence, open-source maturation
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List

from src.models import ContentType, RedditItem

# ── Authors with consistent personas ──────────────────────────────────────────

AUTHORS = {
    "ml_researcher_42": {"subreddits": ["MachineLearning", "artificial"], "lean": "academic"},
    "ai_safety_advocate": {"subreddits": ["artificial", "MachineLearning"], "lean": "safety"},
    "rag_engineer": {"subreddits": ["MachineLearning", "LocalLLaMA"], "lean": "builder"},
    "oss_enthusiast": {"subreddits": ["LocalLLaMA", "MachineLearning"], "lean": "open-source"},
    "policy_wonk": {"subreddits": ["artificial"], "lean": "regulation"},
    "data_scientist_x": {"subreddits": ["MachineLearning"], "lean": "practitioner"},
    "llm_hacker": {"subreddits": ["LocalLLaMA"], "lean": "hacker"},
    "neural_nomad": {"subreddits": ["MachineLearning", "artificial"], "lean": "researcher"},
    "prod_ml_lead": {"subreddits": ["MachineLearning"], "lean": "production"},
    "ethics_prof": {"subreddits": ["artificial"], "lean": "ethics"},
    "startup_cto": {"subreddits": ["MachineLearning", "LocalLLaMA"], "lean": "startup"},
    "quant_researcher": {"subreddits": ["LocalLLaMA", "MachineLearning"], "lean": "quantization"},
    "govtech_analyst": {"subreddits": ["artificial"], "lean": "policy"},
    "inference_guru": {"subreddits": ["LocalLLaMA"], "lean": "optimization"},
    "vector_db_dev": {"subreddits": ["MachineLearning"], "lean": "infrastructure"},
    "alignment_researcher": {"subreddits": ["artificial", "MachineLearning"], "lean": "safety"},
}

# ── Posts per time window (distinct content showing temporal evolution) ────────

Q3_2025_POSTS = [
    {
        "title": "RAG systems are getting really good — here's my experience",
        "body": "I've been building production RAG pipelines for 6 months now. Vector search combined with reranking has improved recall significantly. The key breakthrough was switching from naive chunking to semantic chunking with overlap. Still struggling with the right chunk size though — anyone found the sweet spot?",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "vector search", "chunking"],
        "sentiment": "positive",
    },
    {
        "title": "Comprehensive guide to building RAG with ChromaDB and LangChain",
        "body": "After months of experimentation, here's my complete guide to production RAG. Key lessons: 1) Metadata filtering is crucial for large corpora, 2) Hybrid search (BM25 + dense) outperforms either alone, 3) The embedding model matters more than the vector DB. We saw a 23% improvement switching from all-MiniLM to text-embedding-ada-002.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "ChromaDB", "LangChain", "embedding"],
        "sentiment": "positive",
    },
    {
        "title": "Llama 3 fine-tuning results are incredible for domain-specific tasks",
        "body": "Fine-tuned Llama 3 8B on our legal corpus — it outperforms GPT-4 on our specific benchmarks. The open-source community is making proprietary models less relevant for specialized use cases. QLoRA + 4-bit quantization makes this possible on a single 3090.",
        "subreddit": "LocalLLaMA",
        "topics": ["open-source LLM", "Llama", "fine-tuning", "quantization"],
        "sentiment": "positive",
    },
    {
        "title": "Why vector databases alone aren't enough for complex reasoning",
        "body": "I keep seeing projects that throw everything into a vector DB and call it RAG. But for multi-hop reasoning — where you need to connect information across documents — pure similarity search falls short. We need structured knowledge alongside embeddings. Anyone exploring graph-enhanced approaches?",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "vector search", "knowledge graph"],
        "sentiment": "mixed",
    },
    {
        "title": "r/LocalLLaMA is where open-source AI innovation actually happens",
        "body": "This community has become the de facto hub for quantized model releases, fine-tuning recipes, and inference optimization. The shift from relying on cloud APIs to local inference is accelerating. Last month alone we saw 15 new model variants released by community members.",
        "subreddit": "LocalLLaMA",
        "topics": ["open-source LLM", "community"],
        "sentiment": "positive",
    },
    {
        "title": "Initial thoughts on the EU AI Act — what it means for LLM deployments",
        "body": "The EU AI Act is now official. For those deploying LLMs in production: high-risk classifications will require conformity assessments, transparency obligations for general-purpose AI, and mandatory incident reporting. My take: the framework is reasonable but implementation details matter.",
        "subreddit": "artificial",
        "topics": ["AI regulation", "EU AI Act"],
        "sentiment": "neutral",
    },
    {
        "title": "Embedding model comparison: which one should you use for RAG in 2025?",
        "body": "Tested 12 embedding models on our internal retrieval benchmark. Top performers: 1) Cohere embed-v3, 2) OpenAI text-embedding-3-large, 3) BGE-large-en-v1.5. Surprisingly, the open-source BGE model comes within 3% of the best commercial options. Size matters less than training data quality.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "embedding", "benchmark"],
        "sentiment": "positive",
    },
    {
        "title": "The future of AI is open-source — and it's happening faster than expected",
        "body": "Six months ago, proprietary models had a clear lead. Now? Llama 3, Mistral, and community fine-tunes are competitive on most tasks. The model weights commoditization is real. The value is shifting to application layer and data quality.",
        "subreddit": "LocalLLaMA",
        "topics": ["open-source LLM", "Llama"],
        "sentiment": "positive",
    },
    {
        "title": "Sentiment analysis of AI research papers shows growing safety concerns",
        "body": "Analyzed 2,000 recent ML papers. 34% now include an ethics/safety section, up from 12% two years ago. The most common concerns: hallucination in production systems, bias amplification, and potential for misuse. The field is becoming more self-aware, but action still lags behind awareness.",
        "subreddit": "artificial",
        "topics": ["AI safety", "hallucination", "bias"],
        "sentiment": "mixed",
    },
    {
        "title": "How we scaled our RAG pipeline to handle 10M documents",
        "body": "Our team scaled from 100K to 10M docs in our RAG system. Key architectural decisions: sharded vector indices, async ingestion pipeline, tiered retrieval (BM25 pre-filter → dense retrieval → reranking), and aggressive caching. Latency went from 2s to 200ms per query.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "scaling", "infrastructure"],
        "sentiment": "positive",
    },
    {
        "title": "Community-driven AI governance is more effective than top-down regulation",
        "body": "Rather than waiting for government regulation, AI communities should self-regulate through shared safety benchmarks, model cards, and deployment guidelines. The responsible AI practices emerging from open-source communities are more practical than legislative approaches.",
        "subreddit": "artificial",
        "topics": ["AI regulation", "AI safety", "open-source LLM"],
        "sentiment": "mixed",
    },
    {
        "title": "GGUF quantization breakthrough: run 70B models on consumer hardware",
        "body": "New quantization methods allow running 70B parameter models on 32GB RAM with acceptable quality. The Q4_K_M sweet spot gives ~95% of full precision quality at 1/4 the memory. This democratizes access to frontier-class models.",
        "subreddit": "LocalLLaMA",
        "topics": ["quantization", "open-source LLM", "inference"],
        "sentiment": "positive",
    },
]

Q4_2025_POSTS = [
    {
        "title": "RAG in production is harder than anyone admits",
        "body": "Been running RAG in production for 3 months now. The challenges nobody talks about: 1) Stale embeddings when source docs update, 2) Chunk boundary issues causing hallucination, 3) Retrieval latency spikes under load, 4) Users asking questions that span multiple documents. Pure vector search isn't enough.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "production", "hallucination"],
        "sentiment": "negative",
    },
    {
        "title": "GraphRAG vs traditional vector RAG — benchmark results",
        "body": "Ran benchmarks comparing Microsoft GraphRAG against vanilla ChromaDB retrieval. Graph traversal helped significantly on multi-hop questions (+31% accuracy) but added 2-3x latency. For simple lookup queries, vector-only was faster and nearly as accurate. The hybrid approach seems optimal.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "GraphRAG", "benchmark"],
        "sentiment": "mixed",
    },
    {
        "title": "AI safety discussions are shifting from theoretical to practical",
        "body": "The AI safety conversation has matured. We're no longer debating whether AI is dangerous — we're discussing specific failure modes: prompt injection in production systems, data poisoning in fine-tuning, and the alignment tax of safety measures. This is productive progress.",
        "subreddit": "artificial",
        "topics": ["AI safety", "alignment"],
        "sentiment": "positive",
    },
    {
        "title": "Worried about the pace of open-source model releases without safety testing",
        "body": "Three new 70B+ models released this week alone, none with comprehensive safety evaluations. The open-source community moves fast but safety testing is lagging. We need shared safety benchmarks that every model release should pass before distribution.",
        "subreddit": "LocalLLaMA",
        "topics": ["AI safety", "open-source LLM"],
        "sentiment": "negative",
    },
    {
        "title": "EU AI Act compliance toolkit for ML engineers",
        "body": "Created an open-source toolkit for EU AI Act compliance. Covers: risk classification assessment, model card generation, bias testing automation, and audit trail logging. The regulation is actually pushing better engineering practices.",
        "subreddit": "artificial",
        "topics": ["AI regulation", "EU AI Act"],
        "sentiment": "positive",
    },
    {
        "title": "Knowledge graphs + vector search = the next evolution of RAG",
        "body": "After struggling with pure vector RAG limitations, we added a Neo4j knowledge graph layer. Entity relationships that embeddings miss — like 'Company A acquired Company B which makes Product C' — are now captured explicitly. Retrieval quality jumped 40% on complex queries.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "knowledge graph", "GraphRAG"],
        "sentiment": "positive",
    },
    {
        "title": "Yoshua Bengio and Stuart Russell call for mandatory safety evaluations",
        "body": "In a joint statement, Bengio and Russell advocate for mandatory pre-deployment safety evaluations for models above a capability threshold. Industry pushback is strong — Meta and Google favor voluntary commitments. The debate between regulation and innovation continues.",
        "subreddit": "artificial",
        "topics": ["AI regulation", "AI safety"],
        "sentiment": "neutral",
    },
    {
        "title": "LocalLLaMA community milestones: 500K members and growing",
        "body": "r/LocalLLaMA just hit 500K subscribers. The community has produced 200+ fine-tuned models, 50+ inference frameworks, and countless quantization experiments. We're not just consumers of AI — we're builders. The most active contributors are reshaping the field.",
        "subreddit": "LocalLLaMA",
        "topics": ["open-source LLM", "community"],
        "sentiment": "positive",
    },
    {
        "title": "Retrieval-augmented generation is becoming retrieval-augmented reasoning",
        "body": "The evolution from RAG to RAR: instead of just retrieving context, modern systems reason over retrieved information. Chain-of-thought prompting over retrieved docs, iterative retrieval loops, and self-critique of generated answers. This is where the field is heading.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "reasoning", "LLM"],
        "sentiment": "positive",
    },
    {
        "title": "The real cost of AI regulation: small startups can't compete",
        "body": "New compliance requirements favor large companies with legal teams and resources. A startup deploying a chatbot shouldn't face the same burden as Google deploying a model used by billions. Risk-proportionate regulation is essential.",
        "subreddit": "artificial",
        "topics": ["AI regulation"],
        "sentiment": "negative",
    },
    {
        "title": "Fine-tuning open-source models for enterprise: lessons learned",
        "body": "Deployed fine-tuned Mistral models for three enterprise clients. Key learnings: data quality trumps data quantity, evaluation benchmarks must match production queries, and you need a human-in-the-loop for at least the first month. Open-source is enterprise-ready, but it's not plug-and-play.",
        "subreddit": "LocalLLaMA",
        "topics": ["open-source LLM", "fine-tuning", "deployment"],
        "sentiment": "mixed",
    },
    {
        "title": "Temporal knowledge graphs for tracking AI discourse evolution",
        "body": "Building a system that tracks how AI discussions evolve over time using temporal knowledge graphs. Every entity and relationship has a timestamp. This lets us answer questions like 'When did concern about agentic AI first emerge?' — something pure vector search can't do well.",
        "subreddit": "MachineLearning",
        "topics": ["knowledge graph", "RAG"],
        "sentiment": "positive",
    },
    {
        "title": "AI safety research funding has doubled in 2025",
        "body": "According to the latest analysis, funding for AI safety research has doubled compared to 2024. Major contributions from Open Philanthropy, new government grants, and surprisingly, from AI companies themselves. The question is whether funding translates to actual progress.",
        "subreddit": "artificial",
        "topics": ["AI safety"],
        "sentiment": "positive",
    },
    {
        "title": "Hybrid search implementation: BM25 + dense retrieval + graph traversal",
        "body": "Implemented a three-way hybrid search system. BM25 handles exact keyword matches, dense retrieval captures semantic similarity, and graph traversal provides structural relationships. Reciprocal Rank Fusion combines all three. On our benchmark, this beats any single retriever by 25%+.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "GraphRAG", "vector search"],
        "sentiment": "positive",
    },
]

Q1_2026_POSTS = [
    {
        "title": "New AI safety concerns in Q1 2026: autonomous agentic systems",
        "body": "Unlike Q4 2025 when we focused on model weights and training safety, Q1 2026 discussions center on autonomous AI agents making decisions without human oversight. Agent frameworks that chain multiple LLM calls with tool use create emergent behaviors we can't fully predict. This is qualitatively different from previous safety concerns.",
        "subreddit": "artificial",
        "topics": ["AI safety", "agentic AI"],
        "sentiment": "negative",
    },
    {
        "title": "Sentiment shift: RAG hype cooling, agent frameworks rising",
        "body": "Six months ago everyone was building RAG. Now the conversation has shifted to AI agents with tool use. RAG is still foundational infrastructure but no longer the headline. The new frontier is autonomous systems that retrieve, reason, and act — with RAG as one component of a larger agent architecture.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "agentic AI"],
        "sentiment": "mixed",
    },
    {
        "title": "Agent safety is the new frontier — and we're not ready",
        "body": "With AI agents now executing code, making API calls, and managing infrastructure, the attack surface has exploded. Prompt injection becomes code execution. A hallucinated API call becomes a real action. We need sandboxing, permission systems, and human-in-the-loop safeguards before deploying agents at scale.",
        "subreddit": "artificial",
        "topics": ["AI safety", "agentic AI"],
        "sentiment": "negative",
    },
    {
        "title": "GraphRAG 2.0: what we've learned from deploying graph-enhanced retrieval",
        "body": "After 6 months of running GraphRAG in production: 1) Community detection in the graph reveals topic clusters that embeddings miss, 2) Temporal edges are essential — relationships that were true in Q3 may not hold in Q1, 3) Entity resolution is the hardest unsolved problem. The hybrid approach is clearly superior.",
        "subreddit": "MachineLearning",
        "topics": ["GraphRAG", "RAG", "knowledge graph"],
        "sentiment": "positive",
    },
    {
        "title": "Open-source LLMs now match GPT-4 on most benchmarks — what's next?",
        "body": "The gap has closed. Llama 3.3, Qwen 2.5, and DeepSeek V3 match or exceed GPT-4 on standard benchmarks. The open-source community proved that model capability isn't a moat. What's next? Better tooling, easier deployment, and multimodal capabilities. r/LocalLLaMA is driving this forward.",
        "subreddit": "LocalLLaMA",
        "topics": ["open-source LLM", "benchmark"],
        "sentiment": "positive",
    },
    {
        "title": "AI regulation update: G7 Hiroshima AI Code of Conduct gaining adoption",
        "body": "Beyond the EU AI Act, the G7 Code of Conduct for AI is being adopted voluntarily by major tech companies. Key principles: transparency in AI capabilities, responsible disclosure of safety incidents, and commitment to red-teaming. Is voluntary compliance enough?",
        "subreddit": "artificial",
        "topics": ["AI regulation"],
        "sentiment": "neutral",
    },
    {
        "title": "Who are the most influential voices in AI regulation right now?",
        "body": "Tracking the most-cited people in AI regulation discussions: Yoshua Bengio continues pushing for mandatory evaluations, Stuart Russell advocates for provably beneficial AI, and Timnit Gebru focuses on environmental and social impact. On the industry side, Sam Altman promotes 'iterative deployment' while Meta's Yann LeCun argues regulation is premature.",
        "subreddit": "artificial",
        "topics": ["AI regulation", "AI safety"],
        "sentiment": "neutral",
    },
    {
        "title": "Building agents that are safe by design, not safe by hope",
        "body": "Our approach to agent safety: capability-based access control (agents can only access what they're explicitly granted), execution sandboxing, mandatory human approval for destructive actions, and comprehensive audit logging. Safety shouldn't be an afterthought — it should be architectural.",
        "subreddit": "MachineLearning",
        "topics": ["agentic AI", "AI safety"],
        "sentiment": "positive",
    },
    {
        "title": "The local inference revolution is complete — what did we learn?",
        "body": "Two years of the local LLM movement. We learned: 1) Quantization quality matters more than model size, 2) Community collaboration beats corporate R&D for practical tooling, 3) Privacy-preserving AI is possible without sacrificing quality, 4) The real innovation happens at the application layer.",
        "subreddit": "LocalLLaMA",
        "topics": ["open-source LLM", "quantization", "inference", "privacy"],
        "sentiment": "positive",
    },
    {
        "title": "Temporal RAG: answering questions about how opinions change over time",
        "body": "Built a system that tracks discussion evolution across time windows. By tagging every chunk with temporal metadata and using time-filtered retrieval, we can answer questions like 'How has sentiment about RAG changed?' or 'What new concerns emerged this quarter?' This is a game-changer for market research and competitive intelligence.",
        "subreddit": "MachineLearning",
        "topics": ["RAG", "knowledge graph"],
        "sentiment": "positive",
    },
    {
        "title": "Emerging concern: AI agents as attack vectors in cybersecurity",
        "body": "New threat model: compromised AI agents with access to production systems. Unlike traditional malware, a manipulated agent can use social engineering via natural language. OWASP is developing a top-10 for AI agents. This wasn't on anyone's radar in Q4 2025.",
        "subreddit": "artificial",
        "topics": ["AI safety", "agentic AI"],
        "sentiment": "negative",
    },
    {
        "title": "MachineLearning vs LocalLLaMA vs artificial: how subreddits differ",
        "body": "Analysis of discussion patterns: r/MachineLearning focuses on research and production systems, r/LocalLLaMA on practical deployment and optimization, r/artificial on policy and societal impact. Each community has distinct leaders and conversation styles. Cross-pollination is increasing.",
        "subreddit": "MachineLearning",
        "topics": ["community"],
        "sentiment": "neutral",
    },
    {
        "title": "DeepSeek open-sources their RAG infrastructure — what it means for us",
        "body": "DeepSeek just released their production RAG stack: custom embeddings, hybrid retrieval with learned fusion weights, and a novel iterative refinement loop. Early benchmarks show it outperforms LangChain and LlamaIndex on complex queries. The open-source RAG tooling landscape is about to shift.",
        "subreddit": "LocalLLaMA",
        "topics": ["RAG", "open-source LLM"],
        "sentiment": "positive",
    },
    {
        "title": "AI regulation is fragmenting: US vs EU vs China approaches diverge",
        "body": "Three distinct regulatory philosophies are emerging: EU's risk-based framework (comprehensive but complex), US's sector-specific approach (flexible but patchy), and China's focus on content control and data sovereignty. For global companies, compliance is becoming a competitive advantage — and a cost center.",
        "subreddit": "artificial",
        "topics": ["AI regulation"],
        "sentiment": "mixed",
    },
]

# ── Comments with varied perspectives ─────────────────────────────────────────

SAMPLE_COMMENTS = {
    "Q3_2025": [
        "Totally agree — hybrid retrieval is the future. We switched from pure vector to BM25+dense and saw immediate improvements.",
        "Has anyone tried semantic chunking instead of fixed-size? Our recall improved 15% after switching.",
        "The embedding model choice is underrated. We spent weeks on retrieval architecture when the real bottleneck was embedding quality.",
        "Open-source models are good but still need guardrails. We're using Llama with custom safety filters in production.",
        "RAG is great for factual queries but struggles with subjective or nuanced questions. Adding a reasoning layer on top helps.",
        "Just deployed ChromaDB in production — the metadata filtering is really powerful for our multi-tenant use case.",
        "The EU AI Act timeline is too aggressive. Most companies won't be ready for the first compliance deadline.",
        "Anyone else seeing diminishing returns from larger embedding dimensions? 384 seems to be the sweet spot for most use cases.",
        "Fine-tuned a model on our domain data and it hallucinates less than the base model. Training data quality is everything.",
        "The gap between open-source and proprietary is closing fast. Give it another 6 months.",
    ],
    "Q4_2025": [
        "We switched from pure vector search to graph-enhanced RAG last quarter and the improvement on multi-hop queries is dramatic.",
        "AI safety shouldn't block open research though. We need transparency, not restrictions.",
        "The EU AI Act is actually quite reasonable for RAG use cases — most RAG systems would be classified as limited risk.",
        "Has anyone tried combining Neo4j with ChromaDB? The graph + vector combo seems natural for complex retrieval.",
        "GraphRAG is promising but the entity extraction step is expensive. We're spending more on extraction than on the actual queries.",
        "r/LocalLLaMA is definitely where the action is for OSS models. The community review process is faster than corporate safety teams.",
        "Agent frameworks are interesting but RAG is still essential infrastructure. Agents without good retrieval are just expensive random generators.",
        "Bengio's position on mandatory evaluations is reasonable. The question is who defines the evaluation criteria.",
        "Our hybrid retrieval system (vector + graph + keyword) beats any individual retriever by 25%. RRF fusion is the key.",
        "The safety discussion is finally getting practical. We need tools and benchmarks, not just philosophy papers.",
        "Production RAG is 80% data engineering, 15% retrieval optimization, and 5% prompt engineering. Most tutorials get the ratios backwards.",
        "Knowledge graphs add complexity but they're worth it for domains with structured relationships.",
    ],
    "Q1_2026": [
        "Agentic AI safety is a completely different challenge than model safety. The attack surface is orders of magnitude larger.",
        "The shift from RAG to agents is happening faster than expected. But RAG is the foundation that agents build on.",
        "Open-source LLMs matching GPT-4 was inevitable. The real question is who will be first to match GPT-5.",
        "Temporal knowledge graphs are underrated. Being able to ask 'what changed' is as valuable as asking 'what is'.",
        "Agent sandboxing should be mandatory, not optional. We learned this the hard way in our production deployment.",
        "The regulatory fragmentation is a real problem for startups operating globally. We need harmonized standards.",
        "GraphRAG 2.0 is exactly what we need. Entity resolution is indeed the hardest problem — anyone working on this?",
        "DeepSeek's RAG release is a game-changer. Their learned fusion weights approach is much better than static RRF.",
        "AI safety concerns in Q1 2026 are qualitatively different from Q4 2025. It's not about model weights anymore — it's about autonomous actions.",
        "The local inference movement proved something important: you don't need $100M to build useful AI systems.",
        "Community-driven model evaluation is more rigorous than corporate safety reviews. The crowd catches what internal teams miss.",
        "Privacy-preserving RAG is now possible with local models. No more sending sensitive data to cloud APIs.",
    ],
}

# ── Time Windows ──────────────────────────────────────────────────────────────

WINDOWS = [
    ("Q3_2025", datetime(2025, 7, 1), datetime(2025, 9, 30)),
    ("Q4_2025", datetime(2025, 10, 1), datetime(2025, 12, 31)),
    ("Q1_2026", datetime(2026, 1, 1), datetime(2026, 3, 31)),
]

POSTS_BY_WINDOW = {
    "Q3_2025": Q3_2025_POSTS,
    "Q4_2025": Q4_2025_POSTS,
    "Q1_2026": Q1_2026_POSTS,
}

AUTHOR_LIST = list(AUTHORS.keys())


def generate_sample_data(posts_per_window: int = 12) -> List[RedditItem]:
    """Generate realistic sample Reddit data across 3 time windows.

    Each window has distinct content reflecting temporal evolution of
    AI discourse, with unique posts and contextually relevant comments.
    """
    random.seed(42)
    items: List[RedditItem] = []
    post_counter = 0
    comment_counter = 0

    for window_label, win_start, win_end in WINDOWS:
        window_days = (win_end - win_start).days
        window_posts = POSTS_BY_WINDOW[window_label]
        window_comments = SAMPLE_COMMENTS[window_label]

        # Use all available posts for this window, cycling if needed
        num_posts = min(posts_per_window, len(window_posts))

        for i in range(num_posts):
            template = window_posts[i % len(window_posts)]
            post_counter += 1
            post_id = f"sample_p{post_counter}"

            # Select an author likely to post in this subreddit
            suitable_authors = [
                a for a, info in AUTHORS.items()
                if template["subreddit"] in info["subreddits"]
            ]
            author = random.choice(suitable_authors or AUTHOR_LIST)

            day_offset = random.randint(0, max(window_days - 1, 0))
            created = win_start + timedelta(
                days=day_offset,
                hours=random.randint(8, 22),
                minutes=random.randint(0, 59),
            )

            items.append(RedditItem(
                id=f"post_{post_id}",
                content_type=ContentType.POST,
                title=template["title"],
                body=template["body"],
                author=author,
                subreddit=template["subreddit"],
                created_utc=created.timestamp(),
                score=random.randint(10, 800),
                url=f"/r/{template['subreddit']}/comments/{post_id}/",
                post_id=post_id,
                window_label=window_label,
            ))

            # Add 2-4 comments per post, using window-appropriate comments
            num_comments = random.randint(2, 4)
            for j in range(num_comments):
                comment_counter += 1
                comment_created = created + timedelta(
                    hours=random.randint(1, 72),
                    minutes=random.randint(0, 59),
                )
                comment_author = random.choice(
                    [a for a in AUTHOR_LIST if a != author]
                )
                items.append(RedditItem(
                    id=f"comment_c{comment_counter}",
                    content_type=ContentType.COMMENT,
                    title="",
                    body=random.choice(window_comments),
                    author=comment_author,
                    subreddit=template["subreddit"],
                    created_utc=comment_created.timestamp(),
                    score=random.randint(1, 200),
                    url=f"/r/{template['subreddit']}/comments/{post_id}/c{comment_counter}/",
                    parent_id=f"t3_{post_id}",
                    post_id=post_id,
                    window_label=window_label,
                ))

    return items
