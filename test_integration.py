#!/usr/bin/env python3
"""Quick integration test — validates all major components work together."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all critical imports work."""
    print("[1/5] Testing imports...")
    try:
        from config import get_settings
        from src.models import RedditItem, EnrichedItem, QueryResponse
        from src.ingestion.sample_data import generate_sample_data
        from src.ingestion.llm_extractor import LLMExtractor
        from src.graph.factory import create_graph_store
        from src.vector.factory import create_vector_store
        from src.llm.client import LLMClient
        from src.pipeline.query import QueryEngine
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("[2/5] Testing configuration...")
    try:
        from config import get_settings
        s = get_settings()
        print(f"  ✓ Config loaded (LLM provider: {s.llm_provider})")
        return True
    except Exception as e:
        print(f"  ✗ Config failed: {e}")
        return False

def test_sample_data():
    """Test sample data generation."""
    print("[3/5] Testing sample data generation...")
    try:
        from src.ingestion.sample_data import generate_sample_data
        items = generate_sample_data()
        print(f"  ✓ Generated {len(items)} sample items")
        assert len(items) > 10, "Not enough sample data"
        return True
    except Exception as e:
        print(f"  ✗ Sample data generation failed: {e}")
        return False

def test_storage():
    """Test graph and vector stores initialize."""
    print("[4/5] Testing storage initialization...")
    try:
        from src.graph.factory import create_graph_store
        from src.vector.factory import create_vector_store
        
        graph = create_graph_store()
        vector = create_vector_store()
        
        print(f"  ✓ Graph store: {type(graph).__name__}")
        print(f"  ✓ Vector store: {type(vector).__name__}")
        
        graph.close()
        return True
    except Exception as e:
        print(f"  ✗ Storage initialization failed: {e}")
        return False

def test_llm():
    """Test LLM client initialization."""
    print("[5/5] Testing LLM client...")
    try:
        from src.llm.client import LLMClient
        llm = LLMClient()
        has_key = llm.has_api_key()
        print(f"  ✓ LLM client initialized (API key available: {has_key})")
        return True
    except Exception as e:
        print(f"  ✗ LLM client failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("GraphRAG Integration Test")
    print("="*60 + "\n")
    
    results = [
        test_imports(),
        test_config(),
        test_sample_data(),
        test_storage(),
        test_llm(),
    ]
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All {total} tests passed!")
        print("\nReady to run: python demo.py")
        print("="*60 + "\n")
        return 0
    else:
        print(f"✗ {total - passed}/{total} tests failed")
        print("="*60 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
