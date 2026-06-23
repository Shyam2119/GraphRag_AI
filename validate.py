#!/usr/bin/env python3
"""Validation script — checks project integrity and setup."""

import sys
from pathlib import Path

OK = "[OK]"
FAIL = "[FAIL]"
INFO = "[INFO]"

def check_file_exists(path: str, name: str) -> bool:
    """Check if critical file exists."""
    p = Path(path)
    exists = p.exists()
    status = OK if exists else FAIL
    print(f"  {status} {name}")
    return exists

def check_requirements():
    """Check Python requirements are installed."""
    print("\n[Checking Python dependencies...]")
    required = {
        "pydantic": "pydantic",
        "praw": "praw",
        "neo4j": "neo4j",
        "chromadb": "chromadb",
        "requests": "requests",
        "pytest": "pytest",
        "tenacity": "tenacity",
        "rich": "rich",
        "dateutil": "dateutil",
        "numpy": "numpy",
    }
    
    all_ok = True
    for module_name, display_name in required.items():
        try:
            __import__(module_name)
            print(f"  {OK} {display_name}")
        except ImportError:
            print(f"  {FAIL} {display_name} (run: pip install -r requirements.txt)")
            all_ok = False
    
    return all_ok

def check_files():
    """Check critical project files."""
    print("\n[Checking project files...]")
    files = [
        ("config.py", "Configuration"),
        ("demo.py", "Demo script"),
        (".env.example", "Environment template"),
        ("README.md", "Documentation"),
        ("requirements.txt", "Dependencies"),
        ("docker-compose.yml", "Docker compose"),
        ("src/models.py", "Data models"),
        ("src/pipeline/ingest.py", "Ingestion"),
        ("src/pipeline/query.py", "Query engine"),
        ("src/graph/neo4j_store.py", "Graph store"),
        ("src/vector/chroma_store.py", "Vector store"),
    ]
    
    all_ok = True
    for filepath, name in files:
        if not check_file_exists(filepath, name):
            all_ok = False
    
    return all_ok

def check_env():
    """Check optional .env configuration."""
    print("\n[Checking .env configuration...]")
    env_path = Path(".env")
    
    if not env_path.exists():
        print(f"  {INFO} .env file not found")
        print("    This is OK for the default demo path; built-in safe defaults will be used.")
        return True
    
    print(f"  {OK} .env file exists")
    
    # Basic validation
    with open(env_path) as f:
        content = f.read()
    
    if "USE_SAMPLE_DATA=true" in content or "your_" in content:
        print(f"  {INFO} Using sample data or dummy API keys - this works for demos")
    
    return True

def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("GraphRAG Project Validation")
    print("="*60)
    
    files_ok = check_files()
    deps_ok = check_requirements()
    env_ok = check_env()
    
    print("\n" + "="*60)
    
    if files_ok and deps_ok and env_ok:
        print("[OK] All checks passed! Ready to run: python demo.py")
        print("="*60)
        return 0
    else:
        print("[FAIL] Some checks failed. See above for details.")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
