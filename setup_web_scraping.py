#!/usr/bin/env python3
"""
Setup helper script to configure GraphRAG for web scraping without Reddit API.

This script helps you:
1. Choose a web scraping backend
2. Install required packages
3. Generate .env file
4. Verify setup
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_option(num, title, description):
    """Print an option."""
    print(f"\n  {num}. {title}")
    print(f"     {description}")

def setup_web_scraping():
    """Main setup function."""
    print_header("GraphRAG Web Scraping Setup")
    
    print("\nChoose your web scraping backend:")
    print_option(1, "DuckDuckGo (ddgs) - FREE ⭐", 
                "No API key needed. Works out of the box.")
    print_option(2, "Brave Search - $0-9/month",
                "Privacy-focused. Free tier: 100 requests/day.")
    print_option(3, "Crawl4AI - FREE",
                "Advanced scraping. Good for complex content.")
    print_option(4, "Firecrawl - $99+/month",
                "Professional API. Best for reliability.")
    
    choice = input("\nSelect option (1-4, or press Enter for 1): ").strip()
    if not choice:
        choice = "1"
    
    backends = {
        "1": ("ddgs", "DuckDuckGo", "", "pip install ddgs"),
        "2": ("brave", "Brave Search", "BRAVE_SEARCH_API_KEY", 
              "pip install httpx"),
        "3": ("crawl4ai", "Crawl4AI", "", 
              "pip install crawl4ai"),
        "4": ("firecrawl", "Firecrawl", "FIRECRAWL_API_KEY",
              "pip install firecrawl-py"),
    }
    
    if choice not in backends:
        print("Invalid choice. Using DuckDuckGo.")
        choice = "1"
    
    backend_id, backend_name, api_key_var, install_cmd = backends[choice]
    
    print_header(f"Setting up {backend_name}")
    
    # Show installation info
    print(f"\nInstall packages:")
    print(f"  {install_cmd}")
    
    # Prepare .env configuration
    env_config = f"""# Web Scraping Configuration
SCRAPE_METHOD=web
WEB_SCRAPER_BACKEND={backend_id}
SCRAPE_DELAY=1.5
"""
    
    # Add API key instructions if needed
    if api_key_var:
        env_config += f"\n# Get your API key here:\n# {api_key_var}=<your_key_here>\n"
        
        if backend_id == "brave":
            env_config += "# https://api.search.brave.com\n"
        elif backend_id == "firecrawl":
            env_config += "# https://www.firecrawl.dev\n"
    
    print("\n.env Configuration:")
    print("─" * 60)
    print(env_config)
    print("─" * 60)
    
    # Offer to create .env
    if not Path(".env").exists():
        create_env = input("\nCreate .env file? (y/n): ").strip().lower()
        if create_env == 'y':
            with open(".env", "w") as f:
                f.write(env_config)
            print("✓ Created .env file")
    else:
        print("\n⚠️  .env file already exists. Please manually add the configuration above.")
    
    print_header("Quick Start")
    print("""
1. Install packages:
   pip install -r requirements.txt
   
2. Add API key (if needed):
   Edit .env and add your API key
   
3. Run ingestion:
   python -m src.pipeline.ingest
   
4. Or try the demo first:
   python demo.py

For more details, see: WEB_SCRAPING_ALTERNATIVES.md
""")

if __name__ == "__main__":
    try:
        setup_web_scraping()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
