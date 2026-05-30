"""
CLI interface for tech-doc-rag.

Uses TUI for clean chat interface with SSE streaming.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for TUI import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tui import chat_loop


def main():
    """
    Run the interactive chat loop with TUI.
    
    Usage:
        python -m src.main
        make run
    """
    api_url = os.getenv("API_URL", "http://localhost:8000")
    endpoint = f"{api_url}/api/tech-doc/chat/stream"
    asyncio.run(chat_loop(endpoint, "Tech Doc RAG"))


if __name__ == "__main__":
    main()
