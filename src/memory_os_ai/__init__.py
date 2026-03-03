"""Memory OS AI — Adaptive memory system for AI agents.

Universal MCP server for persistent semantic memory. Works with
Claude Code, Codex CLI, VS Code Copilot, ChatGPT, and any MCP-compatible client.
Document ingestion, FAISS search, chat extraction, cross-project linking.
"""

__version__ = "3.2.0"

__all__ = [
    "__version__",
    "MemoryEngine",
    "ChatExtractor",
    "StorageRouter",
    "TOOL_MODELS",
    "MEMORY_INSTRUCTIONS",
]

from memory_os_ai.engine import MemoryEngine
from memory_os_ai.chat_extractor import ChatExtractor
from memory_os_ai.storage_router import StorageRouter
from memory_os_ai.models import TOOL_MODELS
from memory_os_ai.instructions import MEMORY_INSTRUCTIONS
