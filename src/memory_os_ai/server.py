"""MCP Server for Memory OS AI.

Exposes document ingestion, semantic search, and transcription as MCP tools
consumable by VS Code Copilot. Replaces the local Mistral-7B LLM entirely —
all generation is delegated to Copilot via the MCP protocol.

Usage:
    python -m memory_os_ai.server          # stdio transport (VS Code default)
    python -m memory_os_ai.server --sse    # SSE transport (HTTP)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from .engine import MemoryEngine
from .models import (
    TOOL_MODELS,
    IngestInput,
    SearchInput,
    SearchOccurrencesInput,
    GetContextInput,
    ListDocumentsInput,
    TranscribeInput,
    DocumentStatusInput,
    ChatSyncInput,
    ChatSourceAddInput,
    ChatSourceRemoveInput,
    ChatStatusInput,
    ChatAutoDetectInput,
)
from .chat_extractor import ChatExtractor

# ---------------------------------------------------------------------------
# Global engine instance (singleton per process)
# ---------------------------------------------------------------------------
_engine = MemoryEngine(
    model_name=os.environ.get("MEMORY_MODEL", "all-MiniLM-L6-v2"),
    cache_dir=os.environ.get("MEMORY_CACHE_DIR"),
)

_chat_extractor = ChatExtractor(
    state_dir=os.environ.get("MEMORY_CACHE_DIR", "."),
)

# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------
server = Server("memory-os-ai")


# --- Tool definitions ---
TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_ingest",
        "title": "Ingest Documents",
        "description": (
            "Ingest documents from a folder into the semantic memory. "
            "Supports PDF, TXT, DOCX, DOC, PNG, JPEG, PPT, PPTX, MP3, WAV, OGG, FLAC. "
            "Documents are segmented, embedded via SentenceTransformers, and indexed in FAISS."
        ),
        "inputSchema": IngestInput.model_json_schema(),
    },
    {
        "name": "memory_search",
        "title": "Semantic Search",
        "description": (
            "Search indexed documents using natural language. "
            "Returns the most relevant text segments with their source file and distance score."
        ),
        "inputSchema": SearchInput.model_json_schema(),
    },
    {
        "name": "memory_search_occurrences",
        "title": "Keyword Occurrences",
        "description": (
            "Count exact keyword occurrences across indexed documents. "
            "Uses FAISS for candidate selection then regex for precise counting."
        ),
        "inputSchema": SearchOccurrencesInput.model_json_schema(),
    },
    {
        "name": "memory_get_context",
        "title": "Get Context for Copilot",
        "description": (
            "Retrieve relevant document context for a query. "
            "Designed to provide context to Copilot for summarization, report generation, "
            "or Q&A — replaces the local Mistral-7B LLM entirely."
        ),
        "inputSchema": GetContextInput.model_json_schema(),
    },
    {
        "name": "memory_list_documents",
        "title": "List Documents",
        "description": (
            "List all indexed documents with optional statistics "
            "(page count, segment count, word count)."
        ),
        "inputSchema": ListDocumentsInput.model_json_schema(),
    },
    {
        "name": "memory_transcribe",
        "title": "Transcribe Audio",
        "description": (
            "Transcribe an audio file (MP3, WAV, OGG, FLAC) to text using OpenAI Whisper. "
            "The transcript can then be ingested or analyzed by Copilot."
        ),
        "inputSchema": TranscribeInput.model_json_schema(),
    },
    {
        "name": "memory_status",
        "title": "Engine Status",
        "description": (
            "Return the current status of the Memory OS AI engine: "
            "device, model, index size, document count."
        ),
        "inputSchema": DocumentStatusInput.model_json_schema(),
    },
    {
        "name": "memory_chat_sync",
        "title": "Sync Chat History",
        "description": (
            "Incrementally extract new chat messages from all registered sources "
            "(VS Code Copilot, JSONL, Markdown, folder) and feed them into the "
            "semantic memory. Only new/changed content is processed."
        ),
        "inputSchema": ChatSyncInput.model_json_schema(),
    },
    {
        "name": "memory_chat_source_add",
        "title": "Add Chat Source",
        "description": (
            "Register a new chat source to extract from. Types: "
            "'vscode' (VS Code Copilot DB), 'jsonl' (JSONL log file), "
            "'markdown' (MD export), 'folder' (watch folder for new files)."
        ),
        "inputSchema": ChatSourceAddInput.model_json_schema(),
    },
    {
        "name": "memory_chat_source_remove",
        "title": "Remove Chat Source",
        "description": "Unregister a chat source by its ID.",
        "inputSchema": ChatSourceRemoveInput.model_json_schema(),
    },
    {
        "name": "memory_chat_status",
        "title": "Chat Sync Status",
        "description": (
            "Show all registered chat sources, their sync state, "
            "and the number of messages extracted so far."
        ),
        "inputSchema": ChatStatusInput.model_json_schema(),
    },
    {
        "name": "memory_chat_auto_detect",
        "title": "Auto-Detect VS Code Chats",
        "description": (
            "Automatically detect all VS Code workspace storage directories "
            "on this machine and optionally register them as chat sources. "
            "Discovers all Copilot chat histories without manual configuration."
        ),
        "inputSchema": ChatAutoDetectInput.model_json_schema(),
    },
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Register all Memory OS AI tools."""
    return [
        Tool(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in TOOLS
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool calls to engine methods."""
    try:
        # Validate input via Pydantic
        model_cls = TOOL_MODELS.get(name)
        if model_cls:
            validated = model_cls(**arguments)
            args = validated.model_dump()
        else:
            args = arguments

        result = _dispatch(name, args)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]

    except Exception as e:
        error_result = {"ok": False, "error": str(e), "tool": name}
        return [TextContent(type="text", text=json.dumps(error_result, ensure_ascii=False))]


def _dispatch(name: str, args: dict) -> Any:
    """Route tool name to engine method."""
    if name == "memory_ingest":
        folder = args["folder_path"]
        # Resolve relative to workspace
        if not os.path.isabs(folder):
            workspace = os.environ.get("MEMORY_WORKSPACE", os.getcwd())
            folder = os.path.join(workspace, folder)
        extensions = set(args["extensions"]) if args.get("extensions") else None
        return _engine.ingest(
            folder_path=folder,
            extensions=extensions,
            force_reindex=args.get("force_reindex", False),
        )

    elif name == "memory_search":
        results = _engine.search(
            query=args["query"],
            top_k=args.get("top_k", 10),
            threshold=args.get("threshold", 0.8),
        )
        return {
            "ok": True,
            "count": len(results),
            "results": [
                {
                    "filename": r.filename,
                    "text": r.segment_text,
                    "distance": r.distance,
                }
                for r in results
            ],
        }

    elif name == "memory_search_occurrences":
        occ = _engine.search_occurrences(
            keyword=args["keyword"],
            top_k=args.get("top_k", 500),
        )
        return {
            "ok": True,
            "total_occurrences": occ.total,
            "by_file": occ.by_file,
        }

    elif name == "memory_get_context":
        max_chars = args.get("max_tokens", 3000) * 4  # ~4 chars per token
        context = _engine.get_context(
            query=args["query"],
            max_chars=max_chars,
            top_k=args.get("top_k", 50),
        )
        return {
            "ok": True,
            "context": context,
            "char_count": len(context),
        }

    elif name == "memory_list_documents":
        docs = _engine.list_documents(include_stats=args.get("include_stats", True))
        return {
            "ok": True,
            "count": len(docs),
            "documents": docs,
        }

    elif name == "memory_transcribe":
        file_path = args["file_path"]
        if not os.path.isabs(file_path):
            workspace = os.environ.get("MEMORY_WORKSPACE", os.getcwd())
            file_path = os.path.join(workspace, file_path)
        return _engine.transcribe(
            file_path=file_path,
            language=args.get("language", "fr"),
        )

    elif name == "memory_status":
        return _engine.status()

    elif name == "memory_chat_sync":
        source_id = args.get("source_id")
        result = _chat_extractor.sync(source_id=source_id)
        messages = result.pop("messages", [])
        if messages:
            segments = _chat_extractor.messages_to_segments(messages)
            ingest_result = _engine.ingest_segments(segments, source_label="chat_sync")
            result["ingest"] = ingest_result
        return result

    elif name == "memory_chat_source_add":
        _chat_extractor.add_source(
            source_id=args["source_id"],
            source_type=args["source_type"],
            path=args["path"],
        )
        return {
            "ok": True,
            "message": f"Source '{args['source_id']}' registered ({args['source_type']})",
            "sources": _chat_extractor.list_sources(),
        }

    elif name == "memory_chat_source_remove":
        removed = _chat_extractor.remove_source(args["source_id"])
        return {
            "ok": removed,
            "message": (
                f"Source '{args['source_id']}' removed"
                if removed
                else f"Source '{args['source_id']}' not found"
            ),
            "sources": _chat_extractor.list_sources(),
        }

    elif name == "memory_chat_status":
        return {"ok": True, **_chat_extractor.status()}

    elif name == "memory_chat_auto_detect":
        found = _chat_extractor.auto_detect_vscode()
        registered = []
        if args.get("auto_register", True):
            for i, ws_path in enumerate(found):
                sid = f"vscode-auto-{i}"
                _chat_extractor.add_source(
                    source_id=sid,
                    source_type="vscode",
                    path=ws_path,
                )
                registered.append(sid)
        return {
            "ok": True,
            "workspaces_found": len(found),
            "paths": found,
            "registered": registered,
            "sources": _chat_extractor.list_sources(),
        }

    else:
        return {"ok": False, "error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main():
    """Run the MCP server (stdio transport)."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run():
    """Synchronous entry point."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
