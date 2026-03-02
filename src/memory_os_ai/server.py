"""MCP Server for Memory OS AI.

Universal semantic memory for AI models — works with ANY MCP-compatible client:
VS Code Copilot, Claude Desktop, Claude Code, Codex CLI, ChatGPT, Cursor, etc.

Usage:
    python -m memory_os_ai.server              # stdio transport (default)
    python -m memory_os_ai.server --sse        # SSE transport (HTTP)
    python -m memory_os_ai.server --http       # Streamable HTTP transport
    MEMORY_API_KEY=secret python -m memory_os_ai.server --http  # with auth
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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
    SessionBriefInput,
    ChatSaveInput,
)
from .chat_extractor import ChatExtractor
from .instructions import MEMORY_INSTRUCTIONS

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
# MCP Server setup — instructions teach ANY model how to use memory tools
# ---------------------------------------------------------------------------
server = Server(
    "memory-os-ai",
    instructions=MEMORY_INSTRUCTIONS,
)


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
    {
        "name": "memory_session_brief",
        "title": "Session Brief",
        "description": (
            "CALL THIS TOOL AT THE START OF EVERY CONVERSATION. "
            "Also call it whenever you feel you have lost context or the user "
            "mentions something you should remember but don't. "
            "Generates a comprehensive session briefing from the semantic memory: "
            "project overview, recent activity, pending tasks, key context, "
            "and chat history. Optionally syncs chat sources first and accepts "
            "a focus query to prioritise specific topics."
        ),
        "inputSchema": SessionBriefInput.model_json_schema(),
    },
    {
        "name": "memory_chat_save",
        "title": "Save Conversation to Memory",
        "description": (
            "CALL THIS TOOL REGULARLY during the conversation to persist the "
            "current exchange into semantic memory. Call it: "
            "(1) after completing a significant task, "
            "(2) every ~10 exchanges, "
            "(3) when the user asks you to remember something, "
            "(4) before ending a session. "
            "Messages are indexed in FAISS AND saved to a persistent JSONL log "
            "so they survive process restarts. Include a summary for fast recall."
        ),
        "inputSchema": ChatSaveInput.model_json_schema(),
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

    elif name == "memory_session_brief":
        # 1. Optionally sync chat sources first
        chat_sync_result = None
        if args.get("include_chat_sync", True):
            sync_out = _chat_extractor.sync()
            messages = sync_out.pop("messages", [])
            if messages:
                segments = _chat_extractor.messages_to_segments(messages)
                ingest_out = _engine.ingest_segments(segments, source_label="session_brief")
                sync_out["ingest"] = ingest_out
            chat_sync_result = sync_out

        # 2. Build the brief from the full memory index
        max_chars = args.get("max_tokens", 4000) * 4
        focus = args.get("focus_query")
        brief = _engine.session_brief(max_chars=max_chars, focus_query=focus)

        # 3. Attach chat sync summary
        if chat_sync_result is not None:
            brief["chat_sync"] = chat_sync_result

        return brief

    elif name == "memory_chat_save":
        messages = args["messages"]
        summary = args.get("summary")
        project_label = args.get("project_label", "conversation")
        ts = datetime.now(timezone.utc).isoformat()

        # 1. Build segments for FAISS indexing
        segments: list[str] = []
        for msg in messages:
            prefix = f"[{msg['role']}]"
            segments.append(f"{prefix} {msg['content']}")

        # Add summary as a high-priority segment
        if summary:
            segments.insert(0, f"[summary:{project_label}] {summary}")

        # 2. Ingest into FAISS
        ingest_result = _engine.ingest_segments(
            segments, source_label=f"save_{project_label}"
        )

        # 3. Persist to JSONL log (survives process restart)
        log_dir = os.environ.get("MEMORY_CACHE_DIR", ".")
        log_path = os.path.join(log_dir, "_conversation_log.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                record = {
                    "timestamp": ts,
                    "project": project_label,
                    "summary": summary,
                    "message_count": len(messages),
                    "messages": messages,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # best-effort persistence

        return {
            "ok": True,
            "saved_messages": len(messages),
            "has_summary": summary is not None,
            "project": project_label,
            "timestamp": ts,
            "ingest": ingest_result,
            "log_path": log_path,
        }

    else:
        return {"ok": False, "error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Main entry points — stdio / SSE / Streamable HTTP
# ---------------------------------------------------------------------------
async def main_stdio():
    """Run the MCP server over stdio (default for VS Code, Claude Desktop, Codex)."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _check_api_key(request) -> bool:
    """Validate API key from request headers if MEMORY_API_KEY is set."""
    expected = os.environ.get("MEMORY_API_KEY")
    if not expected:
        return True  # no auth configured
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:] == expected
    return request.headers.get("x-api-key", "") == expected


def run_sse(host: str = "0.0.0.0", port: int = 8765):
    """Run MCP server over SSE transport (for ChatGPT, remote clients)."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route
    from mcp.server.sse import SseServerTransport

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        if not _check_api_key(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )

    async def handle_messages(request):
        if not _check_api_key(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        await sse.handle_post_message(request.scope, request.receive, request._send)

    async def health(request):
        return JSONResponse({"ok": True, "server": "memory-os-ai", "transport": "sse"})

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/sse", handle_sse),
            Route("/messages/", handle_messages, methods=["POST"]),
        ],
    )
    print(f"Memory OS AI — SSE transport on http://{host}:{port}")
    print(f"  Health: http://{host}:{port}/health")
    print(f"  SSE:    http://{host}:{port}/sse")
    if os.environ.get("MEMORY_API_KEY"):
        print("  Auth:   API key required (Bearer token or x-api-key header)")
    uvicorn.run(app, host=host, port=port)


def run_http(host: str = "0.0.0.0", port: int = 8765):
    """Run MCP server over Streamable HTTP transport."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    transport = StreamableHTTPServerTransport("/mcp/")

    async def handle_mcp(request):
        if not _check_api_key(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        await transport.handle_request(request.scope, request.receive, request._send)

    async def health(request):
        return JSONResponse({"ok": True, "server": "memory-os-ai", "transport": "http"})

    app = Starlette(
        routes=[
            Route("/health", health),
            Mount("/mcp", app=transport.app),
        ],
    )
    print(f"Memory OS AI — Streamable HTTP on http://{host}:{port}")
    print(f"  Health:   http://{host}:{port}/health")
    print(f"  MCP:      http://{host}:{port}/mcp/")
    if os.environ.get("MEMORY_API_KEY"):
        print("  Auth:     API key required (Bearer token or x-api-key header)")
    uvicorn.run(app, host=host, port=port)


def run():
    """Entry point — detect transport from CLI args."""
    import asyncio

    host = os.environ.get("MEMORY_HOST", "0.0.0.0")
    port = int(os.environ.get("MEMORY_PORT", "8765"))

    if "--sse" in sys.argv:
        run_sse(host=host, port=port)
    elif "--http" in sys.argv:
        run_http(host=host, port=port)
    else:
        asyncio.run(main_stdio())


# Keep backward compat
main = main_stdio


if __name__ == "__main__":
    run()
