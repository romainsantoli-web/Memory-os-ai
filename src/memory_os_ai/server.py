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
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
    Resource,
    TextResourceContents,
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
    CompactInput,
    ProjectLinkInput,
    ProjectUnlinkInput,
    ProjectListInput,
    CloudConfigureInput,
    CloudStatusInput,
    CloudSyncInput,
)
from .chat_extractor import ChatExtractor
from .storage_router import StorageRouter
from .instructions import MEMORY_INSTRUCTIONS

logger = logging.getLogger("memory_os_ai")

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

# Storage router (local + cloud overflow)
_storage_router = StorageRouter(
    local_dir=os.environ.get("MEMORY_CACHE_DIR", ""),
)

# Cross-project linked memories: alias -> {path, engine}
_linked_projects: dict[str, dict] = {}
_LINKS_FILE = os.path.join(
    os.environ.get("MEMORY_CACHE_DIR", "."), "_project_links.json"
)


def _load_project_links() -> None:
    """Load persisted project links from disk."""
    if os.path.exists(_LINKS_FILE):
        try:
            with open(_LINKS_FILE, "r") as f:
                links = json.load(f)
            for alias, path in links.items():
                if ".." in path:
                    logger.warning("Skipping linked project %s — path traversal in %s", alias, path)
                    continue
                if os.path.isdir(path):
                    _linked_projects[alias] = {"path": path, "engine": None}
                else:
                    logger.info("Linked project %s path missing: %s", alias, path)
        except Exception as e:
            logger.warning("Failed to load project links: %s", e)


def _save_project_links() -> None:
    """Persist project links to disk."""
    try:
        data = {alias: info["path"] for alias, info in _linked_projects.items()}
        os.makedirs(os.path.dirname(_LINKS_FILE) or ".", exist_ok=True)
        with open(_LINKS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def _get_linked_engine(alias: str):
    """Lazy-init a MemoryEngine for a linked project."""
    info = _linked_projects.get(alias)
    if not info:
        return None
    if info["engine"] is None:
        eng = MemoryEngine(
            model_name=os.environ.get("MEMORY_MODEL", "all-MiniLM-L6-v2"),
            cache_dir=info["path"],
        )
        # Try to load persisted index (safe numpy load — no pickle)
        cache_path = os.path.join(info["path"], "embeddings_cache.npy")
        if os.path.exists(cache_path):
            try:
                embeddings = np.load(cache_path, allow_pickle=False)
                if isinstance(embeddings, np.ndarray) and len(embeddings) > 0:
                    eng._build_index(embeddings)
                    # Rebuild segments list from JSONL log if available
                    log_path = os.path.join(info["path"], "_conversation_log.jsonl")
                    if os.path.exists(log_path):
                        segments = []
                        with open(log_path, "r") as lf:
                            for line in lf:
                                try:
                                    rec = json.loads(line)
                                    if rec.get("summary"):
                                        segments.append(f"[summary] {rec['summary']}")
                                    for msg in rec.get("messages", []):
                                        segments.append(f"[{msg.get('role', '?')}] {msg.get('content', '')}")
                                except Exception:
                                    pass
                        eng._segments = segments
                        eng._initialized = True
            except Exception:
                pass
        info["engine"] = eng
    return info["engine"]


# Load any previously persisted links
_load_project_links()

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
        "title": "Get Context",
        "description": (
            "Retrieve relevant document context for a query. "
            "Designed to provide context for summarization, report generation, "
            "or Q&A — the AI model uses this to ground responses in real documents."
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
    {
        "name": "memory_compact",
        "title": "Compact Memory",
        "description": (
            "Compress the in-memory FAISS index by removing near-duplicate segments "
            "and merging short fragments. Use when the memory is getting large and "
            "context retrieval is slow or noisy. Strategies: 'dedup_merge' (default) "
            "removes duplicates and merges short segments; 'top_k' keeps only the "
            "most central segments."
        ),
        "inputSchema": CompactInput.model_json_schema(),
    },
    {
        "name": "memory_project_link",
        "title": "Link External Project Memory",
        "description": (
            "Link another project's memory cache so that searches and session briefs "
            "include context from both projects. Enables cross-project knowledge sharing. "
            "Provide the absolute path to the other project's .memory-os-ai directory."
        ),
        "inputSchema": ProjectLinkInput.model_json_schema(),
    },
    {
        "name": "memory_project_unlink",
        "title": "Unlink External Project",
        "description": "Remove a previously linked project by its alias.",
        "inputSchema": ProjectUnlinkInput.model_json_schema(),
    },
    {
        "name": "memory_project_list",
        "title": "List Linked Projects",
        "description": "List all currently linked external project memories with their status.",
        "inputSchema": ProjectListInput.model_json_schema(),
    },
    {
        "name": "memory_cloud_configure",
        "title": "Configure Cloud Storage",
        "description": (
            "Configure a cloud storage backend for memory overflow when local disk is low. "
            "Supported providers: Google Drive, iCloud, Dropbox, OneDrive, Amazon S3, "
            "Azure Blob, Box, Backblaze B2. Once configured, memory data automatically "
            "syncs to cloud when disk space drops below threshold (default 500 MB)."
        ),
        "inputSchema": CloudConfigureInput.model_json_schema(),
    },
    {
        "name": "memory_cloud_status",
        "title": "Cloud Storage Status",
        "description": (
            "Show local disk usage, cloud storage status, quota, and available providers. "
            "Indicates whether disk is low and whether cloud overflow is active."
        ),
        "inputSchema": CloudStatusInput.model_json_schema(),
    },
    {
        "name": "memory_cloud_sync",
        "title": "Sync Cloud Storage",
        "description": (
            "Synchronize memory data between local disk and cloud. "
            "Directions: 'push' (upload local→cloud), 'pull' (download cloud→local), "
            "'auto' (check disk, offload if needed). Use 'push' for backup, "
            "'pull' to restore on a new machine, 'auto' for hands-free overflow."
        ),
        "inputSchema": CloudSyncInput.model_json_schema(),
    },
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Register all Memory OS AI tools."""
    return [
        Tool(
            name=t["name"],
            title=t.get("title"),
            description=t["description"],
            inputSchema=t["inputSchema"],
        )
        for t in TOOLS
    ]


# ---------------------------------------------------------------------------
# MCP Resources — expose indexed documents as browsable URIs
# ---------------------------------------------------------------------------
@server.list_resources()
async def list_resources() -> list[Resource]:
    """Expose all indexed documents + conversation log as memory:// resources."""
    resources = []

    # 1. Indexed documents
    for name, doc in _engine._documents.items():
        resources.append(Resource(
            uri=f"memory://documents/{name}",
            name=name,
            description=f"{doc.nb_segments} segments, {doc.nb_words} words",
            mimeType="text/plain",
        ))

    # 2. Conversation log (if exists)
    log_dir = os.environ.get("MEMORY_CACHE_DIR", ".")
    log_path = os.path.join(log_dir, "_conversation_log.jsonl")
    if os.path.exists(log_path):
        size = os.path.getsize(log_path)
        resources.append(Resource(
            uri="memory://logs/conversation",
            name="Conversation Log",
            description=f"Persistent JSONL conversation log ({size} bytes)",
            mimeType="application/jsonl",
        ))

    # 3. Linked projects
    for alias in _linked_projects:
        resources.append(Resource(
            uri=f"memory://linked/{alias}",
            name=f"Linked: {alias}",
            description=f"Cross-project memory link → {_linked_projects[alias]['path']}",
            mimeType="text/plain",
        ))

    return resources


@server.read_resource()
async def read_resource(uri: str) -> list[TextResourceContents]:
    """Read the content of a memory resource."""
    from urllib.parse import unquote

    uri_str = str(uri)

    if uri_str.startswith("memory://documents/"):
        doc_name = unquote(uri_str.replace("memory://documents/", "", 1))
        text = _engine.get_segment_text(doc_name)
        if text is None:
            text = f"Document '{doc_name}' not found in index."
        return [TextResourceContents(uri=uri_str, text=text, mimeType="text/plain")]

    elif uri_str == "memory://logs/conversation":
        log_dir = os.environ.get("MEMORY_CACHE_DIR", ".")
        log_path = os.path.join(log_dir, "_conversation_log.jsonl")
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = "No conversation log found."
        return [TextResourceContents(uri=uri_str, text=text, mimeType="application/jsonl")]

    elif uri_str.startswith("memory://linked/"):
        alias = unquote(uri_str.replace("memory://linked/", "", 1))
        info = _linked_projects.get(alias)
        if not info:
            text = f"Linked project '{alias}' not found."
        else:
            eng = _get_linked_engine(alias)
            if eng and eng.is_initialized:
                status = eng.status()
                text = json.dumps(status, indent=2)
            else:
                text = f"Linked project '{alias}' at {info['path']} — index not loaded."
        return [TextResourceContents(uri=uri_str, text=text, mimeType="text/plain")]

    return [TextResourceContents(uri=uri_str, text=f"Unknown resource: {uri_str}", mimeType="text/plain")]


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
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
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
        # Also search linked projects
        linked_results = []
        for alias, info in _linked_projects.items():
            eng = _get_linked_engine(alias)
            if eng and eng.is_initialized:
                lr = eng.search(
                    query=args["query"],
                    top_k=min(args.get("top_k", 10), 5),
                    threshold=args.get("threshold", 0.8),
                )
                for r in lr:
                    linked_results.append({
                        "filename": f"[{alias}] {r.filename}",
                        "text": r.segment_text,
                        "distance": r.distance,
                    })

        all_results = [
            {"filename": r.filename, "text": r.segment_text, "distance": r.distance}
            for r in results
        ] + linked_results
        # Sort by distance
        all_results.sort(key=lambda x: x["distance"])

        return {
            "ok": True,
            "count": len(all_results),
            "results": all_results[:args.get("top_k", 10)],
            "linked_projects_searched": len(_linked_projects),
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

        # 4. Include linked project summaries
        if _linked_projects:
            linked_briefs = {}
            for alias, info in _linked_projects.items():
                eng = _get_linked_engine(alias)
                if eng and eng.is_initialized:
                    lb = eng.session_brief(
                        max_chars=min(max_chars // 4, 4000),
                        focus_query=focus,
                    )
                    linked_briefs[alias] = {
                        "segments": lb.get("unique_segments_retrieved", 0),
                        "context_chars": lb.get("context_chars", 0),
                        "context": lb.get("context", "")[:2000],
                    }
            if linked_briefs:
                brief["linked_projects"] = linked_briefs

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

    elif name == "memory_compact":
        result = _engine.compact(
            max_segments=args.get("max_segments", 500),
            keep_recent_hours=args.get("keep_recent_hours", 24),
            strategy=args.get("strategy", "dedup_merge"),
        )
        return result

    elif name == "memory_project_link":
        project_path = args["project_path"]
        if not os.path.isdir(project_path):
            return {"ok": False, "error": f"Directory not found: {project_path}"}
        alias = args.get("alias") or os.path.basename(project_path.rstrip("/"))
        if alias in _linked_projects:
            return {"ok": False, "error": f"Alias '{alias}' already linked"}
        _linked_projects[alias] = {"path": project_path, "engine": None}
        _save_project_links()
        return {
            "ok": True,
            "alias": alias,
            "path": project_path,
            "linked_count": len(_linked_projects),
        }

    elif name == "memory_project_unlink":
        alias = args["alias"]
        if alias not in _linked_projects:
            return {"ok": False, "error": f"Alias '{alias}' not found"}
        del _linked_projects[alias]
        _save_project_links()
        return {
            "ok": True,
            "removed": alias,
            "linked_count": len(_linked_projects),
        }

    elif name == "memory_project_list":
        projects = []
        for alias, info in _linked_projects.items():
            eng = _get_linked_engine(alias)
            projects.append({
                "alias": alias,
                "path": info["path"],
                "initialized": eng.is_initialized if eng else False,
                "segments": eng.segment_count if eng and eng.is_initialized else 0,
                "documents": eng.document_count if eng and eng.is_initialized else 0,
            })
        return {
            "ok": True,
            "linked_count": len(projects),
            "projects": projects,
        }

    elif name == "memory_cloud_configure":
        result = _storage_router.configure_cloud(
            provider=args["provider"],
            credentials=args.get("credentials", {}),
        )
        return result

    elif name == "memory_cloud_status":
        return _storage_router.status()

    elif name == "memory_cloud_sync":
        direction = args.get("direction", "push")
        if direction == "push":
            sync_result = _storage_router.sync_to_cloud()
        elif direction == "pull":
            sync_result = _storage_router.sync_from_cloud()
        else:  # auto
            return _storage_router.check_and_offload()
        return {
            "ok": len(sync_result.errors) == 0,
            "direction": direction,
            "uploaded": sync_result.uploaded,
            "downloaded": sync_result.downloaded,
            "errors": sync_result.errors,
            "elapsed_seconds": sync_result.elapsed_seconds,
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
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
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
        # Return empty response to avoid NoneType error when client disconnects
        return Response()

    async def _auth_messages_app(scope, receive, send):
        """ASGI middleware: check API key then delegate to SSE post handler."""
        request = Request(scope, receive)
        if not _check_api_key(request):
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
            await response(scope, receive, send)
            return
        await sse.handle_post_message(scope, receive, send)

    async def health(request):
        return JSONResponse({"ok": True, "server": "memory-os-ai", "transport": "sse"})

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/sse", handle_sse),
            Mount("/messages/", app=_auth_messages_app),
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
