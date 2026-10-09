"""MCP stdio server for Memory OS v4.

Wires the pydantic input models in :mod:`memory_os_v4.models` and the
:class:`~memory_os_v4.memory.MemoryV4` façade together behind an
``mcp.server.Server``. Tool dispatch is a flat ``TOOL_HANDLERS`` dict (no
if/elif chain); every handler validates its input, calls the façade, and
returns a JSON-serializable dict. Exceptions never reach the client as a
traceback: they are logged server-side and returned as
``{"ok": False, "error": "<message>"}``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import anyio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, ValidationError

from .config import Config
from .instructions import build_instructions
from .memory import MemoryV4
from .migrate import resolve_under_home as _resolve_under_home
from .store import MemoryItem
from .text import tokenize
from . import models

logger = logging.getLogger("memory_os_v4.server")

DEFAULT_INGEST_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".json", ".py", ".csv", ".log"}
DEFAULT_V3_LOG_PATH = "~/.memory-os-ai/_conversation_log.jsonl"
_TIME_FIELDS = ("created_at", "event_time", "valid_from", "valid_until", "last_accessed")

# --------------------------------------------------------------------- engine
# The façade is created lazily on first tool call (not at import time) so the
# process starts instantly and so tests can swap it out via `set_engine()`.
_engine: MemoryV4 | None = None
# Tool calls run in worker threads (see call_tool), so two first calls could
# race to build two engines over the same data dir: guard the lazy creation.
_engine_init_lock = Lock()


def get_engine() -> MemoryV4:
    global _engine
    if _engine is None:
        with _engine_init_lock:
            if _engine is None:
                _engine = MemoryV4(config=Config.from_env())
    return _engine


def set_engine(engine: MemoryV4 | None) -> None:
    """Test/selfcheck hook: inject (or reset, with None) the module-level engine."""
    global _engine
    with _engine_init_lock:
        old, _engine = _engine, engine
        if old is not None and old is not engine:
            old.close()


def default_project() -> str:
    return os.environ.get("MEMORY_V4_PROJECT", "default")


ALL_PROJECTS = "*"


def _project_of(value: str | None) -> str | None:
    """Read scope. ``"*"`` (explicit, or as MEMORY_V4_PROJECT) means every
    project -> ``None``, which the engine already treats as "no filter".
    A shared server serves many projects, so it runs with MEMORY_V4_PROJECT=*."""
    scope = value if value is not None else default_project()
    return None if scope == ALL_PROJECTS else scope


def _write_project_of(value: str | None) -> str:
    """Write target: always a concrete label, never the ``"*"`` wildcard."""
    if value is not None and value != ALL_PROJECTS:
        return value
    fallback = default_project()
    return "default" if fallback == ALL_PROJECTS else fallback


# ------------------------------------------------------------------- helpers
def parse_iso(value: str | float | int | None) -> float | None:
    """Parse an ISO date/datetime (or unix timestamp) into a unix float.

    Accepts date-only strings ("2026-09-12") and full ISO datetimes with or
    without a timezone offset ("Z" is treated as "+00:00"); naive datetimes
    are interpreted as local time, matching ``datetime.timestamp()``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value!r}") from exc
    return dt.timestamp()


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=None).astimezone().isoformat()


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def _serialize_item(item: MemoryItem) -> dict:
    d = item.to_dict()
    for key in _TIME_FIELDS:
        if d.get(key) is not None:
            d[key] = _iso(d[key])
    return _round_floats(d)


def _convert_item_dict(d: dict) -> dict:
    out = dict(d)
    for key in _TIME_FIELDS:
        if out.get(key) is not None:
            out[key] = _iso(out[key])
    return _round_floats(out)


def _hit_dict(hit) -> dict:
    it = hit.item
    return _round_floats(
        {
            "id": it.id,
            "kind": it.kind,
            "text": it.text,
            "score": hit.score,
            "event_time": _iso(it.event_time),
            "valid_from": _iso(it.valid_from),
            "valid_until": _iso(it.valid_until),
            "entities": it.entities,
            "identity": it.identity,
            "replaces": it.replaces,
            "project": it.project,
            "signals": hit.signals,
        }
    )


# -------------------------------------------------------------------- handlers
def _h_status(engine: MemoryV4, args: models.StatusIn) -> dict:
    s = dict(engine.status(project=_project_of(args.project)))
    s["last_consolidated"] = _iso(s.get("last_consolidated"))
    return _round_floats(s)


def _h_session_brief(engine: MemoryV4, args: models.SessionBriefIn) -> dict:
    project = _project_of(args.project)
    brief = engine.session_brief(focus_query=args.focus_query, max_tokens=args.max_tokens, project=project)
    out = dict(brief)
    for key in ("recent_episodes", "key_facts", "procedures", "focus", "pending"):
        out[key] = [_convert_item_dict(d) for d in out.get(key, [])]
    stats = dict(out.get("stats", {}))
    stats["last_consolidated"] = _iso(stats.get("last_consolidated"))
    out["stats"] = _round_floats(stats)
    return _round_floats(out)


def _h_chat_save(engine: MemoryV4, args: models.ChatSaveIn) -> dict:
    project = _write_project_of(args.project_label)
    ev = parse_iso(args.event_time)

    episode_specs = [
        {
            "text": msg.content,
            "role": msg.role,
            "project": project,
            "event_time": ev,
            "keywords": msg.keywords,
            "facets": [f.model_dump(exclude_none=True) for f in msg.facets] if msg.facets else None,
        }
        for msg in args.messages
    ]
    episodes = engine.add_episodes_bulk(episode_specs) if episode_specs else []
    episode_ids = [it.id for it in episodes]

    summary_id: str | None = None
    if args.summary:
        summary_result = engine.add_facts(
            [{"text": args.summary, "kind": "summary", "importance": 0.8}],
            project=project,
            parent_ids=episode_ids,
            event_time=ev,
        )
        if summary_result["added"]:
            summary_id = summary_result["added"][0]
        elif summary_result["deduped"]:
            summary_id = summary_result["deduped"][0]

    facts_result = {"added": [], "deduped": [], "superseded": []}
    if args.facts:
        fact_dicts = [f.model_dump(exclude_none=True) for f in args.facts]
        facts_result = engine.add_facts(fact_dicts, project=project, parent_ids=episode_ids, event_time=ev)

    engine.flush()
    return {
        "episodes": len(episodes),
        "summary_id": summary_id,
        "facts": {
            "added": facts_result["added"],
            "deduped": facts_result["deduped"],
            "superseded": [list(p) for p in facts_result["superseded"]],
        },
    }


def _h_add_facts(engine: MemoryV4, args: models.AddFactsIn) -> dict:
    project = _write_project_of(args.project)
    fact_dicts = [f.model_dump(exclude_none=True) for f in args.facts]
    result = engine.add_facts(fact_dicts, project=project, parent_ids=args.parent_ids)
    engine.flush()
    return {
        "added": result["added"],
        "deduped": result["deduped"],
        "superseded": [list(p) for p in result["superseded"]],
    }


def _h_search(engine: MemoryV4, args: models.SearchIn) -> dict:
    project = _project_of(args.project)
    at_time = parse_iso(args.at_time)
    facets = [f.model_dump(exclude_none=True) for f in args.facets] if args.facets else None
    hits = engine.search(
        args.query,
        top_k=args.top_k,
        at_time=at_time,
        kinds=args.kinds,
        project=project,
        rerank=args.rerank,
        queries=args.queries,
        hypotheses=args.hypotheses,
        facets=facets,
    )
    return {"hits": [_hit_dict(h) for h in hits]}


def _h_get_context(engine: MemoryV4, args: models.GetContextIn) -> dict:
    project = _project_of(args.project)
    at_time = parse_iso(args.at_time)
    facets = [f.model_dump(exclude_none=True) for f in args.facets] if args.facets else None
    context = engine.get_context(
        args.query,
        max_tokens=args.max_tokens,
        at_time=at_time,
        project=project,
        queries=args.queries,
        hypotheses=args.hypotheses,
        facets=facets,
    )
    n_items = len([line for line in context.split("\n") if line.strip()])
    return {"context": context, "n_items": n_items}


def _h_timeline(engine: MemoryV4, args: models.TimelineIn) -> dict:
    project = _project_of(args.project)
    hits = engine.search(
        args.query, top_k=args.limit, kinds=["fact", "summary", "belief", "procedure"], project=project
    )
    items = sorted((h.item for h in hits), key=lambda it: it.event_time)
    out = []
    for it in items:
        out.append(
            {
                "id": it.id,
                "kind": it.kind,
                "text": it.text,
                "event_time": _iso(it.event_time),
                "valid_from": _iso(it.valid_from),
                "valid_until": _iso(it.valid_until),
                "invalidated_by": it.invalidated_by,
                "entities": it.entities,
                "identity": it.identity,
                "replaces": it.replaces,
                "project": it.project,
            }
        )
    return {"items": out}


def _h_block_get(engine: MemoryV4, args: models.BlockGetIn) -> dict:
    blocks = engine.blocks.all(project=_project_of(args.project), scope=args.scope)
    if args.name:
        blocks = {key: text for key, text in blocks.items() if key == args.name or key.endswith("/" + args.name)}
        if not blocks:
            blocks = {args.name: ""}
    return {"blocks": blocks}


def _h_block_set(engine: MemoryV4, args: models.BlockSetIn) -> dict:
    project = _write_project_of(args.project)
    kwargs = {"project": project, "scope": args.scope}
    limit = engine.blocks.char_limit
    if args.mode == "append":
        history = engine.blocks.history(args.name, **kwargs)
        current = history[-1].text if history else ""
        expected_len = len(f"{current}\n{args.text}".strip()) if current else len(args.text)
        engine.blocks.append(args.name, args.text, **kwargs)
    else:
        expected_len = len(args.text)
        engine.blocks.set(args.name, args.text, **kwargs)
    engine.flush()
    stored = engine.blocks.get(args.name, **kwargs)
    return {"name": args.name, "chars": len(stored), "truncated": expected_len > limit}


def _h_consolidate(engine: MemoryV4, args: models.ConsolidateIn) -> dict:
    result = engine.consolidate(dry_run=args.dry_run, project=_project_of(args.project))
    if not args.dry_run:
        engine.flush()
    return _round_floats(
        {
            "deduped": result["deduped"],
            "deduped_pairs": [list(p) for p in result["deduped_pairs"]],
            "superseded": result["superseded"],
            "superseded_pairs": [list(p) for p in result["superseded_pairs"]],
            "decayed": result["decayed"],
            "summary_proposals": result["summary_proposals"],
        }
    )


def _h_forget(engine: MemoryV4, args: models.ForgetIn) -> dict:
    if not args.item_id and not (args.query and args.confirm):
        raise ValueError("memory_forget requires item_id, or query together with confirm=true")

    # A5 (adversarial review, 2026-09-12): go through the façade's own
    # `forget()` (additive method) instead of reaching into `engine.store`
    # directly -- that used to leave the item's vector/BM25/entity index
    # entries in place until the next consolidate() rebuild, so it kept
    # consuming candidate slots in every retrieval signal in the meantime.
    project = _project_of(args.project)
    forgotten: list[str] = []
    if args.item_id:
        item = engine.store.get(args.item_id)
        if item is None or (project is not None and item.project != project):
            raise ValueError(f"unknown item_id in project scope: {args.item_id}")
        if engine.forget(args.item_id, reason=args.reason, project=project):
            forgotten.append(args.item_id)
    else:
        hits = engine.search(args.query, top_k=1, project=project)
        if not hits:
            raise ValueError(f"no item found matching query: {args.query!r}")
        item_id = hits[0].item.id
        if engine.forget(item_id, reason=args.reason, project=project):
            forgotten.append(item_id)

    engine.flush()
    logger.info("memory_forget: %s marked deleted (reason=%s)", forgotten, args.reason)
    return {"forgotten": forgotten}


def _h_ingest(engine: MemoryV4, args: models.IngestIn) -> dict:
    project = _write_project_of(args.project)
    extensions = {e.lower() for e in args.extensions} if args.extensions else set(DEFAULT_INGEST_EXTENSIONS)
    root = _resolve_under_home(args.path)
    if not root.exists():
        raise ValueError(f"path not found: {args.path}")

    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    documents = 0
    chunks = 0
    skipped: list[str] = []
    for f in files:
        suffix = f.suffix.lower()
        text: str | None = None
        if suffix == ".pdf":
            try:
                import fitz  # type: ignore  # pymupdf, optional
            except ImportError:
                skipped.append(f"{f}: pdf skipped (pymupdf not installed)")
                continue
            try:
                doc = fitz.open(str(f))
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
            except Exception as exc:  # pragma: no cover - depends on pdf contents
                skipped.append(f"{f}: pdf read error ({exc})")
                continue
        elif suffix in extensions:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                skipped.append(f"{f}: read error ({exc})")
                continue
        else:
            continue

        if text is None or not text.strip():
            continue
        result = engine.add_document(f.name, text, project=project, doc_ref=str(f))
        documents += 1
        chunks += result["chunks"]

    engine.flush()
    return {"documents": documents, "chunks": chunks, "skipped": skipped}


def _h_list_documents(engine: MemoryV4, args: models.ListDocumentsIn) -> dict:
    project = _project_of(args.project)
    docs: dict[str, dict] = {}
    for it in engine.store.all(include_deleted=False):
        if it.kind != "doc" or (project is not None and it.project != project):
            continue
        ref = it.doc_ref or it.id
        entry = docs.setdefault(ref, {"doc_ref": ref, "project": it.project, "name": None, "chunks": 0})
        entry["chunks"] += 1
        if entry["name"] is None and it.tags:
            entry["name"] = it.tags[0]
    return {"documents": sorted(docs.values(), key=lambda d: d["doc_ref"])}


def _h_search_occurrences(engine: MemoryV4, args: models.SearchOccurrencesIn) -> dict:
    project = _project_of(args.project)
    keyword_tokens = set(tokenize(args.keyword))
    if not keyword_tokens:
        return {"count": 0, "snippets": []}

    matches = [
        it
        for it in engine.store.all(include_deleted=False)
        if (project is None or it.project == project) and keyword_tokens.issubset(set(tokenize(it.text)))
    ]
    matches.sort(key=lambda it: it.event_time, reverse=True)
    top = matches[: args.limit]
    snippets = [
        {"id": it.id, "kind": it.kind, "text": it.text[:280], "event_time": _iso(it.event_time)} for it in top
    ]
    return {"count": len(matches), "snippets": snippets}


def _h_migrate_v3(engine: MemoryV4, args: models.MigrateV3In) -> dict:
    log_path = args.log_path or DEFAULT_V3_LOG_PATH
    resolved = _resolve_under_home(log_path)
    result = engine.migrate_v3(str(resolved))
    engine.flush()
    return result


# --------------------------------------------------------------- tool tables
TOOL_MODELS: dict[str, type[BaseModel]] = {
    "memory_session_brief": models.SessionBriefIn,
    "memory_chat_save": models.ChatSaveIn,
    "memory_add_facts": models.AddFactsIn,
    "memory_search": models.SearchIn,
    "memory_get_context": models.GetContextIn,
    "memory_timeline": models.TimelineIn,
    "memory_block_get": models.BlockGetIn,
    "memory_block_set": models.BlockSetIn,
    "memory_consolidate": models.ConsolidateIn,
    "memory_compact": models.CompactIn,
    "memory_forget": models.ForgetIn,
    "memory_ingest": models.IngestIn,
    "memory_list_documents": models.ListDocumentsIn,
    "memory_search_occurrences": models.SearchOccurrencesIn,
    "memory_status": models.StatusIn,
    "memory_migrate_v3": models.MigrateV3In,
}

TOOL_HANDLERS: dict[str, Callable[[MemoryV4, BaseModel], dict]] = {
    "memory_session_brief": _h_session_brief,
    "memory_chat_save": _h_chat_save,
    "memory_add_facts": _h_add_facts,
    "memory_search": _h_search,
    "memory_get_context": _h_get_context,
    "memory_timeline": _h_timeline,
    "memory_block_get": _h_block_get,
    "memory_block_set": _h_block_set,
    "memory_consolidate": _h_consolidate,
    "memory_compact": _h_consolidate,
    "memory_forget": _h_forget,
    "memory_ingest": _h_ingest,
    "memory_list_documents": _h_list_documents,
    "memory_search_occurrences": _h_search_occurrences,
    "memory_status": _h_status,
    "memory_migrate_v3": _h_migrate_v3,
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "memory_session_brief": "Recall core blocks, recent episodes, key facts, procedures and pending items. Call this before answering, at the start of every session.",
    "memory_chat_save": "Persist chat turns as episodes, an optional summary, and atomic ADD-only facts. Call regularly, after significant work, and before ending a session.",
    "memory_add_facts": "Add atomic ADD-only facts/summaries/procedures/beliefs directly, without saving chat turns.",
    "memory_search": "Hybrid (dense + BM25 + entity) semantic search over memory, with optional at_time validity filtering.",
    "memory_get_context": "Retrieve a ready-to-paste context block (diversified via MMR) for a query.",
    "memory_timeline": "Trace how an entity or topic evolved: facts/summaries about it, sorted by event_time with validity and supersession.",
    "memory_block_get": "Read one or all core memory blocks (user_profile, project_state, working_rules, ...).",
    "memory_block_set": "Replace or append to a core memory block. Call when the user profile, project state, or working rules change.",
    "memory_consolidate": "Run dedup, supersession, decay sweeps and propose new summaries. Same as memory_compact.",
    "memory_compact": "Alias of memory_consolidate.",
    "memory_forget": "Soft-delete one item by id, or by query top match (requires confirm=true). Never a hard delete.",
    "memory_ingest": "Ingest a text file or folder (under $HOME) into memory as chunked 'doc' items.",
    "memory_list_documents": "List ingested documents grouped by doc_ref, with chunk counts.",
    "memory_search_occurrences": "Count items whose text contains a keyword (tokenized match) and return sample snippets.",
    "memory_status": "Engine status: item counts by kind, embedding model, index sizes, data dir.",
    "memory_migrate_v3": "One-time, read-only migration of a v3 _conversation_log.jsonl into v4 episodes and summaries.",
}


def _request_scope(name: str, payload: dict) -> str:
    """Canonicalize actual operation targets, preserving all-project reads/writes."""
    # Resolve omitted projects before fingerprinting: changing the server default
    # changes the actual target, while explicit and implicit equal targets retry.
    project_field = "project_label" if name == "memory_chat_save" else "project"
    if name in {"memory_chat_save", "memory_add_facts", "memory_ingest", "memory_block_set"}:
        target = _write_project_of(payload.get(project_field))
    elif name == "memory_migrate_v3":
        target = ALL_PROJECTS  # Migration affects the whole store, preserving record projects.
    else:
        target = payload.get(project_field) or default_project()
    if project_field in payload:
        payload[project_field] = target
    if payload.get("scope") == "global":
        target = "global"
        # Global block operations do not use a project target.
        payload.pop("project", None)
    scope = json.dumps([payload.get("scope", "project"), target])
    return scope


def dispatch(name: str, arguments: dict | None) -> dict:
    """Validate ``arguments`` against ``name``'s model, call its handler, and
    normalize any exception into ``{"ok": False, "error": ...}``."""
    arguments = arguments or {}
    model_cls = TOOL_MODELS.get(name)
    handler = TOOL_HANDLERS.get(name)
    if model_cls is None or handler is None:
        return {"ok": False, "error": f"unknown tool: {name!r}"}

    try:
        parsed = model_cls(**arguments)
    except ValidationError as exc:
        logger.warning("validation error for tool %s: %s", name, exc)
        return {"ok": False, "error": str(exc)}
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        engine = get_engine()
        payload = parsed.model_dump(mode="json", exclude={"request_id"})
        scope = _request_scope(name, payload)
        operation = "memory_consolidate" if name == "memory_compact" else name
        return engine.execute_request(operation, scope, payload, getattr(parsed, "request_id", None),
                                      lambda: handler(engine, parsed))
    except Exception as exc:  # noqa: BLE001 - deliberately broad: never leak a traceback to the client
        logger.exception("tool %s failed", name)
        return {"ok": False, "error": str(exc)}


# ------------------------------------------------------------------- mcp glue
# Instructions are resolved once at process start from the env-derived config
# (same lazy-vs-eager tradeoff as everything else here: the engine itself is
# lazy, but `Server(instructions=...)` needs a string at construction time) --
# see instructions.py `build_instructions` for the facet-channel addendum.
server = Server("memory-os-v4", instructions=build_instructions(Config.from_env()))


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=name,
            title=name.replace("memory_", "").replace("_", " ").title(),
            description=TOOL_DESCRIPTIONS.get(name, name),
            inputSchema=model_cls.model_json_schema(),
        )
        for name, model_cls in TOOL_MODELS.items()
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # dispatch is synchronous (embedding, disk I/O): run it off the event loop
    # so one slow call does not stall the other clients of a shared HTTP
    # server. The engine itself is thread-safe (single RLock).
    result = await anyio.to_thread.run_sync(dispatch, name, arguments)
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _build_http_app(host: str, port: int):
    """Streamable HTTP transport: ONE process serves every client, so all
    conversations share one engine and nobody else appends to items.jsonl.
    Stateless mode: no session ids, clients survive a server restart.
    Restricted to loopback with mandatory Host/Origin and token validation."""
    import contextlib

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route
    from .http_security import LocalHTTPGuard, load_http_token, local_security_settings
    from .readiness import build_identity, engine_readiness

    settings = local_security_settings(host, port)
    token = load_http_token(os.environ.get("MEMORY_V4_TOKEN_FILE"))
    identity = build_identity()
    session_manager = StreamableHTTPSessionManager(app=server, stateless=True, security_settings=settings)

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    async def health(request):
        return JSONResponse({"ok": True, "server": "memory-os-v4", "transport": "http",
                             "engine_loaded": _engine is not None, **identity})

    async def ready(request):
        ok, reason = await anyio.to_thread.run_sync(engine_readiness, _engine)
        if ok and (identity["commit"] is None or identity["source_dirty"]):
            ok, reason = False, "build_unqualified"
        return JSONResponse({"ok": ok, "ready": ok, "reason": reason,
                             "server": "memory-os-v4", **identity}, status_code=200 if ok else 503)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        try:
            async with session_manager.run():
                yield
        finally:
            set_engine(None)

    app = Starlette(routes=[Route("/health", health), Route("/ready", ready),
                           Mount("/mcp", app=handle_mcp)], lifespan=lifespan)
    return LocalHTTPGuard(app, settings=settings, token=token)


def _run_http(host: str, port: int, *, warm: bool = False) -> None:
    import uvicorn

    # Validate credentials and transport before opening any data directory.
    app = _build_http_app(host, port)
    if warm:
        get_engine()
    logger.warning("memory-os-v4 streamable HTTP on http://%s:%d/mcp/", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------- selfcheck
def _selfcheck_architecture_doc(tmp_dir: str) -> str:
    """A small read-only file to exercise memory_ingest. In a source checkout this is
    v4/docs/ARCHITECTURE.md; in an installed wheel (no docs/) a tiny note is written into the
    throwaway data dir, which the caller whitelists through MEMORY_WORKSPACE."""
    repo_doc = Path(__file__).resolve().parents[2] / "docs" / "ARCHITECTURE.md"
    if repo_doc.is_file():
        return str(repo_doc)
    note = Path(tmp_dir) / "selfcheck_note.md"
    note.write_text("# Selfcheck\n\nMemory OS v4 selfcheck document for memory_ingest.\n", encoding="utf-8")
    return str(note)


def _selfcheck() -> int:
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="memory_os_v4_selfcheck_")
    engine = MemoryV4(config=Config(data_dir=tmp_dir, embed_model="hash"))
    set_engine(engine)

    rows: list[tuple[str, str]] = []
    any_error = False
    last_fact_id: str | None = None

    def call(name: str, payload: dict) -> dict:
        nonlocal any_error
        result = dispatch(name, payload)
        ok = not (isinstance(result, dict) and result.get("ok") is False)
        rows.append((name, "ok" if ok else f"error: {result.get('error')}"))
        if not ok:
            any_error = True
        return result

    call("memory_status", {})
    call("memory_block_set", {"name": "user_profile", "text": "Selfcheck user.", "mode": "replace"})
    call("memory_block_get", {})
    call(
        "memory_chat_save",
        {
            "messages": [{"role": "user", "content": "Selfcheck message about Paris."}],
            "summary": "Selfcheck session summary.",
            "project_label": "selfcheck",
            "facts": [{"text": "Romain lives in Paris."}],
        },
    )
    add_res = call(
        "memory_add_facts",
        {"facts": [{"text": "Selfcheck fact about NQ futures."}], "project": "selfcheck"},
    )
    if isinstance(add_res, dict) and add_res.get("added"):
        last_fact_id = add_res["added"][0]

    call("memory_search", {"query": "Paris", "project": "selfcheck"})
    call("memory_get_context", {"query": "Paris", "project": "selfcheck"})
    call("memory_timeline", {"query": "Paris", "project": "selfcheck"})
    call("memory_session_brief", {"project": "selfcheck"})
    call("memory_list_documents", {"project": "selfcheck"})
    call("memory_search_occurrences", {"keyword": "Paris", "project": "selfcheck"})
    doc_path = _selfcheck_architecture_doc(tmp_dir)
    in_tmp = doc_path.startswith(tmp_dir)
    prev_workspace = os.environ.get("MEMORY_WORKSPACE")
    if in_tmp:
        os.environ["MEMORY_WORKSPACE"] = tmp_dir
    try:
        call("memory_ingest", {"path": doc_path, "project": "selfcheck"})
    finally:
        if in_tmp:
            if prev_workspace is None:
                os.environ.pop("MEMORY_WORKSPACE", None)
            else:
                os.environ["MEMORY_WORKSPACE"] = prev_workspace
    call("memory_migrate_v3", {})
    call("memory_consolidate", {"dry_run": True})
    call("memory_compact", {"dry_run": True})

    if last_fact_id:
        call("memory_forget", {"item_id": last_fact_id, "reason": "selfcheck cleanup", "project": "selfcheck"})
    else:
        call(
            "memory_forget",
            {"query": "Selfcheck fact about NQ futures", "confirm": True, "reason": "selfcheck cleanup", "project": "selfcheck"},
        )

    engine.flush()
    set_engine(None)

    width = max(len(n) for n, _ in rows)
    print(f"memory-os-v4 selfcheck (data_dir={tmp_dir})")
    for name, status in rows:
        print(f"  {name.ljust(width)}  {status}")
    print("RESULT:", "FAIL" if any_error else "PASS")
    return 1 if any_error else 0


# ------------------------------------------------------------------------ run
def run() -> None:
    parser = argparse.ArgumentParser(prog="python -m memory_os_v4.server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio", help="stdio (one process per client) or http (one shared server).")
    parser.add_argument("--host", default=os.environ.get("MEMORY_V4_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MEMORY_V4_PORT", "8765")))
    parser.add_argument("--warm", action="store_true", help="Load the engine (model + index) before serving.")
    parser.add_argument("--selfcheck", action="store_true", help="Instantiate a throwaway engine and exercise every tool once.")
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("MEMORY_V4_LOG_LEVEL", "WARNING"))

    if args.selfcheck:
        sys.exit(_selfcheck())

    try:
        if args.transport == "http":
            _run_http(args.host, args.port, warm=args.warm)
        else:
            asyncio.run(_run_stdio())
    finally:
        set_engine(None)


if __name__ == "__main__":
    run()
