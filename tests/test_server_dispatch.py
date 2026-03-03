"""Tests for server.py — dispatch logic, resources, tool listing, utilities."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from memory_os_ai.engine import MemoryEngine
from memory_os_ai.chat_extractor import ChatExtractor


# ---------------------------------------------------------------------------
# Helpers — we test _dispatch() directly to avoid asyncio overhead
# ---------------------------------------------------------------------------

def _make_engine(cache_dir: str | None = None) -> MemoryEngine:
    return MemoryEngine(model_name="all-MiniLM-L6-v2", cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Test dispatch via module-level import
# ---------------------------------------------------------------------------
import memory_os_ai.server as srv


class TestDispatch:
    """Test _dispatch routing for each tool."""

    def test_dispatch_memory_ingest_valid(self, tmp_path):
        """Ingest from a real temp folder with a .txt file."""
        (tmp_path / "hello.txt").write_text("hello world test content")
        # Temporarily swap the global engine
        orig = srv._engine
        srv._engine = _make_engine(cache_dir=str(tmp_path))
        try:
            result = srv._dispatch("memory_ingest", {
                "folder_path": str(tmp_path),
                "extensions": None,
                "force_reindex": False,
            })
            assert result["ok"] is True
            assert result["files_indexed"] >= 1
        finally:
            srv._engine = orig

    def test_dispatch_memory_ingest_missing_folder(self):
        result = srv._dispatch("memory_ingest", {
            "folder_path": "/nonexistent/path/xyz",
            "extensions": None,
            "force_reindex": False,
        })
        assert result["ok"] is False
        assert "not found" in result["error"].lower() or "Folder" in result["error"]

    def test_dispatch_memory_search_empty(self):
        """Search on uninitialized engine returns empty."""
        orig = srv._engine
        srv._engine = _make_engine()
        try:
            result = srv._dispatch("memory_search", {
                "query": "test query",
                "top_k": 5,
                "threshold": 0.8,
            })
            assert result["ok"] is True
            assert result["count"] == 0
        finally:
            srv._engine = orig

    def test_dispatch_memory_search_with_data(self, tmp_path):
        """Search with some ingested data."""
        orig = srv._engine
        eng = _make_engine(cache_dir=str(tmp_path))
        eng.ingest_segments(["FAISS indexing is fast", "Memory OS AI rocks"], source_label="test")
        srv._engine = eng
        try:
            result = srv._dispatch("memory_search", {
                "query": "FAISS indexing",
                "top_k": 5,
                "threshold": 2.0,
            })
            assert result["ok"] is True
            assert result["count"] >= 1
        finally:
            srv._engine = orig

    def test_dispatch_memory_search_occurrences(self, tmp_path):
        orig = srv._engine
        eng = _make_engine(cache_dir=str(tmp_path))
        eng.ingest_segments(["hello hello world", "hello again"], source_label="test")
        srv._engine = eng
        try:
            result = srv._dispatch("memory_search_occurrences", {
                "keyword": "hello",
                "top_k": 100,
            })
            assert result["ok"] is True
            assert result["total_occurrences"] >= 1
        finally:
            srv._engine = orig

    def test_dispatch_memory_get_context(self, tmp_path):
        orig = srv._engine
        eng = _make_engine(cache_dir=str(tmp_path))
        eng.ingest_segments(["important context data here"], source_label="test")
        srv._engine = eng
        try:
            result = srv._dispatch("memory_get_context", {
                "query": "important context",
                "max_tokens": 1000,
                "top_k": 10,
            })
            assert result["ok"] is True
            assert len(result["context"]) > 0
        finally:
            srv._engine = orig

    def test_dispatch_memory_list_documents(self, tmp_path):
        orig = srv._engine
        eng = _make_engine(cache_dir=str(tmp_path))
        eng.ingest_segments(["segment one", "segment two"], source_label="test")
        srv._engine = eng
        try:
            result = srv._dispatch("memory_list_documents", {"include_stats": True})
            assert result["ok"] is True
            assert result["count"] >= 1
        finally:
            srv._engine = orig

    def test_dispatch_memory_status(self):
        orig = srv._engine
        srv._engine = _make_engine()
        try:
            result = srv._dispatch("memory_status", {})
            assert "initialized" in result
            assert "device" in result
            assert "model" in result
        finally:
            srv._engine = orig

    def test_dispatch_memory_transcribe_missing(self):
        result = srv._dispatch("memory_transcribe", {
            "file_path": "/nonexistent/audio.mp3",
            "language": "en",
        })
        assert result["ok"] is False

    def test_dispatch_memory_chat_sync(self):
        orig_extractor = srv._chat_extractor
        srv._chat_extractor = ChatExtractor(state_dir=tempfile.mkdtemp())
        try:
            result = srv._dispatch("memory_chat_sync", {"source_id": None})
            assert result["ok"] is True
        finally:
            srv._chat_extractor = orig_extractor

    def test_dispatch_memory_chat_source_add_remove(self):
        orig_extractor = srv._chat_extractor
        srv._chat_extractor = ChatExtractor(state_dir=tempfile.mkdtemp())
        try:
            # Add
            result = srv._dispatch("memory_chat_source_add", {
                "source_id": "test-src",
                "source_type": "jsonl",
                "path": "/tmp/fake.jsonl",
            })
            assert result["ok"] is True

            # Status
            result = srv._dispatch("memory_chat_status", {})
            assert result["ok"] is True
            assert result["total_sources"] >= 1

            # Remove
            result = srv._dispatch("memory_chat_source_remove", {
                "source_id": "test-src",
            })
            assert result["ok"] is True
        finally:
            srv._chat_extractor = orig_extractor

    def test_dispatch_memory_chat_auto_detect(self):
        orig_extractor = srv._chat_extractor
        srv._chat_extractor = ChatExtractor(state_dir=tempfile.mkdtemp())
        try:
            result = srv._dispatch("memory_chat_auto_detect", {
                "auto_register": False,
            })
            assert result["ok"] is True
            assert "workspaces_found" in result
        finally:
            srv._chat_extractor = orig_extractor

    def test_dispatch_memory_session_brief(self, tmp_path):
        orig = srv._engine
        orig_extractor = srv._chat_extractor
        srv._engine = _make_engine(cache_dir=str(tmp_path))
        srv._chat_extractor = ChatExtractor(state_dir=str(tmp_path))
        try:
            result = srv._dispatch("memory_session_brief", {
                "max_tokens": 1000,
                "include_chat_sync": False,
                "focus_query": None,
            })
            assert result["ok"] is True
            assert "overview" in result
        finally:
            srv._engine = orig
            srv._chat_extractor = orig_extractor

    def test_dispatch_memory_chat_save(self, tmp_path):
        orig = srv._engine
        srv._engine = _make_engine(cache_dir=str(tmp_path))
        try:
            with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
                result = srv._dispatch("memory_chat_save", {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "world"},
                    ],
                    "summary": "greeting exchange",
                    "project_label": "test",
                })
            assert result["ok"] is True
            assert result["saved_messages"] == 2
            assert result["has_summary"] is True
            # Check JSONL log was written
            log_path = result["log_path"]
            assert os.path.exists(log_path)
        finally:
            srv._engine = orig

    def test_dispatch_memory_compact(self, tmp_path):
        orig = srv._engine
        eng = _make_engine(cache_dir=str(tmp_path))
        eng.ingest_segments(["seg"] * 5, source_label="test")
        srv._engine = eng
        try:
            result = srv._dispatch("memory_compact", {
                "max_segments": 500,
                "keep_recent_hours": 24,
                "strategy": "dedup_merge",
            })
            assert result["ok"] is True
        finally:
            srv._engine = orig

    def test_dispatch_memory_project_link_unlink_list(self, tmp_path):
        orig_links = srv._linked_projects.copy()
        orig_file = srv._LINKS_FILE
        srv._LINKS_FILE = str(tmp_path / "links.json")
        srv._linked_projects.clear()
        try:
            # Link
            result = srv._dispatch("memory_project_link", {
                "project_path": str(tmp_path),
                "alias": "test-project",
            })
            assert result["ok"] is True
            assert result["alias"] == "test-project"

            # List
            result = srv._dispatch("memory_project_list", {})
            assert result["ok"] is True
            assert result["linked_count"] == 1

            # Unlink
            result = srv._dispatch("memory_project_unlink", {"alias": "test-project"})
            assert result["ok"] is True
            assert result["linked_count"] == 0

            # Unlink nonexistent
            result = srv._dispatch("memory_project_unlink", {"alias": "nope"})
            assert result["ok"] is False
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)
            srv._LINKS_FILE = orig_file

    def test_dispatch_unknown_tool(self):
        result = srv._dispatch("nonexistent_tool", {})
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]

    def test_dispatch_memory_project_link_missing_dir(self):
        result = srv._dispatch("memory_project_link", {
            "project_path": "/nonexistent/dir/12345",
            "alias": "missing",
        })
        assert result["ok"] is False

    def test_dispatch_memory_project_link_duplicate_alias(self, tmp_path):
        orig_links = srv._linked_projects.copy()
        orig_file = srv._LINKS_FILE
        srv._LINKS_FILE = str(tmp_path / "links.json")
        srv._linked_projects.clear()
        try:
            srv._dispatch("memory_project_link", {
                "project_path": str(tmp_path),
                "alias": "dup",
            })
            result = srv._dispatch("memory_project_link", {
                "project_path": str(tmp_path),
                "alias": "dup",
            })
            assert result["ok"] is False
            assert "already linked" in result["error"]
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)
            srv._LINKS_FILE = orig_file


class TestCallTool:
    """Test the async call_tool wrapper."""

    @pytest.mark.asyncio
    async def test_call_tool_valid(self):
        orig = srv._engine
        srv._engine = _make_engine()
        try:
            result = await srv.call_tool("memory_status", {})
            assert len(result) == 1
            data = json.loads(result[0].text)
            assert "initialized" in data
        finally:
            srv._engine = orig

    @pytest.mark.asyncio
    async def test_call_tool_invalid_input(self):
        """Pydantic validation error should be caught and returned as error."""
        result = await srv.call_tool("memory_ingest", {"folder_path": ""})
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["ok"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        result = await srv.call_tool("nonexistent_xyz", {})
        data = json.loads(result[0].text)
        assert data["ok"] is False


class TestListTools:
    """Test tool listing."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_all(self):
        tools = await srv.list_tools()
        assert len(tools) == 21
        names = {t.name for t in tools}
        assert "memory_ingest" in names
        assert "memory_search" in names
        assert "memory_compact" in names
        assert "memory_project_link" in names
        assert "memory_cloud_configure" in names
        assert "memory_cloud_status" in names
        assert "memory_cloud_sync" in names

    @pytest.mark.asyncio
    async def test_list_tools_have_titles(self):
        tools = await srv.list_tools()
        for t in tools:
            # All our tools define titles
            assert t.title is not None, f"Tool {t.name} missing title"


class TestResources:
    """Test MCP resource listing and reading."""

    @pytest.mark.asyncio
    async def test_list_resources_empty(self):
        orig = srv._engine
        srv._engine = _make_engine()
        try:
            resources = await srv.list_resources()
            # No documents, no log, no linked projects = empty or just log
            # depending on whether MEMORY_CACHE_DIR has a log file
            assert isinstance(resources, list)
        finally:
            srv._engine = orig

    @pytest.mark.asyncio
    async def test_list_resources_with_documents(self, tmp_path):
        orig = srv._engine
        eng = _make_engine(cache_dir=str(tmp_path))
        eng.ingest_segments(["resource test"], source_label="test")
        srv._engine = eng
        try:
            resources = await srv.list_resources()
            doc_uris = [r.uri for r in resources if str(r.uri).startswith("memory://documents/")]
            assert len(doc_uris) >= 1
        finally:
            srv._engine = orig

    @pytest.mark.asyncio
    async def test_read_resource_document(self, tmp_path):
        orig = srv._engine
        eng = _make_engine(cache_dir=str(tmp_path))
        eng.ingest_segments(["resource content here"], source_label="test")
        srv._engine = eng
        try:
            # Get doc name
            doc_name = list(eng._documents.keys())[0]
            result = await srv.read_resource(f"memory://documents/{doc_name}")
            assert len(result) == 1
            assert "resource content here" in result[0].text
        finally:
            srv._engine = orig

    @pytest.mark.asyncio
    async def test_read_resource_missing_document(self):
        orig = srv._engine
        srv._engine = _make_engine()
        try:
            result = await srv.read_resource("memory://documents/nonexistent.pdf")
            assert "not found" in result[0].text.lower()
        finally:
            srv._engine = orig

    @pytest.mark.asyncio
    async def test_read_resource_conversation_log(self, tmp_path):
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
            # Create a log file
            log_path = tmp_path / "_conversation_log.jsonl"
            log_path.write_text('{"test": true}\n')
            result = await srv.read_resource("memory://logs/conversation")
            assert len(result) == 1
            assert "test" in result[0].text

    @pytest.mark.asyncio
    async def test_read_resource_no_log(self, tmp_path):
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
            result = await srv.read_resource("memory://logs/conversation")
            assert "No conversation log" in result[0].text

    @pytest.mark.asyncio
    async def test_read_resource_linked_project(self, tmp_path):
        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        srv._linked_projects["test-link"] = {"path": str(tmp_path), "engine": None}
        try:
            result = await srv.read_resource("memory://linked/test-link")
            assert len(result) == 1
            # Engine not initialized, so should say not loaded
            assert "not loaded" in result[0].text.lower() or "index" in result[0].text.lower()
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)

    @pytest.mark.asyncio
    async def test_read_resource_linked_project_missing(self):
        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        try:
            result = await srv.read_resource("memory://linked/nope")
            assert "not found" in result[0].text.lower()
        finally:
            srv._linked_projects.update(orig_links)

    @pytest.mark.asyncio
    async def test_read_resource_unknown_uri(self):
        result = await srv.read_resource("memory://unknown/foo")
        assert "Unknown" in result[0].text or "unknown" in result[0].text.lower()


class TestUtilities:
    """Test server utility functions."""

    def test_check_api_key_no_env(self):
        """No MEMORY_API_KEY set = always passes."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MEMORY_API_KEY", None)
            req = MagicMock()
            req.headers = {}
            assert srv._check_api_key(req) is True

    def test_check_api_key_bearer(self):
        with patch.dict(os.environ, {"MEMORY_API_KEY": "test-secret"}):
            req = MagicMock()
            req.headers = {"authorization": "Bearer test-secret"}
            assert srv._check_api_key(req) is True

    def test_check_api_key_x_header(self):
        with patch.dict(os.environ, {"MEMORY_API_KEY": "test-secret"}):
            req = MagicMock()
            req.headers = {"authorization": "", "x-api-key": "test-secret"}
            assert srv._check_api_key(req) is True

    def test_check_api_key_wrong(self):
        with patch.dict(os.environ, {"MEMORY_API_KEY": "correct"}):
            req = MagicMock()
            req.headers = {"authorization": "Bearer wrong", "x-api-key": "wrong"}
            assert srv._check_api_key(req) is False

    def test_load_project_links_empty(self, tmp_path):
        orig_links = srv._linked_projects.copy()
        orig_file = srv._LINKS_FILE
        srv._LINKS_FILE = str(tmp_path / "links.json")
        srv._linked_projects.clear()
        try:
            srv._load_project_links()
            assert len(srv._linked_projects) == 0
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)
            srv._LINKS_FILE = orig_file

    def test_save_and_load_project_links(self, tmp_path):
        orig_links = srv._linked_projects.copy()
        orig_file = srv._LINKS_FILE
        srv._LINKS_FILE = str(tmp_path / "links.json")
        srv._linked_projects.clear()
        try:
            # Create a real directory to link
            link_dir = tmp_path / "linked_proj"
            link_dir.mkdir()
            srv._linked_projects["saved"] = {"path": str(link_dir), "engine": None}
            srv._save_project_links()

            srv._linked_projects.clear()
            srv._load_project_links()
            assert "saved" in srv._linked_projects
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)
            srv._LINKS_FILE = orig_file

    def test_load_project_links_traversal_blocked(self, tmp_path):
        """Links with '..' should be skipped."""
        links_file = tmp_path / "links.json"
        links_file.write_text(json.dumps({"evil": "../../../etc"}))
        orig_links = srv._linked_projects.copy()
        orig_file = srv._LINKS_FILE
        srv._LINKS_FILE = str(links_file)
        srv._linked_projects.clear()
        try:
            srv._load_project_links()
            assert "evil" not in srv._linked_projects
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)
            srv._LINKS_FILE = orig_file

    def test_get_linked_engine_none(self):
        result = srv._get_linked_engine("nonexistent")
        assert result is None

    def test_ingest_relative_path_resolution(self):
        """Relative folder_path should resolve to MEMORY_WORKSPACE."""
        with patch.dict(os.environ, {"MEMORY_WORKSPACE": "/tmp"}):
            # This will fail because /tmp/nonexistent doesn't have files,
            # but it tests the path resolution logic
            result = srv._dispatch("memory_ingest", {
                "folder_path": "nonexistent_subdir",
                "extensions": None,
                "force_reindex": False,
            })
            assert result["ok"] is False
            assert "nonexistent_subdir" in result.get("error", "")

    def test_transcribe_relative_path_resolution(self):
        """Relative file_path should resolve to MEMORY_WORKSPACE."""
        with patch.dict(os.environ, {"MEMORY_WORKSPACE": "/tmp"}):
            result = srv._dispatch("memory_transcribe", {
                "file_path": "nonexistent.mp3",
                "language": "en",
            })
            assert result["ok"] is False
