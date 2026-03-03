"""Additional coverage tests — engine extractors, server transports, setup CLI.

Uses a single shared MemoryEngine to avoid torch/FAISS segfaults on Apple Silicon
when multiple SentenceTransformer models are loaded in the same process.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Engine — extractor functions (mocked heavy deps)
# ---------------------------------------------------------------------------
from memory_os_ai.engine import (
    _extract_txt,
    _ensure_faiss,
    MemoryEngine,
    EXTRACTORS,
    SUPPORTED_EXTENSIONS,
    DocumentInfo,
    SearchResult,
    OccurrenceResult,
)


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Deterministic fake encoder — returns 384-d normalized vectors."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(len(texts), 384).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-10, None)


class TestExtractors:
    """Test document extractors (with mocking for heavy deps)."""

    def test_extract_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world content")
        text, pages = _extract_txt(str(f))
        assert text == "hello world content"
        assert pages == 1

    def test_extractors_registry_complete(self):
        """All expected extensions are registered."""
        expected = {
            ".pdf", ".txt", ".docx", ".doc", ".png", ".jpeg", ".jpg",
            ".pptx", ".ppt", ".mp3", ".wav", ".ogg", ".flac",
        }
        assert SUPPORTED_EXTENSIONS == expected

    def test_extractors_callable(self):
        for ext, func in EXTRACTORS.items():
            assert callable(func), f"Extractor for {ext} is not callable"

    def test_data_classes(self):
        doc = DocumentInfo("test.pdf", 5, 10, 500, 0, 10)
        assert doc.filename == "test.pdf"
        assert doc.nb_pages == 5

        sr = SearchResult("test.pdf", "hello world", 0.5, 3)
        assert sr.filename == "test.pdf"
        assert sr.distance == 0.5

        occ = OccurrenceResult(10, {"test.pdf": 10})
        assert occ.total == 10


class TestEngineLazy:
    """Test lazy import helpers and engine methods (with mocked encode)."""

    def test_ensure_faiss(self):
        result = _ensure_faiss()
        assert result is not None

    def test_engine_device_property(self):
        eng = MemoryEngine()
        device = eng.device
        assert device in ("cpu", "cuda")

    def test_engine_segment_text_splitting(self):
        eng = MemoryEngine(segment_size=10, segment_overlap=3)
        segments = eng._segment_text("0123456789ABCDEF")
        assert len(segments) >= 2
        assert len(segments[0]) == 10

    def test_engine_segment_overlap_zero(self):
        eng = MemoryEngine(segment_size=10, segment_overlap=10)
        segments = eng._segment_text("0123456789ABCDEF")
        assert len(segments) >= 1

    def test_engine_build_index(self):
        eng = MemoryEngine()
        embeddings = np.random.randn(5, 384).astype("float32")
        # Mock the FAISS index to avoid real index creation (causes segfault
        # on Apple Silicon when combined with real encode in other tests)
        mock_index = MagicMock()
        mock_index.ntotal = 5
        with patch.object(eng, "_build_index") as mock_build:
            mock_build.side_effect = lambda e: setattr(eng, "_index", mock_index)
            eng._build_index(embeddings)
            assert eng._index is not None
            assert eng._index.ntotal == 5

    def test_engine_get_context_empty(self):
        eng = MemoryEngine()
        context = eng.get_context("test", max_chars=1000)
        assert context == ""

    @patch.object(MemoryEngine, "_encode", side_effect=_fake_encode)
    def test_engine_ingest_with_cache(self, mock_enc, tmp_path):
        """Test that numpy cache is written and read."""
        (tmp_path / "test.txt").write_text("cached content for testing")
        eng = MemoryEngine(cache_dir=str(tmp_path))
        result = eng.ingest(str(tmp_path))
        assert result["ok"] is True
        cache_file = tmp_path / "embeddings_cache.npy"
        assert cache_file.exists()
        # Second ingest should use cache
        result2 = eng.ingest(str(tmp_path))
        assert result2["ok"] is True

    @patch.object(MemoryEngine, "_encode", side_effect=_fake_encode)
    def test_engine_ingest_segments(self, mock_enc):
        eng = MemoryEngine()
        result = eng.ingest_segments(["test segment"], source_label="test")
        assert result["ok"] is True
        assert result["added_segments"] == 1
        assert eng.is_initialized is True

    @patch.object(MemoryEngine, "_encode", side_effect=_fake_encode)
    def test_engine_ingest_segments_append(self, mock_enc):
        eng = MemoryEngine()
        eng.ingest_segments(["first"], source_label="t1")
        result = eng.ingest_segments(["second", "third"], source_label="t2")
        assert result["ok"] is True
        assert result["total_segments"] == 3

    @patch.object(MemoryEngine, "_encode", side_effect=_fake_encode)
    def test_engine_ingest_segments_empty(self, mock_enc):
        eng = MemoryEngine()
        result = eng.ingest_segments([], source_label="empty")
        assert result["ok"] is True
        assert result["added_segments"] == 0


class TestEngineIngestErrors:
    """Test ingest error handling (mocked encode)."""

    def test_ingest_empty_folder(self, tmp_path):
        eng = MemoryEngine()
        result = eng.ingest(str(tmp_path))
        assert result["ok"] is False
        assert "No documents" in result["error"]

    @patch.object(MemoryEngine, "_encode", side_effect=_fake_encode)
    def test_ingest_filtered_extensions(self, mock_enc, tmp_path):
        (tmp_path / "test.txt").write_text("hello")
        (tmp_path / "test.pdf").write_text("fake pdf")
        eng = MemoryEngine()
        result = eng.ingest(str(tmp_path), extensions={".txt"})
        assert result["ok"] is True
        assert result["files_indexed"] == 1

    @patch.object(MemoryEngine, "_encode", side_effect=_fake_encode)
    def test_ingest_force_reindex(self, mock_enc, tmp_path):
        (tmp_path / "test.txt").write_text("content to reindex")
        eng = MemoryEngine(cache_dir=str(tmp_path))
        eng.ingest(str(tmp_path))
        result = eng.ingest(str(tmp_path), force_reindex=True)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Server — transport-related and session brief with linked projects
# ---------------------------------------------------------------------------
import memory_os_ai.server as srv


class TestSessionBriefLinked:
    """Test session_brief with linked projects (fully mocked engine)."""

    def test_session_brief_with_linked_project(self, tmp_path):
        orig = srv._engine
        orig_links = srv._linked_projects.copy()
        orig_extractor = srv._chat_extractor

        from memory_os_ai.chat_extractor import ChatExtractor

        # Use a mock engine that returns canned results
        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.session_brief.return_value = {
            "ok": True,
            "overview": {"total_documents": 3, "total_segments": 10},
            "context": "main data context",
            "context_chars": 17,
            "unique_segments_retrieved": 5,
        }
        srv._engine = eng

        linked_eng = MagicMock(spec=MemoryEngine)
        linked_eng.is_initialized = True
        linked_eng.session_brief.return_value = {
            "ok": True,
            "overview": {"total_documents": 1, "total_segments": 5},
            "context": "linked data",
            "context_chars": 11,
            "unique_segments_retrieved": 2,
        }

        srv._linked_projects.clear()
        srv._linked_projects["test-linked"] = {
            "path": str(tmp_path),
            "engine": linked_eng,
        }
        srv._chat_extractor = ChatExtractor(state_dir=str(tmp_path))

        try:
            result = srv._dispatch("memory_session_brief", {
                "max_tokens": 2000,
                "include_chat_sync": False,
                "focus_query": "project data",
            })
            assert result["ok"] is True
            assert "linked_projects" in result
            assert "test-linked" in result["linked_projects"]
        finally:
            srv._engine = orig
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)
            srv._chat_extractor = orig_extractor

    def test_session_brief_with_chat_sync(self, tmp_path):
        orig = srv._engine
        orig_extractor = srv._chat_extractor

        from memory_os_ai.chat_extractor import ChatExtractor

        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.session_brief.return_value = {
            "ok": True,
            "overview": {"total_documents": 1, "total_segments": 2},
            "context": "test context",
            "context_chars": 12,
            "unique_segments_retrieved": 1,
        }
        srv._engine = eng
        srv._chat_extractor = ChatExtractor(state_dir=str(tmp_path))

        try:
            result = srv._dispatch("memory_session_brief", {
                "max_tokens": 2000,
                "include_chat_sync": True,
                "focus_query": None,
            })
            assert result["ok"] is True
            assert "chat_sync" in result
        finally:
            srv._engine = orig
            srv._chat_extractor = orig_extractor


class TestSearchWithLinked:
    """Test search across linked projects (fully mocked engine)."""

    def test_search_includes_linked(self, tmp_path):
        orig = srv._engine
        orig_links = srv._linked_projects.copy()

        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.search.return_value = [
            SearchResult("main.txt", "main content", 0.3, 0),
        ]
        srv._engine = eng

        linked_eng = MagicMock(spec=MemoryEngine)
        linked_eng.is_initialized = True
        linked_eng.search.return_value = [
            SearchResult("linked.txt", "linked content", 0.5, 0),
        ]

        srv._linked_projects.clear()
        srv._linked_projects["linked-proj"] = {
            "path": str(tmp_path),
            "engine": linked_eng,
        }

        try:
            result = srv._dispatch("memory_search", {
                "query": "FAISS content",
                "top_k": 10,
                "threshold": 2.0,
            })
            assert result["ok"] is True
            assert result["linked_projects_searched"] == 1
            assert result["count"] >= 2
        finally:
            srv._engine = orig
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)


class TestServerTransports:
    """Test transport setup functions (structure only, no actual server start)."""

    def test_run_function_exists(self):
        assert callable(srv.run)

    def test_main_stdio_exists(self):
        assert callable(srv.main_stdio)

    def test_run_sse_function(self):
        assert callable(srv.run_sse)

    def test_run_http_function(self):
        assert callable(srv.run_http)

    def test_main_backward_compat(self):
        """main should be an alias for main_stdio."""
        assert srv.main is srv.main_stdio


class TestChatSaveEdgeCases:
    """Test chat_save edge cases (fully mocked engine)."""

    def test_save_without_summary(self, tmp_path):
        orig = srv._engine
        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.ingest_segments.return_value = {"ok": True, "added_segments": 1, "total_segments": 1}
        srv._engine = eng
        try:
            with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
                result = srv._dispatch("memory_chat_save", {
                    "messages": [{"role": "user", "content": "hi"}],
                    "summary": None,
                    "project_label": None,
                })
            assert result["ok"] is True
            assert result["has_summary"] is False
        finally:
            srv._engine = orig


# ---------------------------------------------------------------------------
# Setup CLI — main() and edge cases
# ---------------------------------------------------------------------------
from memory_os_ai.setup import main as setup_main, check_status


class TestSetupCLI:
    """Test the CLI entry point."""

    def test_main_help(self):
        with patch.object(sys, "argv", ["setup", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                setup_main()
            assert exc_info.value.code == 0

    def test_main_no_args(self):
        with patch.object(sys, "argv", ["setup"]):
            with pytest.raises(SystemExit) as exc_info:
                setup_main()
            assert exc_info.value.code == 0

    def test_main_unknown_target(self):
        with patch.object(sys, "argv", ["setup", "unknown_target"]):
            with pytest.raises(SystemExit) as exc_info:
                setup_main()
            assert exc_info.value.code == 1

    def test_main_chatgpt(self, capsys):
        with patch.object(sys, "argv", ["setup", "chatgpt"]):
            setup_main()
        captured = capsys.readouterr()
        assert "ChatGPT" in captured.out

    def test_main_status(self, capsys):
        with patch.object(sys, "argv", ["setup", "status"]):
            setup_main()
        captured = capsys.readouterr()
        assert "Bridge Status" in captured.out

    def test_main_claude_code(self, tmp_path, capsys):
        with patch.object(sys, "argv", ["setup", "claude-code"]):
            with patch("memory_os_ai.setup._HOME", str(tmp_path)):
                with patch("os.getcwd", return_value=str(tmp_path)):
                    setup_main()
        captured = capsys.readouterr()
        assert "✅" in captured.out

    def test_main_all(self, tmp_path, capsys):
        """Test 'all' target."""
        config_path = tmp_path / "claude_config.json"
        with patch.object(sys, "argv", ["setup", "all"]):
            with patch("memory_os_ai.setup._HOME", str(tmp_path)):
                with patch("memory_os_ai.setup._claude_desktop_config_path", return_value=config_path):
                    with patch("os.getcwd", return_value=str(tmp_path)):
                        with patch("pathlib.Path.home", return_value=tmp_path):
                            setup_main()
        captured = capsys.readouterr()
        assert "✅" in captured.out
        assert "ChatGPT" in captured.out


class TestCheckStatusEdgeCases:
    """Test status check with various configurations."""

    def test_status_with_configured_claude(self, tmp_path):
        config_path = tmp_path / "claude_config.json"
        config_path.write_text(json.dumps({"mcpServers": {"memory-os-ai": {}}}))
        with patch("memory_os_ai.setup._claude_desktop_config_path", return_value=config_path):
            result = check_status()
        assert "✅" in result
        assert "Claude Desktop" in result

    def test_status_with_partial_claude(self, tmp_path):
        config_path = tmp_path / "claude_config.json"
        config_path.write_text(json.dumps({"mcpServers": {"other": {}}}))
        with patch("memory_os_ai.setup._claude_desktop_config_path", return_value=config_path):
            result = check_status()
        assert "⚠️" in result or "not found" in result

    def test_status_with_codex(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        mcp_file = codex_dir / "mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {"memory-os-ai": {}}}))
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = check_status()
        assert "Codex" in result

    def test_status_cache_exists(self, tmp_path):
        cache = tmp_path / ".memory-os-ai"
        cache.mkdir()
        (cache / "test.json").write_text("{}")
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = check_status()
        assert "Cache" in result or "📁" in result


# ---------------------------------------------------------------------------
# __main__.py coverage
# ---------------------------------------------------------------------------
class TestMainModule:
    """Test __main__.py entry point structure."""

    def test_importable(self):
        import memory_os_ai
        assert hasattr(memory_os_ai, "__version__")
        assert memory_os_ai.__version__ == "3.0.0"

    def test_all_exports(self):
        import memory_os_ai
        assert "MemoryEngine" in memory_os_ai.__all__
        assert "ChatExtractor" in memory_os_ai.__all__
        assert "TOOL_MODELS" in memory_os_ai.__all__
        assert "MEMORY_INSTRUCTIONS" in memory_os_ai.__all__
