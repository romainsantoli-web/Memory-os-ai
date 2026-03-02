"""Tests for Memory OS AI — models + engine + server."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Models tests
# ---------------------------------------------------------------------------
from memory_os_ai.models import (
    TOOL_MODELS,
    IngestInput,
    SearchInput,
    SearchOccurrencesInput,
    GetContextInput,
    ListDocumentsInput,
    TranscribeInput,
    DocumentStatusInput,
)


class TestModels:
    """Pydantic model validation tests."""

    def test_ingest_valid(self):
        inp = IngestInput(folder_path="pdfs")
        assert inp.folder_path == "pdfs"
        assert inp.extensions is None
        assert inp.force_reindex is False

    def test_ingest_with_extensions(self):
        inp = IngestInput(folder_path="docs", extensions=[".pdf", ".txt"])
        assert inp.extensions == [".pdf", ".txt"]

    def test_ingest_traversal_blocked(self):
        with pytest.raises(ValueError, match="traversal"):
            IngestInput(folder_path="../etc/passwd")

    def test_ingest_absolute_path_blocked(self):
        with pytest.raises(ValueError, match="traversal"):
            IngestInput(folder_path="/tmp/evil")

    def test_ingest_empty_path_rejected(self):
        with pytest.raises(ValueError):
            IngestInput(folder_path="")

    def test_search_valid(self):
        inp = SearchInput(query="quantum computing")
        assert inp.query == "quantum computing"
        assert inp.top_k == 10
        assert inp.threshold == 0.8

    def test_search_constraints(self):
        inp = SearchInput(query="x", top_k=500, threshold=2.0)
        assert inp.top_k == 500

    def test_search_empty_query_rejected(self):
        with pytest.raises(ValueError):
            SearchInput(query="")

    def test_search_top_k_too_high(self):
        with pytest.raises(ValueError):
            SearchInput(query="test", top_k=9999)

    def test_occurrences_valid(self):
        inp = SearchOccurrencesInput(keyword="python")
        assert inp.keyword == "python"
        assert inp.top_k == 500

    def test_occurrences_empty_keyword(self):
        with pytest.raises(ValueError):
            SearchOccurrencesInput(keyword="")

    def test_get_context_valid(self):
        inp = GetContextInput(query="résumé du rapport")
        assert inp.max_tokens == 3000
        assert inp.top_k == 50

    def test_get_context_limits(self):
        inp = GetContextInput(query="x", max_tokens=30000, top_k=500)
        assert inp.max_tokens == 30000

    def test_get_context_too_many_tokens(self):
        with pytest.raises(ValueError):
            GetContextInput(query="x", max_tokens=99999)

    def test_list_documents_defaults(self):
        inp = ListDocumentsInput()
        assert inp.include_stats is True

    def test_transcribe_valid(self):
        inp = TranscribeInput(file_path="audio/test.mp3")
        assert inp.language == "fr"

    def test_transcribe_traversal_blocked(self):
        with pytest.raises(ValueError, match="traversal"):
            TranscribeInput(file_path="../../../etc/passwd")

    def test_status_no_args(self):
        inp = DocumentStatusInput()
        assert inp is not None

    def test_tool_models_registry_complete(self):
        expected = {
            "memory_ingest",
            "memory_search",
            "memory_search_occurrences",
            "memory_get_context",
            "memory_list_documents",
            "memory_transcribe",
            "memory_status",
            "memory_chat_sync",
            "memory_chat_source_add",
            "memory_chat_source_remove",
            "memory_chat_status",
            "memory_chat_auto_detect",
            "memory_session_brief",
            "memory_chat_save",
            "memory_compact",
            "memory_project_link",
            "memory_project_unlink",
            "memory_project_list",
        }
        assert set(TOOL_MODELS.keys()) == expected


# ---------------------------------------------------------------------------
# SessionBriefInput model tests
# ---------------------------------------------------------------------------
from memory_os_ai.models import SessionBriefInput, ChatSaveInput


class TestChatSaveModel:
    """Pydantic validation tests for ChatSaveInput."""

    def test_valid_messages(self):
        inp = ChatSaveInput(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            summary="Greeting exchange",
            project_label="test-project",
        )
        assert len(inp.messages) == 2
        assert inp.summary == "Greeting exchange"
        assert inp.project_label == "test-project"

    def test_defaults(self):
        inp = ChatSaveInput(
            messages=[{"role": "user", "content": "test"}]
        )
        assert inp.summary is None
        assert inp.project_label is None

    def test_empty_messages_rejected(self):
        with pytest.raises(ValueError):
            ChatSaveInput(messages=[])

    def test_too_many_messages_rejected(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(201)]
        with pytest.raises(ValueError):
            ChatSaveInput(messages=msgs)

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError, match="role"):
            ChatSaveInput(
                messages=[{"role": "hacker", "content": "pwned"}]
            )

    def test_empty_content_rejected(self):
        with pytest.raises(ValueError, match="content"):
            ChatSaveInput(
                messages=[{"role": "user", "content": ""}]
            )

    def test_missing_content_rejected(self):
        with pytest.raises(ValueError, match="content"):
            ChatSaveInput(
                messages=[{"role": "user"}]
            )

    def test_missing_role_rejected(self):
        with pytest.raises(ValueError, match="role"):
            ChatSaveInput(
                messages=[{"content": "no role"}]
            )

    def test_summary_too_long_rejected(self):
        with pytest.raises(ValueError):
            ChatSaveInput(
                messages=[{"role": "user", "content": "x"}],
                summary="x" * 2001,
            )

    def test_project_label_too_long_rejected(self):
        with pytest.raises(ValueError):
            ChatSaveInput(
                messages=[{"role": "user", "content": "x"}],
                project_label="x" * 101,
            )

    def test_system_role_accepted(self):
        inp = ChatSaveInput(
            messages=[{"role": "system", "content": "You are helpful"}]
        )
        assert inp.messages[0]["role"] == "system"


class TestSessionBriefModel:
    """Pydantic validation tests for SessionBriefInput."""

    def test_defaults(self):
        inp = SessionBriefInput()
        assert inp.max_tokens == 4000
        assert inp.include_chat_sync is True
        assert inp.focus_query is None

    def test_custom_values(self):
        inp = SessionBriefInput(
            max_tokens=8000,
            include_chat_sync=False,
            focus_query="MCP bridge project",
        )
        assert inp.max_tokens == 8000
        assert inp.include_chat_sync is False
        assert inp.focus_query == "MCP bridge project"

    def test_max_tokens_too_low(self):
        with pytest.raises(ValueError):
            SessionBriefInput(max_tokens=10)

    def test_max_tokens_too_high(self):
        with pytest.raises(ValueError):
            SessionBriefInput(max_tokens=99999)

    def test_focus_query_too_long(self):
        with pytest.raises(ValueError):
            SessionBriefInput(focus_query="x" * 501)


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------
from memory_os_ai.engine import MemoryEngine, SUPPORTED_EXTENSIONS


class TestEngine:
    """Engine unit tests (using text files only — no heavy ML deps needed)."""

    @pytest.fixture
    def tmp_docs(self, tmp_path: Path) -> Path:
        """Create a temp folder with sample .txt files."""
        (tmp_path / "doc1.txt").write_text(
            "Python is a programming language. It was created by Guido van Rossum. "
            "Python is widely used in data science and artificial intelligence.",
            encoding="utf-8",
        )
        (tmp_path / "doc2.txt").write_text(
            "FAISS is a library for similarity search. It was developed by Facebook AI Research. "
            "FAISS supports both CPU and GPU indexing for fast nearest neighbor search.",
            encoding="utf-8",
        )
        (tmp_path / "doc3.txt").write_text(
            "Machine learning is a subset of artificial intelligence. "
            "Deep learning uses neural networks with many layers.",
            encoding="utf-8",
        )
        (tmp_path / "ignored.xyz").write_text("This should be ignored.")
        return tmp_path

    @pytest.fixture
    def engine(self) -> MemoryEngine:
        return MemoryEngine(segment_size=64, segment_overlap=16)

    def test_supported_extensions(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".mp3" in SUPPORTED_EXTENSIONS
        assert ".xyz" not in SUPPORTED_EXTENSIONS

    def test_ingest_success(self, engine: MemoryEngine, tmp_docs: Path):
        result = engine.ingest(str(tmp_docs), extensions={".txt"})
        assert result["ok"] is True
        assert result["files_indexed"] == 3
        assert result["total_segments"] > 0
        assert engine.is_initialized

    def test_ingest_nonexistent_folder(self, engine: MemoryEngine):
        result = engine.ingest("/nonexistent/path/xyz")
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    def test_ingest_ignores_unsupported(self, engine: MemoryEngine, tmp_docs: Path):
        result = engine.ingest(str(tmp_docs), extensions={".txt"})
        doc_names = list(result["documents"].keys())
        assert "ignored.xyz" not in doc_names

    def test_search_after_ingest(self, engine: MemoryEngine, tmp_docs: Path):
        engine.ingest(str(tmp_docs), extensions={".txt"})
        results = engine.search("programming language", top_k=5, threshold=1.5)
        assert len(results) > 0
        assert any("doc1.txt" in r.filename for r in results)

    def test_search_before_ingest_empty(self, engine: MemoryEngine):
        results = engine.search("test")
        assert results == []

    def test_search_occurrences(self, engine: MemoryEngine, tmp_docs: Path):
        engine.ingest(str(tmp_docs), extensions={".txt"})
        occ = engine.search_occurrences("Python")
        assert occ.total >= 1
        assert "doc1.txt" in occ.by_file

    def test_get_context(self, engine: MemoryEngine, tmp_docs: Path):
        engine.ingest(str(tmp_docs), extensions={".txt"})
        context = engine.get_context("artificial intelligence", max_chars=500)
        assert len(context) > 0
        assert len(context) <= 600  # small margin

    def test_list_documents(self, engine: MemoryEngine, tmp_docs: Path):
        engine.ingest(str(tmp_docs), extensions={".txt"})
        docs = engine.list_documents()
        assert len(docs) == 3
        assert all("filename" in d for d in docs)
        assert all("segments" in d for d in docs)

    def test_list_documents_no_stats(self, engine: MemoryEngine, tmp_docs: Path):
        engine.ingest(str(tmp_docs), extensions={".txt"})
        docs = engine.list_documents(include_stats=False)
        assert all("segments" not in d for d in docs)

    def test_status_before_init(self, engine: MemoryEngine):
        status = engine.status()
        assert status["initialized"] is False
        assert status["document_count"] == 0

    def test_status_after_init(self, engine: MemoryEngine, tmp_docs: Path):
        engine.ingest(str(tmp_docs), extensions={".txt"})
        status = engine.status()
        assert status["initialized"] is True
        assert status["document_count"] == 3
        assert status["segment_count"] > 0

    def test_transcribe_missing_file(self, engine: MemoryEngine):
        result = engine.transcribe("/nonexistent/audio.mp3")
        assert result["ok"] is False

    def test_transcribe_unsupported_format(self, engine: MemoryEngine, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("not audio")
        result = engine.transcribe(str(f))
        assert result["ok"] is False
        assert "unsupported" in result["error"].lower()

    def test_force_reindex(self, engine: MemoryEngine, tmp_docs: Path):
        r1 = engine.ingest(str(tmp_docs), extensions={".txt"})
        r2 = engine.ingest(str(tmp_docs), extensions={".txt"}, force_reindex=True)
        assert r1["ok"] is True
        assert r2["ok"] is True

    def test_segment_text(self, engine: MemoryEngine):
        text = "".join(chr(65 + (i % 26)) for i in range(200))  # ABCABC...
        segments = engine._segment_text(text)
        assert len(segments) > 1
        # Segments should overlap (non-identical chars prove it)
        assert segments[0] != segments[1]
        # First segment length should match segment_size
        assert len(segments[0]) == engine.segment_size

    def test_session_brief_empty(self, engine: MemoryEngine):
        """Session brief on an empty engine returns ok but no context."""
        brief = engine.session_brief()
        assert brief["ok"] is True
        assert brief["overview"]["total_documents"] == 0
        assert brief["context"] == ""
        assert brief["unique_segments_retrieved"] == 0

    def test_session_brief_after_ingest(self, engine: MemoryEngine, tmp_docs: Path):
        """Session brief after ingesting docs should return relevant context."""
        engine.ingest(str(tmp_docs), extensions={".txt"})
        brief = engine.session_brief()
        assert brief["ok"] is True
        assert brief["overview"]["total_documents"] == 3
        assert brief["context_chars"] > 0
        assert brief["unique_segments_retrieved"] > 0
        assert len(brief["queries_used"]) >= 6  # default queries
        assert brief["focus_query"] is None

    def test_session_brief_with_focus(self, engine: MemoryEngine, tmp_docs: Path):
        """Focus query should appear first in the queries list."""
        engine.ingest(str(tmp_docs), extensions={".txt"})
        brief = engine.session_brief(focus_query="FAISS library")
        assert brief["ok"] is True
        assert brief["focus_query"] == "FAISS library"
        assert brief["queries_used"][0] == "FAISS library"
        assert brief["context_chars"] > 0

    def test_session_brief_max_chars(self, engine: MemoryEngine, tmp_docs: Path):
        """Context should respect max_chars budget."""
        engine.ingest(str(tmp_docs), extensions={".txt"})
        brief = engine.session_brief(max_chars=200)
        assert brief["context_chars"] <= 250  # small margin for headers

    def test_session_brief_separates_chat_docs(self, engine: MemoryEngine, tmp_docs: Path):
        """Chat-injected docs should be counted separately."""
        engine.ingest(str(tmp_docs), extensions={".txt"})
        # Inject chat segments
        engine.ingest_segments(["chat message one", "chat message two"], source_label="test")
        brief = engine.session_brief()
        assert brief["overview"]["total_documents"] == 3
        assert brief["overview"]["total_chat_sources"] == 1
        assert brief["overview"]["total_chat_segments"] == 2
