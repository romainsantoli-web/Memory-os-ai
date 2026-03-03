"""Tests for gap-closing features: compact, cross-project, resources, AGENTS.md."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------
from memory_os_ai.models import (
    TOOL_MODELS,
    CompactInput,
    ProjectLinkInput,
    ProjectUnlinkInput,
    ProjectListInput,
)


class TestCompactModel:
    """Validate CompactInput Pydantic model."""

    def test_defaults(self):
        m = CompactInput()
        assert m.max_segments == 500
        assert m.keep_recent_hours == 24
        assert m.strategy == "dedup_merge"

    def test_custom_values(self):
        m = CompactInput(max_segments=100, keep_recent_hours=48, strategy="top_k")
        assert m.max_segments == 100
        assert m.strategy == "top_k"

    def test_invalid_strategy(self):
        with pytest.raises(Exception):
            CompactInput(strategy="invalid")

    def test_max_segments_too_low(self):
        with pytest.raises(Exception):
            CompactInput(max_segments=10)

    def test_max_segments_too_high(self):
        with pytest.raises(Exception):
            CompactInput(max_segments=99999)

    def test_keep_recent_hours_too_low(self):
        with pytest.raises(Exception):
            CompactInput(keep_recent_hours=0)


class TestProjectLinkModel:
    """Validate ProjectLinkInput Pydantic model."""

    def test_valid(self):
        m = ProjectLinkInput(project_path="/Users/me/.memory-os-ai")
        assert m.project_path == "/Users/me/.memory-os-ai"
        assert m.alias is None

    def test_with_alias(self):
        m = ProjectLinkInput(project_path="/tmp/proj", alias="other-proj")
        assert m.alias == "other-proj"

    def test_empty_path_rejected(self):
        with pytest.raises(Exception):
            ProjectLinkInput(project_path="")

    def test_path_too_long(self):
        with pytest.raises(Exception):
            ProjectLinkInput(project_path="x" * 1001)


class TestProjectUnlinkModel:
    """Validate ProjectUnlinkInput Pydantic model."""

    def test_valid(self):
        m = ProjectUnlinkInput(alias="my-project")
        assert m.alias == "my-project"

    def test_empty_alias_rejected(self):
        with pytest.raises(Exception):
            ProjectUnlinkInput(alias="")


class TestProjectListModel:
    """Validate ProjectListInput Pydantic model."""

    def test_no_args(self):
        m = ProjectListInput()
        assert m is not None


class TestToolModelsCount:
    """Verify TOOL_MODELS includes new tools."""

    def test_registry_has_21(self):
        assert len(TOOL_MODELS) == 21

    def test_new_tools_registered(self):
        assert "memory_compact" in TOOL_MODELS
        assert "memory_project_link" in TOOL_MODELS
        assert "memory_project_unlink" in TOOL_MODELS
        assert "memory_project_list" in TOOL_MODELS


# ---------------------------------------------------------------------------
# Engine compact tests
# ---------------------------------------------------------------------------
from memory_os_ai.engine import MemoryEngine


class TestCompactEngine:
    """Test the compact method on MemoryEngine."""

    def _make_engine(self) -> MemoryEngine:
        eng = MemoryEngine(model_name="all-MiniLM-L6-v2")
        return eng

    def test_compact_empty(self):
        eng = self._make_engine()
        result = eng.compact()
        assert result["ok"] is True
        assert result["before"] == 0
        assert result["removed"] == 0

    def test_compact_under_target(self):
        eng = self._make_engine()
        # Ingest a few segments (under default 500)
        eng.ingest_segments(["hello world", "foo bar baz"], source_label="test")
        result = eng.compact(max_segments=500)
        assert result["ok"] is True
        assert result["removed"] == 0
        assert "Already under target" in result.get("note", "")

    def test_compact_dedup_merge(self):
        eng = self._make_engine()
        # Create many near-duplicate segments
        segments = [f"the quick brown fox jumps over the lazy dog segment {i}" for i in range(100)]
        # Add exact duplicates
        segments += ["the quick brown fox jumps over the lazy dog segment 0"] * 50
        eng.ingest_segments(segments, source_label="test")
        assert eng.segment_count == 150

        result = eng.compact(max_segments=50, strategy="dedup_merge")
        assert result["ok"] is True
        assert result["after"] <= 50
        assert result["removed"] > 0
        assert eng.segment_count <= 50

    def test_compact_top_k(self):
        eng = self._make_engine()
        segments = [f"unique content number {i} about topic {i % 5}" for i in range(100)]
        eng.ingest_segments(segments, source_label="test")

        result = eng.compact(max_segments=30, strategy="top_k")
        assert result["ok"] is True
        assert result["after"] == 30
        assert result["strategy"] == "top_k"

    def test_get_segment_text(self):
        eng = self._make_engine()
        eng.ingest_segments(["hello world", "foo bar"], source_label="test")
        # Find the doc name (it's timestamped)
        doc_names = list(eng._documents.keys())
        assert len(doc_names) >= 1
        text = eng.get_segment_text(doc_names[0])
        assert text is not None
        assert "hello world" in text

    def test_get_segment_text_missing(self):
        eng = self._make_engine()
        assert eng.get_segment_text("nonexistent") is None


# ---------------------------------------------------------------------------
# Cross-project linking tests (server-level)
# ---------------------------------------------------------------------------
class TestProjectLinking:
    """Test cross-project link/unlink/list functionality."""

    def test_link_unlink_lifecycle(self):
        from memory_os_ai.server import _linked_projects

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate a project dir
            alias = "_test_link"
            _linked_projects[alias] = {"path": tmpdir, "engine": None}
            assert alias in _linked_projects

            # Unlink
            del _linked_projects[alias]
            assert alias not in _linked_projects

    def test_linked_engine_lazy_init(self):
        from memory_os_ai.server import _linked_projects, _get_linked_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            alias = "_test_lazy"
            _linked_projects[alias] = {"path": tmpdir, "engine": None}
            try:
                eng = _get_linked_engine(alias)
                assert eng is not None
                assert not eng.is_initialized  # no data to load
            finally:
                del _linked_projects[alias]


# ---------------------------------------------------------------------------
# MCP Resources tests
# ---------------------------------------------------------------------------
class TestResources:
    """Test MCP resource listing and reading."""

    @pytest.mark.asyncio
    async def test_list_resources_empty(self):
        from memory_os_ai.server import list_resources
        resources = await list_resources()
        # Should return at least 0 resources (might have conv log)
        assert isinstance(resources, list)

    @pytest.mark.asyncio
    async def test_read_resource_unknown(self):
        from memory_os_ai.server import read_resource
        result = await read_resource("memory://unknown/foo")
        assert len(result) == 1
        assert "Unknown resource" in result[0].text


# ---------------------------------------------------------------------------
# Bridge file tests
# ---------------------------------------------------------------------------
class TestAgentsMd:
    """Verify AGENTS.md for Codex exists and has correct content."""

    AGENTS_PATH = Path(__file__).parent.parent / "bridges" / "codex" / "AGENTS.md"

    def test_file_exists(self):
        assert self.AGENTS_PATH.exists()

    def test_mentions_session_brief(self):
        content = self.AGENTS_PATH.read_text()
        assert "memory_session_brief" in content

    def test_mentions_chat_save(self):
        content = self.AGENTS_PATH.read_text()
        assert "memory_chat_save" in content

    def test_mentions_compact(self):
        content = self.AGENTS_PATH.read_text()
        assert "memory_compact" in content

    def test_mentions_project_link(self):
        content = self.AGENTS_PATH.read_text()
        assert "memory_project_link" in content

    def test_mentions_18_tools(self):
        content = self.AGENTS_PATH.read_text()
        assert "18" in content

    def test_has_rules(self):
        content = self.AGENTS_PATH.read_text()
        assert "RULE #1" in content
        assert "Session Start" in content
        assert "Regular Saves" in content


class TestUpdatedBridges:
    """Verify existing bridges were updated."""

    def test_claude_code_md_mentions_18(self):
        path = Path(__file__).parent.parent / "bridges" / "claude-code" / "CLAUDE.md"
        content = path.read_text()
        assert "18" in content
        assert "memory_compact" in content
        assert "memory_project_link" in content

    def test_instructions_mentions_18(self):
        from memory_os_ai.instructions import MEMORY_INSTRUCTIONS
        assert "18" in MEMORY_INSTRUCTIONS
        assert "memory_compact" in MEMORY_INSTRUCTIONS
        assert "Rule 6" in MEMORY_INSTRUCTIONS
        assert "Rule 7" in MEMORY_INSTRUCTIONS
