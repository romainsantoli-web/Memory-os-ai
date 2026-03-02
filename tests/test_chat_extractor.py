"""Tests for the chat extractor module."""

import json
import os
import sqlite3
import tempfile

import pytest

from memory_os_ai.chat_extractor import (
    ChatExtractor,
    ChatMessage,
    SyncCursor,
    _load_state,
    _save_state,
    _sha256,
    extract_folder,
    extract_jsonl,
    extract_markdown,
    extract_vscode_copilot,
)
from memory_os_ai.models import (
    ChatAutoDetectInput,
    ChatSourceAddInput,
    ChatSourceRemoveInput,
    ChatStatusInput,
    ChatSyncInput,
    TOOL_MODELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp(tmp_path):
    return tmp_path


@pytest.fixture
def sample_jsonl(tmp):
    """Create a JSONL chat log."""
    p = tmp / "chat.jsonl"
    lines = [
        json.dumps({"role": "user", "content": "What is FAISS?", "timestamp": "2026-03-01T10:00:00Z"}),
        json.dumps({"role": "assistant", "content": "FAISS is a library for efficient similarity search.", "timestamp": "2026-03-01T10:00:01Z"}),
        json.dumps({"role": "user", "content": "How does it compare to Annoy?"}),
    ]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


@pytest.fixture
def sample_markdown(tmp):
    """Create a Markdown chat export."""
    p = tmp / "chat.md"
    p.write_text(
        "## User\nWhat is vector search?\n\n"
        "## Assistant\nVector search uses embeddings to find similar items.\n\n"
        "## Human\nCan you give an example?\n\n"
        "## AI\nSure, FAISS indexes vectors and returns nearest neighbors.\n"
    )
    return str(p)


@pytest.fixture
def sample_folder(tmp):
    """Create a folder with mixed chat files."""
    d = tmp / "chats"
    d.mkdir()
    (d / "log1.jsonl").write_text(
        json.dumps({"role": "user", "content": "Hello from JSONL"}) + "\n"
    )
    (d / "export.md").write_text("## User\nHello from Markdown\n\n## Assistant\nHi there!\n")
    (d / "notes.txt").write_text("Plain text note about memory systems.")
    (d / "image.png").write_bytes(b"\x89PNG")  # should be ignored
    return str(d)


@pytest.fixture
def mock_vscode_db(tmp):
    """Create a mock VS Code state.vscdb."""
    ws = tmp / "workspace"
    ws.mkdir()
    db_path = ws / "state.vscdb"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE ItemTable (key TEXT PRIMARY KEY, value TEXT)")

    session_id = "test-session-001"
    index_data = {
        "version": 1,
        "entries": {
            session_id: {
                "sessionId": session_id,
                "title": "Test Chat",
                "lastMessageDate": 1772435397364,
            }
        },
    }
    conn.execute(
        "INSERT INTO ItemTable VALUES (?, ?)",
        ("chat.ChatSessionStore.index", json.dumps(index_data)),
    )

    session_data = {
        "requests": [
            {
                "message": {"text": "What is MCP?"},
                "timestamp": 1772435397000,
                "response": {
                    "value": [
                        {"value": "MCP is the Model Context Protocol."},
                    ]
                },
            },
            {
                "message": {"text": "How do I use it?"},
                "timestamp": 1772435398000,
                "response": {
                    "value": "You configure it in .vscode/mcp.json.",
                },
            },
        ]
    }
    conn.execute(
        "INSERT INTO ItemTable VALUES (?, ?)",
        (f"chat.ChatSessionStore.session.{session_id}", json.dumps(session_data)),
    )
    conn.commit()
    conn.close()
    return str(ws)


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_and_load(self, tmp):
        path = str(tmp / "state.json")
        cursors = {
            "src1": SyncCursor(
                source_id="src1",
                last_sync_epoch=1000.0,
                file_hashes={"a.txt": "abc123"},
                message_count=5,
            ),
        }
        _save_state(path, cursors)
        loaded = _load_state(path)
        assert "src1" in loaded
        assert loaded["src1"].last_sync_epoch == 1000.0
        assert loaded["src1"].file_hashes == {"a.txt": "abc123"}
        assert loaded["src1"].message_count == 5

    def test_load_missing_file(self, tmp):
        result = _load_state(str(tmp / "nonexistent.json"))
        assert result == {}

    def test_load_corrupted_file(self, tmp):
        path = tmp / "bad.json"
        path.write_text("not json{{{")
        assert _load_state(str(path)) == {}

    def test_sha256(self, tmp):
        p = tmp / "test.txt"
        p.write_text("hello world")
        h = _sha256(str(p))
        assert len(h) == 64
        assert h == _sha256(str(p))  # deterministic


# ---------------------------------------------------------------------------
# JSONL extractor tests
# ---------------------------------------------------------------------------

class TestJsonlExtractor:
    def test_extract_all(self, sample_jsonl):
        cursor = SyncCursor(source_id="test-jsonl")
        msgs, cursor = extract_jsonl(sample_jsonl, cursor)
        assert len(msgs) == 3
        assert msgs[0].role == "user"
        assert msgs[0].content == "What is FAISS?"
        assert msgs[1].role == "assistant"
        assert msgs[2].content == "How does it compare to Annoy?"
        assert cursor.message_count == 3

    def test_incremental(self, sample_jsonl):
        cursor = SyncCursor(source_id="test-jsonl")
        msgs1, cursor = extract_jsonl(sample_jsonl, cursor)
        assert len(msgs1) == 3

        # No new data → no new messages
        msgs2, cursor = extract_jsonl(sample_jsonl, cursor)
        assert len(msgs2) == 0

        # Append a line → incremental extraction
        with open(sample_jsonl, "a") as f:
            f.write(json.dumps({"role": "user", "content": "New question"}) + "\n")
        msgs3, cursor = extract_jsonl(sample_jsonl, cursor)
        assert len(msgs3) == 1
        assert msgs3[0].content == "New question"
        assert cursor.message_count == 4

    def test_missing_file(self):
        cursor = SyncCursor(source_id="x")
        msgs, cursor = extract_jsonl("/nonexistent/chat.jsonl", cursor)
        assert msgs == []

    def test_malformed_lines(self, tmp):
        p = tmp / "bad.jsonl"
        p.write_text("not json\n{\"role\": \"user\", \"content\": \"ok\"}\n")
        cursor = SyncCursor(source_id="bad")
        msgs, cursor = extract_jsonl(str(p), cursor)
        assert len(msgs) == 1
        assert msgs[0].content == "ok"


# ---------------------------------------------------------------------------
# Markdown extractor tests
# ---------------------------------------------------------------------------

class TestMarkdownExtractor:
    def test_extract_all(self, sample_markdown):
        cursor = SyncCursor(source_id="test-md")
        msgs, cursor = extract_markdown(sample_markdown, cursor)
        assert len(msgs) == 4
        assert msgs[0].role == "user"
        assert "vector search" in msgs[0].content
        assert msgs[1].role == "assistant"
        # "Human" mapped to "user"
        assert msgs[2].role == "user"
        # "AI" mapped to "assistant"
        assert msgs[3].role == "assistant"

    def test_no_change_skips(self, sample_markdown):
        cursor = SyncCursor(source_id="test-md")
        msgs1, cursor = extract_markdown(sample_markdown, cursor)
        assert len(msgs1) == 4
        msgs2, cursor = extract_markdown(sample_markdown, cursor)
        assert len(msgs2) == 0  # hash unchanged

    def test_missing_file(self):
        cursor = SyncCursor(source_id="x")
        msgs, cursor = extract_markdown("/nope.md", cursor)
        assert msgs == []


# ---------------------------------------------------------------------------
# VS Code Copilot extractor tests
# ---------------------------------------------------------------------------

class TestVSCodeExtractor:
    def test_extract(self, mock_vscode_db):
        cursor = SyncCursor(source_id="vscode-test")
        msgs, cursor = extract_vscode_copilot(mock_vscode_db, cursor)
        assert len(msgs) >= 4  # 2 user + 2 assistant
        roles = [m.role for m in msgs]
        assert "user" in roles
        assert "assistant" in roles
        assert any("MCP" in m.content for m in msgs)
        assert cursor.file_hashes.get("state.vscdb")

    def test_no_change_skips(self, mock_vscode_db):
        cursor = SyncCursor(source_id="vscode-test")
        msgs1, cursor = extract_vscode_copilot(mock_vscode_db, cursor)
        assert len(msgs1) >= 4
        msgs2, cursor = extract_vscode_copilot(mock_vscode_db, cursor)
        assert len(msgs2) == 0  # hash unchanged

    def test_missing_db(self, tmp):
        cursor = SyncCursor(source_id="x")
        msgs, cursor = extract_vscode_copilot(str(tmp / "nope"), cursor)
        assert msgs == []


# ---------------------------------------------------------------------------
# Folder extractor tests
# ---------------------------------------------------------------------------

class TestFolderExtractor:
    def test_extract_mixed(self, sample_folder):
        cursor = SyncCursor(source_id="folder-test")
        msgs, cursor = extract_folder(sample_folder, cursor)
        # 1 from JSONL + 2 from MD + 1 from TXT = 4
        assert len(msgs) == 4
        sources = {m.source for m in msgs}
        assert any("jsonl" in s for s in sources)
        assert any("markdown" in s for s in sources)
        assert any("folder" in s for s in sources)

    def test_ignores_unsupported(self, sample_folder):
        cursor = SyncCursor(source_id="folder-test")
        msgs, cursor = extract_folder(sample_folder, cursor)
        # .png should be ignored
        assert not any("png" in m.source.lower() for m in msgs)

    def test_missing_folder(self):
        cursor = SyncCursor(source_id="x")
        msgs, cursor = extract_folder("/nonexistent_dir", cursor)
        assert msgs == []


# ---------------------------------------------------------------------------
# ChatExtractor orchestrator tests
# ---------------------------------------------------------------------------

class TestChatExtractor:
    def test_add_and_list_sources(self, tmp):
        ext = ChatExtractor(state_dir=str(tmp))
        ext.add_source("s1", "jsonl", "/tmp/chat.jsonl")
        ext.add_source("s2", "folder", "/tmp/chats")
        sources = ext.list_sources()
        assert len(sources) == 2
        assert sources[0]["source_id"] == "s1"
        assert sources[1]["type"] == "folder"

    def test_remove_source(self, tmp):
        ext = ChatExtractor(state_dir=str(tmp))
        ext.add_source("s1", "jsonl", "/tmp/chat.jsonl")
        assert ext.remove_source("s1") is True
        assert ext.remove_source("nonexistent") is False
        assert len(ext.list_sources()) == 0

    def test_sync_jsonl(self, tmp, sample_jsonl):
        ext = ChatExtractor(state_dir=str(tmp))
        ext.add_source("log", "jsonl", sample_jsonl)
        result = ext.sync()
        assert result["ok"] is True
        assert result["total_new_messages"] == 3
        assert result["per_source"]["log"] == 3

        # Incremental — no new data
        result2 = ext.sync()
        assert result2["total_new_messages"] == 0

    def test_sync_specific_source(self, tmp, sample_jsonl, sample_markdown):
        ext = ChatExtractor(state_dir=str(tmp))
        ext.add_source("log", "jsonl", sample_jsonl)
        ext.add_source("md", "markdown", sample_markdown)
        result = ext.sync(source_id="log")
        assert result["per_source"].get("log") == 3
        assert "md" not in result["per_source"]

    def test_sync_unknown_source(self, tmp):
        ext = ChatExtractor(state_dir=str(tmp))
        result = ext.sync(source_id="nonexistent")
        assert result["total_new_messages"] == 0

    def test_messages_to_segments(self, tmp):
        ext = ChatExtractor(state_dir=str(tmp))
        msgs = [
            ChatMessage(role="user", content="Hello", source="test", timestamp="2026-03-01"),
            ChatMessage(role="assistant", content="Hi there", source="test"),
        ]
        segments = ext.messages_to_segments(msgs)
        assert len(segments) == 2
        assert "[test]" in segments[0]
        assert "[user]" in segments[0]
        assert "Hello" in segments[0]

    def test_messages_to_segments_no_metadata(self, tmp):
        ext = ChatExtractor(state_dir=str(tmp))
        msgs = [ChatMessage(role="user", content="Hello")]
        segments = ext.messages_to_segments(msgs, include_role=False, include_source=False)
        assert segments == ["Hello"]

    def test_status(self, tmp, sample_jsonl):
        ext = ChatExtractor(state_dir=str(tmp))
        ext.add_source("log", "jsonl", sample_jsonl)
        status = ext.status()
        assert status["total_sources"] == 1
        assert len(status["sources"]) == 1
        assert status["sources"][0]["last_sync"] == "never"

    def test_state_persisted_across_instances(self, tmp, sample_jsonl):
        ext1 = ChatExtractor(state_dir=str(tmp))
        ext1.add_source("log", "jsonl", sample_jsonl)
        ext1.sync()

        # New instance reads persisted state
        ext2 = ChatExtractor(state_dir=str(tmp))
        ext2.add_source("log", "jsonl", sample_jsonl)
        result = ext2.sync()
        assert result["total_new_messages"] == 0  # already synced

    def test_sync_vscode(self, tmp, mock_vscode_db):
        ext = ChatExtractor(state_dir=str(tmp))
        ext.add_source("vsc", "vscode", mock_vscode_db)
        result = ext.sync()
        assert result["ok"] is True
        assert result["total_new_messages"] >= 4

    def test_sync_folder(self, tmp, sample_folder):
        ext = ChatExtractor(state_dir=str(tmp))
        ext.add_source("dir", "folder", sample_folder)
        result = ext.sync()
        assert result["ok"] is True
        assert result["total_new_messages"] == 4

    def test_sync_bad_type(self, tmp):
        ext = ChatExtractor(state_dir=str(tmp))
        ext._sources["bad"] = {"type": "unknown_type", "path": "/tmp"}
        ext._cursors["bad"] = SyncCursor(source_id="bad")
        result = ext.sync()
        assert "bad" in (result.get("errors") or {})


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------

class TestChatModels:
    def test_chat_sync_optional_source(self):
        m = ChatSyncInput()
        assert m.source_id is None

    def test_chat_sync_with_source(self):
        m = ChatSyncInput(source_id="vscode")
        assert m.source_id == "vscode"

    def test_chat_source_add_valid(self):
        m = ChatSourceAddInput(
            source_id="my-log",
            source_type="jsonl",
            path="/tmp/chat.jsonl",
        )
        assert m.source_type == "jsonl"

    def test_chat_source_add_bad_type(self):
        with pytest.raises(Exception):
            ChatSourceAddInput(
                source_id="x",
                source_type="invalid",
                path="/tmp/x",
            )

    def test_chat_source_add_empty_id(self):
        with pytest.raises(Exception):
            ChatSourceAddInput(source_id="", source_type="jsonl", path="/tmp/x")

    def test_chat_source_remove(self):
        m = ChatSourceRemoveInput(source_id="test")
        assert m.source_id == "test"

    def test_chat_status(self):
        m = ChatStatusInput()
        assert m is not None

    def test_chat_auto_detect_default(self):
        m = ChatAutoDetectInput()
        assert m.auto_register is True

    def test_chat_auto_detect_no_register(self):
        m = ChatAutoDetectInput(auto_register=False)
        assert m.auto_register is False

    def test_tool_models_registry(self):
        expected_new = {
            "memory_chat_sync",
            "memory_chat_source_add",
            "memory_chat_source_remove",
            "memory_chat_status",
            "memory_chat_auto_detect",
        }
        assert expected_new.issubset(TOOL_MODELS.keys())
        assert len(TOOL_MODELS) == 13  # 7 original + 5 chat + 1 session_brief
