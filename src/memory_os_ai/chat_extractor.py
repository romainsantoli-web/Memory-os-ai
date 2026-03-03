"""Chat history extractor for Memory OS AI.

Extracts conversations from multiple sources and incrementally feeds them
into the FAISS index so the memory stays perpetually up-to-date.

Supported sources:
  1. VS Code Copilot (state.vscdb SQLite + chat-session-resources/)
  2. JSONL chat logs   (one JSON object per line — role + content)
  3. Markdown exports  (## User / ## Assistant blocks)
  4. Generic folder    (watch any folder for new .txt/.md/.json files)

Incremental strategy:
  - A state file (_chat_sync_state.json) tracks per-source cursors:
    file hashes, last-modified timestamps, byte offsets.
  - On each sync only *new* or *changed* entries are extracted,
    segmented, embedded, and merged into the live FAISS index.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single chat message extracted from any source."""
    role: str                      # "user", "assistant", "system"
    content: str                    # raw text
    timestamp: Optional[str] = None # ISO8601 or epoch string
    source: str = ""                # source identifier
    session_id: str = ""            # session/conversation id
    metadata: dict = field(default_factory=dict)


@dataclass
class SyncCursor:
    """Tracks what has already been synced for one source."""
    source_id: str
    last_sync_epoch: float = 0.0
    file_hashes: dict[str, str] = field(default_factory=dict)   # path → sha256
    byte_offsets: dict[str, int] = field(default_factory=dict)   # path → offset
    message_count: int = 0


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

_DEFAULT_STATE_FILE = "_chat_sync_state.json"


def _load_state(state_path: str) -> dict[str, SyncCursor]:
    """Load sync cursors from disk."""
    if not os.path.isfile(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cursors: dict[str, SyncCursor] = {}
        for sid, data in raw.items():
            cursors[sid] = SyncCursor(
                source_id=sid,
                last_sync_epoch=data.get("last_sync_epoch", 0.0),
                file_hashes=data.get("file_hashes", {}),
                byte_offsets=data.get("byte_offsets", {}),
                message_count=data.get("message_count", 0),
            )
        return cursors
    except Exception:
        return {}


def _save_state(state_path: str, cursors: dict[str, SyncCursor]) -> None:
    """Atomically persist sync cursors."""
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                sid: {
                    "last_sync_epoch": c.last_sync_epoch,
                    "file_hashes": c.file_hashes,
                    "byte_offsets": c.byte_offsets,
                    "message_count": c.message_count,
                }
                for sid, c in cursors.items()
            },
            f,
            indent=2,
        )
    os.replace(tmp, state_path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Extractors — one per source type
# ---------------------------------------------------------------------------

def extract_vscode_copilot(
    workspace_storage_path: str,
    cursor: SyncCursor,
) -> tuple[list[ChatMessage], SyncCursor]:
    """Extract new messages from VS Code Copilot chat state.vscdb.

    VS Code stores chat data in a SQLite DB (state.vscdb) with keys like:
      - chat.ChatSessionStore.index → session list
      - chat.ChatSessionStore.session.<id> → full conversation JSON
    """
    messages: list[ChatMessage] = []
    db_path = os.path.join(workspace_storage_path, "state.vscdb")
    if not os.path.isfile(db_path):
        return messages, cursor

    current_hash = _sha256(db_path)
    if cursor.file_hashes.get("state.vscdb") == current_hash:
        return messages, cursor  # No changes

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        c = conn.cursor()

        # Get session index
        c.execute("SELECT value FROM ItemTable WHERE key = 'chat.ChatSessionStore.index'")
        row = c.fetchone()
        if not row:
            conn.close()
            return messages, cursor

        index_data = json.loads(row[0])
        entries = index_data.get("entries", {})

        # Read each session
        for session_id, meta in entries.items():
            session_key = f"chat.ChatSessionStore.session.{session_id}"
            c.execute("SELECT value FROM ItemTable WHERE key = ?", (session_key,))
            srow = c.fetchone()
            if not srow:
                continue
            try:
                session_data = json.loads(srow[0])
                requests = session_data.get("requests", [])
                for req in requests:
                    # User message
                    user_msg = req.get("message", {})
                    user_text = user_msg.get("text", "")
                    if user_text.strip():
                        ts = req.get("timestamp")
                        messages.append(ChatMessage(
                            role="user",
                            content=user_text.strip(),
                            timestamp=str(ts) if ts else None,
                            source="vscode-copilot",
                            session_id=session_id,
                            metadata={"title": meta.get("title", "")},
                        ))

                    # Assistant response
                    response = req.get("response", {})
                    resp_parts = response.get("value", [])
                    if isinstance(resp_parts, list):
                        for part in resp_parts:
                            if isinstance(part, dict):
                                resp_text = part.get("value", "")
                            elif isinstance(part, str):
                                resp_text = part
                            else:
                                continue
                            if resp_text.strip():
                                messages.append(ChatMessage(
                                    role="assistant",
                                    content=resp_text.strip(),
                                    timestamp=str(ts) if ts else None,
                                    source="vscode-copilot",
                                    session_id=session_id,
                                ))
                    elif isinstance(resp_parts, str) and resp_parts.strip():
                        messages.append(ChatMessage(
                            role="assistant",
                            content=resp_parts.strip(),
                            timestamp=str(ts) if ts else None,
                            source="vscode-copilot",
                            session_id=session_id,
                        ))
            except (json.JSONDecodeError, KeyError):
                continue

        conn.close()
    except Exception:
        pass

    cursor.file_hashes["state.vscdb"] = current_hash
    cursor.last_sync_epoch = time.time()
    cursor.message_count += len(messages)
    return messages, cursor


def extract_jsonl(
    file_path: str,
    cursor: SyncCursor,
) -> tuple[list[ChatMessage], SyncCursor]:
    """Extract messages from a JSONL chat log file (incremental via offset).

    Each line: {"role": "user|assistant", "content": "...", "timestamp": "..."}
    """
    messages: list[ChatMessage] = []
    if not os.path.isfile(file_path):
        return messages, cursor

    offset = cursor.byte_offsets.get(file_path, 0)
    file_size = os.path.getsize(file_path)
    if offset >= file_size:
        return messages, cursor

    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                messages.append(ChatMessage(
                    role=obj.get("role", "unknown"),
                    content=obj.get("content", ""),
                    timestamp=obj.get("timestamp"),
                    source=f"jsonl:{os.path.basename(file_path)}",
                    session_id=obj.get("session_id", ""),
                    metadata=obj.get("metadata", {}),
                ))
            except json.JSONDecodeError:
                continue
        new_offset = f.tell()

    cursor.byte_offsets[file_path] = new_offset
    cursor.last_sync_epoch = time.time()
    cursor.message_count += len(messages)
    return messages, cursor


def extract_markdown(
    file_path: str,
    cursor: SyncCursor,
) -> tuple[list[ChatMessage], SyncCursor]:
    """Extract messages from a Markdown chat export.

    Expected format:
      ## User
      message text

      ## Assistant
      response text
    """
    messages: list[ChatMessage] = []
    if not os.path.isfile(file_path):
        return messages, cursor

    current_hash = _sha256(file_path)
    if cursor.file_hashes.get(file_path) == current_hash:
        return messages, cursor

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by ## headings
    pattern = re.compile(
        r"^##\s+(User|Assistant|System|Human|AI)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    parts = pattern.split(content)
    # parts = [preamble, role1, text1, role2, text2, ...]
    for i in range(1, len(parts) - 1, 2):
        role_raw = parts[i].strip().lower()
        text = parts[i + 1].strip()
        role = {
            "human": "user",
            "ai": "assistant",
        }.get(role_raw, role_raw)
        if text:
            messages.append(ChatMessage(
                role=role,
                content=text,
                source=f"markdown:{os.path.basename(file_path)}",
            ))

    cursor.file_hashes[file_path] = current_hash
    cursor.last_sync_epoch = time.time()
    cursor.message_count += len(messages)
    return messages, cursor


def extract_folder(
    folder_path: str,
    cursor: SyncCursor,
    extensions: set[str] | None = None,
) -> tuple[list[ChatMessage], SyncCursor]:
    """Watch a folder for new/changed chat files (.jsonl, .md, .txt).

    Dispatches to the appropriate extractor based on extension.
    """
    if extensions is None:
        extensions = {".jsonl", ".md", ".txt", ".json"}

    messages: list[ChatMessage] = []
    if not os.path.isdir(folder_path):
        return messages, cursor

    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in extensions:
            continue
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue

        current_hash = _sha256(fpath)
        if cursor.file_hashes.get(fpath) == current_hash:
            continue  # unchanged

        if ext == ".jsonl":
            new_msgs, cursor = extract_jsonl(fpath, cursor)
            messages.extend(new_msgs)
        elif ext == ".md":
            new_msgs, cursor = extract_markdown(fpath, cursor)
            messages.extend(new_msgs)
        elif ext in {".txt", ".json"}:
            # Plain text — treat entire file as one message
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                messages.append(ChatMessage(
                    role="document",
                    content=text,
                    source=f"folder:{fname}",
                ))
            cursor.file_hashes[fpath] = current_hash

    cursor.last_sync_epoch = time.time()
    cursor.message_count += len(messages)
    return messages, cursor


# ---------------------------------------------------------------------------
# ChatExtractor — orchestrates all sources
# ---------------------------------------------------------------------------

class ChatExtractor:
    """Manages incremental chat extraction from multiple sources.

    On each `sync()` call:
      1. Check each registered source for new data
      2. Extract only new/changed messages
      3. Convert messages to text segments
      4. Feed segments into the MemoryEngine (append to FAISS index)
      5. Save cursors to disk
    """

    def __init__(
        self,
        state_dir: str = ".",
        state_file: str = _DEFAULT_STATE_FILE,
    ):
        self.state_path = os.path.join(state_dir, state_file)
        self._cursors = _load_state(self.state_path)
        self._sources: dict[str, dict[str, Any]] = {}

    def add_source(
        self,
        source_id: str,
        source_type: str,
        path: str,
        **kwargs: Any,
    ) -> None:
        """Register a chat source.

        Args:
            source_id: Unique identifier (e.g. "vscode", "slack-export")
            source_type: One of "vscode", "jsonl", "markdown", "folder"
            path: Path to the source (DB, file, or folder)
        """
        self._sources[source_id] = {
            "type": source_type,
            "path": path,
            **kwargs,
        }
        if source_id not in self._cursors:
            self._cursors[source_id] = SyncCursor(source_id=source_id)

    def remove_source(self, source_id: str) -> bool:
        """Unregister a chat source."""
        removed = source_id in self._sources
        self._sources.pop(source_id, None)
        self._cursors.pop(source_id, None)
        return removed

    def list_sources(self) -> list[dict[str, Any]]:
        """Return registered sources with their sync state."""
        result = []
        for sid, cfg in self._sources.items():
            cursor = self._cursors.get(sid, SyncCursor(source_id=sid))
            result.append({
                "source_id": sid,
                "type": cfg["type"],
                "path": cfg["path"],
                "last_sync": (
                    datetime.fromtimestamp(cursor.last_sync_epoch, tz=timezone.utc).isoformat()
                    if cursor.last_sync_epoch > 0
                    else "never"
                ),
                "message_count": cursor.message_count,
            })
        return result

    def sync(self, source_id: str | None = None) -> dict[str, Any]:
        """Run incremental extraction.

        Args:
            source_id: Sync only one source. If None, sync all.

        Returns:
            Summary dict with new messages per source.
        """
        targets = (
            {source_id: self._sources[source_id]}
            if source_id and source_id in self._sources
            else self._sources
        )

        all_messages: list[ChatMessage] = []
        per_source: dict[str, int] = {}
        errors: dict[str, str] = {}

        for sid, cfg in targets.items():
            cursor = self._cursors.get(sid, SyncCursor(source_id=sid))
            try:
                if cfg["type"] == "vscode":
                    msgs, cursor = extract_vscode_copilot(cfg["path"], cursor)
                elif cfg["type"] == "jsonl":
                    msgs, cursor = extract_jsonl(cfg["path"], cursor)
                elif cfg["type"] == "markdown":
                    msgs, cursor = extract_markdown(cfg["path"], cursor)
                elif cfg["type"] == "folder":
                    exts = cfg.get("extensions")
                    if exts:
                        exts = set(exts)
                    msgs, cursor = extract_folder(cfg["path"], cursor, extensions=exts)
                else:
                    errors[sid] = f"Unknown source type: {cfg['type']}"
                    continue

                self._cursors[sid] = cursor
                per_source[sid] = len(msgs)
                all_messages.extend(msgs)
            except Exception as e:
                errors[sid] = str(e)

        # Save state
        _save_state(self.state_path, self._cursors)

        return {
            "ok": True,
            "total_new_messages": len(all_messages),
            "per_source": per_source,
            "errors": errors if errors else None,
            "messages": all_messages,
        }

    def messages_to_segments(
        self,
        messages: list[ChatMessage],
        include_role: bool = True,
        include_source: bool = True,
    ) -> list[str]:
        """Convert chat messages to text segments for FAISS ingestion.

        Each message becomes one or more segments depending on length.
        """
        segments: list[str] = []
        for msg in messages:
            parts = []
            if include_source and msg.source:
                parts.append(f"[{msg.source}]")
            if include_role:
                parts.append(f"[{msg.role}]")
            if msg.timestamp:
                parts.append(f"({msg.timestamp})")
            parts.append(msg.content)
            text = " ".join(parts)
            segments.append(text)
        return segments

    def status(self) -> dict[str, Any]:
        """Return sync status for all sources."""
        return {
            "sources": self.list_sources(),
            "total_sources": len(self._sources),
            "state_file": self.state_path,
        }

    def auto_detect_vscode(self) -> list[str]:
        """Auto-detect VS Code workspace storage directories on macOS/Linux.

        Returns list of paths found.
        """
        found = []
        base_dirs = [
            os.path.expanduser("~/Library/Application Support/Code/User/workspaceStorage"),
            os.path.expanduser("~/.config/Code/User/workspaceStorage"),
            os.path.expanduser("~/.vscode-server/data/User/workspaceStorage"),
        ]
        for base in base_dirs:
            if not os.path.isdir(base):
                continue
            for entry in os.listdir(base):
                ws_path = os.path.join(base, entry)
                db_path = os.path.join(ws_path, "state.vscdb")
                if os.path.isfile(db_path):
                    found.append(ws_path)
        return found
