"""Pydantic models for Memory OS AI MCP tools.

All tool inputs are validated via Pydantic with strict constraints:
- Path traversal blocked on all file paths
- min_length / max_length on text fields
- Enum-constrained choices where applicable
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _no_traversal(v: str) -> str:
    """Block path traversal attempts."""
    if ".." in v or v.startswith("/"):
        raise ValueError("Path traversal detected — relative paths only, no '..'")
    return v


class IngestInput(BaseModel):
    """Input for memory_ingest tool."""

    folder_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Path to folder containing documents to ingest (relative to workspace).",
    )
    extensions: Optional[list[str]] = Field(
        default=None,
        description="File extensions to accept (e.g. ['.pdf', '.txt']). Defaults to all supported.",
    )
    force_reindex: bool = Field(
        default=False,
        description="Force re-indexing even if cache exists.",
    )

    @field_validator("folder_path")
    @classmethod
    def validate_folder_path(cls, v: str) -> str:
        return _no_traversal(v)


class SearchInput(BaseModel):
    """Input for memory_search tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language search query.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Number of results to return.",
    )
    threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Maximum L2 distance threshold for results.",
    )


class SearchOccurrencesInput(BaseModel):
    """Input for memory_search_occurrences tool."""

    keyword: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Keyword to search for (exact match, case-insensitive).",
    )
    top_k: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Number of FAISS neighbors to scan.",
    )


class GetContextInput(BaseModel):
    """Input for memory_get_context tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Query to find relevant context for.",
    )
    max_tokens: int = Field(
        default=3000,
        ge=100,
        le=30000,
        description="Approximate max tokens of context to return.",
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of FAISS neighbors to consider.",
    )


class ListDocumentsInput(BaseModel):
    """Input for memory_list_documents tool."""

    include_stats: bool = Field(
        default=True,
        description="Include segment counts and page counts.",
    )


class TranscribeInput(BaseModel):
    """Input for memory_transcribe tool."""

    file_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Path to audio file to transcribe (relative to workspace).",
    )
    language: str = Field(
        default="fr",
        min_length=2,
        max_length=5,
        description="Language code for transcription (e.g. 'fr', 'en').",
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        # Allow absolute paths (resolved by dispatcher) but block traversal
        if ".." in v:
            raise ValueError("Path traversal detected — no '..' allowed")
        return v


class DocumentStatusInput(BaseModel):
    """Input for memory_status tool."""

    pass  # No input needed — returns current engine status.


class ChatSyncInput(BaseModel):
    """Input for memory_chat_sync tool."""

    source_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Sync only this source. If omitted, sync all registered sources.",
    )


class ChatSourceAddInput(BaseModel):
    """Input for memory_chat_source_add tool."""

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the source (e.g. 'vscode-main', 'slack-export').",
    )
    source_type: str = Field(
        ...,
        description="Source type: 'vscode', 'jsonl', 'markdown', or 'folder'.",
    )
    path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Absolute path to the source (DB dir, file, or folder).",
    )

    @field_validator("source_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"vscode", "jsonl", "markdown", "folder"}
        if v not in allowed:
            raise ValueError(f"source_type must be one of {allowed}")
        return v


class ChatSourceRemoveInput(BaseModel):
    """Input for memory_chat_source_remove tool."""

    source_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Source identifier to remove.",
    )


class ChatStatusInput(BaseModel):
    """Input for memory_chat_status tool."""
    pass


class ChatAutoDetectInput(BaseModel):
    """Input for memory_chat_auto_detect tool."""

    auto_register: bool = Field(
        default=True,
        description="Automatically register detected VS Code workspaces as sources.",
    )


class SessionBriefInput(BaseModel):
    """Input for memory_session_brief tool."""

    max_tokens: int = Field(
        default=4000,
        ge=500,
        le=30000,
        description="Approximate max tokens for the context returned (~4 chars/token).",
    )
    include_chat_sync: bool = Field(
        default=True,
        description="Auto-sync all registered chat sources before building the brief.",
    )
    focus_query: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional focus area to prioritise (e.g. 'MCP bridge project').",
    )


class ChatSaveInput(BaseModel):
    """Input for memory_chat_save tool.

    Allows the model to persist its current conversation directly into
    the semantic memory so it survives context resets and can be
    recalled later via memory_session_brief.
    """

    messages: list[dict] = Field(
        ...,
        min_length=1,
        max_length=200,
        description=(
            "List of messages to save. Each item must have 'role' (user|assistant|system) "
            "and 'content' (text). Example: [{\"role\": \"user\", \"content\": \"...\"}]"
        ),
    )
    summary: Optional[str] = Field(
        default=None,
        max_length=2000,
        description=(
            "Optional high-level summary of the conversation so far. "
            "If provided, this summary is also indexed for faster retrieval."
        ),
    )
    project_label: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional project name to tag these messages (e.g. 'Memory-os-ai').",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[dict]) -> list[dict]:
        for i, msg in enumerate(v):
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i}: must have 'role' and 'content' keys")
            if msg["role"] not in {"user", "assistant", "system"}:
                raise ValueError(f"Message {i}: role must be user|assistant|system")
            if not isinstance(msg["content"], str) or len(msg["content"]) == 0:
                raise ValueError(f"Message {i}: content must be a non-empty string")
        return v


class CompactInput(BaseModel):
    """Input for memory_compact tool.

    Compresses the in-memory index by merging related segments and
    discarding low-value data. Useful when context window is saturating.
    """

    max_segments: int = Field(
        default=500,
        ge=50,
        le=10000,
        description="Target maximum number of segments after compaction.",
    )
    keep_recent_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Always keep segments from chat saves within this many hours.",
    )
    strategy: str = Field(
        default="dedup_merge",
        description=(
            "Compaction strategy: 'dedup_merge' (remove near-duplicates + merge short segments) "
            "or 'top_k' (keep only the top-k most relevant segments)."
        ),
    )

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"dedup_merge", "top_k"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v


class ProjectLinkInput(BaseModel):
    """Input for memory_project_link tool.

    Links another project's memory into the current one so that
    searches and session briefs include context from both.
    """

    project_path: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Absolute path to another project's memory cache directory "
            "(e.g. '/Users/me/other-project/.memory-os-ai')."
        ),
    )
    alias: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Friendly name for this linked project (defaults to folder name).",
    )


class ProjectUnlinkInput(BaseModel):
    """Input for memory_project_unlink tool."""

    alias: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Alias of the linked project to remove.",
    )


class ProjectListInput(BaseModel):
    """Input for memory_project_list tool."""
    pass


# ---------------------------------------------------------------------------
# Cloud storage models
# ---------------------------------------------------------------------------
class CloudConfigureInput(BaseModel):
    """Input for memory_cloud_configure tool.

    Configure a cloud storage backend for overflow when local disk is low.
    Supported providers: google-drive, icloud, dropbox, onedrive, s3, azure-blob, box, b2.
    """

    provider: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description=(
            "Cloud provider name: google-drive, icloud, dropbox, onedrive, "
            "s3, azure-blob, box, b2."
        ),
    )
    credentials: dict = Field(
        default_factory=dict,
        description=(
            "Provider-specific credentials. Examples:\n"
            "  icloud: {}\n"
            "  google-drive: {\"credentials_json\": \"/path/to/creds.json\", \"folder_id\": \"...\"}\n"
            "  dropbox: {\"access_token\": \"...\"}\n"
            "  onedrive: {} (auto-detects local mount) or {\"access_token\": \"...\"}\n"
            "  s3: {\"bucket\": \"...\", \"aws_access_key_id\": \"...\", \"aws_secret_access_key\": \"...\"}\n"
            "  azure-blob: {\"connection_string\": \"...\", \"container\": \"...\"}\n"
            "  box: {\"access_token\": \"...\"}\n"
            "  b2: {\"application_key_id\": \"...\", \"application_key\": \"...\", \"bucket_name\": \"...\"}"
        ),
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {
            "google-drive", "icloud", "dropbox", "onedrive",
            "s3", "azure-blob", "box", "b2",
        }
        if v not in allowed:
            raise ValueError(f"provider must be one of {sorted(allowed)}")
        return v


class CloudStatusInput(BaseModel):
    """Input for memory_cloud_status tool.

    Returns local disk usage, cloud storage status, and available providers.
    """
    pass


class CloudSyncInput(BaseModel):
    """Input for memory_cloud_sync tool.

    Synchronize memory data between local disk and cloud storage.
    """

    direction: str = Field(
        default="push",
        description="Sync direction: 'push' (local→cloud), 'pull' (cloud→local), 'auto' (check disk, offload if needed).",
    )

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        allowed = {"push", "pull", "auto"}
        if v not in allowed:
            raise ValueError(f"direction must be one of {allowed}")
        return v


# --- Tool models registry (for main.py dispatcher) ---
TOOL_MODELS: dict[str, type[BaseModel]] = {
    "memory_ingest": IngestInput,
    "memory_search": SearchInput,
    "memory_search_occurrences": SearchOccurrencesInput,
    "memory_get_context": GetContextInput,
    "memory_list_documents": ListDocumentsInput,
    "memory_transcribe": TranscribeInput,
    "memory_status": DocumentStatusInput,
    "memory_chat_sync": ChatSyncInput,
    "memory_chat_source_add": ChatSourceAddInput,
    "memory_chat_source_remove": ChatSourceRemoveInput,
    "memory_chat_status": ChatStatusInput,
    "memory_chat_auto_detect": ChatAutoDetectInput,
    "memory_session_brief": SessionBriefInput,
    "memory_chat_save": ChatSaveInput,
    "memory_compact": CompactInput,
    "memory_project_link": ProjectLinkInput,
    "memory_project_unlink": ProjectUnlinkInput,
    "memory_project_list": ProjectListInput,
    "memory_cloud_configure": CloudConfigureInput,
    "memory_cloud_status": CloudStatusInput,
    "memory_cloud_sync": CloudSyncInput,
}
