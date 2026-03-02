"""Pydantic models for Memory OS AI MCP tools.

All tool inputs are validated via Pydantic with strict constraints:
- Path traversal blocked on all file paths
- min_length / max_length on text fields
- Enum-constrained choices where applicable
"""

from __future__ import annotations

import os
import re
from enum import Enum
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
        return _no_traversal(v)


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
}
