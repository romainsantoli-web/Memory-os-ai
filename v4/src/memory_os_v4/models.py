"""Pydantic input models for the MCP tool surface (one per tool).

Field descriptions double as the JSON-schema descriptions an LLM client reads
when deciding how to call a tool, so they are written for that audience.
``extra="forbid"`` everywhere so a typo'd argument fails validation instead of
being silently ignored.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

_PROJECT_DESC = (
    "Project label to scope this call to; defaults to MEMORY_V4_PROJECT or 'default'. "
    "'*' reads across all projects (writes then go to 'default')."
)
_ISO_DESC_SUFFIX = " ISO date ('2026-09-12') or datetime, with or without timezone; naive times are local."
_KEYWORDS_DESC = (
    "3-8 lowercase topical keywords (concrete nouns, activities, places, names) describing this "
    "content. Indexed for dense/BM25 recall alongside the text but never shown or appended to it."
)
_QUERIES_DESC = (
    "2-4 short, keyword-rich sub-questions that decompose this query -- name the person/entity for "
    "each piece of evidence needed, one sub-query per part of a comparison/counting/multi-part "
    "question. Each is searched independently and fused with the main query via RRF."
)
_HYPOTHESES_DESC = (
    "1-2 hypothetical sentences the user might have written that would directly answer this query "
    "(HyDE-style). Searched and fused alongside the query and any sub-queries."
)
_ExtraQuery = Annotated[str, Field(max_length=300)]
_FACETS_DESC = (
    "0-8 (subject, predicate, object) facts about this content -- predicate MUST be one of the "
    "closed-vocabulary predicates (see memory_os_v4.facets.PREDICATES / "
    "bench/data/enrich/PREDICATES.md); anything else normalizes to 'other'. Indexed for exact "
    "(subject, predicate) lookup; only fused into ranking when the server's facet channel is "
    "enabled, otherwise stored but unused."
)
_QUERY_FACETS_DESC = (
    "0-8 (subject, predicate) pairs this question is about -- no object, this is a query pattern "
    "not a value -- from the same closed predicate vocabulary as facts' facets. Only takes effect "
    "when the server's facet channel is enabled; otherwise ignored."
)


class FacetIn(BaseModel):
    """One (subject, predicate, object) fact-side facet (see PREDICATES.md)."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., min_length=1, max_length=80, description="Who/what this facet is about.")
    predicate: str = Field(
        ..., min_length=1, description="One of the closed-vocabulary predicates; anything else -> 'other'."
    )
    object: str | None = Field(None, max_length=200, description="The value, if any (e.g. the pet's name).")


class QueryFacetIn(BaseModel):
    """One (subject, predicate) query-side facet -- a pattern, not a value."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., min_length=1, max_length=80, description="Who/what the question is about.")
    predicate: str = Field(
        ..., min_length=1, description="One of the closed-vocabulary predicates; anything else -> 'other'."
    )


class FactIdentityIn(BaseModel):
    """Opaque, caller-declared subject/property identity; never name inference."""
    model_config = ConfigDict(extra="forbid")
    subject_id: str = Field(..., min_length=1, description="Stable unique subject id; distinguish homonyms.")
    predicate: str = Field(..., min_length=1, description="Property represented by this fact.")
    cardinality: Literal["single", "multi"] = Field(..., description="single permits temporal state replacement; multi preserves multiple values.")


class FactIn(BaseModel):
    """One atomic, self-contained fact/summary/procedure/belief to add (ADD-only)."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        min_length=1,
        description=(
            "Atomic, self-contained statement: one sentence, explicit subject, no pronouns. "
            "For corrections, provide replaces or a single-valued identity and validity date. "
            "Text similarity alone never supersedes another fact."
        ),
    )
    kind: Literal["fact", "summary", "procedure", "belief"] = Field(
        "fact",
        description=(
            "Item kind: 'fact' for a statement, 'procedure' for a lesson learned (from a success "
            "or a failure), 'summary' for a session/topic recap, 'belief' for a client-asserted "
            "inference."
        ),
    )
    entities: list[str] | None = Field(
        None,
        description=(
            "Specific entity names mentioned, kept specific (\"Nisha's dad\" not \"dad\"; \"NQ\" "
            "not \"index future\"). Auto-extracted from the text if omitted."
        ),
    )
    event_time: str | None = Field(
        None, description="When this became true." + _ISO_DESC_SUFFIX + " Defaults to now."
    )
    valid_from: str | None = Field(
        None, description="From when this is valid." + _ISO_DESC_SUFFIX + " Defaults to event_time."
    )
    valid_until: str | None = Field(
        None, description="When this stops being valid, if known." + _ISO_DESC_SUFFIX
    )
    importance: float = Field(0.5, ge=0.0, le=1.0, description="0-1 weight used for recall ranking.")
    tags: list[str] | None = Field(
        None, description="Freeform tags; use ['todo'] for a pending task the user or agent owes."
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="0-1 confidence; lower this instead of omitting an uncertain fact."
    )
    item_id: str | None = Field(
        None, description="Caller-supplied id for advanced/idempotent use; must be globally unique."
    )
    keywords: list[str] | None = Field(None, description=_KEYWORDS_DESC)
    facets: list[FacetIn] | None = Field(None, max_length=8, description=_FACETS_DESC)
    identity: FactIdentityIn | None = Field(None, description="Explicit subject/property identity for facts. Names or retrieval facets alone do not authorize replacement.")
    replaces: list[str] | None = Field(None, description="Ids of facts explicitly replaced in the same project; new validity must start later.")


class MessageIn(BaseModel):
    """One raw chat turn."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system", "doc"] = Field(..., description="Speaker role.")
    content: str = Field(..., min_length=1, description="Raw message text.")
    keywords: list[str] | None = Field(None, description=_KEYWORDS_DESC)
    facets: list[FacetIn] | None = Field(None, max_length=8, description=_FACETS_DESC)


class SessionBriefIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus_query: str | None = Field(
        None, description="Optional topic to additionally pull focused search hits for."
    )
    max_tokens: int = Field(
        3000, ge=200, le=20000, description="Approximate token budget for the rendered brief text."
    )
    project: str | None = Field(None, description=_PROJECT_DESC)


class MutationIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(
        None, min_length=1, max_length=200,
        description="Unique retry key within the resolved project/scope. Same operation and payload "
        "replay the completed result, including after restart. Conflicting reuse is rejected; "
        "an interrupted request has an explicit uncertain outcome and must be inspected.",
    )


class ChatSaveIn(MutationIn):
    model_config = ConfigDict(extra="forbid")

    messages: list[MessageIn] = Field(
        ..., min_length=1, description="Raw chat turns; one episode item is stored per message."
    )
    summary: str | None = Field(
        None,
        description="Memo-style summary for your future self, e.g. \"PR #5 merged — 14 tools, 95 tests\". "
        "Stored as a kind='summary' item (importance 0.8).",
    )
    project_label: str | None = Field(None, description=_PROJECT_DESC)
    facts: list[FactIn] | None = Field(
        None, description="Atomic ADD-only facts extracted from this exchange (see the fact-extraction rules)."
    )
    event_time: str | None = Field(
        None, description="When this exchange happened." + _ISO_DESC_SUFFIX + " Defaults to now."
    )


class AddFactsIn(MutationIn):
    model_config = ConfigDict(extra="forbid")

    facts: list[FactIn] = Field(..., min_length=1, description="Atomic ADD-only facts to add.")
    project: str | None = Field(None, description=_PROJECT_DESC)
    parent_ids: list[str] | None = Field(
        None, description="Provenance: ids of the episodes/items these facts were derived from."
    )


class SearchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Natural-language search query.")
    top_k: int = Field(10, ge=1, le=200, description="Maximum number of hits to return.")
    at_time: str | None = Field(
        None,
        description="For \"what was true on X\" queries: only items valid at this time are returned."
        + _ISO_DESC_SUFFIX,
    )
    kinds: list[str] | None = Field(
        None, description="Restrict results to these item kinds, e.g. ['fact', 'procedure']."
    )
    project: str | None = Field(None, description=_PROJECT_DESC)
    rerank: bool | None = Field(
        None, description="Force cross-encoder rerank on/off; omit to use the server's configured default."
    )
    queries: list[_ExtraQuery] | None = Field(None, max_length=8, description=_QUERIES_DESC)
    hypotheses: list[_ExtraQuery] | None = Field(None, max_length=8, description=_HYPOTHESES_DESC)
    facets: list[QueryFacetIn] | None = Field(None, max_length=8, description=_QUERY_FACETS_DESC)


class GetContextIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Natural-language query to build context for.")
    max_tokens: int = Field(1500, ge=200, le=20000, description="Approximate token budget for the context text.")
    at_time: str | None = Field(None, description="Only include items valid at this time." + _ISO_DESC_SUFFIX)
    project: str | None = Field(None, description=_PROJECT_DESC)
    queries: list[_ExtraQuery] | None = Field(None, max_length=8, description=_QUERIES_DESC)
    hypotheses: list[_ExtraQuery] | None = Field(None, max_length=8, description=_HYPOTHESES_DESC)
    facets: list[QueryFacetIn] | None = Field(None, max_length=8, description=_QUERY_FACETS_DESC)


class TimelineIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Entity or topic to trace the evolution of over time.")
    project: str | None = Field(None, description=_PROJECT_DESC)
    limit: int = Field(30, ge=1, le=200, description="Maximum number of facts/summaries to return.")


class BlockGetIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["project", "global"] = Field("project", description="Project-local by default; global explicitly shares across projects.")

    project: str | None = Field(None, description=_PROJECT_DESC)

    name: str | None = Field(
        None, description="Block name, e.g. 'user_profile', 'project_state', 'working_rules'; omit for all blocks."
    )


class BlockSetIn(MutationIn):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["project", "global"] = Field("project", description="Project-local by default; global explicitly shares across projects.")

    project: str | None = Field(None, description=_PROJECT_DESC)

    name: str = Field(..., min_length=1, description="Block name, e.g. 'user_profile', 'project_state'.")
    text: str = Field(..., description="New block text.")
    mode: Literal["replace", "append"] = Field(
        "replace", description="'replace' overwrites the block; 'append' adds to it. Prefer replace to avoid drift."
    )


class ConsolidateIn(MutationIn):
    model_config = ConfigDict(extra="forbid")

    project: str | None = Field(None, description=_PROJECT_DESC)

    dry_run: bool = Field(False, description="If true, report what would change without writing anything.")


class CompactIn(ConsolidateIn):
    """Alias of memory_consolidate."""


class ForgetIn(MutationIn):
    model_config = ConfigDict(extra="forbid")

    project: str | None = Field(None, description=_PROJECT_DESC)

    item_id: str | None = Field(None, description="Exact item id to soft-delete.")
    query: str | None = Field(
        None, description="Query whose single best match will be soft-deleted; requires confirm=true."
    )
    confirm: bool = Field(False, description="Must be true when forgetting by query (safety guard).")
    reason: str = Field(..., min_length=1, description="Why this item is being forgotten (logged server-side).")


class IngestIn(MutationIn):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ..., min_length=1, description="File or folder to ingest; must resolve under $HOME or MEMORY_WORKSPACE."
    )
    project: str | None = Field(None, description=_PROJECT_DESC)
    extensions: list[str] | None = Field(
        None,
        description="File extensions to include, e.g. ['.txt', '.md']; defaults to a safe text-file set. "
        "'.pdf' is supported if pymupdf is installed, otherwise skipped.",
    )


class MigrateV3In(MutationIn):
    model_config = ConfigDict(extra="forbid")

    log_path: str | None = Field(
        None,
        description="Path to the v3 _conversation_log.jsonl; defaults to "
        "~/.memory-os-ai/_conversation_log.jsonl. Read-only.",
    )


class StatusIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str | None = Field(None, description=_PROJECT_DESC)


class ListDocumentsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str | None = Field(None, description=_PROJECT_DESC)


class SearchOccurrencesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(..., min_length=1, description="Keyword or short phrase to count exact occurrences of.")
    project: str | None = Field(None, description=_PROJECT_DESC)
    limit: int = Field(20, ge=1, le=200, description="Maximum number of matching snippets to return.")
