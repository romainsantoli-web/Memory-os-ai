"""Shared data model and retriever protocol for the bench.

Exact shapes are defined in docs/ARCHITECTURE.md under "Bench retriever
protocol". Kept dependency-free (stdlib dataclasses + typing only) so that
every loader/retriever module can import it without pulling in numpy/torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class BenchItem:
    """One retrievable unit (a dialogue turn / haystack turn)."""

    id: str
    text: str
    speaker: str
    event_time: Optional[float]
    session_id: str
    meta: dict = field(default_factory=dict)


@dataclass
class BenchQuestion:
    """One evaluation question with its gold evidence item ids."""

    qid: str
    text: str
    evidence_ids: list[str]
    category: str
    query_time: Optional[float]
    meta: dict = field(default_factory=dict)


@dataclass
class BenchConversation:
    """One conversation (LoCoMo) or one per-question haystack (LongMemEval)."""

    conv_id: str
    items: list[BenchItem]
    questions: list[BenchQuestion]


@runtime_checkable
class Retriever(Protocol):
    """Protocol every bench retriever adapter must satisfy."""

    name: str

    def reset(self) -> None:
        """Drop all indexed state; called once per BenchConversation."""
        ...

    def add(self, items: list[BenchItem]) -> None:
        """Index a batch of items."""
        ...

    def search(self, query: str, top_k: int, at_time: Optional[float] = None) -> list[str]:
        """Return up to top_k item ids, best first."""
        ...
