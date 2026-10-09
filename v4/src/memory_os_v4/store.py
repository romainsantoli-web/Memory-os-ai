"""Data model and append-only persistence for Memory OS v4.

``items.jsonl`` is the source of truth: every mutation (create or update)
appends a *full* record; :meth:`Store.load` replays the log keeping the last
record per id. Embeddings are cached separately in ``embeddings.npy`` +
``ids.json`` and ``meta.json`` records the embedding model/dim/version.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("memory_os_v4")

VALID_KINDS = {"episode", "fact", "summary", "procedure", "doc", "belief"}


def text_hash(text: str) -> str:
    """sha1 of whitespace-collapsed, lowercased, accent-stripped text."""
    normalized = unicodedata.normalize("NFKD", text.strip().lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = " ".join(normalized.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


@dataclass
class MemoryItem:
    """A single memory record. See docs/ARCHITECTURE.md for field semantics."""

    id: str
    kind: str
    text: str
    role: str = "user"
    project: str = "default"
    source: str = "chat_save"
    doc_ref: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    event_time: float = field(default_factory=time.time)
    valid_from: float | None = None
    valid_until: float | None = None
    invalidated_by: str | None = None
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # Keywords (production port of the LoCoMo-validated bench mechanism,
    # 2026-09-14): distinct from `tags` (freeform, caller-set labels like
    # "todo") and from `entities` (heuristically extracted spans). Appended
    # to the INDEXED text (see memory.py `index_text`) for dense/BM25 recall
    # only -- `text` itself (display/context) is never touched. Backward
    # compatible: absent in old `items.jsonl` records, `from_dict` then
    # leaves this at its default (empty list).
    keywords: list[str] = field(default_factory=list)
    # Facets (production port of the LoCoMo-validated bench mechanism,
    # 2026-09-19 -- see facets.py and bench/retrievers.py's
    # `build_facet_index`/`rank_facet_pool`): closed-predicate-vocabulary
    # (subject, predicate, object) triples, already normalized by
    # `facets.normalize_facets` (see memory.py `_build_items`). Indexed by
    # `facets.FacetIndex` for exact (subject, predicate) lookup only --
    # NEVER appended to the indexed text (see memory.py `index_text`), so
    # dense/BM25 recall is completely unaffected by this field. Backward
    # compatible exactly like `keywords`: absent in old `items.jsonl`
    # records, `from_dict` then leaves this at its default (empty list).
    facets: list[dict] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: float | None = None
    hash: str = ""
    deleted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class Store:
    """Append-only JSONL log of :class:`MemoryItem` plus embedding cache."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.items_path = self.data_dir / "items.jsonl"
        self.meta_path = self.data_dir / "meta.json"
        self.emb_path = self.data_dir / "embeddings.npy"
        self.ids_path = self.data_dir / "ids.json"
        self.chunk_map_path = self.data_dir / "chunk_map.json"

        self.items: dict[str, MemoryItem] = {}
        self.meta: dict = {}
        self._dirty_access: set[str] = set()
        # E1 multi_vector: chunk_id -> owning item_id (only populated when
        # config.multi_vector is on; empty dict elsewhere, so vector ids are
        # always item ids and this map is simply never consulted).
        self.chunk_owner: dict[str, str] = {}

        self.load()
        self.load_chunk_map()

    # ------------------------------------------------------------------ log
    def load(self) -> None:
        """Replay ``items.jsonl`` (last record per id wins) and load meta."""
        self.items.clear()
        if self.items_path.exists():
            with self.items_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        item = MemoryItem.from_dict(rec)
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning("skipping malformed items.jsonl line: %s", exc)
                        continue
                    self.items[item.id] = item
        if self.meta_path.exists():
            try:
                self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.meta = {}
        else:
            self.meta = {}

    def append(self, item: MemoryItem) -> None:
        self.append_many([item])

    def append_many(self, items: list[MemoryItem]) -> None:
        if not items:
            return
        with self.items_path.open("a", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                self.items[item.id] = item

    def get(self, item_id: str) -> MemoryItem | None:
        return self.items.get(item_id)

    def all(self, include_deleted: bool = True) -> list[MemoryItem]:
        if include_deleted:
            return list(self.items.values())
        return [it for it in self.items.values() if not it.deleted]

    # ------------------------------------------------------------ access log
    def touch(self, item_id: str, t: float | None = None) -> None:
        """Record an access: increments access_count, sets last_accessed.

        Kept in-memory (dirty set) and only persisted on the next
        :meth:`flush_access` call, to avoid rewriting the log per search.
        """
        item = self.items.get(item_id)
        if item is None:
            return
        item.access_count += 1
        item.last_accessed = t if t is not None else time.time()
        self._dirty_access.add(item_id)

    def flush_access(self) -> None:
        dirty_items = [self.items[i] for i in self._dirty_access if i in self.items]
        self._dirty_access.clear()
        if dirty_items:
            self.append_many(dirty_items)

    # -------------------------------------------------------------- meta
    def save_meta(self, meta: dict) -> None:
        self.meta = meta
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------------------------------------------------------- embeddings
    def save_embeddings(self, ids: list[str], vecs: np.ndarray) -> None:
        np.save(self.emb_path, vecs.astype(np.float32))
        self.ids_path.write_text(json.dumps(ids), encoding="utf-8")

    def load_embeddings(self) -> tuple[list[str], np.ndarray] | tuple[None, None]:
        if not (self.emb_path.exists() and self.ids_path.exists()):
            return None, None
        try:
            vecs = np.load(self.emb_path)
            ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("failed to load embedding cache: %s", exc)
            return None, None
        return ids, vecs

    # ------------------------------------------------------- chunk ownership
    def save_chunk_map(self, chunk_owner: dict[str, str] | None = None) -> None:
        """Persist chunk_id -> item_id (E1 multi_vector). No-op file (``{}``)
        when multi_vector is off, so single-vector-per-item stores are
        unaffected and this file is simply absent/empty for them."""
        if chunk_owner is not None:
            self.chunk_owner = chunk_owner
        self.chunk_map_path.write_text(
            json.dumps(self.chunk_owner, ensure_ascii=False), encoding="utf-8"
        )

    def load_chunk_map(self) -> None:
        if not self.chunk_map_path.exists():
            self.chunk_owner = {}
            return
        try:
            self.chunk_owner = json.loads(self.chunk_map_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("failed to load chunk map: %s", exc)
            self.chunk_owner = {}
