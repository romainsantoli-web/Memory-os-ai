"""``MemoryV4`` façade: the single entry point wiring store, embedder, vector
index, BM25, entities, retrieval, blocks, and consolidation together.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
import weakref
from functools import wraps
from pathlib import Path
from threading import RLock

import numpy as np

from . import consolidate as consolidate_mod
from .blocks import BlockStore
from .bm25 import BM25Index
from .config import Config
from .embed import Embedder, HashEmbedder
from .entities import EntityIndex, extract_entities
from .extract import normalize_client_facts
from .facets import FacetIndex, normalize_facets
from .fact_updates import duplicate_compatible, equivalent_assertions, validate_replacements, replacement_sweep
from .llm import LLMClient
from .idempotency import RequestJournal, sync_sources
from .writer_lock import WriterLock
from .migrate import read_v3_log
from .retrieval import HybridRetriever, SearchHit, mmr
from .store import MemoryItem, Store, text_hash
from .temporal import is_valid_at, recency_weight
from .text import chunk_text, contextual_header, date_str, tokenize
from .vector_index import VectorIndex

logger = logging.getLogger("memory_os_v4")

_PENDING_RE = re.compile(r"\bTODO\b|\ba faire\b|\bà faire\b|\bnext step\b|\bprochaine etape\b", re.IGNORECASE)


def _make_embedder(config: Config) -> Embedder | HashEmbedder:
    if config.embed_model == "hash":
        return HashEmbedder(dim=config.hash_dim)
    return Embedder(config.embed_model, config.device, config.embed_max_seq_length, config.embed_batch_size)


def index_text(item: MemoryItem, config: Config) -> str:
    """The ONE string every signal indexes `item` under: dense embedding,
    BM25 tokenization (production
    port of the LoCoMo-validated bench mechanism, 2026-09-14 -- see
    `bench/retrievers.py`'s `V4EnrichedRetriever` "facts+keywords" mode).

    ``contextual_header(item) + "\\n" + item.text``, plus
    ``" | keywords: k1, k2, ..."`` appended when `config.index_keywords` is
    on AND `item.keywords` is non-empty. `item.text` itself is never
    touched -- this only changes what gets embedded/tokenized, never what is
    displayed or returned to a caller. Duplicate decisions use normalized
    assertion equality, independently of these retrieval signals. A no-op (byte-identical
    to the pre-keywords header+text string) for any item with no keywords,
    so every existing caller is unaffected by default.
    """
    text = f"{contextual_header(item)}\n{item.text}"
    if config.index_keywords and item.keywords:
        text += " | keywords: " + ", ".join(item.keywords)
    return text


def _requires_open(method):
    """Hold the facade lock and reject stale handles after their lease was released."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            self._ensure_open()
            return method(self, *args, **kwargs)
    return guarded


class MemoryV4:
    """Local-first long-term memory engine façade. Thread-safe via a single RLock."""

    def __init__(self, data_dir: str | None = None, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        if data_dir is not None:
            self.config.data_dir = data_dir
        self._lock = RLock()

        self.data_dir = Path(self.config.data_dir).expanduser()
        if self.config.store_backend not in {"jsonl", "sqlite_s1"}:
            raise ValueError("Unknown store backend; choose jsonl or experimental sqlite_s1")
        if self.config.store_backend == "sqlite_s1":
            from .sqlite_store import validate_sqlite_target
            validate_sqlite_target(self.data_dir)
        elif (self.data_dir / "store.sqlite3").exists():
            raise ValueError("SQLite store present: explicit backend selection required; no implicit downgrade")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._closed = False
        self._writer_lock = WriterLock(self.data_dir)
        self._writer_finalizer = weakref.finalize(self, self._writer_lock.close)
        try:
            self.requests = RequestJournal(self.data_dir)
            if self.config.store_backend == "sqlite_s1":
                from .sqlite_store import SQLiteStore
                self.store = SQLiteStore(self.data_dir)
            else:
                self.store = Store(self.data_dir)
            self.embedder = _make_embedder(self.config)
            self.blocks = BlockStore(self.data_dir / "blocks.jsonl", self.config.block_char_limit)
            self.llm = LLMClient()

            self._bm25_path = self.data_dir / "bm25.json"
            self._entities_path = self.data_dir / "entities.json"
            # facet_channel (2026-09-19): derived state only, never persisted to
            # its own file (see facets.py's `FacetIndex` docstring) -- always
            # starts empty and is fully repopulated by `_reconcile_indexes()`
            # below from the live items' own (already-durable) `.facets`.
            self.facet_index = FacetIndex()

            self._init_indexes()

            self.retriever = HybridRetriever(
                self.store,
                self.embedder,
                self.vector_index,
                self.bm25,
                self.entities,
                self.config,
                facet_index=self.facet_index,
            )
            self.last_consolidated: float | None = self.store.meta.get("last_consolidated")
        except BaseException:
            self._closed = True
            try:
                if hasattr(self, "store"):
                    self.store.close()
            finally:
                self._writer_finalizer()
            raise

    def _ensure_open(self) -> None:
        self._writer_lock.assert_owner()
        if self._closed:
            raise RuntimeError("Memory engine is closed; open a new engine or use the shared HTTP server")

    def close(self) -> None:
        """Flush and release the lifetime writer lease, even when flushing fails."""
        with self._lock:
            if self._closed:
                return
            try:
                self.flush()
            finally:
                self._closed = True
                try:
                    self.store.close()
                finally:
                    self._writer_finalizer()

    def __enter__(self) -> "MemoryV4":
        self._ensure_open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def execute_request(self, operation: str, scope: str, payload: dict, request_id: str | None,
                        action) -> dict:
        """Serialize the whole MCP handler, including direct block/store access."""
        with self._lock:
            self._ensure_open()
            if request_id is None:
                return action()
            return self.requests.execute(scope, request_id, operation, payload, action,
                                         lambda: sync_sources(self.data_dir))

    # ------------------------------------------------------------- startup
    def _init_indexes(self) -> None:
        dim = self.embedder.dim
        stored_model = self.store.meta.get("embed_model")
        stored_dim = self.store.meta.get("dim")
        need_reembed = stored_model is not None and stored_model != self.config.embed_model
        if stored_dim is not None and stored_dim != dim:
            need_reembed = True

        self.bm25 = BM25Index.load(self._bm25_path) if self._bm25_path.exists() else BM25Index()
        self.entities = (
            EntityIndex.load(self._entities_path) if self._entities_path.exists() else EntityIndex()
        )

        self.vector_index = VectorIndex(dim)
        ids, vecs = self.store.load_embeddings()
        if ids is not None and vecs is not None:
            # A1 (adversarial review, 2026-09-12): a corrupt/stale embedding
            # cache -- id count != vector row count (e.g. a hand-edited
            # ids.json), or vector dim != the current embedder's dim -- used
            # to reach faiss's `add_with_ids` as a shape AssertionError,
            # bricking the store on every future load. Detect it here and
            # fall back to a full re-embed instead of crashing.
            if len(ids) != vecs.shape[0] or (vecs.shape[0] and vecs.shape[1] != dim):
                logger.warning(
                    "embedding cache corrupt (%d ids, vecs shape %s, expected dim %d): forcing full re-embed",
                    len(ids), tuple(vecs.shape), dim,
                )
                need_reembed = True
                ids, vecs = None, None

        if need_reembed or ids is None:
            self._reembed_all()
        elif ids:
            self.vector_index.add(ids, vecs)

        self._reconcile_indexes()

        self.store.meta["embed_model"] = self.config.embed_model
        self.store.meta["dim"] = dim
        self.store.meta["version"] = "4.0.1"

    def _reconcile_indexes(self) -> None:
        """A1 (adversarial review, 2026-09-12): after loading whatever was
        on disk, make every index (vector/bm25/entity) exactly match the
        live (non-deleted) item set -- silently and deterministically,
        never a crash. Covers: an item appended to ``items.jsonl`` but
        never flushed (live in the store, absent from the index caches); a
        truncated last JSONL line leaving an orphan index entry for an id no
        longer in the store; a soft-deleted item (``deleted=True``, e.g. from
        an older session before :meth:`forget` existed) whose index entries
        were never dropped.
        """
        live_items = {it.id: it for it in self.store.all(include_deleted=False)}
        live_ids = set(live_items)
        chunk_owner = self.store.chunk_owner

        # ---- vector index: drop stale entries, embed+add missing ones ----
        indexed_ids = set(self.vector_index.ids())
        stale_vec_ids = [sid for sid in indexed_ids if chunk_owner.get(sid, sid) not in live_ids]
        if stale_vec_ids:
            self.vector_index.remove(stale_vec_ids)
            indexed_ids -= set(stale_vec_ids)
            for sid in stale_vec_ids:
                self.store.chunk_owner.pop(sid, None)
        represented = {chunk_owner.get(sid, sid) for sid in indexed_ids}
        missing_ids = [iid for iid in live_ids if iid not in represented]
        if missing_ids:
            missing_items = [live_items[iid] for iid in missing_ids]
            texts = [index_text(it, self.config) for it in missing_items]
            vecs = self.embedder.encode(texts)
            self.vector_index.add(missing_ids, vecs)
            self._add_chunk_vectors(missing_items)

        # ---- bm25: drop stale docs, add missing ones ----
        for doc_id in [d for d in self.bm25.doc_len if d not in live_ids]:
            self.bm25.remove(doc_id)
        for iid, it in live_items.items():
            if iid not in self.bm25.doc_len:
                self.bm25.add(iid, tokenize(index_text(it, self.config), stem=self.config.use_stemming))

        # ---- entities: drop stale docs, add missing ones ----
        for doc_id in [d for d in self.entities.doc_entities if d not in live_ids]:
            self.entities.remove(doc_id)
        for iid, it in live_items.items():
            if iid not in self.entities.doc_entities:
                self.entities.add(iid, it.entities)

        # ---- facets: drop stale entries, add missing ones ----
        for doc_id in [d for d in self.facet_index.ids() if d not in live_ids]:
            self.facet_index.remove(doc_id)
        for iid, it in live_items.items():
            if iid not in self.facet_index.ids():
                self.facet_index.add(it)

    def _reembed_all(self) -> None:
        items = list(self.store.all(include_deleted=False))
        self.vector_index = VectorIndex(self.embedder.dim)
        self.store.chunk_owner = {}
        if not items:
            return
        texts = [index_text(it, self.config) for it in items]
        vecs = self.embedder.encode(texts)
        self.vector_index.add([it.id for it in items], vecs)
        self._add_chunk_vectors(items)

    def _add_chunk_vectors(self, items: list[MemoryItem]) -> None:
        """E1 multi_vector: index extra overlapping chunk-vectors per item.

        A whole-item "representative" vector is always added under ``item.id``
        (by the caller) so every existing consumer of
        ``vector_index.get_one(item_id)`` (dedup, supersession, MMR,
        consolidate.py) keeps working unchanged; this adds *additional*
        vectors under ``f"{item.id}::{chunk_idx}"`` purely for dense search,
        which max-pools chunk scores back to their owning item id (see
        ``retrieval.HybridRetriever._dense_candidates``). No-op unless
        ``config.multi_vector`` is set.
        """
        if not self.config.multi_vector or not items:
            return
        chunk_ids: list[str] = []
        chunk_texts: list[str] = []
        owners: list[str] = []
        for it in items:
            chunks = chunk_text(
                it.text, max_tokens=self.config.chunk_tokens, overlap=self.config.chunk_overlap
            )
            if len(chunks) <= 1:
                continue  # single chunk == the representative vector already covers it
            header = contextual_header(it)
            for ci, chunk in enumerate(chunks):
                chunk_ids.append(f"{it.id}::{ci}")
                chunk_texts.append(f"{header}\n{chunk}")
                owners.append(it.id)
        if not chunk_ids:
            return
        vecs = self.embedder.encode(chunk_texts)
        self.vector_index.add(chunk_ids, vecs)
        for cid, owner in zip(chunk_ids, owners):
            self.store.chunk_owner[cid] = owner

    def _rebuild_bm25_entities(self) -> None:
        self.bm25 = BM25Index()
        self.entities = EntityIndex()
        self.facet_index.rebuild(self.store.all(include_deleted=False))
        for it in self.store.all(include_deleted=False):
            self.bm25.add(
                it.id, tokenize(index_text(it, self.config), stem=self.config.use_stemming)
            )
            self.entities.add(it.id, it.entities)

    # ------------------------------------------------------------ internals
    def _build_items(self, specs: list[dict], kind: str) -> list[MemoryItem]:
        now = time.time()
        items: list[MemoryItem] = []
        seen_ids: set[str] = set()
        for spec in specs:
            text = spec["text"]
            item_id = spec.get("item_id") or uuid.uuid4().hex
            if item_id in seen_ids:
                # A4 (adversarial review, 2026-09-12): a duplicate item_id
                # WITHIN this one batch (e.g. add_episodes_bulk called with a
                # log replaying a duplicated line) used to build 2 MemoryItems
                # for the same id -- the store dict silently kept only the
                # last one, but both got embedded/indexed, so
                # VectorIndex.ntotal (and bm25/entity doc counts) drifted
                # above the true stored-item count. First occurrence wins;
                # later duplicates are silently skipped (caller-visible via a
                # shorter returned items list, e.g. migrate_v3's "skipped").
                continue
            if item_id in self.store.items:
                raise ValueError(f"duplicate item id: {item_id}")
            seen_ids.add(item_id)
            event_time = spec.get("event_time")
            event_time = float(event_time) if event_time is not None else now
            entities = spec.get("entities")
            if entities is None:
                entities = extract_entities(text)
            valid_from = spec.get("valid_from")
            if valid_from is None:
                valid_from = event_time
            item = MemoryItem(
                id=item_id,
                kind=kind,
                text=text,
                role=spec.get("role", "user" if kind == "episode" else "system"),
                project=spec.get("project", "default"),
                source=spec.get("source", "chat_save"),
                doc_ref=spec.get("doc_ref"),
                parent_ids=list(spec.get("parent_ids") or []),
                created_at=now,
                event_time=event_time,
                valid_from=valid_from,
                valid_until=spec.get("valid_until"),
                entities=list(entities),
                tags=list(spec.get("tags") or []),
                keywords=list(spec.get("keywords") or []),
                facets=normalize_facets(spec.get("facets") or []),
                identity=spec.get("identity"),
                replaces=list(spec.get("replaces") or []),
                importance=float(spec.get("importance", 0.5)),
                confidence=float(spec.get("confidence", 1.0)),
                hash=text_hash(text),
            )
            items.append(item)
        return items

    def _index_items(self, items: list[MemoryItem], vecs: np.ndarray | None = None) -> None:
        """Embed (unless ``vecs`` is already provided) and index ``items``.

        A3 (adversarial review, 2026-09-12): ``add_facts`` computes its
        dedup probe vector with the exact header this method would embed
        with anyway; passing it in via ``vecs`` avoids encoding the same
        text twice. Every other caller leaves ``vecs=None`` and this
        encodes ``items`` itself, unchanged from before.
        """
        if not items:
            return
        if vecs is None:
            texts = [index_text(it, self.config) for it in items]
            vecs = self.embedder.encode(texts)
        ids = [it.id for it in items]
        self.vector_index.add(ids, vecs)
        self._add_chunk_vectors(items)
        for it in items:
            self.bm25.add(
                it.id, tokenize(index_text(it, self.config), stem=self.config.use_stemming)
            )
            self.entities.add(it.id, it.entities)
            self.facet_index.add(it)
        self.store.append_many(items)

    # --------------------------------------------------------------- writes
    @_requires_open
    def add_episode(
        self,
        text: str,
        role: str = "user",
        project: str = "default",
        event_time: float | None = None,
        entities: list[str] | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        item_id: str | None = None,
        source: str = "chat_save",
        doc_ref: str | None = None,
        keywords: list[str] | None = None,
        facets: list[dict] | None = None,
    ) -> MemoryItem:
        with self._lock:
            items = self._build_items(
                [
                    {
                        "text": text,
                        "role": role,
                        "project": project,
                        "event_time": event_time,
                        "entities": entities,
                        "tags": tags,
                        "importance": importance,
                        "item_id": item_id,
                        "source": source,
                        "doc_ref": doc_ref,
                        "keywords": keywords,
                        "facets": facets,
                    }
                ],
                kind="episode",
            )
            self._index_items(items)
            return items[0]

    @_requires_open
    def add_episodes_bulk(self, episodes: list[dict]) -> list[MemoryItem]:
        with self._lock:
            items = self._build_items(episodes, kind="episode")
            self._index_items(items)
            return items

    def _find_duplicate(self, item: MemoryItem) -> str | None:
        """Cosine similarity never proves duplicate identity.

        An identity names a property, not its value: differing values must reach
        explicit replacement logic. Equal normalized text is required for all kinds.
        """
        for it in self.store.all(include_deleted=False):
            if duplicate_compatible(it, item) and equivalent_assertions(it, item):
                return it.id
        return None

    @_requires_open
    def add_facts(
        self,
        facts: list[dict],
        project: str = "default",
        parent_ids: list[str] | None = None,
        event_time: float | None = None,
        source: str = "client",
    ) -> dict:
        with self._lock:
            normalized = normalize_client_facts(facts)
            # `normalize_client_facts` (extract.py) whitelists a fixed set of
            # output keys and doesn't know about `keywords` -- it always
            # returns exactly one normalized dict per input dict, in order
            # (raises before appending anything for an invalid fact, never a
            # partial list), so re-attaching `keywords` by position here is
            # safe and keeps extract.py untouched.
            for nf, raw in zip(normalized, facts):
                kw = raw.get("keywords") if isinstance(raw, dict) else None
                if kw:
                    nf["keywords"] = list(kw)
                fac = raw.get("facets") if isinstance(raw, dict) else None
                if fac:
                    nf["facets"] = fac

            added: list[str] = []
            deduped: list[str] = []
            superseded: list[tuple[str, str]] = []

            for nf in normalized:
                kind = nf.get("kind", "fact")
                ev_time = nf.get("event_time")
                if ev_time is None:
                    ev_time = event_time if event_time is not None else time.time()

                spec = dict(nf)
                spec["item_id"] = nf.get("item_id")
                spec["project"] = project
                spec["parent_ids"] = parent_ids
                spec["event_time"] = ev_time
                spec["source"] = source
                item = self._build_items([spec], kind=kind)[0]
                validate_replacements(item, self.store)
                duplicate_id = self._find_duplicate(item)
                if duplicate_id:
                    deduped.append(duplicate_id)
                    continue

                # Embeddings serve retrieval, never duplicate decisions. Reuse
                # this vector for indexing rather than computing it twice.
                probe_text = index_text(item, self.config)
                vec = self.embedder.encode([probe_text])[0]

                self._index_items([item], vecs=np.asarray([vec]))
                added.append(item.id)
                if kind == "fact" and (item.replaces or
                                       (item.identity and item.identity["cardinality"] == "single")):
                    superseded.extend(replacement_sweep(self.store, project=project, focus=item))

            return {"added": added, "deduped": deduped, "superseded": superseded}

    @_requires_open
    def forget(self, item_id: str, reason: str = "", project: str | None = None) -> bool:
        """Soft-delete ``item_id`` (A5, additive façade method): marks it
        ``deleted=True`` in the store AND removes it from the vector/BM25/
        entity indexes immediately, so it stops consuming candidate slots in
        every retrieval signal right away instead of only after the next
        :meth:`consolidate` rebuild. Returns False if ``item_id`` is unknown
        or already deleted (idempotent no-op), True if it was just forgotten.
        Never a hard delete: the tombstone record is still appended to
        ``items.jsonl`` like any other mutation.
        """
        with self._lock:
            item = self.store.get(item_id)
            if item is None or item.deleted or (project is not None and item.project != project):
                return False
            item.deleted = True
            self.store.append(item)
            chunk_ids = [cid for cid, owner in self.store.chunk_owner.items() if owner == item_id]
            self.vector_index.remove([item_id, *chunk_ids])
            for cid in chunk_ids:
                self.store.chunk_owner.pop(cid, None)
            self.bm25.remove(item_id)
            self.entities.remove(item_id)
            self.facet_index.remove(item_id)
            logger.info("forget: %s marked deleted (reason=%s)", item_id, reason)
            return True

    @_requires_open
    def add_document(
        self, name: str, text: str, project: str = "default", doc_ref: str | None = None
    ) -> dict:
        with self._lock:
            ref = doc_ref or name
            chunks = chunk_text(text, max_tokens=180, overlap=30)
            specs = [
                {
                    "text": chunk,
                    "role": "doc",
                    "project": project,
                    "doc_ref": ref,
                    "source": "ingest",
                    "tags": [name],
                }
                for chunk in chunks
            ]
            items = self._build_items(specs, kind="doc")
            self._index_items(items)
            return {"doc_ref": ref, "chunk_ids": [it.id for it in items], "chunks": len(items)}

    # -------------------------------------------------------------- reads
    @_requires_open
    def search(
        self,
        query: str,
        top_k: int = 10,
        at_time: float | None = None,
        kinds: list[str] | None = None,
        project: str | None = None,
        rerank: bool | None = None,
        queries: list[str] | None = None,
        hypotheses: list[str] | None = None,
        facets: list[dict] | None = None,
    ) -> list[SearchHit]:
        """`queries`/`hypotheses` (B, production port of `V4PlannedRetriever`'s
        RRF query-planning fusion, +0.075 R@10 on LoCoMo): when either is
        given (non-empty), search the original `query` AND every extra
        string, fuse with RRF (`HybridRetriever.search_multi`); otherwise
        (both omitted/empty, the default) this is byte-identical to the
        pre-existing single-query path.

        `facets` (C, production port of the facet channel, 2026-09-19):
        ``[{"subject", "predicate"}, ...]`` describing what the question is
        about. Routes through `search_multi` -- even with no `queries`/
        `hypotheses` -- ONLY when `config.facet_channel` is on AND the
        resulting facet pool (`HybridRetriever.facet_pool`) is non-empty;
        otherwise (flag off, no facets, or an empty pool) this call is
        byte-identical to omitting `facets` entirely.
        """
        with self._lock:
            facet_pool = self.retriever.facet_pool(facets, kinds=kinds, project=project, at_time=at_time)
            if queries or hypotheses or facet_pool:
                return self.retriever.search_multi(
                    query,
                    queries=queries,
                    hypotheses=hypotheses,
                    top_k=top_k,
                    at_time=at_time,
                    kinds=kinds,
                    project=project,
                    rerank=rerank,
                    facets=facets,
                )
            return self.retriever.search(
                query, top_k=top_k, at_time=at_time, kinds=kinds, project=project, rerank=rerank
            )

    @_requires_open
    def get_context(
        self,
        query: str,
        max_tokens: int = 1500,
        at_time: float | None = None,
        project: str | None = None,
        queries: list[str] | None = None,
        hypotheses: list[str] | None = None,
        facets: list[dict] | None = None,
    ) -> str:
        with self._lock:
            facet_pool = self.retriever.facet_pool(facets, project=project, at_time=at_time)
            if queries or hypotheses or facet_pool:
                hits = self.retriever.search_multi(
                    query,
                    queries=queries,
                    hypotheses=hypotheses,
                    top_k=self.config.candidates,
                    at_time=at_time,
                    project=project,
                    facets=facets,
                )
            else:
                hits = self.retriever.search(
                    query, top_k=self.config.candidates, at_time=at_time, project=project
                )
            vectors = {h.item.id: v for h in hits if (v := self.vector_index.get_one(h.item.id)) is not None}
            diversified = mmr(hits, vectors, lambda_=0.7, k=len(hits))
            return self._render_hits(diversified, max_tokens)

    def _render_hits(self, hits: list[SearchHit], max_tokens: int) -> str:
        budget_chars = max_tokens * 4
        lines: list[str] = []
        total = 0
        for h in hits:
            tag = f"[{h.item.kind}|{date_str(h.item.event_time)}|{h.item.project}]"
            line = f"{tag} {h.item.text}"
            if lines and total + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def _is_pending(self, item: MemoryItem) -> bool:
        if "todo" in item.tags or "pending" in item.tags:
            return True
        return bool(_PENDING_RE.search(item.text))

    @_requires_open
    def session_brief(
        self,
        focus_query: str | None = None,
        max_tokens: int = 3000,
        project: str | None = None,
        queries: list[str] | None = None,
        hypotheses: list[str] | None = None,
    ) -> dict:
        with self._lock:
            now = time.time()
            live = [
                it for it in self.store.all(include_deleted=False) if project is None or it.project == project
            ]

            episodes = sorted(
                (it for it in live if it.kind == "episode"), key=lambda it: it.event_time, reverse=True
            )
            recent_episodes = episodes[:8]

            def fact_score(it: MemoryItem) -> float:
                age_days = max(0.0, (now - it.event_time) / 86400.0)
                rec = recency_weight(age_days, self.config.half_life_days("fact"))
                return it.importance * (0.5 + 0.5 * rec)

            facts = [it for it in live if it.kind == "fact" and is_valid_at(it, now)]
            facts.sort(key=fact_score, reverse=True)
            key_facts = facts[:12]

            procedures = [it for it in live if it.kind == "procedure"][:6]

            focus_hits: list[SearchHit] = []
            if focus_query:
                if queries or hypotheses:
                    focus_hits = self.retriever.search_multi(
                        focus_query, queries=queries, hypotheses=hypotheses, top_k=8, project=project
                    )
                else:
                    focus_hits = self.retriever.search(focus_query, top_k=8, project=project)

            pending = [it for it in live if self._is_pending(it)][:6]

            sections: dict = {
                "blocks": self.blocks.all(project=project),
                "recent_episodes": [it.to_dict() for it in recent_episodes],
                "key_facts": [it.to_dict() for it in key_facts],
                "procedures": [it.to_dict() for it in procedures],
                "focus": [h.item.to_dict() for h in focus_hits],
                "pending": [it.to_dict() for it in pending],
                "stats": self.status(project=project),
            }
            sections["text"] = self._render_brief(sections, max_tokens)
            return sections

    def _render_brief(self, sections: dict, max_tokens: int) -> str:
        budget = max_tokens * 4
        parts: list[str] = []

        def add(title: str, lines: list[str]) -> bool:
            block = f"## {title}\n" + "\n".join(lines) if lines else f"## {title}\n(none)"
            current = "\n\n".join(parts)
            if current and len(current) + len(block) + 2 > budget:
                return False
            parts.append(block)
            return True

        block_lines = [f"[{name}] {text}" for name, text in sections["blocks"].items() if text]
        add("Core blocks", block_lines)
        add(
            "Recent episodes",
            [
                f"[{date_str(e['event_time'])}|{e['project']}] {e['text']}"
                for e in sections["recent_episodes"]
            ],
        )
        add(
            "Key facts",
            [f"[{date_str(f['event_time'])}|{f['project']}] {f['text']}" for f in sections["key_facts"]],
        )
        add("Procedures", [f"[{p['project']}] {p['text']}" for p in sections["procedures"]])
        add("Focus", [f"[{date_str(fo['event_time'])}] {fo['text']}" for fo in sections["focus"]])
        add("Pending", [f"[{date_str(p['event_time'])}] {p['text']}" for p in sections["pending"]])
        return "\n\n".join(parts)

    # ------------------------------------------------------------ upkeep
    @_requires_open
    def consolidate(self, dry_run: bool = False, project: str | None = None) -> dict:
        with self._lock:
            deduped = consolidate_mod.dedup_sweep(self.store, self.vector_index, self.config, dry_run, project=project)
            superseded = consolidate_mod.supersede_sweep(self.store, self.vector_index, self.config, dry_run, project=project)
            decayed = consolidate_mod.decay(self.store, self.config, time.time(), dry_run, project=project)
            proposals = consolidate_mod.summary_proposals(self.store, self.vector_index, project=project)

            if not dry_run and (deduped or superseded):
                self._reconcile_indexes()

            if not dry_run:
                self.last_consolidated = time.time()
                self.store.meta["last_consolidated"] = self.last_consolidated
                self.store.meta.setdefault("last_consolidated_by_project", {})[project or "*"] = self.last_consolidated

            return {
                "deduped": len(deduped),
                "deduped_pairs": deduped,
                "superseded": len(superseded),
                "superseded_pairs": superseded,
                "decayed": decayed,
                "summary_proposals": proposals,
            }

    @_requires_open
    def status(self, project: str | None = None) -> dict:
        items = [it for it in self.store.all(False) if project is None or it.project == project]
        ids = {it.id for it in items}
        counts: dict[str, int] = {}
        for it in items:
            counts[it.kind] = counts.get(it.kind, 0) + 1
        vector_count = sum(self.store.chunk_owner.get(vid, vid) in ids for vid in self.vector_index.ids())
        entity_terms = len({entity for it in items for entity in self.entities.doc_entities.get(it.id, [])})
        return {
            "counts": counts,
            "total": sum(counts.values()),
            "embed_model": getattr(self.embedder, "model_name", "unknown"),
            "dim": self.embedder.dim,
            "data_dir": str(self.data_dir),
            "bm25_docs": sum(iid in ids for iid in self.bm25.doc_len),
            "entity_terms": entity_terms,
            "vector_count": vector_count,
            "last_consolidated": (self.last_consolidated if project is None else
                                  self.store.meta.get("last_consolidated_by_project", {}).get(project)),
        }

    @_requires_open
    def flush(self) -> None:
        with self._lock:
            ids, vecs = self.vector_index.export()
            self.store.save_embeddings(ids, vecs)
            self.store.save_chunk_map()
            self.bm25.save(self._bm25_path)
            self.entities.save(self._entities_path)
            meta = dict(self.store.meta)
            meta["embed_model"] = self.config.embed_model
            meta["dim"] = self.embedder.dim
            meta["version"] = "4.0.1"
            if self.last_consolidated is not None:
                meta["last_consolidated"] = self.last_consolidated
            self.store.save_meta(meta)
            self.store.flush_access()

    @_requires_open
    def migrate_v3(self, log_path: str) -> dict:
        with self._lock:
            episode_specs, summary_specs, skipped = read_v3_log(log_path, self.store.items)

            n_episodes = 0
            if episode_specs:
                added_items = self.add_episodes_bulk(episode_specs)
                n_episodes = len(added_items)
                # A4: a duplicated log line can still slip 2 identical specs
                # (same deterministic id) past read_v3_log's own idempotence
                # check WITHIN one run (that check only sees ids already
                # persisted from a PRIOR call); add_episodes_bulk now dedupes
                # them (first wins), so the true "skipped" count must include
                # whatever it silently dropped -- reporting `len(episode_specs)`
                # here would otherwise overstate how many episodes actually
                # got stored (adversarial review, 2026-09-12: migrate_dup.py).
                skipped += len(episode_specs) - n_episodes

            n_summaries = 0
            seen_summary_ids = set(self.store.items)
            for spec in summary_specs:
                summary_id = spec.get("item_id")
                # Migration ids identify source records, including duplicate
                # lines within this batch. Handle replay here; ordinary
                # add_facts retains strict caller-supplied id collision checks.
                if summary_id in seen_summary_ids:
                    skipped += 1
                    continue
                seen_summary_ids.add(summary_id)
                result = self.add_facts(
                    [spec], project=spec.get("project", "default"), event_time=spec.get("event_time")
                )
                n_summaries += len(result["added"])

            return {"episodes": n_episodes, "summaries": n_summaries, "skipped": skipped}
