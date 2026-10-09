"""Retriever adapters implementing bench.protocol.Retriever.

- V3EquivRetriever: faithful from-scratch reproduction of the OLD v3.2.0
  engine's indexing scheme (128-char sliding-window chunks, MiniLM, FlatL2)
  WITHOUT importing memory_os_v3 — see PLAN.md diagnostic table.
- DenseTurnRetriever: whole-turn dense embedding baseline (IndexFlatIP).
- BM25TurnRetriever: self-contained BM25 (k1=1.2, b=0.75) over whole turns.
- V4Retriever: thin adapter over the (separately developed) memory_os_v4
  core package's public façade. Import of THIS module must always succeed;
  only *instantiating* V4Retriever may raise if the core isn't ready yet.
- V4EnrichedRetriever: V4Retriever + offline LLM enrichment (A-Mem/Mem0
  style facts extracted per turn by bench/enrich.py's producers) indexed as
  memory_os_v4 `fact` items alongside the raw turns.

Only ML model used anywhere here: the locally cached
sentence-transformers/all-MiniLM-L6-v2 (HF_HUB_OFFLINE=1 enforced below).
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import dataclasses
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from bench.enrich import (
    EnrichRecord,
    FactFacetRecord,
    MAX_DUMP_CANDIDATES,
    QueryPlan,
    fid_to_turn_id,
    load_candidate_dump,
    load_enrichment,
    load_fact_facets,
    load_llm_rankings,
    load_qfacets,
    load_query_plans,
)
from bench.protocol import BenchItem

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache: the model is expensive to load (~90 MB) and is shared
# by V3EquivRetriever and DenseTurnRetriever across every conversation in a
# run, so it is loaded at most once per process regardless of how many
# retriever instances / conversations are processed.
_model_cache: dict[str, Any] = {}


def _get_embedder(model_name: str = EMBED_MODEL_NAME):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


EmbedFn = Callable[[list[str]], np.ndarray]


def _default_embed_fn(model_name: str = EMBED_MODEL_NAME) -> EmbedFn:
    def _embed(texts: list[str]) -> np.ndarray:
        model = _get_embedder(model_name)
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    return _embed


# ---------------------------------------------------------------------------
# V3-equivalent: 128-char / 28-char-overlap chunking, MiniLM, FAISS FlatL2.
# ---------------------------------------------------------------------------
class V3EquivRetriever:
    """A **charitable v3-style baseline** -- NOT a literal reproduction of
    the real v3.2.0 engine (A10, adversarial review 2026-09-12: see
    `V3ExactRetriever` below for that, and `bench/review/chunk_diff.py`'s
    measurement of how much the two actually diverge on LoCoMo). It shares
    v3's 128-CHARACTER / 100-char-step chunking idea and MiniLM+FlatL2
    indexing, but is charitable to v3 in three concrete ways every reader of
    a comparison table should know about:

    1. **Chunking drops the tiny/redundant tail window real v3 keeps.**
       `chunk_text` breaks as soon as a window would reach the text's end
       (``i + CHUNK_SIZE >= n``), so it never emits the ragged, mostly- or
       entirely-overlap-only final chunk the real v3
       ``range(0, len(text), 100)`` slicing still produces (e.g. a
       128-char text: real v3 emits a second chunk of just the last
       28 overlap chars; this class emits none). It matches real v3 exactly
       only when the natural last window happens to cover genuinely new
       text. Measured on the full LoCoMo corpus: 23.8% of turns chunk
       differently, 1400 fewer chunks overall (9863 vs v3's 11263) --
       slightly less index noise than the real engine had to search through.
    2. **Chunk->turn dedup is padded to `top_k`.** Search oversamples
       ``top_k * 8`` raw chunk hits, then walks them in ranked order keeping
       the first occurrence of each parent id, stopping once `top_k`
       *distinct* ids are collected -- smoothing over the raw chunk-level
       ranking so a heavily-chunked item's multiple hits never "crowd out"
       a real top_k of distinct turns the way a smaller, un-padded search
       could.
    3. **No absolute-distance cutoff.** Real v3 hard-dropped any chunk whose
       (squared) L2 distance exceeded 0.8, so a query with no good match in
       the corpus still returned nothing there; this class returns whatever
       is nearest regardless of how far that still is.

    v3 had no temporal awareness at all (see PLAN.md diagnostic table row
    "Temporel: aucun"), so `at_time` is accepted for protocol compatibility
    but ignored here -- that part IS the faithful behaviour, not an
    oversight.
    """

    CHUNK_SIZE = 128
    CHUNK_OVERLAP = 28
    CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP  # 100
    CANDIDATE_MULTIPLIER = 8

    def __init__(self, model_name: str = EMBED_MODEL_NAME, embed_fn: Optional[EmbedFn] = None, name: str = "v3equiv"):
        self.name = name
        self.model_name = model_name
        self._embed_fn = embed_fn or _default_embed_fn(model_name)
        self.reset()

    def reset(self) -> None:
        self._chunk_texts: list[str] = []
        self._chunk_item_ids: list[str] = []
        self._index = None  # built lazily: needs the embedding dim from the first batch

    @classmethod
    def chunk_text(cls, text: str) -> list[str]:
        """Split `text` into 128-char windows, 28-char overlap (100-char step)."""
        if not text:
            return [""]
        n = len(text)
        if n <= cls.CHUNK_SIZE:
            return [text]
        chunks = []
        i = 0
        while i < n:
            chunks.append(text[i : i + cls.CHUNK_SIZE])
            if i + cls.CHUNK_SIZE >= n:
                break
            i += cls.CHUNK_STEP
        return chunks

    def add(self, items: list[BenchItem]) -> None:
        import faiss

        new_texts: list[str] = []
        new_ids: list[str] = []
        for item in items:
            for chunk in self.chunk_text(item.text):
                new_texts.append(chunk)
                new_ids.append(item.id)
        if not new_texts:
            return
        vecs = self._embed_fn(new_texts)
        if self._index is None:
            self._index = faiss.IndexFlatL2(vecs.shape[1])
        self._index.add(vecs)
        self._chunk_texts.extend(new_texts)
        self._chunk_item_ids.extend(new_ids)

    def search(self, query: str, top_k: int, at_time: Optional[float] = None) -> list[str]:
        if self._index is None or self._index.ntotal == 0:
            return []
        qvec = self._embed_fn([query])
        k = min(top_k * self.CANDIDATE_MULTIPLIER, self._index.ntotal)
        _, idxs = self._index.search(qvec, k)
        seen: set[str] = set()
        result: list[str] = []
        for idx in idxs[0]:
            if idx < 0:
                continue
            item_id = self._chunk_item_ids[idx]
            if item_id in seen:
                continue
            seen.add(item_id)
            result.append(item_id)
            if len(result) >= top_k:
                break
        return result


# ---------------------------------------------------------------------------
# V3-EXACT: literal reproduction of the real v3.2.0 engine, warts and all.
# ---------------------------------------------------------------------------
class V3ExactRetriever:
    """Literal reproduction of the real v3.2.0 chunking/search (A10,
    adversarial review 2026-09-12) -- unlike `V3EquivRetriever` above, this
    class makes none of that class's three charitable choices:

    - ``chunk_text(t) = [t[i:i+128] for i in range(0, len(t), 100)]``, with
      no early break, so a text can end in a tiny ragged tail chunk (down to
      1 char) that only overlaps already-covered text.
    - Search fetches exactly ``k = min(top_k, n)`` chunk hits -- no ``* 8``
      oversampling to backfill enough distinct parent ids.
    - Any chunk whose (squared) L2 distance exceeds 0.8 is dropped, exactly
      matching the real engine's absolute-distance cutoff.
    - Surviving chunks are mapped to their parent turn id in ranked order,
      deduped only to avoid literal repeats (no re-fetching to compensate
      for collisions) -- so the result can, and often does, come back with
      FEWER than `top_k` distinct turns. This is real v3's actual, sometimes
      worse, behaviour, not a bug in this class.

    Like v3, it has no temporal awareness: `at_time` is accepted for
    protocol compatibility and ignored.
    """

    CHUNK_SIZE = 128
    CHUNK_STEP = 100
    DISTANCE_THRESHOLD = 0.8  # real v3's absolute (squared) L2 cutoff

    def __init__(self, model_name: str = EMBED_MODEL_NAME, embed_fn: Optional[EmbedFn] = None, name: str = "v3exact"):
        self.name = name
        self.model_name = model_name
        self._embed_fn = embed_fn or _default_embed_fn(model_name)
        self.reset()

    def reset(self) -> None:
        self._chunk_texts: list[str] = []
        self._chunk_item_ids: list[str] = []
        self._index = None  # built lazily: needs the embedding dim from the first batch

    @classmethod
    def chunk_text(cls, text: str) -> list[str]:
        """Literal v3 slicing: fixed 128-char windows, 100-char step, no
        early break -- can end in a short/ragged, overlap-only tail chunk."""
        if not text:
            return [""]
        return [text[i : i + cls.CHUNK_SIZE] for i in range(0, len(text), cls.CHUNK_STEP)]

    def add(self, items: list[BenchItem]) -> None:
        import faiss

        new_texts: list[str] = []
        new_ids: list[str] = []
        for item in items:
            for chunk in self.chunk_text(item.text):
                new_texts.append(chunk)
                new_ids.append(item.id)
        if not new_texts:
            return
        vecs = self._embed_fn(new_texts)
        if self._index is None:
            self._index = faiss.IndexFlatL2(vecs.shape[1])
        self._index.add(vecs)
        self._chunk_texts.extend(new_texts)
        self._chunk_item_ids.extend(new_ids)

    def search(self, query: str, top_k: int, at_time: Optional[float] = None) -> list[str]:
        if self._index is None or self._index.ntotal == 0:
            return []
        qvec = self._embed_fn([query])
        k = min(top_k, self._index.ntotal)  # no oversampling multiplier, unlike V3EquivRetriever
        distances, idxs = self._index.search(qvec, k)
        seen: set[str] = set()
        result: list[str] = []
        for dist, idx in zip(distances[0], idxs[0]):
            if idx < 0:
                continue
            if dist > self.DISTANCE_THRESHOLD:
                continue  # real v3's absolute distance cutoff
            item_id = self._chunk_item_ids[idx]
            if item_id in seen:
                continue
            seen.add(item_id)
            result.append(item_id)
        return result  # may be shorter than top_k -- see class docstring


# ---------------------------------------------------------------------------
# Dense, whole-turn baseline.
# ---------------------------------------------------------------------------
class DenseTurnRetriever:
    """Whole-turn MiniLM embeddings, FAISS IndexFlatIP (cosine via normalized dot)."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME, embed_fn: Optional[EmbedFn] = None, name: str = "dense"):
        self.name = name
        self.model_name = model_name
        self._embed_fn = embed_fn or _default_embed_fn(model_name)
        self.reset()

    def reset(self) -> None:
        self._ids: list[str] = []
        self._index = None

    def add(self, items: list[BenchItem]) -> None:
        import faiss

        if not items:
            return
        texts = [item.text for item in items]
        vecs = self._embed_fn(texts)
        if self._index is None:
            self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)
        self._ids.extend(item.id for item in items)

    def search(self, query: str, top_k: int, at_time: Optional[float] = None) -> list[str]:
        if self._index is None or self._index.ntotal == 0:
            return []
        qvec = self._embed_fn([query])
        k = min(top_k, self._index.ntotal)
        _, idxs = self._index.search(qvec, k)
        return [self._ids[i] for i in idxs[0] if i >= 0]


# ---------------------------------------------------------------------------
# Self-contained BM25 (no dependency on memory_os_v4.bm25 or any package).
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25TurnRetriever:
    """Textbook Okapi BM25 (k1=1.2, b=0.75) over whole-turn text, inverted-index based."""

    K1 = 1.2
    B = 0.75

    def __init__(self, name: str = "bm25"):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self._ids: list[str] = []
        self._doc_len: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc_idx, tf)]
        self._df: dict[str, int] = defaultdict(int)
        self._avgdl = 0.0
        self._n_docs = 0

    def add(self, items: list[BenchItem]) -> None:
        for item in items:
            tokens = _tokenize(item.text)
            doc_idx = len(self._ids)
            self._ids.append(item.id)
            self._doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((doc_idx, tf))
                self._df[term] += 1
        self._n_docs = len(self._ids)
        self._avgdl = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int, at_time: Optional[float] = None) -> list[str]:
        if not self._n_docs:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        avgdl = self._avgdl or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in Counter(q_tokens).items():
            idf = self._idf(term)
            if idf <= 0:
                continue
            for doc_idx, tf in self._postings.get(term, ()):
                dl = self._doc_len[doc_idx]
                denom = tf + self.K1 * (1 - self.B + self.B * dl / avgdl)
                scores[doc_idx] += qtf * idf * (tf * (self.K1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self._ids[i] for i, _ in ranked]


# ---------------------------------------------------------------------------
# v4 core adapter.
# ---------------------------------------------------------------------------
class V4Retriever:
    """Adapter over memory_os_v4.MemoryV4 (see docs/ARCHITECTURE.md façade).

    Importing this module never fails because of memory_os_v4: the core
    package (written in parallel by another agent) is only imported lazily,
    inside __init__, so that other retrievers stay usable/testable even
    while memory_os_v4 has no MemoryV4/Config yet.
    """

    def __init__(self, config_overrides: Optional[dict] = None, name: str = "v4"):
        self.name = name
        self.config_overrides = dict(config_overrides or {})
        try:
            from memory_os_v4 import MemoryV4
            from memory_os_v4.config import Config
        except ImportError as exc:
            raise ImportError(
                "memory_os_v4 does not expose MemoryV4/Config yet (core package "
                "still being written in parallel, per v4/PLAN.md P1). "
                "V4Retriever cannot be instantiated until it lands; "
                "other bench retrievers (v3equiv/dense/bm25) are unaffected."
            ) from exc
        self._MemoryV4 = MemoryV4
        self._Config = Config
        self._tmpdir: Optional[str] = None
        self._memory = None

    def reset(self) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._tmpdir = tempfile.mkdtemp(prefix="memv4_bench_")
        config = self._Config(**self.config_overrides)
        self._memory = self._MemoryV4(data_dir=self._tmpdir, config=config)

    def add(self, items: list[BenchItem]) -> None:
        episodes = []
        for item in items:
            # Two BenchItems worth of turns can already carry a "speaker: "
            # prefix baked into `text` by the loader (LoCoMo does this; see
            # bench/locomo.py) while others don't (LongMemEval does not; see
            # bench/longmemeval.py). Per the bench brief: role is always
            # "user" (both LoCoMo speakers are humans, not a user/assistant
            # pair; collapsing one of them to "assistant" would be wrong),
            # and the speaker identity goes at the start of `text` instead —
            # guarded here so an already-prefixed text is not prefixed twice.
            prefix = f"{item.speaker}: " if item.speaker else ""
            text = item.text if (prefix and item.text.startswith(prefix)) else f"{prefix}{item.text}"
            episodes.append(
                {
                    "text": text,
                    "role": "user",
                    "project": "bench",
                    "event_time": item.event_time,
                    "item_id": item.id,
                    # E3 neighbor_expansion: doc_ref groups items for the
                    # +-window time-adjacency lookup (HybridRetriever._doc_ref_order).
                    "doc_ref": item.session_id,
                }
            )
        self._memory.add_episodes_bulk(episodes)

    def search(self, query: str, top_k: int, at_time: Optional[float] = None) -> list[str]:
        hits = self._memory.search(query, top_k=top_k, at_time=at_time)
        return [hit.item.id for hit in hits]

    def __del__(self):  # best-effort cleanup of the per-conversation tmp dir
        try:
            if getattr(self, "_tmpdir", None):
                shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# v4 + offline LLM enrichment ("A-Mem/Mem0-style" facts).
# ---------------------------------------------------------------------------
class V4EnrichedRetriever(V4Retriever):
    """`V4Retriever` plus per-turn offline LLM enrichment loaded from
    ``bench/data/enrich/locomo_facts/<conv_id>.jsonl`` (see `bench.enrich`).

    Two modes:

    - ``"facts"``: index the raw turns exactly like `V4Retriever`, then ALSO
      index each turn's extracted facts as `memory_os_v4` ``fact`` items
      (``parent_ids=[turn_id]``) so search can surface a paraphrased/
      distilled statement the raw turn's wording wouldn't match directly.
    - ``"facts+keywords"``: same, plus the raw turn's own indexed text is
      augmented with `` | keywords: k1, k2, ...`` before it is embedded/
      BM25-indexed (so lexical/dense signals over the RAW turn also see the
      extracted keywords, not just the separate fact items).

    Conversation identity for `add()` comes from ``item.meta["conv_id"]``
    (set by `bench.locomo.load_locomo`) rather than a separate hook, so this
    class needs no change to the `Retriever.add(items)` protocol; a dataset
    whose loader doesn't set `meta["conv_id"]` (e.g. LongMemEval) simply
    gets no enrichment (recorded as `enrich_file_found=False` in
    `self.coverage`, never an error).

    **Cost note** (asked for explicitly, see docstring on `add_facts` in
    `memory_os_v4/memory.py`): every `add_facts` call runs, per new fact,
    an exact-hash dedup scan, a near-duplicate embedding-similarity scan,
    AND (for `kind="fact"`) a full supersession sweep -- each an O(items
    already in the store) linear scan over `self.store.all(...)`. So
    enriching a conversation with `T` turns and `F` facts/turn costs
    roughly O(T * F * (T + prior_facts)) store scans by the time the last
    turn's facts are added, not O(T*F). This is deliberately called **once
    per turn** (one `add_facts` batch per turn, not one call per individual
    fact, and not one call for the whole conversation's facts at once) so
    that the batch size handed to any single `add_facts` call stays small
    and bounded by "facts extracted from one turn" (typically 0-3) --
    calling it once per fact would multiply the *number* of these already-
    quadratic scans for no benefit, and calling it once for the whole
    conversation would not reduce the scan cost (still O(total facts) per
    scan) while making `parent_ids` (one turn id per call) awkward to keep
    correct per fact. On LoCoMo's few-hundred-turn conversations with a
    handful of facts per turn this stays well under the bench's per-question
    latency budget; it is NOT a scheme that would scale to a conversation
    with thousands of turns.
    """

    VALID_MODES = ("facts", "facts+keywords")

    def __init__(
        self,
        config_overrides: Optional[dict] = None,
        name: str = "v4_enriched",
        mode: str = "facts",
        enrich_dir: Optional[str] = None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {self.VALID_MODES}")
        super().__init__(config_overrides=config_overrides, name=name)
        self.mode = mode
        self.enrich_dir = Path(enrich_dir) if enrich_dir is not None else None
        # Per-conversation snapshot, refreshed by every add() call (mirrors
        # the reset()-then-add() lifecycle run_bench.py drives).
        self.coverage: dict[str, Any] = {
            "conv_id": None,
            "n_turns": 0,
            "n_turns_with_facts": 0,
            "n_facts_added": 0,
            "enrich_file_found": False,
        }
        # Cumulative across the WHOLE run (every conversation's add()/
        # search() calls), never cleared by reset() -- run_bench.py reports
        # the final totals once in the run's metadata, per the bench brief
        # ("expose a search_stats counter ... reported in run metadata").
        self.search_stats: dict[str, int] = {
            "n_searches": 0,
            "top_hit_facts": 0,
            "top_hit_episodes": 0,
            "top_hit_other": 0,
            "all_hits_facts": 0,
            "all_hits_episodes": 0,
            "all_hits_other": 0,
        }

    def reset(self) -> None:
        super().reset()
        self.coverage = {
            "conv_id": None,
            "n_turns": 0,
            "n_turns_with_facts": 0,
            "n_facts_added": 0,
            "enrich_file_found": False,
        }

    def add(self, items: list[BenchItem]) -> None:
        if not items:
            return

        conv_id = (items[0].meta or {}).get("conv_id")
        enrichment: dict[str, EnrichRecord] = {}
        enrich_file_found = False
        if self.enrich_dir is not None and conv_id is not None:
            enrichment = load_enrichment(self.enrich_dir, conv_id)
            enrich_file_found = (self.enrich_dir / f"{conv_id}.jsonl").exists()

        # 1) Raw turns, exactly as V4Retriever indexes them (same item_ids)
        #    -- optionally with extracted keywords appended to the text so
        #    BM25/dense signals over the raw turn see them too.
        raw_items = items
        if self.mode == "facts+keywords":
            raw_items = []
            for item in items:
                record = enrichment.get(item.id)
                if record and record.keywords:
                    augmented_text = f"{item.text} | keywords: {', '.join(record.keywords)}"
                    raw_items.append(dataclasses.replace(item, text=augmented_text))
                else:
                    raw_items.append(item)
        super().add(raw_items)

        # 2) Extracted facts, one add_facts() batch per turn (see class
        #    docstring for why per-turn, not per-fact or per-conversation).
        n_turns_with_facts = 0
        n_facts_added = 0
        for item in items:
            record = enrichment.get(item.id)
            if not record or not record.facts:
                continue
            n_turns_with_facts += 1
            facts_payload = [
                {
                    "text": fact,
                    "entities": record.entities,
                    "event_time": item.event_time,
                    "importance": 0.6,
                    "tags": ["enriched"],
                }
                for fact in record.facts
            ]
            self._memory.add_facts(
                facts_payload,
                project="bench",
                parent_ids=[item.id],
                source="bench_enrich",
            )
            n_facts_added += len(facts_payload)

        self.coverage = {
            "conv_id": conv_id,
            "n_turns": len(items),
            "n_turns_with_facts": n_turns_with_facts,
            "n_facts_added": n_facts_added,
            "enrich_file_found": enrich_file_found,
        }

    def search(self, query: str, top_k: int, at_time: Optional[float] = None) -> list[str]:
        if self._memory is None:
            return []
        # Oversample 3x candidates: a fact hit maps to (possibly several)
        # parent turn ids, and several fact/episode hits can collapse onto
        # the same turn id once deduped, so raw hit count << final turn
        # count without headroom.
        hits = self._memory.search(query, top_k=top_k * 3, at_time=at_time)

        seen: set[str] = set()
        turn_ids: list[str] = []
        top_hit_kind: Optional[str] = None
        for hit in hits:
            kind = getattr(hit.item, "kind", None)
            if top_hit_kind is None:
                top_hit_kind = kind
            if kind == "fact":
                self.search_stats["all_hits_facts"] += 1
                mapped_ids = list(hit.item.parent_ids) or [hit.item.id]
            elif kind == "episode":
                self.search_stats["all_hits_episodes"] += 1
                mapped_ids = [hit.item.id]
            else:
                self.search_stats["all_hits_other"] += 1
                mapped_ids = list(hit.item.parent_ids) or [hit.item.id]
            for turn_id in mapped_ids:
                if turn_id not in seen:
                    seen.add(turn_id)
                    turn_ids.append(turn_id)

        self.search_stats["n_searches"] += 1
        if top_hit_kind == "fact":
            self.search_stats["top_hit_facts"] += 1
        elif top_hit_kind == "episode":
            self.search_stats["top_hit_episodes"] += 1
        elif top_hit_kind is not None:
            self.search_stats["top_hit_other"] += 1

        return turn_ids[:top_k]


# ---------------------------------------------------------------------------
# P9 step A/B/C: query planning + RRF fusion, fact links, stronger reranker.
# ---------------------------------------------------------------------------
def _map_hits_to_turn_ids(hits: list[Any]) -> list[str]:
    """Map a list of `memory_os_v4.SearchHit` to a deduped, order-preserving
    list of parent turn ids: a ``fact``/other-kind hit maps via
    ``item.parent_ids`` (falling back to its own id if empty), an
    ``episode`` hit maps to its own id. Same mapping rule as
    `V4EnrichedRetriever.search` above, factored out here so
    `V4PlannedRetriever` can call it once per sub-query/hypothesis without
    duplicating (and without disturbing `V4EnrichedRetriever`'s own tested
    `search_stats` bookkeeping, which stays inline in that method).
    """
    seen: set[str] = set()
    turn_ids: list[str] = []
    for hit in hits:
        kind = getattr(hit.item, "kind", None)
        mapped_ids = [hit.item.id] if kind == "episode" else (list(hit.item.parent_ids) or [hit.item.id])
        for turn_id in mapped_ids:
            if turn_id not in seen:
                seen.add(turn_id)
                turn_ids.append(turn_id)
    return turn_ids


def rrf_fuse(ranked_lists: list[tuple[list[str], float]], k: int = 60) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion over several best-first id lists.

    Each element of `ranked_lists` is ``(ranked_ids, weight)``. RRF uses
    POSITION, not any underlying score: id at (0-based) rank ``r`` in a list
    with weight ``w`` contributes ``w / (k + r + 1)`` (1-based rank, standard
    Cormack et al. 2009 form). An id absent from a given list simply gets no
    contribution from it -- never a penalty. Returns ``{id: fused_score}``,
    summed across every list the id appears in; use `_rank_by_score` to turn
    this into a final ranked list with the codebase's standard
    (score desc, id asc) deterministic tiebreak.

    ``k=60`` is the standard RRF constant; P9 §A weights each *component*
    (original question / subqueries / hypotheses) via `weight` on that
    component's list rather than varying `k` per list.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids, weight in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids):
            scores[doc_id] += weight / (k + rank + 1)
    return dict(scores)


def _rank_by_score(scores: dict[str, float]) -> list[str]:
    """Deterministic (score desc, id asc) ordering -- the same tiebreak
    convention `src/memory_os_v4/retrieval.py`'s `HybridRetriever.search`
    uses, so fused rankings don't jitter with dict/set iteration order."""
    return sorted(scores.keys(), key=lambda d: (-scores[d], d))


# ---------------------------------------------------------------------------
# Idées inter-disciplines (docs/IDEAS.md, 2026-09-14): five opt-in mechanisms
# layered onto V4PlannedRetriever's fusion pipeline, each a pure/testable
# module-level function so tests/test_bench_plan.py can hand-verify them
# without building a whole retriever.
# ---------------------------------------------------------------------------
def quota_fuse(ranked_lists: list[tuple[list[str], float]], seats: int) -> list[str]:
    """Idea 1 (Sainte-Laguë proportional fusion): treat each element of
    `ranked_lists` as a "party" with a fixed weight (its RRF weight --
    original/sub/hyp), and allocate `seats` (top-K positions) one at a time
    by the Sainte-Laguë divisor method: at each round, the party with the
    highest quotient ``weight / (2*seats_won + 1)`` wins the next seat and
    contributes its own next NOT-YET-SELECTED id (in its own ranked order);
    a party with no more unselected ids drops out of contention. Ties in
    quotient are broken by party order (first list -- typically the
    original question -- wins), which is what keeps this deterministic
    without depending on dict/set iteration order.

    This guarantees every non-empty party gets at least one of the first
    `len(parties)` seats (proportional representation -- a multi-hop
    question's own sub-queries each get a turn instead of the single
    "loudest" list sweeping the top-K under plain RRF). Returns fewer than
    `seats` ids only if every party is exhausted first (small/duplicate
    corpora); duplicates across parties are never double-counted (an id
    already selected by one party can never be selected again by another).
    """
    parties = [
        {"ids": list(ids), "weight": weight, "seats_won": 0, "next_idx": 0}
        for ids, weight in ranked_lists
        if weight > 0 and ids
    ]
    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < seats:
        best_idx: Optional[int] = None
        best_quotient = -1.0
        for i, party in enumerate(parties):
            while party["next_idx"] < len(party["ids"]) and party["ids"][party["next_idx"]] in selected_set:
                party["next_idx"] += 1
            if party["next_idx"] >= len(party["ids"]):
                continue
            quotient = party["weight"] / (2 * party["seats_won"] + 1)
            if quotient > best_quotient:
                best_quotient = quotient
                best_idx = i
        if best_idx is None:
            break  # every party exhausted
        party = parties[best_idx]
        candidate = party["ids"][party["next_idx"]]
        selected.append(candidate)
        selected_set.add(candidate)
        party["seats_won"] += 1
        party["next_idx"] += 1
    return selected


def quota_then_rrf_fuse(
    ranked_lists: list[tuple[list[str], float]],
    fused_scores: dict[str, float],
    quota_seats: int,
    total_seats: int,
) -> list[str]:
    """Idea 1 variant `quota_then_rrf`: Sainte-Laguë quota for the first
    `quota_seats` positions only (see `quota_fuse`), plain RRF
    (`fused_scores`, `_rank_by_score` order) for the remaining positions up
    to `total_seats` (and beyond -- every remaining scored id is appended,
    RRF-ordered, after the quota head)."""
    head = quota_fuse(ranked_lists, quota_seats)
    head_set = set(head)
    tail = [doc_id for doc_id in _rank_by_score(fused_scores) if doc_id not in head_set]
    return head + tail


def _compute_owner_map(labeled_lists: list[tuple[str, list[str]]]) -> dict[str, str]:
    """Idea 2 support: for every id appearing in at least one of
    `labeled_lists` (``[(label, ranked_ids), ...]``), its "owner" is the
    label of the list in which it ranks best (lowest 0-based index); a tie
    keeps the FIRST list to claim that rank (so `labeled_lists` order --
    original, then sub-queries, then hypotheses -- is the tiebreak)."""
    best_rank: dict[str, int] = {}
    owner: dict[str, str] = {}
    for label, ids in labeled_lists:
        for rank, doc_id in enumerate(ids):
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
                owner[doc_id] = label
    return owner


def subquery_mmr_reorder(
    ranked_ids: list[str],
    scores: dict[str, float],
    owner_of: dict[str, str],
    lam: float = 0.6,
) -> tuple[list[str], dict[str, float]]:
    """Idea 2 (portfolio-diversification MMR): greedily re-order
    `ranked_ids` so a candidate whose owner sub-query (`owner_of`, see
    `_compute_owner_map`) is already represented `n` times among the
    already-selected candidates is penalised by `lam ** n` on its `scores`
    value. An id with no owner (not present in any of the original
    query-lists -- e.g. one added later by fact-link/adjacency expansion)
    gets a singleton owner group of its own id, so it is never penalised
    against (or for) any other candidate.

    Deterministic: at each step the candidate with the highest
    ``score * lam**n`` wins; ties are broken by the UNPENALISED fused score
    (higher wins), then by id ascending -- "ties broken by fused score" per
    docs/IDEAS.md idea 2. Returns ``(new_order, new_scores)`` where
    `new_scores[id]` is the penalised score AT THE MOMENT it was selected
    (so a downstream consumer -- e.g. the §C rerank blend -- sees the
    diversified ranking, not just a reshuffled list with stale scores).
    """
    counts: dict[str, int] = defaultdict(int)
    remaining = list(ranked_ids)
    result: list[str] = []
    new_scores: dict[str, float] = {}
    while remaining:
        best_id = None
        best_key = None
        for doc_id in remaining:
            owner = owner_of.get(doc_id, doc_id)
            n = counts[owner]
            base = scores.get(doc_id, 0.0)
            adjusted = base * (lam**n)
            key = (-adjusted, -base, doc_id)
            if best_key is None or key < best_key:
                best_key = key
                best_id = doc_id
        result.append(best_id)
        new_scores[best_id] = -best_key[0]
        counts[owner_of.get(best_id, best_id)] += 1
        remaining.remove(best_id)
    return result, new_scores


def signal_confidence(top_scores: list[float]) -> float:
    """Idea 3 (sensor-fusion confidence): normalised margin between a
    query list's own top-1 and top-2 ENGINE scores (`top_scores`, best
    first -- e.g. `[hit.score for hit in hits]` before any turn-id mapping),
    clipped to ``[0.3, 1.0]``. A "flat" list (top-1 barely beats top-2, or
    top-1 is negative/near-zero) is uninformative and loses weight down to
    the 0.3 floor; a list with a clear top-1 never gets boosted above 1.0
    (this only ever DOWN-weights a list relative to its configured base
    weight, never up-weights it). Fewer than 2 scores means there is no
    margin to judge: a single hit gets full confidence (nothing to compare
    against, no evidence it's flat), zero hits get the floor (irrelevant
    either way -- an empty list contributes nothing to RRF regardless of
    its weight).
    """
    if len(top_scores) >= 2:
        top1, top2 = top_scores[0], top_scores[1]
        denom = abs(top1) if abs(top1) > 1e-9 else 1e-9
        margin = (top1 - top2) / denom
        return max(0.3, min(1.0, margin))
    return 1.0 if top_scores else 0.3


# Idea 4 (adjacent-pair expansion): a turn "looks like a question" if its
# raw text (speaker prefix stripped) ends with "?" or opens with one of
# these (English LoCoMo dialogues; kept intentionally small/explicit rather
# than a POS tagger -- this is a bench-only heuristic, never src/).
QUESTION_WORDS = frozenset(
    {
        "who", "whom", "whose", "what", "when", "where", "why", "how", "which",
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "will", "should", "have", "has", "had",
    }
)


def _is_question_turn(text: str, speaker: str) -> bool:
    """Idea 4 support: strip the ``"speaker: "`` prefix `V4Retriever.add`
    bakes into `text` (if present) and check whether what remains ends with
    "?" or opens with a `QUESTION_WORDS` token."""
    stripped = text.strip()
    prefix = f"{speaker}: " if speaker else ""
    if prefix and stripped.startswith(prefix):
        stripped = stripped[len(prefix):]
    stripped = stripped.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    first_word = re.split(r"[\s,]+", stripped, maxsplit=1)[0].strip(".,!?;:\"'").lower()
    return first_word in QUESTION_WORDS


def apply_llm_ranking(
    candidate_ids: list[str], llm_ranked_ids: Optional[list[str]], top_k: int
) -> list[str]:
    """P9 step D: combine a candidate pool with an offline LLM's listwise
    ranking of (some of) those candidates.

    Returns the LLM-ranked ids first (deduped, in the LLM's own order), then
    every remaining `candidate_ids` id not already included (in their
    original -- i.e. the retriever's/dump's -- order), cut to `top_k`.
    `llm_ranked_ids` is None/empty (no LLM ranking exists yet for this
    question, or every one of its labels failed to map to a turn id) ->
    returns `candidate_ids[:top_k]` unchanged (parent/dump order), never an
    error. Used identically by `V4LLMRerankedRetriever.search` (candidates
    freshly recomputed) and `bench.run_bench`'s `--from-dump` evaluation
    (candidates read straight from a `--dump-candidates` file), so the two
    code paths can never disagree on the combination rule.
    """
    if not llm_ranked_ids:
        return candidate_ids[:top_k]
    seen: set[str] = set()
    ordered: list[str] = []
    for turn_id in llm_ranked_ids:
        if turn_id not in seen:
            seen.add(turn_id)
            ordered.append(turn_id)
    for turn_id in candidate_ids:
        if turn_id not in seen:
            seen.add(turn_id)
            ordered.append(turn_id)
    return ordered[:top_k]


# ---------------------------------------------------------------------------
# Idea 8 (docs/IDEAS.md, 2026-09-19): facet channel -- a faceted index with a
# closed predicate vocabulary raises the recall CEILING for multi-hop
# questions beyond what the same number of ordinary candidates gives (see
# docs/IDEAS.md "témoin à taille égale"); these two pure/testable functions
# are the bench-side implementation V4PlannedRetriever's `facet_channel`
# option wires into its RRF fusion.
# ---------------------------------------------------------------------------
def build_facet_index(fact_facets: dict[str, FactFacetRecord]) -> dict[tuple[str, str], set[str]]:
    """``(subject, predicate) -> {turn_id, ...}`` over one conversation's own
    `bench.enrich.load_fact_facets` output. Predicate ``"other"`` (the
    closed vocabulary's fallback bucket, `bench/data/enrich/PREDICATES.md`)
    is always skipped -- exactly the rule `bench/facet_size_matched.py`'s
    offline "prédicat exact" analysis already validated (docs/IDEAS.md).
    Pure function of `fact_facets`, so it is naturally per-conversation: it
    is rebuilt from scratch by `V4PlannedRetriever.add` on every `reset()`
    -> `add()` cycle (bench.run_bench's per-conversation loop), never leaked
    or accumulated across conversations.
    """
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fid, record in fact_facets.items():
        turn_id = fid_to_turn_id(fid)
        for facet in record.facets:
            predicate = facet.get("predicate")
            if not predicate or predicate == "other":
                continue
            subject = facet.get("subject")
            if not subject:
                continue
            index[(subject, predicate)].add(turn_id)
    return dict(index)


def rank_facet_pool(pool: set[str], hit_lists: list[list[str]]) -> list[str]:
    """Turn a facet `pool` (turn ids matching >=1 of the question's own
    (subject, predicate) facets, see `build_facet_index`) into a RANKED list
    for RRF fusion: ordered by the best (lowest) 0-based rank a member
    achieves across `hit_lists` -- the per-query hit lists already computed
    for THIS question (original question, then sub-queries, then
    hypotheses, in that order; see `V4PlannedRetriever.search`), never
    recomputed. A member that appears in none of `hit_lists` goes last; ties
    (equal best rank, or "absent from everything") are broken by turn id
    ascending, matching this codebase's standard determinism convention
    (`_rank_by_score`).
    """
    best_rank: dict[str, int] = {}
    for ids in hit_lists:
        for rank, doc_id in enumerate(ids):
            if doc_id not in pool:
                continue
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
    return sorted(pool, key=lambda d: (best_rank.get(d, math.inf), d))


class V4PlannedRetriever(V4EnrichedRetriever):
    """P9 §A-C (PLAN.md): `V4EnrichedRetriever` plus

    - **§A query planning**: for a question with a precomputed plan in
      ``bench/data/enrich/locomo_queries/<conv_id>.jsonl`` (see
      `bench.enrich.load_query_plans`), run `engine.search` (reranking
      disabled -- ``rerank=False``, ``top_k=top_k*4`` oversample) once for
      the ORIGINAL question, once per subquery, once per hypothesis
      (selected by `plan_mode`), map every hit list to turn ids exactly like
      `V4EnrichedRetriever.search` does (fact hit -> parent turn), then fuse
      all the resulting ranked turn-id lists with weighted RRF
      (`rrf_fuse`/`_rank_by_score`; `rrf_k`, `weight_original`, `weight_sub`,
      `weight_hyp` are all configurable). A question with no plan for its
      `qid` (not yet produced by the concurrent worker, or `--queries-dir`
      not given) falls back to searching the original question alone --
      counted in `plan_stats["n_fallback_missing_plan"]`, never an error.
    - **§B fact links** (`fact_links=True`): an A-Mem-style link graph over
      this conversation's enriched facts (`_build_link_graph`), used after
      fusion to add the turns of facts linked to the top `link_seed` fused
      turns, at ``seed_score * link_decay`` (`_expand_fact_links`).
    - **§C reranker**: an optional cross-encoder (`rerank_model`, default
      unset -- e.g. ``tomaarsen/ms-marco-ettin-17m-reranker`` or the P9 §C
      ``-68m`` variant) applied by the BENCH itself, once, to the top
      `rerank_top_n` FUSED (and, if enabled, link-expanded) turns --
      deliberately independent of `memory_os_v4.Config.rerank_model` (never
      set here; every `engine.search` call above passes ``rerank=False``),
      per the task brief: "apply the reranker yourself at the end". Scores
      ``(question, turn_text [+ its facts joined])`` pairs and either blends
      0.5/0.5 with the pre-rerank fused score (`plan_rerank_blend=True`,
      default, mirrors `retrieval.py`'s own blend) or ranks by the
      cross-encoder score alone (`plan_rerank_blend=False`).
    - **Idées inter-disciplines** (`docs/IDEAS.md`, 2026-09-14; all default
      OFF, applied in this order after §A fusion and before §C rerank):
      `fusion_mode` (idea 1, Sainte-Laguë `quota`/`quota_then_rrf` fusion,
      see `quota_fuse`/`quota_then_rrf_fuse`) replaces the plain RRF
      ordering of the fused list; `iterative_budget` (idea 5, foraging: a
      sub-query with none of its own hits in the fused top-20 is re-run
      alone at `top_k*8` and its own top-5 merged into positions 11-20,
      capped at 2 extra searches/question) fills sub-query coverage gaps;
      `fact_links`/§B (existing, above) and `adjacency_pair` (idea 4:
      a top-10 turn that reads as a question, asked by a different speaker
      than the immediately-following same-session turn, pulls that
      following turn in at `score*0.8`) both ADD candidates; `subquery_mmr`
      (idea 2: greedy `lam**n` penalty on a candidate whose owner
      sub-query/hypothesis/original-question list is already represented n
      times, see `subquery_mmr_reorder`) re-orders the result one last time
      before §C rerank. `signal_confidence` (idea 3) instead modifies §A's
      OWN weighted-RRF fusion step: each query list's configured weight
      (`weight_original`/`weight_sub`/`weight_hyp`) is scaled by
      `signal_confidence(engine_scores)` -- the normalised top1/top2 margin
      of that list's own raw engine scores, clipped to [0.3, 1.0] -- so a
      "flat" (uninformative) list is down-weighted in the fusion itself.
    - **Idea 8 -- facet channel + implications** (`docs/IDEAS.md`, idea 8,
      2026-09-19; both default OFF, independent of each other and of every
      mechanism above):
      `facet_channel` builds, per conversation, a closed-predicate index
      ``(subject, predicate) -> {turn_id, ...}`` from
      ``bench/data/enrich/locomo_facets/<conv_id>.jsonl`` (`build_facet_index`,
      predicate ``"other"`` always skipped -- exact match only, no
      predicate-family widening, per the task brief). At search time the
      question's own facets (``bench/data/enrich/locomo_qfacets/<conv_id>.jsonl``,
      looked up by `qid`) select the union of matching turns (the "facet
      pool"), which `rank_facet_pool` turns into ONE MORE RRF party (weight
      `facet_weight`, default 0.5) ranked by each pool member's own best
      rank across the original/sub-query/hypothesis hit lists already
      computed for this question -- added to `ranked_lists` BEFORE
      `rrf_fuse` runs, so it takes part in fusion exactly like any other
      query list, and downstream steps (§B/§C, `--rerank-top-n`) see its
      candidates like any other fused candidate. A question with no facets,
      a `qid` not in `_qfacets`, or `facet_weight <= 0` contributes an
      empty/omitted party -- byte-identical to `facet_channel=False`.
      `implications` indexes each fact's implications (same file) as
      additional `add_facts` items (``parent_ids=[turn_id]``, ``tags=
      ["implication"]``, ``importance=0.5``) -- one `add_facts` batch per
      turn, same mechanism/cost profile as `V4EnrichedRetriever`'s own
      enriched facts (see that class's docstring) -- so a hit on an
      implication is credited to its parent turn by the SAME fact->parent
      mapping every other fact-kind hit already goes through
      (`_map_hits_to_turn_ids` / `V4EnrichedRetriever.search`); no new
      credit-mapping code needed. `facet_stats` (cumulative across the
      whole run, like `search_stats`) records facet coverage (how many
      searches had >=1 facet, mean pool size, mean pool members NOT already
      in the pre-facet fused top-50) and how many implication items were
      indexed -- reported in `bench.run_bench`'s ``metadata["facets"]``.

    `search()` needs the question's `qid` to look up its plan, which
    `bench.protocol.Retriever.search(query, top_k, at_time)` doesn't carry --
    `accepts_qid = True` is a documented, additive signal `bench/run_bench.py`
    checks (via `getattr`) to pass `qid=q.qid` as an extra keyword only to
    retrievers that declare support for it; every other retriever's 3-arg
    call is unaffected.
    """

    def __init__(
        self,
        config_overrides: Optional[dict] = None,
        name: str = "v4_planned",
        mode: str = "facts+keywords",
        enrich_dir: Optional[str] = None,
        queries_dir: Optional[str] = None,
        plan_mode: str = "sub+hyp",
        weight_original: float = 1.0,
        weight_sub: float = 1.0,
        weight_hyp: float = 0.7,
        rrf_k: int = 60,
        rerank_model: Optional[str] = None,
        rerank_device: str = "auto",
        rerank_top_n: int = 30,
        plan_rerank_blend: bool = True,
        fact_links: bool = False,
        link_seed: int = 8,
        link_decay: float = 0.5,
        rare_entity_max_df_frac: float = 0.05,
        rare_entity_min_len: int = 3,
        link_same_session_max_turn_gap: int = 2,
        reranker: Optional[Any] = None,
        # ---- Idées inter-disciplines (docs/IDEAS.md, 2026-09-14): all off by default ----
        fusion_mode: str = "rrf",
        quota_then_rrf_boundary: int = 10,
        subquery_mmr: bool = False,
        mmr_lambda: float = 0.6,
        signal_confidence: bool = False,
        adjacency_pair: bool = False,
        adjacency_pair_top_n: int = 10,
        adjacency_pair_decay: float = 0.8,
        iterative_budget: bool = False,
        iterative_budget_max_extra_searches: int = 2,
        iterative_budget_oversample_mult: int = 8,
        iterative_budget_merge_top: int = 5,
        iterative_budget_check_top: int = 20,
        # ---- Idea 8 (docs/IDEAS.md, 2026-09-19): off by default ----
        facet_channel: bool = False,
        facet_weight: float = 0.5,
        facets_dir: Optional[str] = None,
        qfacets_dir: Optional[str] = None,
        implications: bool = False,
    ):
        valid_plan_modes = ("sub", "sub+hyp", "hyp")
        if plan_mode not in valid_plan_modes:
            raise ValueError(f"unknown plan_mode {plan_mode!r}, expected one of {valid_plan_modes}")
        valid_fusion_modes = ("rrf", "quota", "quota_then_rrf")
        if fusion_mode not in valid_fusion_modes:
            raise ValueError(f"unknown fusion_mode {fusion_mode!r}, expected one of {valid_fusion_modes}")
        super().__init__(config_overrides=config_overrides, name=name, mode=mode, enrich_dir=enrich_dir)
        self.queries_dir = Path(queries_dir) if queries_dir is not None else None
        self.plan_mode = plan_mode
        self.weight_original = weight_original
        self.weight_sub = weight_sub
        self.weight_hyp = weight_hyp
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.plan_rerank_blend = plan_rerank_blend
        self.fact_links = fact_links
        self.link_seed = link_seed
        self.link_decay = link_decay
        self.rare_entity_max_df_frac = rare_entity_max_df_frac
        self.rare_entity_min_len = rare_entity_min_len
        self.link_same_session_max_turn_gap = link_same_session_max_turn_gap
        # ---- Idées inter-disciplines ----
        self.fusion_mode = fusion_mode
        self.quota_then_rrf_boundary = quota_then_rrf_boundary
        self.subquery_mmr = subquery_mmr
        self.mmr_lambda = mmr_lambda
        self.signal_confidence = signal_confidence
        self.adjacency_pair = adjacency_pair
        self.adjacency_pair_top_n = adjacency_pair_top_n
        self.adjacency_pair_decay = adjacency_pair_decay
        self.iterative_budget = iterative_budget
        self.iterative_budget_max_extra_searches = iterative_budget_max_extra_searches
        self.iterative_budget_oversample_mult = iterative_budget_oversample_mult
        self.iterative_budget_merge_top = iterative_budget_merge_top
        self.iterative_budget_check_top = iterative_budget_check_top
        # ---- Idea 8 (docs/IDEAS.md, 2026-09-19) ----
        self.facet_channel = facet_channel
        self.facet_weight = facet_weight
        self.facets_dir = Path(facets_dir) if facets_dir is not None else None
        self.qfacets_dir = Path(qfacets_dir) if qfacets_dir is not None else None
        self.implications = implications
        # bench/run_bench.py: thread q.qid through to search() -- see class
        # docstring for why the base 3-arg Retriever protocol can't carry it.
        self.accepts_qid = True

        # `reranker` (a pre-built object exposing `.available()`/`.score()`,
        # e.g. a test's fake reranker or memory_os_v4.rerank.CrossEncoderReranker)
        # takes precedence when given; otherwise build a real CrossEncoderReranker
        # lazily from `rerank_model` (None -> no reranking at all, same as
        # `rerank_top_n=0`). Never imports memory_os_v4.rerank unless a real
        # model name is actually requested, so this class stays importable/
        # instantiable even before that module exists.
        self._reranker: Optional[Any] = reranker
        if self._reranker is None and rerank_model:
            from memory_os_v4.rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(rerank_model, rerank_device)

        self.plan_stats: dict[str, int] = {
            "n_searches": 0,
            "n_plan_used": 0,
            "n_fallback_missing_plan": 0,
        }
        # Idea 8 (docs/IDEAS.md, 2026-09-19): cumulative across the WHOLE
        # run (every conversation's add()/search() calls), never cleared by
        # reset() -- same convention as `V4EnrichedRetriever.search_stats`
        # (bench/run_bench.py reports the final totals once, in
        # metadata["facets"]). `n_implications_indexed` is tracked
        # independently of `facet_channel` (the `implications` option is
        # its own independent flag, see class docstring).
        self.facet_stats: dict[str, int] = {
            "n_searches": 0,
            "n_questions_with_facets": 0,
            "total_pool_size": 0,
            "total_new_vs_top50": 0,
            "n_implications_indexed": 0,
        }
        self._reset_plan_state()

    def _reset_plan_state(self) -> None:
        self._queries: dict[str, QueryPlan] = {}
        self._facts_by_turn: dict[str, list[str]] = {}
        self._fact_text: dict[str, str] = {}
        self._fact_parent_turn: dict[str, str] = {}
        self._fact_links_graph: dict[str, set[str]] = {}
        # Idea 4 (adjacency-pair): this conversation's own transcript order,
        # per session, plus raw text/speaker -- built by `add()` from the
        # SAME `items` argument used to build the store (not re-derived from
        # the v4 store, which may hold keyword-augmented text under
        # mode="facts+keywords").
        self._turns_by_session: dict[str, list[str]] = {}
        self._turn_order_in_session: dict[str, int] = {}
        self._turn_session: dict[str, str] = {}
        self._turn_text_raw: dict[str, str] = {}
        self._turn_speaker: dict[str, str] = {}
        # Idea 8 (docs/IDEAS.md, 2026-09-19): rebuilt fresh per conversation
        # by `add()` -- see `build_facet_index`/`load_qfacets`.
        self._facet_index: dict[tuple[str, str], set[str]] = {}
        self._qfacets: dict[str, list[tuple[str, str]]] = {}

    def reset(self) -> None:
        super().reset()
        self._reset_plan_state()

    def add(self, items: list[BenchItem]) -> None:
        super().add(items)  # indexes raw turns + facts exactly like V4EnrichedRetriever
        if not items:
            return

        conv_id = (items[0].meta or {}).get("conv_id")
        if self.queries_dir is not None and conv_id is not None:
            self._queries = load_query_plans(self.queries_dir, conv_id)

        enrichment: dict[str, EnrichRecord] = {}
        if self.enrich_dir is not None and conv_id is not None:
            # Re-loaded here (rather than threading it out of the parent
            # `add()` call) because it's cheap (small per-conversation JSONL,
            # already read once by `super().add()`) and keeps this class's
            # own bookkeeping (`_facts_by_turn`, the link graph) independent
            # of `V4EnrichedRetriever`'s internals.
            enrichment = load_enrichment(self.enrich_dir, conv_id)

        self._build_fact_index(items, enrichment)
        if self.fact_links:
            self._build_link_graph(items, enrichment)
        if self.adjacency_pair:
            self._build_session_order(items)

        # Idea 8 (docs/IDEAS.md, 2026-09-19): facet channel + implications
        # both read the SAME `locomo_facets/<conv_id>.jsonl` file, loaded at
        # most once per add() call regardless of how many of the two options
        # are on.
        if (self.facet_channel or self.implications) and self.facets_dir is not None and conv_id is not None:
            fact_facets = load_fact_facets(self.facets_dir, conv_id)
        else:
            fact_facets = {}
        if self.facet_channel:
            self._facet_index = build_facet_index(fact_facets)
            self._qfacets = (
                load_qfacets(self.qfacets_dir, conv_id)
                if (self.qfacets_dir is not None and conv_id is not None)
                else {}
            )
        if self.implications:
            self.facet_stats["n_implications_indexed"] += self._index_implications(items, fact_facets)

    def _index_implications(self, items: list[BenchItem], fact_facets: dict[str, FactFacetRecord]) -> int:
        """Idea 8 (docs/IDEAS.md, implications channel): index each fact's
        implications as additional `add_facts` items, ``parent_ids=
        [turn_id]``, ``tags=["implication"]``, ``importance=0.5`` -- same
        mechanism/batching rationale as `V4EnrichedRetriever.add`'s own
        enriched facts (one `add_facts` batch per turn, see that class's
        docstring for why per-turn rather than per-fact/per-conversation).
        A hit on an indexed implication is credited to its parent turn by
        the SAME fact->parent mapping every other fact-kind hit already
        goes through (`_map_hits_to_turn_ids` / `V4EnrichedRetriever.
        search`) -- nothing else needs to change for that credit to happen.
        Returns the number of implication items actually added, for
        `facet_stats["n_implications_indexed"]`.
        """
        by_turn: dict[str, list[str]] = defaultdict(list)
        valid_turn_ids = {item.id for item in items}
        event_time_by_turn = {item.id: item.event_time for item in items}
        for fid, record in fact_facets.items():
            if not record.implications:
                continue
            turn_id = fid_to_turn_id(fid)
            if turn_id not in valid_turn_ids:
                continue  # defensive: a facets-file entry outside this batch
            by_turn[turn_id].extend(record.implications)

        n_added = 0
        for turn_id, implications in by_turn.items():
            facts_payload = [
                {
                    "text": implication,
                    "event_time": event_time_by_turn.get(turn_id),
                    "importance": 0.5,
                    "tags": ["implication"],
                }
                for implication in implications
            ]
            self._memory.add_facts(
                facts_payload,
                project="bench",
                parent_ids=[turn_id],
                source="bench_enrich",
            )
            n_added += len(facts_payload)
        return n_added

    def _build_session_order(self, items: list[BenchItem]) -> None:
        """Idea 4 support: record each turn's raw text/speaker and its
        0-based position within its own session's transcript order (the
        order `items` -- this conversation's own turn list -- already comes
        in, i.e. insertion order, per the task brief)."""
        by_session: dict[str, list[str]] = defaultdict(list)
        for item in items:
            idx = len(by_session[item.session_id])
            by_session[item.session_id].append(item.id)
            self._turn_order_in_session[item.id] = idx
            self._turn_session[item.id] = item.session_id
            self._turn_text_raw[item.id] = item.text
            self._turn_speaker[item.id] = item.speaker
        self._turns_by_session = dict(by_session)

    def _build_fact_index(self, items: list[BenchItem], enrichment: dict[str, EnrichRecord]) -> None:
        for item in items:
            record = enrichment.get(item.id)
            if not record or not record.facts:
                continue
            fact_ids = []
            for i, fact_text in enumerate(record.facts):
                fid = f"{item.id}#f{i}"
                self._fact_text[fid] = fact_text
                self._fact_parent_turn[fid] = item.id
                fact_ids.append(fid)
            self._facts_by_turn[item.id] = fact_ids

    # ------------------------------------------------------------ §B links
    def _build_link_graph(self, items: list[BenchItem], enrichment: dict[str, EnrichRecord]) -> None:
        """Build the A-Mem-style fact link graph for one conversation (see
        class docstring §B): two facts are linked if they (1) share >=1
        "rare" entity -- document frequency (over TURNS, not facts) <=
        `rare_entity_max_df_frac`, >= `rare_entity_min_len` chars, and not a
        speaker name (this conversation's own `item.speaker` values,
        case-insensitive) -- OR (2) belong to the same session (`doc_ref`,
        i.e. `item.session_id`) and are within `link_same_session_max_turn_gap`
        turns of each other (turn position = index in `items`, this
        conversation's own transcript order). Populates
        `self._fact_links_graph: {fact_id: {linked_fact_id, ...}}`.
        """
        n_turns = len(items)
        if n_turns == 0:
            self._fact_links_graph = {}
            return
        speaker_names = {it.speaker.strip().lower() for it in items if it.speaker}

        entity_df: dict[str, int] = defaultdict(int)
        turn_entities: dict[str, set[str]] = {}
        for item in items:
            record = enrichment.get(item.id)
            ents = {e.strip().lower() for e in (record.entities if record else []) if e.strip()}
            turn_entities[item.id] = ents
            for e in ents:
                entity_df[e] += 1

        def is_rare(entity: str) -> bool:
            if len(entity) < self.rare_entity_min_len:
                return False
            if entity in speaker_names:
                return False
            df = entity_df.get(entity, 0)
            return df > 0 and (df / n_turns) <= self.rare_entity_max_df_frac

        order_by_turn = {item.id: idx for idx, item in enumerate(items)}
        session_by_turn = {item.id: item.session_id for item in items}

        # entity -> [fact_id, ...], rare entities only.
        entity_buckets: dict[str, list[str]] = defaultdict(list)
        for turn_id, fact_ids in self._facts_by_turn.items():
            rare = {e for e in turn_entities.get(turn_id, ()) if is_rare(e)}
            for fid in fact_ids:
                for e in rare:
                    entity_buckets[e].append(fid)

        links: dict[str, set[str]] = defaultdict(set)
        for fids in entity_buckets.values():
            for a in fids:
                for b in fids:
                    if a != b:
                        links[a].add(b)

        all_fact_ids = list(self._fact_parent_turn.keys())
        for a in all_fact_ids:
            ta = self._fact_parent_turn[a]
            sa, oa = session_by_turn.get(ta), order_by_turn.get(ta)
            for b in all_fact_ids:
                if a == b:
                    continue
                tb = self._fact_parent_turn[b]
                if session_by_turn.get(tb) != sa:
                    continue
                if abs(order_by_turn.get(tb, 0) - (oa or 0)) <= self.link_same_session_max_turn_gap:
                    links[a].add(b)

        self._fact_links_graph = dict(links)

    def _expand_fact_links(
        self, ranked_ids: list[str], scores: dict[str, float]
    ) -> tuple[list[str], dict[str, float]]:
        """For the top `link_seed` fused turns, hop (one step) to every turn
        owning a fact linked (`self._fact_links_graph`) to one of the seed
        turn's own facts, scored at ``seed_score * link_decay`` -- a turn
        already present anywhere in `ranked_ids` is left untouched (never
        overridden, matching `src/memory_os_v4/retrieval.py`'s
        `_expand_entities` convention of only ever ADDING new candidates). A
        turn reachable from several seeds/facts keeps the highest such
        score. No-op (returns the inputs unchanged) if nothing was added.
        """
        seeds = ranked_ids[: self.link_seed]
        added: dict[str, float] = {}
        for seed_turn_id in seeds:
            seed_score = scores.get(seed_turn_id, 0.0)
            for fid in self._facts_by_turn.get(seed_turn_id, ()):
                for linked_fid in self._fact_links_graph.get(fid, ()):
                    linked_turn = self._fact_parent_turn.get(linked_fid)
                    if linked_turn is None or linked_turn in scores:
                        continue
                    new_score = seed_score * self.link_decay
                    if linked_turn not in added or added[linked_turn] < new_score:
                        added[linked_turn] = new_score
        if not added:
            return ranked_ids, scores
        merged_scores = dict(scores)
        merged_scores.update(added)
        merged_ranked = _rank_by_score(merged_scores)
        return merged_ranked, merged_scores

    # ------------------------------------------------------------ idea 4
    def _adjacency_pair_expand(
        self, ranked_ids: list[str], scores: dict[str, float]
    ) -> tuple[list[str], dict[str, float]]:
        """Idea 4 (adjacent-pair expansion, docs/IDEAS.md): for each of the
        top `adjacency_pair_top_n` fused turns whose raw text reads as a
        question (`_is_question_turn`) AND whose speaker differs from the
        speaker of the immediately-following SAME-SESSION turn (transcript
        order from `_build_session_order`, i.e. `add()`'s own insertion
        order -- never re-derived from the v4 store), add that following
        turn at ``seed_score * adjacency_pair_decay`` -- never overriding a
        turn already present anywhere in `ranked_ids` (same "only ever ADD"
        convention as `_expand_fact_links`). A turn reachable as the
        follow-up of several qualifying seeds keeps the highest such score.
        No-op if nothing was added, or if `_build_session_order` was never
        run for this conversation (`adjacency_pair=False`).
        """
        seeds = ranked_ids[: self.adjacency_pair_top_n]
        added: dict[str, float] = {}
        for seed_id in seeds:
            text = self._turn_text_raw.get(seed_id)
            speaker = self._turn_speaker.get(seed_id)
            if text is None or not _is_question_turn(text, speaker or ""):
                continue
            session_id = self._turn_session.get(seed_id)
            idx = self._turn_order_in_session.get(seed_id)
            if session_id is None or idx is None:
                continue
            session_turns = self._turns_by_session.get(session_id, [])
            if idx + 1 >= len(session_turns):
                continue
            next_id = session_turns[idx + 1]
            if self._turn_speaker.get(next_id) == speaker:
                continue  # same speaker following themselves -> not a partner's answer
            if next_id in scores:
                continue  # already present -- never override
            new_score = scores.get(seed_id, 0.0) * self.adjacency_pair_decay
            if next_id not in added or added[next_id] < new_score:
                added[next_id] = new_score
        if not added:
            return ranked_ids, scores
        merged_scores = dict(scores)
        merged_scores.update(added)
        merged_ranked = _rank_by_score(merged_scores)
        return merged_ranked, merged_scores

    # ------------------------------------------------------------ idea 5
    def _iterative_forage(
        self,
        plan_ranked_lists: list[tuple[str, list[str], float]],
        fused_ranked: list[str],
        fused_scores: dict[str, float],
        top_k: int,
        at_time: Optional[float],
    ) -> tuple[list[str], dict[str, float]]:
        """Idea 5 (optimal-foraging iterative budget, docs/IDEAS.md): for
        each sub-query LABEL in `plan_ranked_lists` (label starts with
        ``"sub"``) whose own hit list has NO id in `fused_ranked`'s top
        `iterative_budget_check_top` (a "starved" sub-query -- its own
        evidence never surfaced in the fused top-20), re-run that
        sub-query's TEXT alone at ``top_k * iterative_budget_oversample_mult``
        and merge its own top `iterative_budget_merge_top` ids at fused
        positions 11-20 (after the existing top-10, ahead of whatever
        already occupied position 11+ -- a duplicate of an id already
        anywhere in `fused_ranked` is dropped, never double-counted).
        Capped at `iterative_budget_max_extra_searches` extra
        `self._memory.search()` calls per question (starved sub-queries
        beyond the cap are simply left starved). No-op if there are no
        starved sub-queries or no plan (`plan_ranked_lists` has no "sub*"
        labels).
        """
        check_top = set(fused_ranked[: self.iterative_budget_check_top])
        starved: list[tuple[str, str]] = []  # (label, text)
        for label, ids, _weight in plan_ranked_lists:
            if not label.startswith("sub"):
                continue
            if check_top.isdisjoint(ids):
                starved.append((label, self._sub_query_text_by_label.get(label, "")))
        if not starved:
            return fused_ranked, fused_scores

        head = fused_ranked[:10]
        tail = fused_ranked[10:]
        merged_scores = dict(fused_scores)
        new_head_extras: list[str] = []
        present = set(fused_ranked)
        n_extra_searches = 0
        for label, qtext in starved:
            if n_extra_searches >= self.iterative_budget_max_extra_searches:
                break
            if not qtext or not qtext.strip():
                continue
            n_extra_searches += 1
            oversample_k = top_k * self.iterative_budget_oversample_mult
            hits = self._memory.search(qtext, top_k=oversample_k, at_time=at_time, rerank=False)
            forage_ids = _map_hits_to_turn_ids(hits)[: self.iterative_budget_merge_top]
            base_score = merged_scores.get(head[-1], 0.0) * 0.5 if head else 0.0
            for rank, doc_id in enumerate(forage_ids):
                if doc_id in present:
                    continue
                present.add(doc_id)
                new_head_extras.append(doc_id)
                merged_scores[doc_id] = base_score / (rank + 1)

        if not new_head_extras:
            return fused_ranked, fused_scores
        merged_tail = new_head_extras + [d for d in tail if d not in set(new_head_extras)]
        return head + merged_tail, merged_scores

    # ------------------------------------------------------------ P9 step D
    def facts_for_turn(self, turn_id: str, max_facts: int = 2) -> list[str]:
        """Up to `max_facts` extracted fact texts for `turn_id` (this
        conversation's own `_facts_by_turn`/`_fact_text` index, built by
        `_build_fact_index`), best-effort/empty list if the turn has none.
        Used by `bench/run_bench.py --dump-candidates` (P9 step D) to attach
        a candidate's own facts to its dump entry without callers needing to
        reload/re-parse the enrichment file themselves.
        """
        fact_ids = self._facts_by_turn.get(turn_id, ())
        return [self._fact_text[fid] for fid in fact_ids[:max_facts]]

    # ------------------------------------------------------------ §C rerank
    def _rerank_text(self, turn_id: str) -> str:
        """``turn text [+ its facts joined]`` -- `turn_id`'s own indexed text
        (already keyword-augmented under `mode="facts+keywords"`) plus its
        extracted facts' text, space-joined, for the cross-encoder to score
        against the ORIGINAL question."""
        item = self._memory.store.get(turn_id) if self._memory is not None else None
        base = item.text if item is not None else ""
        fact_ids = self._facts_by_turn.get(turn_id, ())
        if not fact_ids:
            return base
        facts_text = " ".join(self._fact_text[fid] for fid in fact_ids)
        return f"{base} {facts_text}".strip()

    def _apply_rerank(self, query: str, ranked_ids: list[str], scores: dict[str, float]) -> list[str]:
        if self._reranker is None or self.rerank_top_n <= 0 or not ranked_ids:
            return ranked_ids
        top_n_ids = ranked_ids[: self.rerank_top_n]
        rest_ids = ranked_ids[self.rerank_top_n :]
        if not self._reranker.available():
            return ranked_ids
        texts = [self._rerank_text(tid) for tid in top_n_ids]
        ce_scores = self._reranker.score(query, texts)
        if ce_scores is None:
            return ranked_ids
        ce_by_id = dict(zip(top_n_ids, ce_scores))
        if self.plan_rerank_blend:
            from memory_os_v4.retrieval import minmax_normalize

            base_norm = minmax_normalize({tid: scores.get(tid, 0.0) for tid in top_n_ids})
            ce_norm = minmax_normalize(ce_by_id)
            blended = {tid: 0.5 * base_norm.get(tid, 0.0) + 0.5 * ce_norm.get(tid, 0.0) for tid in top_n_ids}
        else:
            blended = ce_by_id
        top_n_sorted = sorted(top_n_ids, key=lambda d: (-blended[d], d))
        return top_n_sorted + rest_ids

    # ------------------------------------------------------------ search
    def search(self, query: str, top_k: int, at_time: Optional[float] = None, qid: Optional[str] = None) -> list[str]:
        if self._memory is None:
            return []
        self.plan_stats["n_searches"] += 1

        plan = self._queries.get(qid) if qid is not None else None
        # (label, text, weight) -- label feeds idea 2's owner map and idea
        # 5's "which sub-query is starved" check; weight is the CONFIGURED
        # weight_original/_sub/_hyp, before any idea-3 confidence scaling.
        labeled_queries: list[tuple[str, str, float]] = [("orig", query, self.weight_original)]
        if plan is None:
            self.plan_stats["n_fallback_missing_plan"] += 1
        else:
            self.plan_stats["n_plan_used"] += 1
            if self.plan_mode in ("sub", "sub+hyp"):
                labeled_queries.extend(
                    (f"sub{i}", sq, self.weight_sub) for i, sq in enumerate(plan.subqueries)
                )
            if self.plan_mode in ("hyp", "sub+hyp"):
                labeled_queries.extend(
                    (f"hyp{i}", hq, self.weight_hyp) for i, hq in enumerate(plan.hypotheses)
                )

        oversample_k = top_k * 4
        ranked_lists: list[tuple[list[str], float]] = []
        labeled_lists: list[tuple[str, list[str]]] = []  # idea 2: owner map
        plan_ranked_lists: list[tuple[str, list[str], float]] = []  # idea 5: starved lookup
        self._sub_query_text_by_label: dict[str, str] = {}
        for label, qtext, base_weight in labeled_queries:
            if not qtext or not qtext.strip():
                continue
            self._sub_query_text_by_label[label] = qtext
            hits = self._memory.search(qtext, top_k=oversample_k, at_time=at_time, rerank=False)
            turn_ids = _map_hits_to_turn_ids(hits)
            weight = base_weight
            if self.signal_confidence:
                # Idea 3: down-weight (never up-weight) a "flat" list -- one
                # whose own engine top1/top2 scores barely differ -- in the
                # RRF fusion itself, before rank-position fusion happens.
                weight = base_weight * signal_confidence([hit.score for hit in hits])
            ranked_lists.append((turn_ids, weight))
            labeled_lists.append((label, turn_ids))
            plan_ranked_lists.append((label, turn_ids, weight))

        if not ranked_lists:
            return []

        # Idea 8 (docs/IDEAS.md, 2026-09-19): facet channel -- built and
        # appended to `ranked_lists` BEFORE `rrf_fuse` runs, so the facet
        # pool takes part in fusion as one more party (never a post-fusion
        # patch): downstream steps (§B/§C, `--rerank-top-n`) then see its
        # candidates exactly like any other fused candidate, with no special
        # casing needed anywhere else in this method.
        if self.facet_channel:
            self.facet_stats["n_searches"] += 1
            qfacets = self._qfacets.get(qid, []) if qid is not None else []
            if qfacets:
                self.facet_stats["n_questions_with_facets"] += 1
            pool: set[str] = set()
            for subject, predicate in qfacets:
                pool |= self._facet_index.get((subject, predicate), set())
            self.facet_stats["total_pool_size"] += len(pool)
            if pool:
                # Informational only (facet_coverage bookkeeping, task
                # brief) -- how many pool members are NOT already in the
                # PLAIN (pre-facet) fused top-50; never affects ranking.
                pre_facet_scores = rrf_fuse(ranked_lists, k=self.rrf_k)
                pre_facet_top50 = set(_rank_by_score(pre_facet_scores)[:50])
                self.facet_stats["total_new_vs_top50"] += len(pool - pre_facet_top50)
                if self.facet_weight > 0:
                    # weight<=0 (or an empty pool, handled by `if pool:`
                    # above) intentionally leaves `ranked_lists` untouched --
                    # not just zero-weighted -- so `rrf_fuse` never creates a
                    # new zero-score dict entry for a facet-only id (which
                    # would otherwise appear in `_rank_by_score`'s output
                    # even at weight 0). This is what makes `facet_weight=0`
                    # reproduce the baseline ranking byte-for-byte.
                    facet_ranked = rank_facet_pool(pool, [ids for ids, _w in ranked_lists])
                    ranked_lists.append((facet_ranked, self.facet_weight))

        fused_scores = rrf_fuse(ranked_lists, k=self.rrf_k)

        if self.fusion_mode == "quota":
            # Idea 1: Sainte-Laguë quota decides the first `rerank_top_n`
            # positions; anything RRF-scored but not quota-selected is
            # appended after, RRF-ordered (never dropped).
            quota_head = quota_fuse(ranked_lists, seats=self.rerank_top_n)
            quota_set = set(quota_head)
            leftover = [doc_id for doc_id in _rank_by_score(fused_scores) if doc_id not in quota_set]
            fused_ranked = quota_head + leftover
        elif self.fusion_mode == "quota_then_rrf":
            # Idea 1 variant: quota for the first `quota_then_rrf_boundary`
            # positions only, plain RRF for the rest.
            fused_ranked = quota_then_rrf_fuse(
                ranked_lists,
                fused_scores,
                quota_seats=self.quota_then_rrf_boundary,
                total_seats=self.rerank_top_n,
            )
        else:
            fused_ranked = _rank_by_score(fused_scores)

        if self.iterative_budget and plan is not None and fused_ranked:
            fused_ranked, fused_scores = self._iterative_forage(
                plan_ranked_lists, fused_ranked, fused_scores, top_k, at_time
            )

        if self.fact_links and fused_ranked:
            fused_ranked, fused_scores = self._expand_fact_links(fused_ranked, fused_scores)

        if self.adjacency_pair and fused_ranked:
            fused_ranked, fused_scores = self._adjacency_pair_expand(fused_ranked, fused_scores)

        if self.subquery_mmr and fused_ranked:
            # Idea 2: final diversification pass before §C rerank.
            owner_of = _compute_owner_map(labeled_lists)
            fused_ranked, fused_scores = subquery_mmr_reorder(
                fused_ranked, fused_scores, owner_of, lam=self.mmr_lambda
            )

        if self._reranker is not None:
            fused_ranked = self._apply_rerank(query, fused_ranked, fused_scores)

        return fused_ranked[:top_k]


# ---------------------------------------------------------------------------
# P9 step D: LLM listwise reranking, offline (PLAN.md).
# ---------------------------------------------------------------------------
class V4LLMRerankedRetriever(V4PlannedRetriever):
    """`V4PlannedRetriever` plus an offline LLM's listwise reranking of its
    own top-`bench.enrich.MAX_DUMP_CANDIDATES` (20) candidates (P9 step D).

    Two file dirs, both optional (missing either -> pure `V4PlannedRetriever`
    behaviour, i.e. parent order, every question falls back):

    - ``candidates_dir``: a `--dump-candidates` output (`bench.enrich.
      load_candidate_dump`) -- one line per question, this conversation's
      top-20 candidate turn ids *and* their ``{"c01": turn_id, ...}``
      ``label_map``. This is intentionally the SAME dump an LLM ranking
      worker was shown, read back here rather than recomputed, so a label
      like ``"c07"`` is always translated using the exact mapping the LLM
      saw -- not a freshly-recomputed candidate list's Nth position, which
      could subtly disagree with the dump (a different `rerank_top_n`
      cutoff windowed differently, a non-deterministic model call, etc.).
    - ``llmrank_dir``: the offline LLM's own output (`bench.enrich.
      load_llm_rankings`) -- one line per question, ``{"qid", "ranked":
      ["c07", "c02", ...]}``, produced by other worker sessions later.

    `search()` still runs the full `V4PlannedRetriever` pipeline (§A
    planning + §B links + §C rerank) to get this question's top-20
    candidates itself -- **the same deterministic config always reproduces
    the same top-20 order as the dump** (no randomness anywhere in that
    pipeline; the whole point of `candidates_dir`/`label_map` is to not have
    to *trust* that assumption for the label->turn_id translation itself).
    That freshly-recomputed top-20 list is also the fallback/"remaining
    candidates" pool `apply_llm_ranking` appends after the LLM-ranked ids
    (see there) -- so a question with no LLM ranking yet, or a `qid` not
    present in `candidates_dir` at all, degrades gracefully to plain
    `V4PlannedRetriever` behaviour.

    ``llm_ranked_questions`` (`self.llm_stats`) counts how many searches
    actually had a usable LLM ranking applied (>=1 label mapped to a turn
    id); `bench/run_bench.py` reports it in `metadata.llm_rerank`.

    Note: `search()`'s candidate pool is always exactly the top-20 (never
    more), so a `top_k` above `MAX_DUMP_CANDIDATES` (20) still returns at
    most 20 ids -- this retriever is deliberately scoped to reordering the
    dump's own top-20, not to serving an arbitrarily large `top_k`.
    """

    def __init__(
        self,
        config_overrides: Optional[dict] = None,
        name: str = "v4_llmrank",
        mode: str = "facts+keywords",
        enrich_dir: Optional[str] = None,
        queries_dir: Optional[str] = None,
        plan_mode: str = "sub+hyp",
        weight_original: float = 1.0,
        weight_sub: float = 1.0,
        weight_hyp: float = 0.7,
        rrf_k: int = 60,
        rerank_model: Optional[str] = None,
        rerank_device: str = "auto",
        rerank_top_n: int = 30,
        plan_rerank_blend: bool = True,
        fact_links: bool = False,
        link_seed: int = 8,
        link_decay: float = 0.5,
        rare_entity_max_df_frac: float = 0.05,
        rare_entity_min_len: int = 3,
        link_same_session_max_turn_gap: int = 2,
        reranker: Optional[Any] = None,
        llmrank_dir: Optional[str] = None,
        candidates_dir: Optional[str] = None,
    ):
        super().__init__(
            config_overrides=config_overrides,
            name=name,
            mode=mode,
            enrich_dir=enrich_dir,
            queries_dir=queries_dir,
            plan_mode=plan_mode,
            weight_original=weight_original,
            weight_sub=weight_sub,
            weight_hyp=weight_hyp,
            rrf_k=rrf_k,
            rerank_model=rerank_model,
            rerank_device=rerank_device,
            rerank_top_n=rerank_top_n,
            plan_rerank_blend=plan_rerank_blend,
            fact_links=fact_links,
            link_seed=link_seed,
            link_decay=link_decay,
            rare_entity_max_df_frac=rare_entity_max_df_frac,
            rare_entity_min_len=rare_entity_min_len,
            link_same_session_max_turn_gap=link_same_session_max_turn_gap,
            reranker=reranker,
        )
        self.llmrank_dir = Path(llmrank_dir) if llmrank_dir is not None else None
        self.candidates_dir = Path(candidates_dir) if candidates_dir is not None else None
        self.llm_stats: dict[str, int] = {
            "n_searches": 0,
            "llm_ranked_questions": 0,
            "n_fallback_no_llm_ranking": 0,
        }
        self._reset_llmrank_state()

    def _reset_llmrank_state(self) -> None:
        self._llm_rankings: dict[str, list[str]] = {}
        self._label_maps: dict[str, dict[str, str]] = {}

    def reset(self) -> None:
        super().reset()
        self._reset_llmrank_state()

    def add(self, items: list[BenchItem]) -> None:
        super().add(items)  # indexes raw turns/facts + loads query plans, exactly like V4PlannedRetriever
        if not items:
            return
        conv_id = (items[0].meta or {}).get("conv_id")
        if conv_id is None:
            return
        if self.llmrank_dir is not None:
            self._llm_rankings = load_llm_rankings(self.llmrank_dir, conv_id)
        if self.candidates_dir is not None:
            dump_records = load_candidate_dump(self.candidates_dir, conv_id)
            self._label_maps = {qid: record.label_map for qid, record in dump_records.items()}

    def _llm_ranked_turn_ids(self, qid: Optional[str]) -> Optional[list[str]]:
        """Translate `qid`'s LLM-ranked labels (if any) into turn ids via
        THIS conversation's dump `label_map` (never by recomputed position --
        see class docstring). Unknown labels are dropped; returns None if
        there is no ranking for `qid`, or every label failed to map (both
        treated identically by `apply_llm_ranking`: fall back to parent
        order)."""
        if qid is None:
            return None
        labels = self._llm_rankings.get(qid)
        if not labels:
            return None
        label_map = self._label_maps.get(qid, {})
        mapped = [label_map[label] for label in labels if label in label_map]
        return mapped or None

    def search(self, query: str, top_k: int, at_time: Optional[float] = None, qid: Optional[str] = None) -> list[str]:
        self.llm_stats["n_searches"] += 1
        candidate_ids = super().search(query, top_k=MAX_DUMP_CANDIDATES, at_time=at_time, qid=qid)
        llm_ranked_ids = self._llm_ranked_turn_ids(qid)
        if llm_ranked_ids:
            self.llm_stats["llm_ranked_questions"] += 1
        else:
            self.llm_stats["n_fallback_no_llm_ranking"] += 1
        return apply_llm_ranking(candidate_ids, llm_ranked_ids, top_k)
