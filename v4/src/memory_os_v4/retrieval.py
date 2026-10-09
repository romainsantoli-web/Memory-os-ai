"""Hybrid retrieval: dense + BM25 + entity candidates, configurable fusion,
recency/importance weighting, validity filtering, optional cross-encoder
rerank, and MMR diversification.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25Index
from .config import Config
from .embed import Embedder, HashEmbedder
from .entities import JUNK_ENTITY_WORDS, EntityIndex, extract_entities
from .facets import FacetIndex, normalize_facet, rank_facet_pool
from .rerank import CrossEncoderReranker
from .store import MemoryItem, Store
from .temporal import is_valid_at, parse_dates, recency_weight
from .text import tokenize
from .vector_index import VectorIndex

logger = logging.getLogger("memory_os_v4")

# Part B (LongMemEval temporal-reasoning targeted experiment, docs/ABLATIONS.md
# "Hardening 2026-09-13"): cues that mark a query as date-heavy even when
# `parse_dates` finds nothing explicit ("How long ago did we ship it?" has no
# parseable date but is still asking about time). EN + FR, matched
# case-insensitively as whole words/phrases.
_TEMPORAL_CUE_RE = re.compile(
    r"\b(when|what date|how long|days|weeks|months|years|ago|last|first time"
    r"|quand|combien de temps|il y a)\b",
    re.IGNORECASE,
)


@dataclass
class SearchHit:
    item: MemoryItem
    score: float
    signals: dict = field(default_factory=dict)


def _passes_filters(
    item: MemoryItem | None,
    kinds: list[str] | None,
    project: str | None,
    at_time: float | None,
) -> bool:
    """The candidate-acceptance rule `HybridRetriever.search` applies to
    every fused candidate: not deleted/missing, kind, project, and
    bi-temporal validity. Factored out so any OTHER candidate source --
    `search_multi`'s facet-pool synthesis (facet_channel) is the only one
    today -- applies exactly this rule instead of a looser ad-hoc copy."""
    if item is None or item.deleted:
        return False
    if kinds is not None and item.kind not in kinds:
        return False
    if project is not None and item.project != project:
        return False
    if at_time is not None and not is_valid_at(item, at_time):
        return False
    return True


def cosine_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _ranks(hits: list[tuple[str, float]]) -> dict[str, int]:
    return {doc_id: rank for rank, (doc_id, _score) in enumerate(hits, start=1)}


def _raw_scores(hits: list[tuple[str, float]]) -> dict[str, float]:
    return dict(hits)


def _rank_fallback(values: dict[str, float]) -> dict[str, float]:
    """Rank-based normalized score (1.0 best .. 0.0 worst) for a degenerate
    signal whose raw scores are all equal (adversarial review, 2026-09-12):
    min-max/z-score would otherwise flatten every candidate to the same
    constant 1.0, making an all-tied signal contribute nothing
    discriminative even though its constituents still carry an implicit
    rank order from the signal that produced them. Relies on `values`
    insertion order matching that original best-first rank order (true for
    every caller here: `_raw_scores` builds a dict from an already-ranked
    list). A single candidate has nothing to rank against and keeps 1.0.
    """
    n = len(values)
    if n <= 1:
        return {k: 1.0 for k in values}
    return {k: 1.0 - i / (n - 1) for i, k in enumerate(values)}


def minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    # A8 (adversarial review, 2026-09-12): a NaN/±inf raw score (e.g. a
    # corrupt embedding, or a divide-by-zero upstream) poisons min()/max()
    # unpredictably (NaN comparisons are always False) and can produce NaN
    # results (inf - inf, inf / inf) that would then silently corrupt the
    # final ranking. Treat "any non-finite value present" the same as the
    # existing degenerate-signal case: fall back to the rank-based score,
    # which never touches the raw values arithmetically.
    if not all(math.isfinite(v) for v in values.values()):
        return _rank_fallback(values)
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return _rank_fallback(values)
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _zscore_clip_minmax(values: dict[str, float], clip: float = 3.0) -> dict[str, float]:
    if not values:
        return {}
    if not all(math.isfinite(v) for v in values.values()):  # A8, see minmax_normalize
        return _rank_fallback(values)
    arr = np.array(list(values.values()), dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-9:
        return _rank_fallback(values)
    clipped = {k: max(-clip, min(clip, (v - mean) / std)) for k, v in values.items()}
    return minmax_normalize(clipped)


def _fixed_cosine_normalize(values: dict[str, float]) -> dict[str, float]:
    """Config.dense_norm="fixed" for the dense signal: map a raw cosine
    (range [-1, 1], normalized embeddings) linearly onto [0, 1] by simply
    clipping negatives to 0 -- NOT a per-query min-max stretch. A date-heavy
    query where every candidate's cosine is low (weak dense evidence
    overall) then stays low after normalization instead of the worst
    candidate being floored to 0.0 and the best stretched up to 1.0 as if it
    were a strong match (docs/ABLATIONS.md "Full-candidate scoring")."""
    return {k: max(0.0, v) for k, v in values.items()}


def _max_normalize(values: dict[str, float]) -> dict[str, float]:
    """Config.dense_norm="fixed" companion for bm25: score / per-query max,
    the sparse-signal analogue of `_fixed_cosine_normalize` -- preserves how
    far below the query's own best match each candidate is, rather than a
    min-max stretch that also floors the worst candidate to 0 regardless of
    how close it actually was to the best."""
    if not values:
        return {}
    if not all(math.isfinite(v) for v in values.values()):  # A8, see minmax_normalize
        return _rank_fallback(values)
    m = max(values.values())
    if m <= 1e-9:
        return {k: 0.0 for k in values}
    return {k: v / m for k, v in values.items()}


def _normalize_signal(name: str, raw: dict[str, float], method: str, dense_norm: str) -> dict[str, float]:
    """Per-signal normalization for the "cc"/"dbsf" fusion branch, shared by
    `fuse` and `fuse_full`. `dense_norm="fixed"` overrides the dense/bm25
    channels with a fixed (non-per-query) normalization; entity is always
    normalized the same way `method` would otherwise normalize every
    signal (min-max for "cc", z-score-clip-minmax for "dbsf")."""
    if dense_norm == "fixed" and name == "dense":
        return _fixed_cosine_normalize(raw)
    if dense_norm == "fixed" and name == "bm25":
        return _max_normalize(raw)
    return _zscore_clip_minmax(raw) if method == "dbsf" else minmax_normalize(raw)


def fuse(
    candidate_ids: set[str],
    signal_hits: dict[str, list[tuple[str, float]]],
    weights: dict[str, float],
    method: str,
    dense_norm: str = "minmax",
) -> dict[str, float]:
    """Combine per-signal ranked lists into one fused score per candidate.

    Each signal's raw scores only cover the ids in its OWN top-N hit list
    (`signal_hits[name]`); a candidate missing from that list gets an
    imputed 0.0 contribution from it. See `fuse_full` for the
    Config.full_candidate_scoring variant that scores every candidate
    against every signal instead of imputing.
    """
    result = {cid: 0.0 for cid in candidate_ids}
    if method == "rrf":
        k = 60
        for name, hits in signal_hits.items():
            w = weights.get(name, 1.0)
            ranks = _ranks(hits)
            for cid in candidate_ids:
                r = ranks.get(cid)
                if r is not None:
                    result[cid] += w / (k + r)
        return result
    if method in ("cc", "dbsf"):
        for name, hits in signal_hits.items():
            w = weights.get(name, 1.0)
            raw = _raw_scores(hits)
            norm = _normalize_signal(name, raw, method, dense_norm)
            for cid in candidate_ids:
                result[cid] += w * norm.get(cid, 0.0)
        return result
    raise ValueError(f"unknown fusion method: {method!r}")


def fuse_full(
    candidate_ids: set[str],
    full_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    method: str,
    dense_norm: str = "minmax",
) -> dict[str, float]:
    """Like `fuse`, but `full_scores[name]` already carries a TRUE score for
    every id in `candidate_ids` (no imputed zeros) -- used when
    ``Config.full_candidate_scoring`` is set (see retrieval.py
    `HybridRetriever._dense_full_scores` / `_bm25_full_scores` /
    `_entity_full_scores`). For "rrf", ranks are computed over this full
    per-signal scoring instead of the top-N search() ranks.
    """
    result = {cid: 0.0 for cid in candidate_ids}
    if method == "rrf":
        k = 60
        for name, raw in full_scores.items():
            w = weights.get(name, 1.0)
            ranked = sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))
            ranks = _ranks(ranked)
            for cid in candidate_ids:
                r = ranks.get(cid)
                if r is not None:
                    result[cid] += w / (k + r)
        return result
    if method in ("cc", "dbsf"):
        for name, raw in full_scores.items():
            w = weights.get(name, 1.0)
            norm = _normalize_signal(name, raw, method, dense_norm)
            for cid in candidate_ids:
                result[cid] += w * norm.get(cid, 0.0)
        return result
    raise ValueError(f"unknown fusion method: {method!r}")


def mmr(
    hits: list[SearchHit],
    vectors: dict[str, np.ndarray],
    lambda_: float = 0.7,
    k: int | None = None,
) -> list[SearchHit]:
    """Maximal Marginal Relevance re-ordering of ``hits`` using item embeddings."""
    if not hits:
        return []
    k = len(hits) if k is None else min(k, len(hits))
    relevance = minmax_normalize({h.item.id: h.score for h in hits})
    remaining = list(hits)
    selected: list[SearchHit] = []
    while remaining and len(selected) < k:
        best_hit, best_val = None, float("-inf")
        for h in remaining:
            rel = relevance.get(h.item.id, 0.0)
            vec = vectors.get(h.item.id)
            if not selected or vec is None:
                div = 0.0
            else:
                sims = [
                    cosine_sim(vec, vectors.get(s.item.id))
                    for s in selected
                    if vectors.get(s.item.id) is not None
                ]
                div = max(sims) if sims else 0.0
            val = lambda_ * rel - (1 - lambda_) * div
            if val > best_val:
                best_val, best_hit = val, h
        selected.append(best_hit)  # type: ignore[arg-type]
        remaining.remove(best_hit)  # type: ignore[arg-type]
    return selected


class HybridRetriever:
    """Dense + BM25 + entity retrieval with fusion, temporal/importance scoring."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder | HashEmbedder,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        entity_index: EntityIndex,
        config: Config,
        facet_index: FacetIndex | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_index = vector_index
        self.bm25 = bm25_index
        self.entities = entity_index
        self.config = config
        # facet_channel (2026-09-19): defaults to a fresh, empty FacetIndex
        # when not given, so a caller that hasn't been updated (e.g. a
        # direct programmatic instantiation, never a MemoryV4 façade user)
        # still works -- just with an always-empty facet pool.
        self.facet_index = facet_index if facet_index is not None else FacetIndex()
        self._reranker: CrossEncoderReranker | None = None
        # E3 neighbor_expansion cache: doc_ref -> [item_id, ...] ordered by
        # (event_time, insertion order). Rebuilt lazily, invalidated whenever
        # the store's item count changes (see `_doc_ref_order`).
        self._nbr_cache: dict[str, list[str]] | None = None
        self._nbr_cache_n: int = -1
        # E6 recency: corpus event_time span (days), cached the same way.
        self._span_cache: float | None = None
        self._span_cache_n: int = -1
        # full_candidate_scoring: item_id -> [chunk_id, ...] (multi_vector
        # only), the reverse of store.chunk_owner, cached/invalidated the
        # same way (see `_chunk_ids_by_owner`).
        self._chunk_cache: dict[str, list[str]] | None = None
        self._chunk_cache_n: int = -1

    def _corpus_span_days(self) -> float:
        """max(event_time) - min(event_time) across live items, in days.

        Used only when ``recency_only_with_at_time`` is set and an explicit
        ``at_time`` is given (adversarial review, 2026-09-12): on LoCoMo,
        item ages run ~1200 days against a 30-day episode half-life, so
        ``recency_weight`` underflows to ~0 for every item alike — a
        constant, not a ranking signal (`use_recency` on/off is then
        byte-identical). Flooring the half-life to a quarter of the corpus's
        own time span keeps recency numerically alive for datasets whose
        natural span dwarfs the configured half-life.
        """
        n = len(self.store.items)
        if self._span_cache is not None and self._span_cache_n == n:
            return self._span_cache
        times = [
            it.event_time
            for it in self.store.items.values()
            if not it.deleted and it.event_time is not None
        ]
        span = (max(times) - min(times)) / 86400.0 if len(times) >= 2 else 0.0
        self._span_cache = span
        self._span_cache_n = n
        return span

    def _get_reranker(self) -> CrossEncoderReranker | None:
        if not self.config.rerank_model:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.config.rerank_model, self.config.rerank_device)
        return self._reranker

    # --------------------------------------------------------- E1 multi_vector
    def _dense_candidates(self, qvec, n: int) -> list[tuple[str, float]]:
        """Dense candidate ids+scores for the query vector.

        Plain path: one vector per item, straight FAISS top-n. multi_vector
        path: the index also holds several chunk-vectors per item
        (``f"{item_id}::{i}"``, see memory.py `_add_chunk_vectors`); fetch
        more raw hits, map each chunk id back to its owning item id via
        ``store.chunk_owner`` (identity if it's already an item id), and keep
        the MAX score per item (the ablation's specified pooling rule).
        """
        if not self.config.multi_vector:
            return self.vector_index.search(qvec, n)
        mult = max(1, self.config.multi_vector_candidate_mult)
        raw = self.vector_index.search(qvec, n * mult)
        owner = self.store.chunk_owner
        best: dict[str, float] = {}
        for cid, score in raw:
            item_id = owner.get(cid, cid)
            if item_id not in best or score > best[item_id]:
                best[item_id] = score
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:n]

    # ------------------------------------------------------- full_candidate_scoring
    def _chunk_ids_by_owner(self) -> dict[str, list[str]]:
        """item_id -> [chunk_id, ...], the reverse of ``store.chunk_owner``
        (chunk_id -> item_id). Empty unless ``config.multi_vector`` is/was
        on. Cached, invalidated whenever ``chunk_owner``'s size changes
        (same pattern as `_doc_ref_order` / `_corpus_span_days`)."""
        owner = self.store.chunk_owner
        n = len(owner)
        if self._chunk_cache is not None and self._chunk_cache_n == n:
            return self._chunk_cache
        grouped: dict[str, list[str]] = defaultdict(list)
        for cid, iid in owner.items():
            grouped[iid].append(cid)
        self._chunk_cache = dict(grouped)
        self._chunk_cache_n = n
        return self._chunk_cache

    def _dense_full_scores(self, qvec: np.ndarray, candidate_ids: set[str]) -> dict[str, float]:
        """True dense cosine of `qvec` against EVERY candidate (not just
        those in the dense top-N list), for ``Config.full_candidate_scoring``.
        Every live item always has at least its own representative vector
        (see memory.py `_add_chunk_vectors`), so this never has to impute a
        missing score the way `fuse`'s top-N-only path does. multi_vector:
        max over the item's representative vector plus its chunk vectors --
        the same pooling rule `_dense_candidates` applies via faiss's own
        top-N search, just computed directly here instead.

        Candidate ids are sorted before building the score dict so its
        insertion order (and therefore `_rank_fallback`'s tie order in the
        rare degenerate all-equal-cosine case) never depends on
        PYTHONHASHSEED's iteration order over the `candidate_ids` set
        (adversarial-review convention already established elsewhere in
        this module for the same reason).
        """
        ids = sorted(candidate_ids)
        if not self.config.multi_vector:
            return self.vector_index.scores_for(qvec, ids)
        chunk_map = self._chunk_ids_by_owner()
        lookup_ids: list[str] = []
        for iid in ids:
            lookup_ids.append(iid)
            lookup_ids.extend(chunk_map.get(iid, []))
        raw = self.vector_index.scores_for(qvec, lookup_ids)
        out: dict[str, float] = {}
        for iid in ids:
            pool = [raw[iid]] + [raw[cid] for cid in chunk_map.get(iid, [])]
            out[iid] = max(pool)
        return out

    def _bm25_full_scores(self, query_tokens: list[str], candidate_ids: set[str]) -> dict[str, float]:
        """True BM25 score of every candidate against `query_tokens`
        (``Config.full_candidate_scoring``). Empty dict (not a same-shaped
        all-zero dict) when the query has no tokens or the index is empty,
        matching `signal_hits["bm25"]` being effectively absent in that
        case -- `fuse_full`'s normalization then contributes 0 to every
        candidate from this signal instead of routing through
        `_rank_fallback` on a meaningless all-tied score."""
        if not query_tokens or self.bm25.n_docs == 0:
            return {}
        return {cid: self.bm25.score_doc(query_tokens, cid) for cid in sorted(candidate_ids)}

    def _entity_full_scores(self, query_entities: list[str], candidate_ids: set[str]) -> dict[str, float]:
        """True entity-overlap score of every candidate against
        `query_entities` (``Config.full_candidate_scoring``). Empty dict
        when the query has no entities or the index is empty -- see
        `_bm25_full_scores` docstring for why that (not an all-zero dict)
        is the correct "signal absent" representation."""
        if not query_entities or self.entities.n_docs == 0:
            return {}
        return {cid: self.entities.score_doc(query_entities, cid) for cid in sorted(candidate_ids)}

    # ------------------------------------------------------ E3 neighbor_expansion
    def _doc_ref_order(self) -> dict[str, list[str]]:
        """doc_ref -> item ids ordered by (event_time, insertion order).

        ``self.store.items`` is a plain dict populated in insertion order
        (Store.append_many), so enumerating its values gives a stable
        insertion-order tiebreak for free without adding any field to
        MemoryItem.
        """
        n = len(self.store.items)
        if self._nbr_cache is not None and self._nbr_cache_n == n:
            return self._nbr_cache
        groups: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
        for idx, it in enumerate(self.store.items.values()):
            if it.deleted or not it.doc_ref:
                continue
            groups[it.doc_ref].append((it.event_time, idx, it.id))
        ordered = {doc_ref: [iid for _, _, iid in sorted(entries)] for doc_ref, entries in groups.items()}
        self._nbr_cache = ordered
        self._nbr_cache_n = n
        return ordered

    def _expand_neighbors(
        self,
        scored: list["SearchHit"],
        kinds: list[str] | None,
        project: str | None,
        at_time: float | None,
    ) -> list["SearchHit"]:
        """For the top `neighbor_seed` hits, add +-`neighbor_window` items
        adjacent in time within the same doc_ref, scored at
        `hit.score * neighbor_decay`, skipping ids already present (or
        reached by an earlier, higher-scoring seed in this same pass)."""
        cfg = self.config
        if not scored:
            return scored
        order = self._doc_ref_order()
        pos_of: dict[str, int] = {}
        for doc_ref, ids in order.items():
            for i, iid in enumerate(ids):
                pos_of[iid] = i
        existing_ids = {h.item.id for h in scored}
        extra: dict[str, SearchHit] = {}
        for h in scored[: cfg.neighbor_seed]:
            doc_ref = h.item.doc_ref
            if not doc_ref or doc_ref not in order:
                continue
            ids_in_doc = order[doc_ref]
            pos = pos_of.get(h.item.id)
            if pos is None:
                continue
            for delta in range(1, cfg.neighbor_window + 1):
                for npos in (pos - delta, pos + delta):
                    if npos < 0 or npos >= len(ids_in_doc):
                        continue
                    nid = ids_in_doc[npos]
                    if nid in existing_ids:
                        continue
                    nscore = h.score * cfg.neighbor_decay
                    if nid in extra and extra[nid].score >= nscore:
                        continue
                    nitem = self.store.get(nid)
                    if nitem is None or nitem.deleted:
                        continue
                    if kinds is not None and nitem.kind not in kinds:
                        continue
                    if project is not None and nitem.project != project:
                        continue
                    if at_time is not None and not is_valid_at(nitem, at_time):
                        continue
                    extra[nid] = SearchHit(
                        item=nitem,
                        score=nscore,
                        signals={"neighbor_of": h.item.id, "neighbor_delta": delta},
                    )
        if not extra:
            return scored
        combined = scored + list(extra.values())
        combined.sort(key=lambda hh: (-hh.score, hh.item.id))
        return combined

    # ------------------------------------------------------ E12 entity_expansion
    def _rare_entities(self, item_id: str) -> set[str]:
        """The subset of `item_id`'s own indexed entities that are "rare"
        enough to be worth hopping on for `Config.entity_expansion`: present
        in at most 5% of the corpus (same "discriminative" spirit as
        `EntityIndex.search`'s >10% skip, just a tighter bar since this
        drives multi-hop fan-out rather than a single ranked score), at
        least 3 characters, and not a stopword/question word (the same
        `JUNK_ENTITY_WORDS` filter `extract_entities` applies to single-word
        spans -- a query's own leaked interrogative should never seed a
        cross-corpus hop).
        """
        n = self.entities.n_docs
        if n == 0:
            return set()
        ents = self.entities.doc_entities.get(item_id, [])
        rare: set[str] = set()
        for e in ents:
            if len(e) < 3:
                continue
            if " " not in e and e in JUNK_ENTITY_WORDS:
                continue
            bucket = self.entities.postings.get(e)
            df = len(bucket) if bucket else 0
            if df == 0 or df / n > 0.05:
                continue
            rare.add(e)
        return rare

    def _expand_entities(
        self,
        scored: list["SearchHit"],
        kinds: list[str] | None,
        project: str | None,
        at_time: float | None,
        top_k: int,
    ) -> list["SearchHit"]:
        """For the top `config.expansion_seed` hits, hop to every OTHER item
        sharing >=1 "rare" entity (`_rare_entities`) with that seed, via
        `EntityIndex.postings` -- a corpus-wide entity co-occurrence hop,
        distinct from E3's `neighbor_expansion` (time-adjacency within a
        single doc_ref). A candidate's score is
        ``seed.score * config.expansion_decay * (shared / max_shared)``,
        where `shared` is how many of the seed's rare entities that
        candidate shares and `max_shared` is the largest such count for that
        seed -- so a candidate overlapping on more of the seed's rare
        entities scores closer to the seed's own decayed score. A candidate
        already present (in `scored`, or added by an earlier, higher-scoring
        seed in this same pass) is skipped/kept-at-max, never duplicated.
        Total additions across all seeds are capped at ``2 * top_k``,
        highest-scored first, both for a deterministic (score, id)-sorted
        cap point and to bound extra rerank cost downstream.
        """
        cfg = self.config
        if not scored or self.entities.n_docs == 0:
            return scored
        seeds = scored[: cfg.expansion_seed]
        existing_ids = {h.item.id for h in scored}
        cap = 2 * top_k
        extra: dict[str, SearchHit] = {}
        for h in seeds:
            seed_rare = sorted(self._rare_entities(h.item.id))
            if not seed_rare:
                continue
            shared_count: dict[str, int] = defaultdict(int)
            for e in seed_rare:
                bucket = self.entities.postings.get(e)
                if not bucket:
                    continue
                for doc_id in bucket:
                    if doc_id == h.item.id:
                        continue
                    shared_count[doc_id] += 1
            if not shared_count:
                continue
            max_shared = max(shared_count.values())
            for doc_id in sorted(shared_count):
                if doc_id in existing_ids:
                    continue
                shared = shared_count[doc_id]
                nscore = h.score * cfg.expansion_decay * (shared / max_shared)
                if doc_id in extra and extra[doc_id].score >= nscore:
                    continue
                nitem = self.store.get(doc_id)
                if nitem is None or nitem.deleted:
                    continue
                if kinds is not None and nitem.kind not in kinds:
                    continue
                if project is not None and nitem.project != project:
                    continue
                if at_time is not None and not is_valid_at(nitem, at_time):
                    continue
                extra[doc_id] = SearchHit(
                    item=nitem,
                    score=nscore,
                    signals={"expansion_of": h.item.id, "shared_entities": shared},
                )
        if not extra:
            return scored
        ranked_extra = sorted(extra.values(), key=lambda hh: (-hh.score, hh.item.id))[:cap]
        combined = scored + ranked_extra
        combined.sort(key=lambda hh: (-hh.score, hh.item.id))
        return combined

    def search(
        self,
        query: str,
        top_k: int = 10,
        at_time: float | None = None,
        kinds: list[str] | None = None,
        project: str | None = None,
        rerank: bool | None = None,
    ) -> list[SearchHit]:
        cfg = self.config
        n = cfg.candidates
        query_time = at_time if at_time is not None else time.time()

        # A7(b): parse any date the query itself carries (absolute or
        # relative, anchored on the query's own time), exposed to callers via
        # SearchHit.signals["query_dates"]. Also the trigger signal for Part B's
        # temporal_query_mode (docs/ABLATIONS.md "Hardening 2026-09-13").
        query_dates = parse_dates(query, reference_time=query_time)
        temporal_query = bool(query_dates) or bool(_TEMPORAL_CUE_RE.search(query))
        temporal_boost = cfg.temporal_query_mode == "auto" and temporal_query

        # Computed once, reused below both for the top-N candidate lists
        # (candidate GENERATION -- always top-N, per Config.candidates,
        # regardless of full_candidate_scoring) and, when
        # `full_candidate_scoring` is on, for re-scoring every candidate
        # against each signal directly (candidate SCORING).
        query_tokens = tokenize(query, stem=cfg.use_stemming) if cfg.use_bm25 else None
        query_entities = extract_entities(query) if cfg.use_entities else None

        signal_hits: dict[str, list[tuple[str, float]]] = {}
        if cfg.use_dense:
            qvec = self.embedder.encode([query])[0]
            signal_hits["dense"] = self._dense_candidates(qvec, n)
        else:
            qvec = None
        if cfg.use_bm25:
            signal_hits["bm25"] = self.bm25.search(query_tokens, n)
        if cfg.use_entities:
            signal_hits["entity"] = self.entities.search(query_entities, n)

        candidate_ids: set[str] = set()
        for hits in signal_hits.values():
            candidate_ids.update(doc_id for doc_id, _ in hits)
        if not candidate_ids:
            return []

        weights = {"dense": cfg.w_dense, "bm25": cfg.w_bm25, "entity": cfg.w_entity}
        if temporal_boost:
            # Part B hypothesis 2: dense/MiniLM scores dominate date-heavy
            # queries and drown out BM25's exact date/number match; boosting
            # BM25 for just this query (never mutates `cfg`) is the targeted
            # counter-measure being A/B-tested (lme_oracle_v4_tempauto).
            weights["bm25"] = cfg.w_bm25 * cfg.temporal_bm25_boost

        if cfg.full_candidate_scoring:
            # See Config.full_candidate_scoring / docs/ABLATIONS.md
            # "Full-candidate scoring": score every candidate against every
            # enabled signal directly instead of imputing 0 for a candidate
            # missing from that signal's own top-N list.
            full_scores: dict[str, dict[str, float]] = {}
            if cfg.use_dense and qvec is not None:
                full_scores["dense"] = self._dense_full_scores(qvec, candidate_ids)
            if cfg.use_bm25:
                full_scores["bm25"] = self._bm25_full_scores(query_tokens, candidate_ids)
            if cfg.use_entities:
                full_scores["entity"] = self._entity_full_scores(query_entities, candidate_ids)
            fused = fuse_full(candidate_ids, full_scores, weights, cfg.fusion, cfg.dense_norm)
            raw_maps = full_scores
        else:
            fused = fuse(candidate_ids, signal_hits, weights, cfg.fusion, cfg.dense_norm)
            raw_maps = {name: _raw_scores(hits) for name, hits in signal_hits.items()}

        # rank_maps stays top-N-based even under full_candidate_scoring: it
        # is diagnostic metadata only (SearchHit.signals[...]["rank"]), not
        # part of score computation (fuse_full re-derives its own ranks
        # internally for the "rrf" method).
        rank_maps = {name: _ranks(hits) for name, hits in signal_hits.items()}

        scored: list[SearchHit] = []
        for cid in candidate_ids:
            item = self.store.get(cid)
            if not _passes_filters(item, kinds, project, at_time):
                continue

            base = fused.get(cid, 0.0)

            # Part B hypothesis 1: the recency multiplier penalises older
            # sessions holding the evidence for date-heavy queries (a
            # "when/how long ago" question is often precisely about an OLD
            # event) — disable it for this query only when temporal_boost
            # fired, same scope as the bm25 weight boost above.
            skip_recency = (cfg.recency_only_with_at_time and at_time is None) or temporal_boost
            if cfg.use_recency and not skip_recency:
                half_life = cfg.half_life_days(item.kind)
                if cfg.recency_only_with_at_time and at_time is not None:
                    # Adaptive floor so recency stays numerically alive on
                    # corpora whose time span dwarfs the configured
                    # half-life (see `_corpus_span_days`).
                    half_life = max(half_life, 0.25 * self._corpus_span_days())
                age_days = max(0.0, (query_time - item.event_time) / 86400.0)
                recency = recency_weight(age_days, half_life)
            else:
                recency = 1.0
            importance = max(0.0, min(1.0, item.importance))
            final = base * (0.7 + 0.3 * recency) * (0.8 + 0.4 * importance)

            if item.invalidated_by is not None and not (
                at_time is not None and is_valid_at(item, at_time)
            ):
                final *= 0.5

            signals = {
                name: {"rank": rank_maps.get(name, {}).get(cid), "score": raw_maps.get(name, {}).get(cid)}
                for name in ("dense", "bm25", "entity")
            }
            signals["fused"] = base
            signals["recency"] = recency
            signals["query_dates"] = query_dates
            signals["temporal_boost"] = temporal_boost
            scored.append(SearchHit(item=item, score=final, signals=signals))

        # Deterministic total order: `candidate_ids` is a set (assembled from
        # per-signal hit lists via set.update), so its iteration order is
        # PYTHONHASHSEED-dependent; ties in `.score` must break on item id,
        # not on that random insertion order (adversarial review, 2026-09-12:
        # otherwise seed-to-seed jitter of up to 0.007 R@10 on LoCoMo).
        scored.sort(key=lambda h: (-h.score, h.item.id))

        if cfg.neighbor_expansion:
            scored = self._expand_neighbors(scored, kinds, project, at_time)

        if cfg.entity_expansion:
            scored = self._expand_entities(scored, kinds, project, at_time, top_k)

        scored = self._rerank_hits(query, scored, rerank)

        result = scored[:top_k]
        now = time.time()
        for h in result:
            self.store.touch(h.item.id, now)
        return result

    # --------------------------------------------------- optional CE rerank
    def _rerank_hits(
        self, query: str, scored: list[SearchHit], rerank: bool | None
    ) -> list[SearchHit]:
        """Cross-encoder rerank of the top `config.rerank_top_n` of `scored`
        against `query`, blended with the pre-rerank score per
        `config.rerank_weight` -- factored out of `search()` so
        `search_multi` (multi-query RRF fusion) can apply the exact same
        final rerank step to its fused list. A no-op (returns `scored`
        unchanged) unless `rerank` (or, when None, `config.rerank_model`)
        enables it and a reranker is actually available.
        """
        cfg = self.config
        do_rerank = rerank if rerank is not None else bool(cfg.rerank_model)
        if not (do_rerank and cfg.rerank_model):
            return scored
        reranker = self._get_reranker()
        top_n = scored[: cfg.rerank_top_n]
        if reranker is None or not reranker.available() or not top_n:
            return scored
        ce_scores = reranker.score(query, [h.item.text for h in top_n])
        if ce_scores is None:
            return scored
        base_norm = minmax_normalize({h.item.id: h.score for h in top_n})
        ce_norm = minmax_normalize({h.item.id: s for h, s in zip(top_n, ce_scores)})
        w = cfg.rerank_weight
        for h in top_n:
            h.signals["rerank"] = ce_norm.get(h.item.id)
            h.score = (1 - w) * base_norm.get(h.item.id, 0.0) + w * ce_norm.get(h.item.id, 0.0)
        scored = top_n + scored[cfg.rerank_top_n :]
        scored.sort(key=lambda h: (-h.score, h.item.id))
        return scored

    # -------------------------------------------------------- facet_channel
    def facet_pool(
        self,
        facets: list[dict] | None,
        kinds: list[str] | None = None,
        project: str | None = None,
        at_time: float | None = None,
    ) -> set[str]:
        """The facet-channel candidate pool for `facets` (client-supplied
        ``{"subject", "predicate"}`` dicts; ``object`` is ignored on the
        query side, see `facets.normalize_facet`). Empty unless
        `config.facet_channel` is on AND `facets` normalizes to >=1 usable
        pair AND at least one indexed item matches -- this single method IS
        the "should the facet channel do anything for this call" contract,
        reused both to DECIDE whether `MemoryV4.search()`/`get_context()`
        route through `search_multi` at all (see memory.py) and, internally,
        by `search_multi` to build the ranked list it fuses -- so there is
        exactly one place this decision is made, never a second copy that
        could drift from it.

        Every pool member is filtered through `_passes_filters`, the EXACT
        rule `search()` applies to its own candidates -- a facet-only hit
        can never bypass project/kind/bi-temporal scoping that an ordinary
        candidate would be held to.
        """
        if not self.config.facet_channel or not facets:
            return set()
        qfacets = [nf for f in facets if (nf := normalize_facet(f)) is not None]
        if not qfacets:
            return set()
        pool = self.facet_index.pool(qfacets)
        if not pool:
            return pool
        return {
            doc_id for doc_id in pool if _passes_filters(self.store.get(doc_id), kinds, project, at_time)
        }

    # ------------------------------------------------------ B. multi-query
    # Guard (task spec): at most this many extra query strings (queries +
    # hypotheses combined), each truncated to this many characters.
    MAX_EXTRA_QUERIES = 8
    MAX_QUERY_CHARS = 300

    def search_multi(
        self,
        query: str,
        queries: list[str] | None = None,
        hypotheses: list[str] | None = None,
        top_k: int = 10,
        at_time: float | None = None,
        kinds: list[str] | None = None,
        project: str | None = None,
        rerank: bool | None = None,
        facets: list[dict] | None = None,
    ) -> list[SearchHit]:
        """Multi-query retrieval: run `search()` for the original `query`
        plus every client-supplied sub-query/hypothesis (rerank disabled,
        `top_k * 4` candidates each -- reranking once, at the end, on the
        fused list is both cheaper and avoids a cross-encoder call per
        sub-query), fuse the resulting ranked id lists with weighted RRF
        (`config.multi_query_rrf_k`; weights: original=1.0, `queries`=1.0,
        `hypotheses`=0.7 -- matches `bench/retrievers.py`'s
        `V4PlannedRetriever`/`rrf_fuse`), then apply the same optional
        cross-encoder rerank `search()` would on the fused top
        `config.rerank_top_n`. Recency/importance are already baked into
        each sub-search's own `.score` (computed by `search()` exactly as
        for a plain query) -- RRF fuses on RANK, not raw score, so they
        influence fusion the same way they always influence a ranked list,
        with no separate re-application needed.

        `queries`/`hypotheses` are guarded to at most `MAX_EXTRA_QUERIES`
        combined (queries first, then hypotheses filling any remaining
        slots) and `MAX_QUERY_CHARS` characters each -- silently clipped,
        never an error, matching this module's established "never crash the
        caller" convention. Blank/whitespace-only strings are dropped.

        `facets` (C, production port of `V4PlannedRetriever`'s facet
        channel, +0.036 R@10 on top of the query-planning combo on LoCoMo):
        when `config.facet_channel` is on and `self.facet_pool(...)` is
        non-empty, that pool -- ranked by `facets.rank_facet_pool` against
        the id lists already computed above -- is appended as one more
        weighted party (`config.facet_weight`) to the RRF fusion, exactly
        like an extra sub-query's hit list. A pool member with no
        `SearchHit` yet (found by no query/sub-query/hypothesis at all) gets
        one synthesized here, after passing the same filters `search()`
        applies to every candidate. Every facet-pool member (pre-existing or
        synthesized) is marked `signals["facet"] = True`.
        """
        cfg = self.config
        weighted: list[tuple[str, float]] = [(query, 1.0)]
        extra: list[tuple[str, float]] = []
        for s in queries or []:
            if s and s.strip():
                extra.append((s[: self.MAX_QUERY_CHARS], 1.0))
        remaining = max(0, self.MAX_EXTRA_QUERIES - len(extra))
        for s in (hypotheses or [])[:remaining] if remaining else []:
            if s and s.strip():
                extra.append((s[: self.MAX_QUERY_CHARS], 0.7))
        weighted.extend(extra[: self.MAX_EXTRA_QUERIES])

        oversample_k = top_k * 4
        id_lists: list[tuple[list[str], float]] = []
        hits_by_id: dict[str, SearchHit] = {}
        for qtext, weight in weighted:
            hits = self.search(
                qtext, top_k=oversample_k, at_time=at_time, kinds=kinds, project=project, rerank=False
            )
            ids: list[str] = []
            for h in hits:
                ids.append(h.item.id)
                if h.item.id not in hits_by_id or h.score > hits_by_id[h.item.id].score:
                    hits_by_id[h.item.id] = h
            id_lists.append((ids, weight))

        # facet_channel: append the facet-ranked pool BEFORE the early
        # "nothing found" return below, so a facet-only match (every
        # query/sub-query/hypothesis missed, but the facet index has it) is
        # never silently dropped -- see facets.py's module docstring for why
        # this is precisely the case the channel exists for.
        filtered_pool = self.facet_pool(facets, kinds=kinds, project=project, at_time=at_time)
        if filtered_pool:
            facet_ranked = rank_facet_pool(filtered_pool, [ids for ids, _w in id_lists])
            # Weight as a SHARE of the total vote (config.facet_share), so
            # the channel votes instead of filtering when there is a single
            # query list -- see config.py's comment for the measurement.
            share = cfg.facet_share
            if share is not None and 0.0 < share < 1.0:
                facet_w = share / (1.0 - share) * sum(w for _ids, w in id_lists)
            else:
                facet_w = cfg.facet_weight
            id_lists.append((facet_ranked, facet_w))
            for doc_id in filtered_pool:
                hit = hits_by_id.get(doc_id)
                if hit is None:
                    hit = SearchHit(item=self.store.get(doc_id), score=0.0, signals={})
                    hits_by_id[doc_id] = hit
                hit.signals = dict(hit.signals)
                hit.signals["facet"] = True

        if not hits_by_id:
            return []

        k = cfg.multi_query_rrf_k
        fused_scores: dict[str, float] = defaultdict(float)
        for ids, weight in id_lists:
            for rank, doc_id in enumerate(ids, start=1):
                fused_scores[doc_id] += weight / (k + rank)

        n_queries = len(weighted)
        ordered_ids = sorted(fused_scores, key=lambda d: (-fused_scores[d], d))
        result: list[SearchHit] = []
        for doc_id in ordered_ids:
            h = hits_by_id[doc_id]
            h.score = fused_scores[doc_id]
            h.signals = dict(h.signals)
            h.signals["multi_query"] = {"n_queries": n_queries}
            result.append(h)

        result = self._rerank_hits(query, result, rerank)
        return result[:top_k]
