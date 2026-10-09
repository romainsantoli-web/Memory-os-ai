"""Runtime configuration for Memory OS v4.

All fields can be overridden via environment variables prefixed ``MEMORY_V4_``
(see :meth:`Config.from_env`). Defaults follow ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_opt_str(name: str, default: str | None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None else default


def _env_optional_float(name: str, default: float | None) -> float | None:
    """`_env_float` that also accepts the literal "none"/"" -> None."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip().lower() in ("", "none", "null"):
        return None
    return float(raw)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    """All tunables for the retrieval and memory pipeline.

    Defaults as of the 2026-09-12 ablation campaign (see docs/ABLATIONS.md):
    `fusion="cc"`, `w_bm25=1.5`, `candidates=100`, `use_stemming=True` are
    ``locomo_v4_combo_v2``, the config that beat the pre-campaign v4 default
    on BOTH LoCoMo (macro R@10 0.435 -> 0.527, +0.092) and a stratified
    150-question LongMemEval oracle sample (R@10 0.776 -> 0.832, +0.056) —
    see docs/ARCHITECTURE.md's "Retrieval scoring" section for the RRF
    formula that was the *previous* default (kept there as the `fusion="rrf"`
    illustration; still fully supported, just no longer what ships). All
    other fields (`w_dense`, `w_entity`, `use_recency`, the E1/E3/E6 ablation
    flags below, ...) are unchanged from pre-campaign defaults: they showed
    no reproducible, above-noise gain on their own (see docs/ABLATIONS.md).
    """

    data_dir: str = "~/.memory-os-v4"
    embed_model: str = "ibm-granite/granite-embedding-97m-multilingual-r2"  # promoted 2026-09-13: wins LoCoMo 0.538 vs 0.527 and LME 0.8369 vs 0.832; multilingual (FR/EN); MiniLM remains valid via MEMORY_V4_EMBED_MODEL
    device: str = "auto"
    hash_dim: int = 64

    # Palier 2-3 (2026-09-13, E10): cap on SentenceTransformer.max_seq_length,
    # applied after model load (embed.py `Embedder._ensure_model`). Needed for
    # `ibm-granite/granite-embedding-97m-multilingual-r2`, whose tokenizer
    # config reports 32768 tokens (ModernBERT-based) -- encoding the short
    # turns/questions this benchmark actually uses at that length would be
    # needlessly slow. 512 is generous headroom over any single turn/question
    # here while keeping encode throughput fast; a no-op for MiniLM, whose own
    # native max_seq_length (256) is already below 512.
    embed_max_seq_length: int = 512
    # Operational knob (2026-09-13, resource-constrained machine: 16 GB RAM,
    # MPS-wired memory pressure during encoding) -- SentenceTransformer.encode
    # batch size. Lower than sentence-transformers' own default (32) trades a
    # little throughput for a smaller MPS working set; does not change
    # embedding VALUES (batching is purely a compute-chunking detail), so it
    # never appears in ledger comparisons as its own variable.
    embed_batch_size: int = 32

    fusion: str = "cc"  # "rrf" | "dbsf" | "cc" — was "rrf" pre-campaign (docs/ABLATIONS.md E4/E7)
    w_dense: float = 1.0
    w_bm25: float = 1.5  # was 1.0 pre-campaign
    w_entity: float = 0.5
    candidates: int = 100  # was 50 pre-campaign

    rerank_model: str | None = None
    rerank_top_n: int = 30
    # Palier 2-3 (E11): blend weight for the cross-encoder score against the
    # pre-rerank fused score, both minmax-normalized over the reranked slice:
    # final = (1 - rerank_weight) * minmax(fused) + rerank_weight * minmax(ce).
    # Spec default is the symmetric 0.5/0.5 blend; see docs/ABLATIONS.md E11
    # for the 0.7-weighted-toward-CE variant tested as `rerank_weight=0.7`.
    rerank_weight: float = 0.5
    # "auto" resolves to mps (Apple Silicon) when available, else cpu -- same
    # resolution rule as `device`/embed.py's `_resolve_device`, kept
    # independent of `device` because the dense embedder and the reranker
    # are two different models that could need different fallbacks.
    rerank_device: str = "auto"

    dedup_cos: float = 0.92
    supersede_cos: float = 0.80

    half_life_episode_days: float = 30.0
    half_life_fact_days: float = 180.0
    half_life_doc_days: float = 365.0
    half_life_default_days: float = 90.0

    use_recency: bool = True
    use_entities: bool = True
    use_bm25: bool = True
    use_dense: bool = True

    block_char_limit: int = 2000
    episode_ttl_days: int = 120

    # --- Ablation flags (bench/ABLATIONS.md, 2026-09-12 campaign) -----------
    # Every flag below defaults to the pre-campaign behaviour EXCEPT
    # `use_stemming` (E2), which won on its own (+0.027-0.029 R@10) and again
    # as part of the winning combo above — it is now part of the shipped
    # default, not an opt-in ablation.

    # E1 multi_vector: index several overlapping chunk-vectors per item
    # (text.chunk_text, sentence-aware) instead of one whole-item vector;
    # dense score per item = max over its chunk scores. See memory.py
    # `_add_chunk_vectors` / retrieval.py `HybridRetriever._dense_candidates`.
    multi_vector: bool = False
    chunk_tokens: int = 48
    chunk_overlap: int = 12
    multi_vector_candidate_mult: int = 4  # FAISS raw fetch = candidates * this, before max-pooling to items

    # E2 stem: Porter (English) + light suffix stripping (French) applied in
    # tokenize() for BM25 indexing/querying only (never for display text).
    use_stemming: bool = True  # was False pre-campaign; part of the winning combo

    # E3 neighbor_expansion: after fusion+scoring, pull in the +-window items
    # adjacent in time within the same doc_ref (episodes ordered by
    # event_time then insertion order), scored at hit.score * neighbor_decay.
    neighbor_expansion: bool = False
    neighbor_seed: int = 20
    neighbor_decay: float = 0.6
    neighbor_window: int = 1

    # E6 recency (two bugs found by adversarial review, 2026-09-12, fixed
    # together behind one flag): (a) at_time=None means "no explicit query
    # time" (e.g. a bench replaying historical dialogue with no wall-clock
    # meaning); the pre-campaign default instead silently fell back to
    # time.time(), which makes the recency multiplier vary with real
    # wall-clock age instead of being a no-op; (b) when at_time IS given,
    # a half-life much shorter than the corpus's own event_time span (e.g.
    # LoCoMo: ~1200-day-old turns vs a 30-day episode half-life) makes
    # recency_weight underflow to ~0 for every item alike — a constant, not
    # a ranking signal — so half-life is floored to 25% of the corpus span.
    # Kept OFF by default (unchanged legacy behaviour) until proven on the
    # ledger; see docs/ABLATIONS.md ("locomo_v4_norec").
    recency_only_with_at_time: bool = False

    # Part B (LongMemEval temporal-reasoning targeted experiment, docs/ABLATIONS.md
    # "Hardening 2026-09-13"): v4 combo scores 0.513 R@10 on LME's
    # temporal-reasoning category vs plain BM25's 0.873. Hypotheses: (1) the
    # recency multiplier penalises older sessions holding the evidence when
    # `at_time` is given; (2) dense MiniLM scores dominate date-heavy queries
    # and drown out BM25's literal date/number match. "auto": when a query's
    # own text yields a non-empty `parse_dates()` OR matches a small
    # EN/FR temporal-cue regex (retrieval.py `_TEMPORAL_CUE_RE`), disable
    # recency and multiply `w_bm25` by `temporal_bm25_boost` for THAT QUERY
    # ONLY (never mutates this Config). Kept OFF ("off") by default per the
    # established ablation-flag convention until B1-B4 prove a win on the
    # ledger with no LoCoMo regression (see docs/ABLATIONS.md decision).
    temporal_query_mode: str = "off"  # "off" | "auto"
    temporal_bm25_boost: float = 1.5

    # Full-candidate scoring experiment (docs/ABLATIONS.md "Full-candidate
    # scoring 2026-09-13"). Hypothesis: `fuse()` imputes 0.0 for any
    # candidate missing from a signal's own top-`candidates` list (the
    # candidate union is dense top-N ∪ bm25 top-N ∪ entity top-N), so a turn
    # found only by BM25 is penalised against one found by both signals; and
    # per-query min-max normalization stretches a noisy dense channel to the
    # full [0,1] range even when every cosine in the query is low (typical
    # of date-heavy queries), letting dense noise dominate the fused score.
    #
    # `full_candidate_scoring`: after building the candidate union from the
    # top-N lists as usual (candidate GENERATION is unchanged), compute each
    # enabled signal's TRUE score for EVERY candidate instead of imputing 0
    # for the ones outside that signal's own top-N — dense via
    # `VectorIndex.scores_for`, bm25 via `BM25Index.score_doc`, entity via
    # `EntityIndex.score_doc` (see retrieval.py `_dense_full_scores` /
    # `_bm25_full_scores` / `_entity_full_scores` and `fuse_full`). Kept OFF
    # by default until proven on the ledger.
    full_candidate_scoring: bool = False

    # E12 entity_expansion (Palier 2-3, docs/ABLATIONS.md, new mechanism):
    # after fusion+scoring, take the top `expansion_seed` hits, collect each
    # seed's "rare" entities (df/n_docs <= 0.05, >=3 chars, not a
    # stopword/question word -- see entities.py `JUNK_ENTITY_WORDS`), and add
    # every OTHER item sharing >=1 rare entity with a seed (via
    # `EntityIndex.postings`) that isn't already present, scored at
    # `seed.score * expansion_decay * (shared / max_shared)` (`shared` = how
    # many of that seed's rare entities the candidate shares; `max_shared` =
    # the largest such count for that seed) -- a multi-hop expansion distinct
    # from E3's `neighbor_expansion` (time-adjacency within a doc_ref): this
    # hops via shared rare entities across the whole corpus, seed-scoped so a
    # seed's own additions are capped by that seed's own relevance. Additions
    # across all seeds are capped at `2 * top_k` (highest-scored first).
    # See retrieval.py `HybridRetriever._expand_entities` /
    # `_rare_entities`. Kept OFF by default until proven on the ledger.
    entity_expansion: bool = False
    expansion_seed: int = 10
    expansion_decay: float = 0.5

    # `dense_norm`: "minmax" (default, unchanged) normalizes each signal's
    # raw scores per-query via min-max (or the dbsf z-score-clip-minmax
    # variant). "fixed" replaces just the dense channel's normalization with
    # a fixed linear map of cosine similarity onto [0,1] (negatives clipped
    # to 0, no per-query min-max stretch) so the dense channel's absolute
    # magnitude carries confidence — a query where every cosine is low (weak
    # dense evidence) no longer gets artificially stretched to fill [0,1].
    # In that same mode bm25 is normalized by its own per-query max
    # (score / max) instead of min-max, for the analogous reason. Only
    # affects fusion methods "cc"/"dbsf" (rrf uses ranks, not normalized
    # scores, so this is a no-op there); entity normalization is unchanged.
    dense_norm: str = "minmax"  # "minmax" | "fixed"

    # --- Production port of two LoCoMo-validated bench mechanisms (2026-09-14) ---
    # A. index_keywords: when True (default), `memory.index_text()` appends
    # `" | keywords: k1, k2, ..."` to the contextual-header+text string used
    # for dense embedding, BM25 tokenization, and the add_facts dedup probe,
    # for any item whose `MemoryItem.keywords` is non-empty (bench
    # `V4EnrichedRetriever` "facts+keywords" mode, +0.03 R@10 on LoCoMo).
    # `MemoryItem.text` itself is never touched -- this only changes what
    # dense/BM25 index. A no-op (byte-identical indexed text) for any item
    # with no keywords, so existing callers are unaffected by default.
    index_keywords: bool = True

    # B. multi_query_rrf_k: the RRF constant (see retrieval.py
    # `HybridRetriever.search_multi`) used to fuse the original query's hit
    # list with the client-supplied `queries` (sub-questions) and
    # `hypotheses` (HyDE-style statements) hit lists (bench
    # `V4PlannedRetriever` query planning, +0.075 R@10 on LoCoMo). Only
    # exercised when a caller passes `queries` or `hypotheses` to
    # `search()`/`get_context()`/`session_brief()`; a plain `search(query)`
    # call is completely unaffected (single-query path, unchanged).
    multi_query_rrf_k: int = 20

    # C. facet_channel / facet_weight (2026-09-19): production port of the
    # LoCoMo-validated bench mechanism -- see `facets.py`,
    # `bench/retrievers.py`'s `build_facet_index`/`rank_facet_pool`/
    # `V4PlannedRetriever.search`'s facet branch, and
    # `bench/data/enrich/PREDICATES.md` (the closed vocabulary). When a
    # caller supplies `facets` to `search()`/`get_context()` AND this flag
    # is on AND the resulting facet pool is non-empty, the facet-ranked pool
    # (`facets.rank_facet_pool`) is fused as one more party in the RRF
    # combination alongside the query/sub-query/hypothesis hit lists (see
    # retrieval.py `HybridRetriever.search_multi`). Measured on LoCoMo ONLY
    # (2026-09-19): R@10 0.721 -> 0.757 with `facet_weight=2.0`, no
    # reranker; `facet_weight` was tuned on that SAME benchmark. Honestly:
    # this is a single-benchmark result -- kept OFF by default until a
    # second benchmark (e.g. LongMemEval) confirms the gain, per this
    # module's established ablation-flag convention.
    facet_channel: bool = False
    facet_weight: float = 2.0
    # facet_share (2026-10-09): the LongMemEval second benchmark
    # (`bench/lme_facets_eval.py`, docs/IDEAS.md) showed that an ABSOLUTE
    # facet weight does not transfer: 2.0 was ~30% of the total RRF vote on
    # LoCoMo (mean query-list weight 4.79 with sub-queries + hypotheses) but
    # 67% with a single query list, where the channel then acts as a filter
    # (R@10 0.688 -> 0.665). When `facet_share` is set (default), the facet
    # party's weight is derived so that it is exactly this SHARE of the
    # total vote, whatever the number of query lists:
    # w_facet = share / (1 - share) * sum(query-list weights). 0.30 is the
    # LoCoMo-equivalent of 2.0 (2.0 / (2.0 + 4.79) = 0.295). `None` falls
    # back to the absolute `facet_weight` (bench reproduction only).
    facet_share: float | None = 0.30

    def half_life_days(self, kind: str) -> float:
        """Recency half-life (days) applicable to a given item kind."""
        table = {
            "episode": self.half_life_episode_days,
            "fact": self.half_life_fact_days,
            "belief": self.half_life_fact_days,
            "summary": self.half_life_fact_days,
            "doc": self.half_life_doc_days,
            "procedure": self.half_life_doc_days,
        }
        return table.get(kind, self.half_life_default_days)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config`, applying ``MEMORY_V4_*`` overrides."""
        d = cls()
        return cls(
            data_dir=_env_str("MEMORY_V4_DIR", d.data_dir),
            embed_model=_env_str("MEMORY_V4_EMBED_MODEL", d.embed_model),
            device=_env_str("MEMORY_V4_DEVICE", d.device),
            hash_dim=_env_int("MEMORY_V4_HASH_DIM", d.hash_dim),
            embed_max_seq_length=_env_int("MEMORY_V4_EMBED_MAX_SEQ_LENGTH", d.embed_max_seq_length),
            embed_batch_size=_env_int("MEMORY_V4_EMBED_BATCH_SIZE", d.embed_batch_size),
            fusion=_env_str("MEMORY_V4_FUSION", d.fusion),
            w_dense=_env_float("MEMORY_V4_W_DENSE", d.w_dense),
            w_bm25=_env_float("MEMORY_V4_W_BM25", d.w_bm25),
            w_entity=_env_float("MEMORY_V4_W_ENTITY", d.w_entity),
            candidates=_env_int("MEMORY_V4_CANDIDATES", d.candidates),
            rerank_model=_env_opt_str("MEMORY_V4_RERANK_MODEL", d.rerank_model),
            rerank_top_n=_env_int("MEMORY_V4_RERANK_TOP_N", d.rerank_top_n),
            rerank_weight=_env_float("MEMORY_V4_RERANK_WEIGHT", d.rerank_weight),
            rerank_device=_env_str("MEMORY_V4_RERANK_DEVICE", d.rerank_device),
            dedup_cos=_env_float("MEMORY_V4_DEDUP_COS", d.dedup_cos),
            supersede_cos=_env_float("MEMORY_V4_SUPERSEDE_COS", d.supersede_cos),
            half_life_episode_days=_env_float(
                "MEMORY_V4_HALF_LIFE_EPISODE", d.half_life_episode_days
            ),
            half_life_fact_days=_env_float("MEMORY_V4_HALF_LIFE_FACT", d.half_life_fact_days),
            half_life_doc_days=_env_float("MEMORY_V4_HALF_LIFE_DOC", d.half_life_doc_days),
            half_life_default_days=_env_float(
                "MEMORY_V4_HALF_LIFE_DEFAULT", d.half_life_default_days
            ),
            use_recency=_env_bool("MEMORY_V4_USE_RECENCY", d.use_recency),
            use_entities=_env_bool("MEMORY_V4_USE_ENTITIES", d.use_entities),
            use_bm25=_env_bool("MEMORY_V4_USE_BM25", d.use_bm25),
            use_dense=_env_bool("MEMORY_V4_USE_DENSE", d.use_dense),
            block_char_limit=_env_int("MEMORY_V4_BLOCK_CHAR_LIMIT", d.block_char_limit),
            episode_ttl_days=_env_int("MEMORY_V4_EPISODE_TTL_DAYS", d.episode_ttl_days),
            multi_vector=_env_bool("MEMORY_V4_MULTI_VECTOR", d.multi_vector),
            chunk_tokens=_env_int("MEMORY_V4_CHUNK_TOKENS", d.chunk_tokens),
            chunk_overlap=_env_int("MEMORY_V4_CHUNK_OVERLAP", d.chunk_overlap),
            multi_vector_candidate_mult=_env_int(
                "MEMORY_V4_MULTI_VECTOR_CANDIDATE_MULT", d.multi_vector_candidate_mult
            ),
            use_stemming=_env_bool("MEMORY_V4_USE_STEMMING", d.use_stemming),
            neighbor_expansion=_env_bool("MEMORY_V4_NEIGHBOR_EXPANSION", d.neighbor_expansion),
            neighbor_seed=_env_int("MEMORY_V4_NEIGHBOR_SEED", d.neighbor_seed),
            neighbor_decay=_env_float("MEMORY_V4_NEIGHBOR_DECAY", d.neighbor_decay),
            neighbor_window=_env_int("MEMORY_V4_NEIGHBOR_WINDOW", d.neighbor_window),
            recency_only_with_at_time=_env_bool(
                "MEMORY_V4_RECENCY_ONLY_WITH_AT_TIME", d.recency_only_with_at_time
            ),
            temporal_query_mode=_env_str("MEMORY_V4_TEMPORAL_QUERY_MODE", d.temporal_query_mode),
            temporal_bm25_boost=_env_float("MEMORY_V4_TEMPORAL_BM25_BOOST", d.temporal_bm25_boost),
            full_candidate_scoring=_env_bool(
                "MEMORY_V4_FULL_CANDIDATE_SCORING", d.full_candidate_scoring
            ),
            dense_norm=_env_str("MEMORY_V4_DENSE_NORM", d.dense_norm),
            entity_expansion=_env_bool("MEMORY_V4_ENTITY_EXPANSION", d.entity_expansion),
            expansion_seed=_env_int("MEMORY_V4_EXPANSION_SEED", d.expansion_seed),
            expansion_decay=_env_float("MEMORY_V4_EXPANSION_DECAY", d.expansion_decay),
            index_keywords=_env_bool("MEMORY_V4_INDEX_KEYWORDS", d.index_keywords),
            multi_query_rrf_k=_env_int("MEMORY_V4_MULTI_QUERY_RRF_K", d.multi_query_rrf_k),
            facet_channel=_env_bool("MEMORY_V4_FACET_CHANNEL", d.facet_channel),
            facet_weight=_env_float("MEMORY_V4_FACET_WEIGHT", d.facet_weight),
            facet_share=_env_optional_float("MEMORY_V4_FACET_SHARE", d.facet_share),
        )
