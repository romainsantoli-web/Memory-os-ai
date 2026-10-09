"""Fusion methods, MMR, and the HybridRetriever end-to-end via MemoryV4."""

import math
import time

import numpy as np
import pytest

from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4
from memory_os_v4.retrieval import (
    SearchHit,
    _fixed_cosine_normalize,
    _max_normalize,
    _zscore_clip_minmax,
    fuse,
    fuse_full,
    minmax_normalize,
    mmr,
)
from memory_os_v4.store import MemoryItem
from memory_os_v4.vector_index import VectorIndex


def test_minmax_normalize_basic():
    out = minmax_normalize({"a": 1.0, "b": 3.0, "c": 2.0})
    assert out["a"] == 0.0
    assert out["b"] == 1.0
    assert 0 < out["c"] < 1


def test_minmax_normalize_constant_values_falls_back_to_rank_order():
    """Adversarial review, 2026-09-12: an all-tied signal used to flatten
    every candidate to 1.0, which made that signal contribute nothing
    discriminative even though its inputs still had an implicit best-first
    rank order. Falls back to rank-based scoring instead (1.0 first, then
    linearly down to 0.0 for the last), using dict insertion order as rank."""
    out = minmax_normalize({"a": 5.0, "b": 5.0})
    assert out["a"] == 1.0
    assert out["b"] == 0.0


def test_minmax_normalize_single_value_stays_at_one():
    out = minmax_normalize({"solo": 3.0})
    assert out["solo"] == 1.0


def test_zscore_clip_minmax_constant_values_falls_back_to_rank_order():
    out = _zscore_clip_minmax({"a": 2.0, "b": 2.0, "c": 2.0})
    assert out["a"] == 1.0
    assert out["c"] == 0.0
    assert out["a"] > out["b"] > out["c"]


# ---------------------------------------------------------------------------
# A8: minmax_normalize / _zscore_clip_minmax must guard NaN/±inf (adversarial
# review, 2026-09-12: a poisoned raw score could produce NaN results that
# silently corrupted the final ranking, e.g. inf - inf or inf / inf).
# ---------------------------------------------------------------------------
def test_minmax_normalize_guards_nan():
    out = minmax_normalize({"a": float("nan"), "b": 0.5})
    assert all(math.isfinite(v) for v in out.values())
    # falls back to rank order (insertion order = "a" first, "b" second)
    assert out == {"a": 1.0, "b": 0.0}


def test_minmax_normalize_guards_inf():
    out = minmax_normalize({"a": float("inf"), "b": 0.0})
    assert all(math.isfinite(v) for v in out.values())
    assert out == {"a": 1.0, "b": 0.0}

    out2 = minmax_normalize({"a": float("-inf"), "b": 0.0})
    assert all(math.isfinite(v) for v in out2.values())


def test_zscore_clip_minmax_guards_nan_and_inf():
    out = _zscore_clip_minmax({"a": float("nan"), "b": 1.0, "c": 2.0})
    assert all(math.isfinite(v) for v in out.values())
    out2 = _zscore_clip_minmax({"a": float("inf"), "b": 1.0})
    assert all(math.isfinite(v) for v in out2.values())


def test_fuse_cc_guards_nan_raw_score():
    """End-to-end through `fuse`: a NaN raw score in one signal must never
    produce a NaN fused score for any candidate."""
    fused = fuse({"a", "b"}, {"dense": [("a", float("nan")), ("b", 0.5)]}, {"dense": 1.0}, "cc")
    assert all(math.isfinite(v) for v in fused.values())


def test_fuse_rrf_orders_candidates_present_in_more_signals_higher():
    candidates = {"a", "b", "c"}
    signals = {
        "dense": [("a", 0.9), ("b", 0.5), ("c", 0.1)],
        "bm25": [("a", 5.0), ("c", 1.0)],
    }
    weights = {"dense": 1.0, "bm25": 1.0, "entity": 0.5}
    fused = fuse(candidates, signals, weights, "rrf")
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]


def test_fuse_cc_respects_weights():
    candidates = {"a", "b"}
    signals = {"dense": [("a", 1.0), ("b", 0.0)], "bm25": [("a", 0.0), ("b", 1.0)]}
    heavy_dense = fuse(candidates, signals, {"dense": 5.0, "bm25": 0.1}, "cc")
    heavy_bm25 = fuse(candidates, signals, {"dense": 0.1, "bm25": 5.0}, "cc")
    assert heavy_dense["a"] > heavy_dense["b"]
    assert heavy_bm25["b"] > heavy_bm25["a"]


def test_fuse_dbsf_respects_weights():
    candidates = {"a", "b", "c"}
    signals = {"dense": [("a", 10.0), ("b", 1.0), ("c", 0.5)]}
    fused = fuse(candidates, signals, {"dense": 1.0}, "dbsf")
    assert fused["a"] > fused["b"] > fused["c"]


def test_mmr_diversifies_similar_high_scoring_items():
    vectors = {}
    a = MemoryItem(id="a", kind="fact", text="a")
    b = MemoryItem(id="b", kind="fact", text="b")  # near-duplicate of a
    c = MemoryItem(id="c", kind="fact", text="c")  # different but relevant
    hits = [SearchHit(item=a, score=1.0), SearchHit(item=b, score=0.95), SearchHit(item=c, score=0.5)]
    vectors["a"] = np.array([1.0, 0.0], dtype=np.float32)
    vectors["b"] = np.array([0.99, 0.01], dtype=np.float32)
    vectors["c"] = np.array([0.0, 1.0], dtype=np.float32)
    result = mmr(hits, vectors, lambda_=0.5, k=2)
    ids = [h.item.id for h in result]
    assert ids[0] == "a"
    assert "c" in ids  # diversified away from near-duplicate b


def _make_memory(tmp_path, **cfg_kwargs) -> MemoryV4:
    cfg = Config(embed_model="hash", data_dir=str(tmp_path), **cfg_kwargs)
    return MemoryV4(config=cfg)


def test_hybrid_retriever_finds_relevant_episode(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("We chose FAISS FlatL2 for the vector index.", project="p1")
    m.add_episode("The weather was nice today.", project="p1")
    hits = m.search("vector index choice FAISS", top_k=5)
    assert hits
    assert "FAISS" in hits[0].item.text


def test_hybrid_retriever_project_filter(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("mm_cross_v2 rewrite finished.", project="p1")
    m.add_episode("mm_cross_v2 rewrite finished.", project="p2", item_id="dup-other-project")
    hits = m.search("mm_cross_v2 rewrite", top_k=5, project="p1")
    assert all(h.item.project == "p1" for h in hits)


def test_hybrid_retriever_kind_filter(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("Romain likes Python.", project="p1")
    m.add_facts([{"text": "Romain uses Python daily."}], project="p1")
    hits = m.search("Python", top_k=5, kinds=["fact"])
    assert hits
    assert all(h.item.kind == "fact" for h in hits)


def test_fusion_methods_all_return_results(tmp_path):
    for fusion in ("rrf", "cc", "dbsf"):
        m = _make_memory(tmp_path / fusion, fusion=fusion)
        m.add_episode("The Kraken exchange API rate limit is 60 requests per minute.", project="p1")
        m.add_episode("Unrelated note about gardening.", project="p1")
        hits = m.search("Kraken API rate limit", top_k=5)
        assert hits
        assert "Kraken" in hits[0].item.text


def test_search_dampens_superseded_fact_by_default(tmp_path):
    cfg = Config(embed_model="hash", data_dir=str(tmp_path), supersede_cos=0.5, dedup_cos=0.92)
    m = MemoryV4(config=cfg)
    t0 = time.time() - 200000
    t1 = time.time() - 100000
    m.add_facts([{"text": "Romain lives in Paris."}], project="p1", event_time=t0)
    result = m.add_facts([{"text": "Romain lives in Lyon."}], project="p1", event_time=t1)
    assert result["superseded"]
    hits = m.search("Where does Romain live?", top_k=5)
    lyon = next(h for h in hits if "Lyon" in h.item.text)
    paris = next(h for h in hits if "Paris" in h.item.text)
    assert lyon.score > paris.score


# ---------------------------------------------------------------------------
# E1 multi_vector: chunk vectors + max-pooling (ablation, docs/ABLATIONS.md)
# ---------------------------------------------------------------------------
def test_dense_candidates_max_pools_chunk_scores_to_owning_item(tmp_path):
    m = _make_memory(tmp_path, hash_dim=4, multi_vector=True)
    m.vector_index.add(
        ["itemA::0", "itemA::1", "itemB::0"],
        np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]], dtype="float32"),
    )
    m.store.chunk_owner = {"itemA::0": "itemA", "itemA::1": "itemA", "itemB::0": "itemB"}
    hits = m.retriever._dense_candidates(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), 10)
    scores = dict(hits)
    assert abs(scores["itemA"] - 1.0) < 1e-6  # max over its two chunks (1.0 and 0.5), not their sum/mean
    assert abs(scores["itemB"] - 0.9) < 1e-6
    assert set(scores) == {"itemA", "itemB"}  # chunk ids never leak out as candidate ids


def test_dense_candidates_plain_path_unaffected_when_multi_vector_off(tmp_path):
    m = _make_memory(tmp_path)  # multi_vector defaults to False
    m.add_episode("A single whole-turn episode about Kraken rate limits.", project="p1", item_id="only")
    qvec = m.embedder.encode(["Kraken rate limits"])[0]
    hits = m.retriever._dense_candidates(qvec, 5)
    assert [i for i, _ in hits] == ["only"]


def test_multi_vector_indexing_populates_chunk_owner_for_long_items(tmp_path):
    m = _make_memory(tmp_path, multi_vector=True, chunk_tokens=8, chunk_overlap=2)
    long_text = " ".join(f"sentence number {i} about the rollout." for i in range(20))
    m.add_episode(long_text, project="p1", item_id="long1")
    owners = set(m.store.chunk_owner.values())
    assert "long1" in owners
    assert any(cid.startswith("long1::") for cid in m.store.chunk_owner)
    # the whole-item representative vector is still present (dedup/MMR rely on it)
    assert m.vector_index.get_one("long1") is not None


def test_multi_vector_short_item_gets_no_extra_chunk_vectors(tmp_path):
    m = _make_memory(tmp_path, multi_vector=True, chunk_tokens=180, chunk_overlap=30)
    m.add_episode("Short note.", project="p1", item_id="short1")
    assert not any(cid.startswith("short1::") for cid in m.store.chunk_owner)


# ---------------------------------------------------------------------------
# E3 neighbor_expansion (ablation, docs/ABLATIONS.md)
# ---------------------------------------------------------------------------
def test_expand_neighbors_adds_window_one_neighbor_with_decayed_score(tmp_path):
    m = _make_memory(
        tmp_path, neighbor_expansion=True, neighbor_decay=0.5, neighbor_window=1, neighbor_seed=10
    )
    m.add_episode("first turn", project="p1", doc_ref="sess1", item_id="t1", event_time=1000)
    m.add_episode("second turn", project="p1", doc_ref="sess1", item_id="t2", event_time=1001)
    m.add_episode("third turn", project="p1", doc_ref="sess1", item_id="t3", event_time=1002)

    seed_hit = SearchHit(item=m.store.get("t1"), score=1.0, signals={})
    expanded = m.retriever._expand_neighbors([seed_hit], kinds=None, project=None, at_time=None)
    ids = [h.item.id for h in expanded]
    assert "t2" in ids  # +-1 neighbor pulled in
    assert "t3" not in ids  # window=1: t3 is 2 positions away from the seed
    t2_hit = next(h for h in expanded if h.item.id == "t2")
    assert abs(t2_hit.score - 0.5) < 1e-9
    assert t2_hit.signals["neighbor_of"] == "t1"


def test_expand_neighbors_window_two_reaches_further(tmp_path):
    m = _make_memory(
        tmp_path, neighbor_expansion=True, neighbor_decay=0.5, neighbor_window=2, neighbor_seed=10
    )
    m.add_episode("first turn", project="p1", doc_ref="sess1", item_id="t1", event_time=1000)
    m.add_episode("second turn", project="p1", doc_ref="sess1", item_id="t2", event_time=1001)
    m.add_episode("third turn", project="p1", doc_ref="sess1", item_id="t3", event_time=1002)

    seed_hit = SearchHit(item=m.store.get("t1"), score=1.0, signals={})
    expanded = m.retriever._expand_neighbors([seed_hit], kinds=None, project=None, at_time=None)
    ids = {h.item.id for h in expanded}
    assert {"t2", "t3"} <= ids


def test_expand_neighbors_never_overrides_an_already_present_id(tmp_path):
    m = _make_memory(
        tmp_path, neighbor_expansion=True, neighbor_decay=0.5, neighbor_window=1, neighbor_seed=10
    )
    m.add_episode("first turn", project="p1", doc_ref="sess1", item_id="t1", event_time=1000)
    m.add_episode("second turn", project="p1", doc_ref="sess1", item_id="t2", event_time=1001)

    seed1 = SearchHit(item=m.store.get("t1"), score=1.0, signals={})
    seed2 = SearchHit(item=m.store.get("t2"), score=0.9, signals={})  # already present, own native score
    expanded = m.retriever._expand_neighbors([seed1, seed2], kinds=None, project=None, at_time=None)
    t2_hit = next(h for h in expanded if h.item.id == "t2")
    assert t2_hit.score == 0.9  # kept its own score, not overwritten by the neighbor-decay score


def test_expand_neighbors_no_doc_ref_is_a_no_op(tmp_path):
    m = _make_memory(tmp_path, neighbor_expansion=True)
    m.add_episode("solo turn, no doc_ref", project="p1", item_id="solo")
    seed_hit = SearchHit(item=m.store.get("solo"), score=1.0, signals={})
    expanded = m.retriever._expand_neighbors([seed_hit], kinds=None, project=None, at_time=None)
    assert [h.item.id for h in expanded] == ["solo"]


# ---------------------------------------------------------------------------
# E6 recency: at_time=None should not silently vary with real wall-clock time
# ---------------------------------------------------------------------------
def test_recency_varies_with_wallclock_by_default_when_at_time_is_none(tmp_path):
    """Documents the pre-fix behaviour: at_time=None still falls back to
    time.time(), so two equally-relevant items with different event_time get
    different recency multipliers even though the caller gave no query time."""
    now = time.time()
    m = _make_memory(tmp_path)
    m.add_episode(
        "Kraken exchange rate limit note.", project="p1", event_time=now - 400 * 86400, item_id="old"
    )
    m.add_episode("Kraken exchange rate limit note.", project="p1", event_time=now, item_id="new")
    hits = m.search("Kraken exchange rate limit", top_k=5)
    recency = {h.item.id: h.signals["recency"] for h in hits}
    assert recency["old"] != recency["new"]


def test_recency_only_with_at_time_makes_recency_constant_without_at_time(tmp_path):
    now = time.time()
    m = _make_memory(tmp_path, recency_only_with_at_time=True)
    m.add_episode(
        "Kraken exchange rate limit note.", project="p1", event_time=now - 400 * 86400, item_id="old"
    )
    m.add_episode("Kraken exchange rate limit note.", project="p1", event_time=now, item_id="new")
    hits = m.search("Kraken exchange rate limit", top_k=5)
    recency = {h.item.id: h.signals["recency"] for h in hits}
    assert recency["old"] == recency["new"] == 1.0


def test_recency_only_with_at_time_still_applies_decay_when_at_time_given(tmp_path):
    m = _make_memory(tmp_path, recency_only_with_at_time=True)
    m.add_episode("Kraken exchange rate limit note.", project="p1", event_time=1000.0, item_id="old")
    m.add_episode("Kraken exchange rate limit note.", project="p1", event_time=2_000_000.0, item_id="new")
    hits = m.search("Kraken exchange rate limit", top_k=5, at_time=2_000_000.0)
    recency = {h.item.id: h.signals["recency"] for h in hits}
    assert recency["new"] == 1.0
    assert recency["old"] < 1.0


def test_recency_adaptive_half_life_floors_to_corpus_span(tmp_path):
    """Adversarial review, 2026-09-12: half_life=30d against a corpus whose
    span is hundreds of days makes recency underflow to ~0 for every older
    item alike (a constant, not a signal). recency_only_with_at_time floors
    half_life to 25% of the corpus's own event_time span so it stays
    numerically meaningful."""
    day = 86400.0
    span_days = 400.0
    at_time = span_days * day

    m_fixed = _make_memory(tmp_path / "fixed", recency_only_with_at_time=True)
    m_fixed.add_episode("Kraken exchange rate limit note.", project="p1", event_time=0.0, item_id="old")
    m_fixed.add_episode("Kraken exchange rate limit note.", project="p1", event_time=at_time, item_id="new")
    hits_fixed = m_fixed.search("Kraken exchange rate limit", top_k=5, at_time=at_time)
    recency_fixed = {h.item.id: h.signals["recency"] for h in hits_fixed}

    m_legacy = _make_memory(tmp_path / "legacy")  # recency_only_with_at_time defaults False
    m_legacy.add_episode("Kraken exchange rate limit note.", project="p1", event_time=0.0, item_id="old")
    m_legacy.add_episode("Kraken exchange rate limit note.", project="p1", event_time=at_time, item_id="new")
    hits_legacy = m_legacy.search("Kraken exchange rate limit", top_k=5, at_time=at_time)
    recency_legacy = {h.item.id: h.signals["recency"] for h in hits_legacy}

    # legacy fixed 30-day half-life against a 400-day-old item: underflows near 0
    assert recency_legacy["old"] < 0.001
    # adaptive half-life (0.25 * 400 = 100d) keeps a 400-day-old item's decay
    # numerically meaningful instead of collapsing to ~0
    assert recency_fixed["old"] > 0.05


# ---------------------------------------------------------------------------
# A7(b): query_dates exposed on every SearchHit.signals; Part B temporal_query_mode
# (docs/ABLATIONS.md "Hardening 2026-09-13")
# ---------------------------------------------------------------------------
def test_search_exposes_query_dates_signal(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("Some note about the Kraken exchange.", project="p1")
    hits = m.search("What happened on 2026-01-02?", top_k=5)
    assert hits
    assert len(hits[0].signals["query_dates"]) == 1


def test_search_query_dates_empty_for_non_temporal_query(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("Some note about the Kraken exchange.", project="p1")
    hits = m.search("Kraken exchange rate limit", top_k=5)
    assert hits
    assert hits[0].signals["query_dates"] == []


def test_temporal_query_mode_off_by_default(tmp_path):
    m = _make_memory(tmp_path)  # temporal_query_mode defaults to "off"
    m.add_episode("Note about the Kraken exchange rate limit.", project="p1")
    hits = m.search("When did we discuss the Kraken exchange rate limit?", top_k=5)
    assert hits
    assert hits[0].signals["temporal_boost"] is False


def test_temporal_query_mode_auto_fires_on_temporal_cue(tmp_path):
    m = _make_memory(tmp_path, temporal_query_mode="auto")
    m.add_episode("Note about the Kraken exchange rate limit.", project="p1")
    hits = m.search("When did we discuss the Kraken exchange rate limit?", top_k=5)
    assert hits
    assert hits[0].signals["temporal_boost"] is True
    assert hits[0].signals["recency"] == 1.0  # recency disabled for this query


def test_temporal_query_mode_auto_does_not_fire_on_plain_query(tmp_path):
    m = _make_memory(tmp_path, temporal_query_mode="auto")
    m.add_episode("Note about the Kraken exchange rate limit.", project="p1")
    hits = m.search("Kraken exchange rate limit", top_k=5)
    assert hits
    assert hits[0].signals["temporal_boost"] is False


def test_temporal_query_mode_auto_boosts_bm25_weight_for_temporal_query():
    """A query that fires temporal_boost must use w_bm25 * temporal_bm25_boost
    for fusion -- verified by checking the resulting weighted contribution
    directly via `fuse`, isolating it from full-pipeline noise."""
    cfg = Config(embed_model="hash", temporal_query_mode="auto", temporal_bm25_boost=2.0, w_bm25=1.5)
    signals = {"bm25": [("a", 4.0)]}
    weights = {"dense": cfg.w_dense, "bm25": cfg.w_bm25 * cfg.temporal_bm25_boost, "entity": cfg.w_entity}
    boosted = fuse({"a"}, signals, weights, cfg.fusion)
    unboosted = fuse({"a"}, signals, {"dense": cfg.w_dense, "bm25": cfg.w_bm25, "entity": cfg.w_entity}, cfg.fusion)
    assert boosted["a"] >= unboosted["a"]


def test_temporal_query_mode_auto_never_mutates_config(tmp_path):
    m = _make_memory(tmp_path, temporal_query_mode="auto")
    m.add_episode("Note about the Kraken exchange rate limit.", project="p1")
    before = (m.config.w_bm25, m.config.use_recency)
    m.search("When did we discuss the Kraken exchange rate limit?", top_k=5)
    after = (m.config.w_bm25, m.config.use_recency)
    assert before == after


# ---------------------------------------------------------------------------
# Full-candidate scoring (docs/ABLATIONS.md "Full-candidate scoring
# 2026-09-13"): VectorIndex.scores_for, the fixed/max normalizers, fuse_full,
# and the HybridRetriever._{dense,bm25,entity}_full_scores helpers + the
# Config.full_candidate_scoring / dense_norm flags end-to-end.
# ---------------------------------------------------------------------------
def test_vector_index_scores_for_matches_search_score_for_same_id():
    vi = VectorIndex(dim=4)
    vi.add(
        ["x", "y", "z"],
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0]], dtype="float32"),
    )
    qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    search_scores = dict(vi.search(qvec, top_k=10))
    full = vi.scores_for(qvec, ["x", "y", "z"])
    for sid in ("x", "y", "z"):
        assert abs(full[sid] - search_scores[sid]) < 1e-6


def test_vector_index_scores_for_matches_search_outside_a_tiny_top_k():
    vi = VectorIndex(dim=4)
    vi.add(
        ["x", "y", "z"],
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0]], dtype="float32"),
    )
    qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    full_search = dict(vi.search(qvec, top_k=10))
    top1 = dict(vi.search(qvec, top_k=1))
    dropped = next(sid for sid in full_search if sid not in top1)
    assert abs(vi.scores_for(qvec, [dropped])[dropped] - full_search[dropped]) < 1e-6


def test_vector_index_scores_for_missing_id_scores_zero():
    vi = VectorIndex(dim=4)
    vi.add(["x"], np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    out = vi.scores_for(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), ["x", "never-added"])
    assert out["x"] > 0.0
    assert out["never-added"] == 0.0


def test_vector_index_scores_for_empty_ids_returns_empty_dict():
    vi = VectorIndex(dim=4)
    vi.add(["x"], np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"))
    assert vi.scores_for(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), []) == {}


def test_fixed_cosine_normalize_clips_negatives_and_passes_through_positives():
    out = _fixed_cosine_normalize({"a": 0.9, "b": 0.0, "c": -0.3, "d": 0.05})
    assert out["a"] == 0.9
    assert out["b"] == 0.0
    assert out["c"] == 0.0  # clipped
    assert out["d"] == 0.05
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_fixed_cosine_normalize_does_not_stretch_uniformly_low_scores():
    """The whole point of dense_norm="fixed": unlike minmax_normalize, a
    signal where every candidate is a weak match must NOT get stretched so
    the best of a bad lot reads as 1.0."""
    out_fixed = _fixed_cosine_normalize({"a": 0.05, "b": 0.02})
    out_minmax = minmax_normalize({"a": 0.05, "b": 0.02})
    assert out_fixed["a"] == 0.05  # stays low
    assert out_minmax["a"] == 1.0  # stretched to look like a strong match


def test_max_normalize_divides_by_query_max():
    out = _max_normalize({"a": 4.0, "b": 2.0, "c": 0.0})
    assert out["a"] == 1.0
    assert out["b"] == 0.5
    assert out["c"] == 0.0


def test_max_normalize_all_zero_stays_zero():
    out = _max_normalize({"a": 0.0, "b": 0.0})
    assert out == {"a": 0.0, "b": 0.0}


def test_max_normalize_empty_returns_empty():
    assert _max_normalize({}) == {}


def test_fuse_full_scores_every_candidate_no_imputed_zero():
    """This is the hypothesis under test: `fuse` imputes 0.0 for a
    candidate missing from a signal's own top-N hit list; `fuse_full` is
    given that signal's TRUE score for every candidate instead, so a
    candidate outside the dense top-N is no longer flatly zeroed there.
    Three candidates so the dense top-N genuinely omits "b" (only "a" and
    "c" made the top-N) while full scoring still has its real cosine."""
    candidates = {"a", "b", "c"}
    full_scores = {"dense": {"a": 0.9, "b": 0.5, "c": 0.3}}
    weights = {"dense": 1.0}
    fused = fuse_full(candidates, full_scores, weights, "cc")
    # imputed-zero equivalent via plain `fuse`: dense top-N search only
    # surfaced "a" and "c" -- "b" never appears in the hit list at all.
    signal_hits = {"dense": [("a", 0.9), ("c", 0.3)]}
    fused_imputed = fuse(candidates, signal_hits, weights, "cc")
    assert fused_imputed["b"] == 0.0  # literal imputation
    assert fused["b"] > 0.0  # full scoring credits b's real (nonzero) dense score
    assert fused["b"] > fused_imputed["b"]


def test_fuse_full_rrf_ranks_over_the_full_scored_union():
    candidates = {"a", "b", "c"}
    full_scores = {"dense": {"a": 0.9, "b": 0.5, "c": 0.1}}
    fused = fuse_full(candidates, full_scores, {"dense": 1.0}, "rrf")
    assert fused["a"] > fused["b"] > fused["c"]


def test_fuse_full_dense_norm_fixed_uses_fixed_normalizers():
    candidates = {"a", "b"}
    full_scores = {"dense": {"a": 0.05, "b": 0.02}, "bm25": {"a": 4.0, "b": 2.0}}
    weights = {"dense": 1.0, "bm25": 1.0}
    fused_fixed = fuse_full(candidates, full_scores, weights, "cc", dense_norm="fixed")
    # dense contributes its raw (unstretched) value, bm25 contributes score/max
    assert abs(fused_fixed["a"] - (0.05 + 1.0)) < 1e-9
    assert abs(fused_fixed["b"] - (0.02 + 0.5)) < 1e-9


def test_fuse_full_unknown_fusion_method_raises():
    with pytest.raises(ValueError):
        fuse_full({"a"}, {"dense": {"a": 1.0}}, {"dense": 1.0}, "bogus")


def test_dense_full_scores_matches_dense_candidates_for_plain_path(tmp_path):
    m = _make_memory(tmp_path, full_candidate_scoring=True)
    m.add_episode("A note about Kraken rate limits.", project="p1", item_id="a")
    m.add_episode("Something else entirely about gardening.", project="p1", item_id="b")
    qvec = m.embedder.encode(["Kraken rate limits"])[0]
    topn = dict(m.retriever._dense_candidates(qvec, 10))
    full = m.retriever._dense_full_scores(qvec, {"a", "b"})
    for iid in ("a", "b"):
        assert abs(full[iid] - topn[iid]) < 1e-6


def test_dense_full_scores_max_pools_multi_vector_chunks(tmp_path):
    m = _make_memory(tmp_path, hash_dim=4, multi_vector=True)
    m.vector_index.add(
        ["itemA", "itemA::0", "itemB"],
        np.array([[0.2, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0]], dtype="float32"),
    )
    m.store.chunk_owner = {"itemA::0": "itemA"}
    qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    out = m.retriever._dense_full_scores(qvec, {"itemA", "itemB"})
    assert abs(out["itemA"] - 1.0) < 1e-6  # max(0.2, 1.0) from its chunk, not the representative vector alone
    assert abs(out["itemB"] - 0.9) < 1e-6


def test_bm25_full_scores_covers_candidates_outside_bm25_topn(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("Kraken exchange API rate limit details here.", project="p1", item_id="a")
    m.add_episode("Totally unrelated gardening note.", project="p1", item_id="b")
    from memory_os_v4.text import tokenize

    q_tokens = tokenize("Kraken API", stem=m.config.use_stemming)
    out = m.retriever._bm25_full_scores(q_tokens, {"a", "b"})
    assert out["a"] > 0.0
    assert out["b"] == 0.0  # no overlap, true score is 0 (not merely absent/imputed)


def test_bm25_full_scores_empty_query_returns_empty_dict(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("Some note.", project="p1", item_id="a")
    assert m.retriever._bm25_full_scores([], {"a"}) == {}


def test_entity_full_scores_covers_candidates_outside_entity_topn(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("A note mentioning Marie Curie and radium.", project="p1", item_id="a")
    m.add_episode("Unrelated note about gardening.", project="p1", item_id="b")
    # dilute df/n well under the 10% commonality cutoff (entities.py), same
    # convention as tests/test_core_entities.py, so "marie curie" isn't
    # skipped as a too-common (non-discriminative) entity.
    for i in range(20):
        m.add_episode(f"Filler note number {i} about nothing in particular.", project="p1", item_id=f"filler{i}")
    from memory_os_v4.entities import extract_entities

    q_ents = extract_entities("Marie Curie")
    out = m.retriever._entity_full_scores(q_ents, {"a", "b"})
    assert out["a"] > 0.0
    assert out["b"] == 0.0


def test_entity_full_scores_empty_query_returns_empty_dict(tmp_path):
    m = _make_memory(tmp_path)
    m.add_episode("Some note.", project="p1", item_id="a")
    assert m.retriever._entity_full_scores([], {"a"}) == {}


def test_full_candidate_scoring_off_by_default():
    cfg = Config()
    assert cfg.full_candidate_scoring is False
    assert cfg.dense_norm == "minmax"


def test_full_candidate_scoring_does_not_change_candidate_generation(tmp_path):
    """Point 3 of the spec: candidate ids depend only on each signal's
    top-N list, built BEFORE the full_candidate_scoring branch -- toggling
    the flag changes SCORING, never which candidates make it into the
    union. Two independently-built stores (same corpus/query/config except
    the flag) must therefore surface the exact same candidate id set."""
    off = _make_memory(tmp_path / "off", full_candidate_scoring=False, candidates=2)
    on = _make_memory(tmp_path / "on", full_candidate_scoring=True, candidates=2)
    for m in (off, on):
        m.add_episode("Kraken exchange API rate limit is 60 requests per minute.", project="p1", item_id="a")
        m.add_episode("A note about the Kraken migration timeline and rate caps.", project="p1", item_id="b")
        m.add_episode("Completely unrelated note about gardening tomatoes.", project="p1", item_id="c")

    hits_off = {h.item.id for h in off.search("Kraken rate limit migration timeline", top_k=10)}
    hits_on = {h.item.id for h in on.search("Kraken rate limit migration timeline", top_k=10)}
    assert hits_off  # sanity: got something
    assert hits_off == hits_on


def test_full_candidate_scoring_still_uses_topn_for_candidate_generation(tmp_path):
    """Point 3 of the spec: candidate GENERATION must still use the top-N
    lists (Config.candidates) -- only scoring changes. With candidates=1,
    at most 1 dense + 1 bm25 + 1 entity hit feed the candidate union
    regardless of full_candidate_scoring."""
    m = _make_memory(tmp_path, full_candidate_scoring=True, candidates=1)
    for i in range(5):
        m.add_episode(f"Kraken note number {i} about rate limits.", project="p1", item_id=f"i{i}")
    hits = m.search("Kraken rate limits", top_k=10)
    # at most dense(1) + bm25(1) + entity(1) distinct candidates can surface
    assert len(hits) <= 3


def test_dense_norm_fixed_does_not_change_default_off_behaviour(tmp_path):
    """dense_norm="fixed" without full_candidate_scoring only changes how
    the TOP-N dense/bm25 raw scores are normalized (still via `fuse`, still
    imputing 0 for out-of-top-N candidates) -- it must not raise and must
    still return the obviously relevant hit first."""
    m = _make_memory(tmp_path, dense_norm="fixed")
    m.add_episode("Kraken exchange API rate limit is 60 requests per minute.", project="p1")
    m.add_episode("The weather was nice today.", project="p1")
    hits = m.search("Kraken API rate limit", top_k=5)
    assert hits
    assert "Kraken" in hits[0].item.text


def test_full_candidate_scoring_with_rrf_fusion_returns_results(tmp_path):
    m = _make_memory(tmp_path, full_candidate_scoring=True, fusion="rrf")
    m.add_episode("Kraken exchange API rate limit is 60 requests per minute.", project="p1")
    m.add_episode("The weather was nice today.", project="p1")
    hits = m.search("Kraken API rate limit", top_k=5)
    assert hits
    assert "Kraken" in hits[0].item.text


def test_full_candidate_scoring_with_multi_vector_returns_results(tmp_path):
    m = _make_memory(tmp_path, full_candidate_scoring=True, multi_vector=True, chunk_tokens=8, chunk_overlap=2)
    long_text = " ".join(f"sentence number {i} about the Kraken rollout." for i in range(20))
    m.add_episode(long_text, project="p1", item_id="long1")
    m.add_episode("Unrelated short note.", project="p1", item_id="short1")
    hits = m.search("Kraken rollout", top_k=5)
    assert hits
    assert hits[0].item.id == "long1"


# ---------------------------------------------------------------------------
# Palier 2-3 E11: cross-encoder rerank blend (fake reranker injected --
# docs/ABLATIONS.md; real CrossEncoder behaviour is unit-tested separately
# in test_core_rerank.py).
# ---------------------------------------------------------------------------
class _FakeReranker:
    """Drop-in for CrossEncoderReranker: records the exact texts it was
    asked to score (so we can assert item.text is passed WITHOUT the
    contextual header being duplicated in), returns fixed scores."""

    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def available(self) -> bool:
        return True

    def score(self, query: str, texts: list[str]):
        self.calls.append((query, list(texts)))
        return [self.scores_by_text.get(t, 0.0) for t in texts]


def test_rerank_blend_uses_default_symmetric_weight(tmp_path):
    """With exactly two reranked candidates, minmax normalizes each signal
    to {0.0, 1.0}; a 0.5/0.5 blend of two fully-disagreeing binary signals
    ties by construction (0.5 each) -- broken deterministically by item id,
    per the codebase's established (score desc, id asc) convention. This
    test checks the mechanics (reranker invoked, raw item.text with no
    contextual-header duplication, exact blend arithmetic via the recorded
    `signals["rerank"]`); the weight=0.7 test below demonstrates an actual
    order flip once the blend is no longer symmetric."""
    m = _make_memory(tmp_path, rerank_model="fake/model", rerank_top_n=10)
    m.add_episode("Kraken exchange API rate limit is 60 requests per minute.", project="p1", item_id="a")
    m.add_episode("A different note about Kraken deposits.", project="p1", item_id="b")
    fake = _FakeReranker({"Kraken exchange API rate limit is 60 requests per minute.": -5.0,
                           "A different note about Kraken deposits.": 5.0})
    m.retriever._reranker = fake
    hits = m.search("Kraken API rate limit", top_k=5)
    assert fake.calls  # reranker was actually invoked
    query, texts = fake.calls[0]
    assert query == "Kraken API rate limit"
    # Exactly the raw item texts, no contextual header ("[date | project: ... ]") prefix.
    assert all(not t.startswith("[") for t in texts)
    by_id = {h.item.id: h for h in hits}
    assert by_id["a"].signals["rerank"] == 0.0  # worst raw CE score -> minmax 0.0
    assert by_id["b"].signals["rerank"] == 1.0  # best raw CE score -> minmax 1.0
    assert abs(by_id["a"].score - 0.5) < 1e-9  # 0.5*base_norm(1.0) + 0.5*ce_norm(0.0)
    assert abs(by_id["b"].score - 0.5) < 1e-9  # 0.5*base_norm(0.0) + 0.5*ce_norm(1.0)
    assert {h.item.id for h in hits} == {"a", "b"}


def test_rerank_weight_shifts_blend_toward_cross_encoder(tmp_path):
    """E11: Config.rerank_weight=0.7 weights the cross-encoder score more
    heavily than the symmetric 0.5/0.5 default."""
    m = _make_memory(tmp_path, rerank_model="fake/model", rerank_top_n=10, rerank_weight=0.7)
    m.add_episode("aaa", project="p1", item_id="a")
    m.add_episode("bbb", project="p1", item_id="b")
    fake = _FakeReranker({"aaa": 0.0, "bbb": 1.0})
    m.retriever._reranker = fake
    hits = m.search("q", top_k=5, rerank=True)
    fused_scores = {h.item.id: h.score for h in hits}
    # base_norm and ce_norm are each minmax'd to {0.0, 1.0} over {a, b};
    # final = (1-w)*base + w*ce. With ce favoring b (1.0) and w=0.7, b should
    # score higher than under the 0.5 default regardless of the fused order.
    assert hits[0].item.id == "b"
    assert fused_scores["b"] > fused_scores["a"]


def test_rerank_off_when_no_rerank_model_configured(tmp_path):
    m = _make_memory(tmp_path)  # rerank_model defaults to None
    m.add_episode("Kraken exchange API rate limit.", project="p1")
    hits = m.search("Kraken API rate limit", top_k=5)
    assert hits
    assert "rerank" not in hits[0].signals


# ---------------------------------------------------------------------------
# Palier 2-3 E12: entity_expansion, new multi-hop mechanism (docs/ABLATIONS.md)
# ---------------------------------------------------------------------------
def _add_filler(m, n, project="p1"):
    episodes = [
        {"text": f"filler note number {i} about nothing in particular", "role": "user", "project": project}
        for i in range(n)
    ]
    m.add_episodes_bulk(episodes)


def test_rare_entities_includes_low_df_identifier_shared_by_two_items(tmp_path):
    m = _make_memory(tmp_path)
    _add_filler(m, 40)  # + seed + target = 42 docs; df=2 -> 2/42 ~= 0.048 <= 0.05
    m.add_episode("Results for xgb_v14 look strong.", project="p1", item_id="seed")
    m.add_episode("Compare xgb_v14 against the baseline.", project="p1", item_id="target")
    rare = m.retriever._rare_entities("seed")
    assert "xgb_v14" in rare


def test_rare_entities_excludes_entity_above_five_percent_of_corpus(tmp_path):
    m = _make_memory(tmp_path)
    _add_filler(m, 10)
    for i in range(10):
        m.add_episode(f"Kraken update {i}.", project="p1", item_id=f"kraken{i}")
    rare = m.retriever._rare_entities("kraken0")
    assert "kraken" not in rare  # df=10/20=0.5 >> 0.05


def test_rare_entities_excludes_short_and_stopwords(tmp_path):
    m = _make_memory(tmp_path)
    _add_filler(m, 40)
    m.add_episode('Did "ok" work?', project="p1", item_id="seed")  # "ok" < 3 chars
    rare = m.retriever._rare_entities("seed")
    assert "ok" not in rare


def test_expand_entities_adds_item_sharing_rare_entity_with_decayed_score(tmp_path):
    m = _make_memory(tmp_path, entity_expansion=True, expansion_seed=10, expansion_decay=0.5)
    _add_filler(m, 40)
    m.add_episode("Results for xgb_v14 look strong.", project="p1", item_id="seed")
    m.add_episode("Compare xgb_v14 against the baseline.", project="p1", item_id="target")

    seed_hit = SearchHit(item=m.store.get("seed"), score=1.0, signals={})
    expanded = m.retriever._expand_entities([seed_hit], kinds=None, project=None, at_time=None, top_k=10)
    ids = [h.item.id for h in expanded]
    assert "target" in ids
    target_hit = next(h for h in expanded if h.item.id == "target")
    assert abs(target_hit.score - 0.5) < 1e-9  # 1.0 (seed) * decay(0.5) * (shared=1 / max_shared=1)
    assert target_hit.signals["expansion_of"] == "seed"


def test_expand_entities_never_overrides_an_already_present_id(tmp_path):
    m = _make_memory(tmp_path, entity_expansion=True)
    _add_filler(m, 40)
    m.add_episode("Results for xgb_v14 look strong.", project="p1", item_id="seed")
    m.add_episode("Compare xgb_v14 against the baseline.", project="p1", item_id="target")

    seed_hit = SearchHit(item=m.store.get("seed"), score=1.0, signals={})
    target_native = SearchHit(item=m.store.get("target"), score=0.9, signals={})
    expanded = m.retriever._expand_entities(
        [seed_hit, target_native], kinds=None, project=None, at_time=None, top_k=10
    )
    target_hit = next(h for h in expanded if h.item.id == "target")
    assert target_hit.score == 0.9  # kept its own native score, not overwritten


def test_expand_entities_caps_additions_at_2x_top_k(tmp_path):
    m = _make_memory(tmp_path, entity_expansion=True, expansion_seed=10, expansion_decay=0.5)
    n_share = 8
    # keep the shared entity "rare" (<=5% of corpus) despite many sharers:
    # df = 1 (seed) + n_share, so filler count must push n_docs high enough.
    df = 1 + n_share
    n_filler = 4 * df * 20  # generous margin under the 0.05 threshold
    _add_filler(m, n_filler)
    m.add_episode("Results for xgb_v14 look strong.", project="p1", item_id="seed")
    for i in range(n_share):
        m.add_episode(f"xgb_v14 variant {i} benchmark.", project="p1", item_id=f"share{i}")

    seed_hit = SearchHit(item=m.store.get("seed"), score=1.0, signals={})
    expanded = m.retriever._expand_entities([seed_hit], kinds=None, project=None, at_time=None, top_k=2)
    assert len(expanded) - 1 <= 2 * 2  # seed + at most 2*top_k additions


def test_expand_entities_no_rare_entities_is_a_no_op(tmp_path):
    m = _make_memory(tmp_path, entity_expansion=True)
    m.add_episode("plain lowercase note with no entities at all", project="p1", item_id="solo")
    seed_hit = SearchHit(item=m.store.get("solo"), score=1.0, signals={})
    expanded = m.retriever._expand_entities([seed_hit], kinds=None, project=None, at_time=None, top_k=10)
    assert [h.item.id for h in expanded] == ["solo"]


def test_entity_expansion_off_by_default(tmp_path):
    cfg = Config(embed_model="hash", data_dir=str(tmp_path))
    assert cfg.entity_expansion is False


def test_entity_expansion_end_to_end_pulls_in_a_result_dense_bm25_missed(tmp_path):
    m = _make_memory(tmp_path, entity_expansion=True, expansion_seed=10, expansion_decay=0.5)
    _add_filler(m, 40)
    m.add_episode("The xgb_v14 run finished overnight.", project="p1", item_id="seed")
    # No lexical/semantic overlap with the query below beyond the shared
    # rare entity -- only entity_expansion should be able to surface it.
    m.add_episode("xgb_v14: totally unrelated phrasing about something else entirely.",
                  project="p1", item_id="target")
    hits = m.search("xgb_v14 run finished", top_k=20)
    assert any(h.item.id == "target" for h in hits)


# ---------------------------------------------------------------------------
# Production port B: multi-query RRF fusion (`HybridRetriever.search_multi`,
# `MemoryV4.search(queries=..., hypotheses=...)`, 2026-09-14)
# ---------------------------------------------------------------------------
def _bm25_only_memory(tmp_path, **cfg_kwargs) -> MemoryV4:
    """A single-signal (BM25-only) engine: dense/entity candidates would add
    ranking noise from HashEmbedder that isn't relevant to testing RRF
    fusion's own arithmetic, which operates purely on rank position."""
    return _make_memory(tmp_path, use_dense=False, use_entities=False, use_bm25=True, **cfg_kwargs)


def test_search_multi_rrf_fuses_original_and_subquery_hit_lists(tmp_path):
    m = _bm25_only_memory(tmp_path)
    m.add_episode("alpha topic note.", project="p1", item_id="a")
    m.add_episode("beta topic note.", project="p1", item_id="b")
    m.add_episode("gamma topic note.", project="p1", item_id="c")

    hits = m.search("alpha", queries=["gamma"], top_k=5, project="p1")
    ids = {h.item.id for h in hits}
    assert ids == {"a", "c"}  # union of both hit lists; "b" matches neither

    scores = {h.item.id: h.score for h in hits}
    k = m.config.multi_query_rrf_k
    assert scores["a"] == pytest.approx(1.0 / (k + 1))  # rank 1 of the original query, weight 1.0
    assert scores["c"] == pytest.approx(1.0 / (k + 1))  # rank 1 of the sub-query, weight 1.0
    assert hits[0].signals["multi_query"] == {"n_queries": 2}  # original + 1 sub-query


def test_search_multi_hypotheses_weighted_lower_than_queries(tmp_path):
    m = _bm25_only_memory(tmp_path)
    m.add_episode("queryhit unique term.", project="p1", item_id="qh")
    m.add_episode("hypothesishit unique term.", project="p1", item_id="hh")

    hits = m.search(
        "nomatchatall000", queries=["queryhit"], hypotheses=["hypothesishit"], top_k=5, project="p1"
    )
    scores = {h.item.id: h.score for h in hits}
    k = m.config.multi_query_rrf_k
    assert scores["qh"] == pytest.approx(1.0 / (k + 1))
    assert scores["hh"] == pytest.approx(0.7 / (k + 1))
    assert scores["qh"] > scores["hh"]


def test_search_multi_guard_caps_total_extra_strings_at_eight(tmp_path):
    m = _bm25_only_memory(tmp_path)
    m.add_episode("alpha main topic.", project="p1", item_id="a")
    queries = [f"sub{i}" for i in range(10)]  # 10 requested, guard caps combined extras at 8
    hypotheses = ["hyp one", "hyp two"]  # no slots left once 8 queries are taken

    hits = m.search("alpha", queries=queries, hypotheses=hypotheses, top_k=5, project="p1")
    assert hits and hits[0].item.id == "a"
    assert hits[0].signals["multi_query"]["n_queries"] == 1 + 8  # original + 8 (queries win priority)


def test_search_multi_truncates_extra_query_strings_past_300_chars(tmp_path):
    m = _bm25_only_memory(tmp_path)
    m.add_episode("zzzmagic token here.", project="p1", item_id="a")
    # "zzzmagic" (the only token that would actually match item "a") sits
    # well past the 300-char truncation boundary, so it must never be tokenized.
    padding = "filler " * 50  # 350 chars
    long_query = padding + "zzzmagic"
    hits = m.search("no match at all", queries=[long_query], top_k=5, project="p1")
    assert hits == []


def test_search_multi_off_by_default_matches_plain_search(tmp_path):
    """Omitting `queries`/`hypotheses` (the default) must be byte-identical
    to the pre-existing single-query path -- no behaviour change for
    existing callers."""
    m = _make_memory(tmp_path)
    m.add_episode("We chose FAISS FlatL2 for the vector index.", project="p1")
    plain = m.search("vector index choice FAISS", top_k=5, project="p1")
    explicit = m.search("vector index choice FAISS", top_k=5, project="p1", queries=None, hypotheses=None)
    assert [h.item.id for h in plain] == [h.item.id for h in explicit]
    assert "multi_query" not in plain[0].signals
