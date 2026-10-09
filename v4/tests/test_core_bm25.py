"""BM25 ranking sanity and persistence."""

from memory_os_v4.bm25 import BM25Index
from memory_os_v4.text import tokenize


def _index():
    idx = BM25Index()
    idx.add("d1", tokenize("the quick brown fox jumps over the lazy dog"))
    idx.add("d2", tokenize("foxes are quick and clever animals"))
    idx.add("d3", tokenize("completely unrelated text about cooking pasta"))
    return idx


def test_bm25_ranks_relevant_doc_first():
    idx = _index()
    results = idx.search(tokenize("quick fox"), top_k=3)
    ids = [d for d, _ in results]
    assert ids[0] in ("d1", "d2")
    assert "d3" not in ids or results[-1][0] == "d3"


def test_bm25_empty_query_or_empty_index():
    idx = _index()
    assert idx.search([], top_k=3) == []
    assert BM25Index().search(tokenize("anything"), top_k=3) == []


def test_bm25_remove_updates_postings():
    idx = _index()
    idx.remove("d1")
    assert idx.n_docs == 2
    results = idx.search(tokenize("quick fox"), top_k=3)
    ids = [d for d, _ in results]
    assert "d1" not in ids


def test_bm25_incremental_add_replaces_document():
    idx = _index()
    idx.add("d1", tokenize("nothing to do with foxes at all"))
    results = idx.search(tokenize("quick fox"), top_k=3)
    ids = [d for d, _ in results]
    assert "d1" not in ids or results[0][0] != "d1"


def test_bm25_persistence_roundtrip(tmp_path):
    idx = _index()
    path = tmp_path / "bm25.json"
    idx.save(path)
    loaded = BM25Index.load(path)
    assert loaded.n_docs == idx.n_docs
    original = idx.search(tokenize("quick fox"), top_k=3)
    restored = loaded.search(tokenize("quick fox"), top_k=3)
    assert original == restored


def test_bm25_scales_to_many_documents():
    idx = BM25Index()
    for i in range(2000):
        idx.add(f"doc{i}", tokenize(f"document number {i} about topic {i % 10} and filler words"))
    results = idx.search(tokenize("topic 3"), top_k=10)
    assert len(results) == 10
    assert all(score > 0 for _, score in results)


# ---------------------------------------------------------------------------
# A9: persist per-doc term-frequency maps, not full (possibly redundant)
# token lists (adversarial review, 2026-09-12)
# ---------------------------------------------------------------------------
def test_bm25_does_not_persist_full_token_lists():
    idx = BM25Index()
    assert not hasattr(idx, "doc_tokens")
    idx.add("d1", tokenize("the quick brown fox jumps over the lazy dog"))
    assert idx.doc_term_freqs["d1"] == {"quick": 1, "brown": 1, "fox": 1, "jumps": 1, "lazy": 1, "dog": 1}


def test_bm25_persisted_size_scales_with_distinct_terms_not_total_tokens(tmp_path):
    """A doc with heavy word repetition must not bloat bm25.json
    proportionally to its total token count -- only distinct terms matter."""
    idx = BM25Index()
    idx.add("repetitive", ["word"] * 5000)  # 1 distinct term, 5000 occurrences
    path = tmp_path / "bm25.json"
    idx.save(path)
    size_bytes = path.stat().st_size
    assert size_bytes < 2000  # a 5000-token doc_tokens list would dwarf this


def test_bm25_persistence_roundtrip_preserves_term_frequencies():
    idx = _index()
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bm25.json"
        idx.save(path)
        raw = json.loads(path.read_text())
        assert "doc_term_freqs" in raw
        assert "doc_tokens" not in raw
        loaded = BM25Index.load(path)
        assert loaded.doc_term_freqs == idx.doc_term_freqs


def test_bm25_remove_after_reload_still_updates_postings(tmp_path):
    """remove() must work off the persisted per-doc term-frequency map, not
    a token list that no longer exists after a round-trip."""
    idx = _index()
    path = tmp_path / "bm25.json"
    idx.save(path)
    loaded = BM25Index.load(path)
    loaded.remove("d1")
    assert loaded.n_docs == 2
    results = loaded.search(tokenize("quick fox"), top_k=3)
    assert "d1" not in [d for d, _ in results]


# ---------------------------------------------------------------------------
# score_doc (Config.full_candidate_scoring, docs/ABLATIONS.md "Full-candidate
# scoring 2026-09-13"): must equal the score `search()` computes for the
# same doc, but work for a doc outside `search()`'s own top-k too.
# ---------------------------------------------------------------------------
def test_bm25_score_doc_matches_search_score_for_same_doc():
    idx = _index()
    q = tokenize("quick fox")
    search_scores = dict(idx.search(q, top_k=10))
    for doc_id in ("d1", "d2", "d3"):
        assert idx.score_doc(q, doc_id) == search_scores.get(doc_id, 0.0)


def test_bm25_score_doc_matches_search_even_outside_a_tiny_top_k():
    """score_doc must give the doc's TRUE score even when it wouldn't
    survive search()'s own top_k=1 cutoff -- this is the whole point of the
    helper (full_candidate_scoring re-scores every candidate, not just the
    signal's own top-N)."""
    idx = _index()
    q = tokenize("quick fox")
    full_search = dict(idx.search(q, top_k=10))
    top1 = dict(idx.search(q, top_k=1))
    assert len(top1) == 1
    dropped = next(d for d in full_search if d not in top1)
    assert idx.score_doc(q, dropped) == full_search[dropped]
    assert idx.score_doc(q, dropped) > 0.0


def test_bm25_score_doc_unknown_doc_returns_zero():
    idx = _index()
    assert idx.score_doc(tokenize("quick fox"), "does-not-exist") == 0.0


def test_bm25_score_doc_empty_query_returns_zero():
    idx = _index()
    assert idx.score_doc([], "d1") == 0.0


def test_bm25_score_doc_no_term_overlap_is_zero():
    idx = _index()
    assert idx.score_doc(tokenize("quick fox"), "d3") == 0.0
