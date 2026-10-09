"""Tests for bench.retrievers.

Importing bench.retrievers (done at module scope below, for every test in
this file) must never fail even before memory_os_v4.MemoryV4 exists -- that
is itself exercised implicitly by this file collecting at all. V3Equiv/Dense
tests use a fake, injectable embed_fn so no model weights are ever loaded.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from bench.protocol import BenchItem
from bench.retrievers import (
    BM25TurnRetriever,
    DenseTurnRetriever,
    V3EquivRetriever,
    V3ExactRetriever,
    V4Retriever,
)


def _hash_vec(text: str, dim: int = 4) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    ints = np.frombuffer(h[: dim * 4], dtype=">i4").astype("float32")
    return ints


def make_fake_embed_fn(overrides: dict[str, np.ndarray], dim: int = 4):
    """Deterministic fake embedder: exact string -> vector via `overrides`,
    else a stable hash-derived vector (never a real model, so tests are fast
    and don't touch the filesystem cache / MPS)."""

    def _embed(texts: list[str]) -> np.ndarray:
        vecs = [overrides[t].astype("float32") if t in overrides else _hash_vec(t, dim) for t in texts]
        return np.stack(vecs).astype("float32")

    return _embed


def _item(id_: str, text: str) -> BenchItem:
    return BenchItem(id=id_, text=text, speaker="s", event_time=None, session_id="s1")


# ---------------------------------------------------------------------------
# V3EquivRetriever: chunking math
# ---------------------------------------------------------------------------
def test_chunk_text_short_text_is_single_chunk():
    assert V3EquivRetriever.chunk_text("short text") == ["short text"]
    exact = "x" * 128
    assert V3EquivRetriever.chunk_text(exact) == [exact]


def test_chunk_text_empty_text():
    assert V3EquivRetriever.chunk_text("") == [""]


def test_chunk_text_windows_and_28char_overlap():
    text = "".join(str(i % 10) for i in range(250))  # length 250, all distinct-ish content
    chunks = V3EquivRetriever.chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0] == text[0:128]
    assert chunks[1] == text[100:228]  # 100-char step (128 - 28 overlap)
    assert chunks[2] == text[200:250]  # final short remainder chunk
    # verify the declared 28-char overlap between consecutive windows
    assert chunks[0][-28:] == chunks[1][:28] == text[100:128]
    assert chunks[1][-28:] == chunks[2][:28] == text[200:228]


def test_chunk_text_step_and_size_constants():
    assert V3EquivRetriever.CHUNK_SIZE == 128
    assert V3EquivRetriever.CHUNK_OVERLAP == 28
    assert V3EquivRetriever.CHUNK_STEP == 100


# ---------------------------------------------------------------------------
# V3EquivRetriever: search behaviour (dedupe to parent id, cut to top_k)
# ---------------------------------------------------------------------------
def test_v3equiv_search_ranks_by_distance_and_respects_top_k():
    overrides = {
        "ALPHA": np.array([1, 0, 0, 0], dtype="float32"),
        "BETA": np.array([0, 0, 0, 0], dtype="float32"),
        "GAMMA": np.array([2, 0, 0, 0], dtype="float32"),
        "QUERY": np.array([0, 0, 0, 0], dtype="float32"),  # == BETA exactly
    }
    retriever = V3EquivRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    retriever.add([_item("ida", "ALPHA"), _item("idb", "BETA"), _item("idc", "GAMMA")])

    assert retriever.search("QUERY", top_k=1) == ["idb"]
    assert retriever.search("QUERY", top_k=2) == ["idb", "ida"]
    assert retriever.search("QUERY", top_k=10) == ["idb", "ida", "idc"]


def test_v3equiv_dedupes_multiple_chunks_of_same_item_to_one_id():
    # "X"*200 chunks into 2 windows ("X"*128 and "X"*100), both near the query;
    # "Y"*50 is a single far chunk. The near item must appear only ONCE.
    overrides = {
        "X" * 128: np.array([0, 0, 0, 0], dtype="float32"),
        "X" * 100: np.array([0.01, 0, 0, 0], dtype="float32"),
        "Y" * 50: np.array([10, 0, 0, 0], dtype="float32"),
        "Q": np.array([0, 0, 0, 0], dtype="float32"),
    }
    retriever = V3EquivRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    retriever.add([_item("idlong", "X" * 200), _item("idfar", "Y" * 50)])

    assert retriever.search("Q", top_k=1) == ["idlong"]
    result_top2 = retriever.search("Q", top_k=2)
    assert result_top2 == ["idlong", "idfar"]
    assert len(result_top2) == len(set(result_top2))  # no duplicate ids ever


def test_v3equiv_reset_clears_index():
    overrides = {"A": np.array([0, 0, 0, 0], dtype="float32"), "q": np.array([0, 0, 0, 0], dtype="float32")}
    retriever = V3EquivRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    retriever.add([_item("a", "A")])
    assert retriever.search("q", top_k=5) == ["a"]
    retriever.reset()
    assert retriever.search("q", top_k=5) == []  # empty index after reset


def test_v3equiv_search_on_empty_index_returns_empty():
    retriever = V3EquivRetriever(embed_fn=make_fake_embed_fn({}))
    retriever.reset()
    assert retriever.search("anything", top_k=5) == []


def test_v3equiv_ignores_at_time_faithfully(monkeypatch):
    # v3 had zero temporal awareness (PLAN.md diagnostic: "Temporel: aucun");
    # at_time must be accepted but change nothing about the result.
    overrides = {"A": np.array([0, 0, 0, 0], dtype="float32"), "q": np.array([0, 0, 0, 0], dtype="float32")}
    retriever = V3EquivRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    retriever.add([_item("a", "A")])
    assert retriever.search("q", top_k=5, at_time=None) == retriever.search("q", top_k=5, at_time=123456789.0)


# ---------------------------------------------------------------------------
# A10: V3ExactRetriever -- literal reproduction of real v3 chunking/search
# ---------------------------------------------------------------------------
def test_v3exact_chunk_text_matches_literal_v3_formula():
    text = "".join(str(i % 10) for i in range(250))
    expected = [text[i : i + 128] for i in range(0, len(text), 100)]
    assert V3ExactRetriever.chunk_text(text) == expected


def test_v3exact_chunk_text_keeps_the_ragged_tail_v3equiv_drops():
    """V3EquivRetriever drops the tiny overlap-only tail chunk for a text
    exactly CHUNK_SIZE long; the literal v3 formula still emits it."""
    text = "x" * 128
    assert V3ExactRetriever.chunk_text(text) == [text, text[100:128]]
    assert V3EquivRetriever.chunk_text(text) == [text]  # charitable: no tail


def test_v3exact_chunk_text_empty_text():
    assert V3ExactRetriever.chunk_text("") == [""]


def test_v3exact_search_drops_chunks_past_distance_threshold():
    overrides = {
        "NEAR": np.array([0.1, 0, 0, 0], dtype="float32"),
        "FAR": np.array([5.0, 0, 0, 0], dtype="float32"),  # squared L2 distance to q is >> 0.8
        "q": np.array([0, 0, 0, 0], dtype="float32"),
    }
    retriever = V3ExactRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    retriever.add([_item("near_id", "NEAR"), _item("far_id", "FAR")])
    result = retriever.search("q", top_k=5)
    assert "near_id" in result
    assert "far_id" not in result  # dropped by the 0.8 distance cutoff


def test_v3exact_search_no_oversampling_can_return_fewer_than_top_k():
    """With k = min(top_k, n) chunks (no `* 8` oversampling) and no
    dedup-and-pad expansion, a corpus whose chunks map to few DISTINCT
    parent ids can legitimately return fewer than top_k results."""
    overrides = {
        "A1": np.array([0, 0, 0, 0], dtype="float32"),
        "q": np.array([0, 0, 0, 0], dtype="float32"),
    }
    retriever = V3ExactRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    # both chunks belong to the SAME parent id -- with k=2 (top_k) and no
    # oversampling, both nearest hits map to "only", so the result has just
    # 1 distinct id even though top_k=2 was requested.
    retriever.add([_item("only", "A1")])
    retriever._chunk_texts.append("A2")
    retriever._chunk_item_ids.append("only")
    retriever._index.add(np.array([[0.01, 0, 0, 0]], dtype="float32"))
    result = retriever.search("q", top_k=2)
    assert result == ["only"]
    assert len(result) < 2


def test_v3exact_search_on_empty_index_returns_empty():
    retriever = V3ExactRetriever(embed_fn=make_fake_embed_fn({}))
    retriever.reset()
    assert retriever.search("anything", top_k=5) == []


def test_v3exact_reset_clears_index():
    overrides = {"A": np.array([0, 0, 0, 0], dtype="float32"), "q": np.array([0, 0, 0, 0], dtype="float32")}
    retriever = V3ExactRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    retriever.add([_item("a", "A")])
    assert retriever.search("q", top_k=5) == ["a"]
    retriever.reset()
    assert retriever.search("q", top_k=5) == []


def test_v3exact_ignores_at_time_faithfully():
    overrides = {"A": np.array([0, 0, 0, 0], dtype="float32"), "q": np.array([0, 0, 0, 0], dtype="float32")}
    retriever = V3ExactRetriever(embed_fn=make_fake_embed_fn(overrides))
    retriever.reset()
    retriever.add([_item("a", "A")])
    assert retriever.search("q", top_k=5, at_time=None) == retriever.search("q", top_k=5, at_time=123456789.0)


# ---------------------------------------------------------------------------
# DenseTurnRetriever
# ---------------------------------------------------------------------------
def test_dense_turn_retriever_whole_turn_ranking():
    overrides = {
        "one": np.array([1, 0], dtype="float32"),
        "two": np.array([0, 1], dtype="float32"),
        "q": np.array([0, 1], dtype="float32"),
    }
    retriever = DenseTurnRetriever(embed_fn=make_fake_embed_fn(overrides, dim=2))
    retriever.reset()
    retriever.add([_item("a", "one"), _item("b", "two")])
    assert retriever.search("q", top_k=1) == ["b"]
    assert retriever.search("q", top_k=2) == ["b", "a"]


def test_dense_turn_retriever_empty_index():
    retriever = DenseTurnRetriever(embed_fn=make_fake_embed_fn({}, dim=2))
    retriever.reset()
    assert retriever.search("q", top_k=5) == []


# ---------------------------------------------------------------------------
# BM25TurnRetriever
# ---------------------------------------------------------------------------
def test_bm25_ranks_docs_containing_the_query_term():
    retriever = BM25TurnRetriever()
    retriever.reset()
    retriever.add(
        [
            _item("cat_doc", "the cat sat on the mat"),
            _item("dog_doc", "dogs are friendly animals"),
            _item("weather_doc", "completely unrelated weather content"),
        ]
    )
    assert retriever.search("cat", top_k=5) == ["cat_doc"]


def test_bm25_top_k_truncates():
    retriever = BM25TurnRetriever()
    retriever.reset()
    retriever.add([_item(f"d{i}", "alpha beta gamma") for i in range(5)])
    result = retriever.search("alpha", top_k=2)
    assert len(result) == 2


def test_bm25_empty_query_or_empty_index():
    retriever = BM25TurnRetriever()
    assert retriever.search("anything", top_k=5) == []  # empty index
    retriever.add([_item("x", "hello world")])
    assert retriever.search("", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []


def test_bm25_idf_matches_okapi_formula():
    retriever = BM25TurnRetriever()
    retriever.reset()
    retriever.add([_item("d1", "alpha beta"), _item("d2", "alpha gamma")])
    # "alpha": df=2, n=2 ; "beta": df=1, n=2 ; unseen term: df=0
    expected_idf_alpha = math.log(1.0 + (2 - 2 + 0.5) / (2 + 0.5))
    expected_idf_beta = math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    assert retriever._idf("alpha") == pytest.approx(expected_idf_alpha)
    assert retriever._idf("beta") == pytest.approx(expected_idf_beta)
    assert retriever._idf("nonexistent_term") == 0.0


def test_bm25_reset_clears_state():
    retriever = BM25TurnRetriever()
    retriever.add([_item("x", "hello world")])
    assert retriever.search("hello", top_k=5) == ["x"]
    retriever.reset()
    assert retriever.search("hello", top_k=5) == []


# ---------------------------------------------------------------------------
# V4Retriever: import-safety contract
# ---------------------------------------------------------------------------
def test_v4retriever_matches_core_availability_contract():
    """bench.retrievers must always be importable (already proven by this
    file's top-level import). Whether *instantiating* V4Retriever raises
    ImportError depends only on whether memory_os_v4 currently exposes
    MemoryV4/Config -- written by design as a conditional check so this test
    stays correct whether it runs before or after the parallel core lands.
    """
    try:
        from memory_os_v4 import MemoryV4  # noqa: F401
        from memory_os_v4.config import Config  # noqa: F401

        core_ready = True
    except ImportError:
        core_ready = False

    if not core_ready:
        with pytest.raises(ImportError):
            V4Retriever()
    else:
        try:
            V4Retriever()
        except ImportError:
            pytest.fail("V4Retriever raised ImportError even though memory_os_v4 core is importable")
        except Exception:
            pass  # anything else (e.g. Config validation) is out of scope here
