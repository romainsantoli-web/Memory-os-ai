"""Unit tests for bench.metrics — hand-computed examples."""
from __future__ import annotations

import math

import pytest

from bench.metrics import aggregate, context_tokens, mrr, ndcg_at_k, recall_at_k


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------
def test_recall_at_k_basic():
    ranked = ["a", "b", "c", "d", "e"]
    evidence = ["b", "d", "z"]  # 'z' is never retrievable
    assert recall_at_k(ranked, evidence, 3) == pytest.approx(1 / 3)  # only 'b' in top3
    assert recall_at_k(ranked, evidence, 5) == pytest.approx(2 / 3)  # 'b' and 'd' in top5


def test_recall_at_k_perfect_and_zero():
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert recall_at_k(["x", "y"], ["a", "b"], 2) == 0.0


def test_recall_at_k_empty_evidence_is_zero_not_nan():
    assert recall_at_k(["a", "b"], [], 5) == 0.0


def test_recall_at_k_k_larger_than_ranked_list():
    assert recall_at_k(["a"], ["a", "b"], 10) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------
def test_mrr_first_hit_rank():
    assert mrr(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert mrr(["a", "b", "c"], ["a", "c"]) == pytest.approx(1.0)  # 'a' hits at rank 1
    assert mrr(["a", "b", "c"], ["z"]) == 0.0
    assert mrr([], ["a"]) == 0.0
    assert mrr(["a"], []) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k (binary relevance)
# ---------------------------------------------------------------------------
def test_ndcg_at_k_hand_computed():
    ranked = ["a", "b", "c", "d"]
    evidence = ["b", "d"]
    # rel = [0, 1, 0, 1] at ranks 1..4
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(5)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)  # 2 relevant items, ideally at ranks 1,2
    expected = dcg / idcg
    assert ndcg_at_k(ranked, evidence, 4) == pytest.approx(expected)


def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = ["a", "b", "c", "d"]
    evidence = ["a", "b"]
    assert ndcg_at_k(ranked, evidence, 4) == pytest.approx(1.0)


def test_ndcg_at_k_no_hits_is_zero():
    assert ndcg_at_k(["a", "b"], ["z"], 2) == 0.0


def test_ndcg_at_k_empty_evidence_is_zero():
    assert ndcg_at_k(["a", "b"], [], 2) == 0.0


def test_ndcg_at_k_truncates_ideal_by_k():
    # 3 relevant items but k=1: ideal DCG uses only 1 relevant item.
    ranked = ["a", "b", "c"]
    evidence = ["a", "b", "c"]
    dcg = 1.0 / math.log2(2)  # single slot, hit
    idcg = 1.0 / math.log2(2)
    assert ndcg_at_k(ranked, evidence, 1) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(ranked, evidence, 1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def test_aggregate_macro_vs_micro_with_imbalanced_categories():
    # category "A": 1 question, r10=0.0 ; category "B": 3 questions, r10=1.0
    rows = [
        {"category": "A", "r5": 0.0, "r10": 0.0, "mrr": 0.0, "ndcg10": 0.0, "latency_ms": 10.0},
        {"category": "B", "r5": 1.0, "r10": 1.0, "mrr": 1.0, "ndcg10": 1.0, "latency_ms": 20.0},
        {"category": "B", "r5": 1.0, "r10": 1.0, "mrr": 1.0, "ndcg10": 1.0, "latency_ms": 30.0},
        {"category": "B", "r5": 1.0, "r10": 1.0, "mrr": 1.0, "ndcg10": 1.0, "latency_ms": 40.0},
    ]
    agg = aggregate(rows)
    assert agg["n_questions"] == 4
    assert agg["per_category"]["A"]["r10"] == pytest.approx(0.0)
    assert agg["per_category"]["A"]["n"] == 1
    assert agg["per_category"]["B"]["r10"] == pytest.approx(1.0)
    assert agg["per_category"]["B"]["n"] == 3
    # macro: mean of per-category means -> (0.0 + 1.0) / 2 = 0.5
    assert agg["macro"]["r10"] == pytest.approx(0.5)
    # micro: mean over all 4 questions -> 3/4 = 0.75 (category B dominates)
    assert agg["micro"]["r10"] == pytest.approx(0.75)
    assert agg["macro"]["r10"] != agg["micro"]["r10"]


def test_aggregate_latency_percentiles():
    rows = [{"category": "A", "r5": 0, "r10": 0, "mrr": 0, "ndcg10": 0, "latency_ms": v} for v in [10, 20, 30, 40, 100]]
    agg = aggregate(rows)
    lat = agg["latency_ms"]
    assert lat["p50"] == 30  # median of [10,20,30,40,100]
    assert lat["p95"] == 100
    assert lat["mean"] == pytest.approx(40.0)


def test_aggregate_mean_context_tokens():
    rows = [
        {"category": "A", "r5": 0, "r10": 0, "mrr": 0, "ndcg10": 0, "latency_ms": 1, "context_tokens": 100.0},
        {"category": "A", "r5": 0, "r10": 0, "mrr": 0, "ndcg10": 0, "latency_ms": 1, "context_tokens": 200.0},
    ]
    agg = aggregate(rows)
    assert agg["mean_context_tokens"] == pytest.approx(150.0)


def test_aggregate_empty_rows():
    agg = aggregate([])
    assert agg["n_questions"] == 0
    assert agg["macro"]["r10"] == 0.0
    assert agg["mean_context_tokens"] is None


def test_context_tokens_is_chars_over_four():
    assert context_tokens(["abcd", "abcdefgh"]) == pytest.approx((4 + 8) / 4.0)
    assert context_tokens([]) == 0.0
