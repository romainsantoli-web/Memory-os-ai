"""Retrieval metrics: Recall@k, MRR, nDCG@k (binary relevance), and aggregation.

All functions are pure / stdlib-only (no numpy) so they are trivial to unit
test with hand-computed examples (see tests/test_bench_metrics.py).
"""
from __future__ import annotations

import math
from statistics import mean
from typing import Any, Iterable, Sequence


def recall_at_k(ranked: Sequence[str], evidence: Sequence[str], k: int) -> float:
    """Fraction of gold evidence ids found in the top-k ranked ids.

    Returns 0.0 if `evidence` is empty (caller is expected to filter those
    questions out beforehand; this is a safe default rather than NaN so the
    bench never crashes on an edge case slipping through).
    """
    if not evidence:
        return 0.0
    top = set(ranked[:k])
    gold = set(evidence)
    hit = len(top & gold)
    return hit / len(gold)


def mrr(ranked: Sequence[str], evidence: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant item found in `ranked` (0 if none)."""
    if not evidence:
        return 0.0
    gold = set(evidence)
    for i, item_id in enumerate(ranked, start=1):
        if item_id in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], evidence: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG@k: rel(i) = 1 if ranked[i] in evidence else 0."""
    if not evidence:
        return 0.0
    gold = set(evidence)
    top = ranked[:k]
    dcg = 0.0
    for i, item_id in enumerate(top, start=1):
        rel = 1.0 if item_id in gold else 0.0
        if rel:
            dcg += rel / math.log2(i + 1)
    n_rel = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _p(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (0<=pct<=100), stdlib-only, stable & simple."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(pct / 100.0 * len(s)) - 1))
    return s[idx]


def aggregate(
    per_question_rows: Iterable[dict[str, Any]],
    latency_key: str = "latency_ms",
    category_key: str = "category",
    metric_keys: Sequence[str] = ("r5", "r10", "mrr", "ndcg10"),
    tokens_key: str = "context_tokens",
) -> dict[str, Any]:
    """Aggregate per-question metric rows into per-category / macro / micro summaries.

    `per_question_rows`: iterable of dicts each containing at least the keys
    in `metric_keys`, plus `category_key` and `latency_key`. `tokens_key` is
    optional (mean returned-context size in ~tokens, chars/4).

    Returns:
        {
          "n_questions": int,
          "per_category": {cat: {metric: mean, ..., "n": int}},
          "macro": {metric: mean over category means},
          "micro": {metric: mean over all questions},
          "latency_ms": {"p50": ..., "p95": ..., "mean": ...},
          "mean_context_tokens": float | None,
        }
    """
    rows = list(per_question_rows)
    n = len(rows)
    per_category: dict[str, dict[str, Any]] = {}
    for row in rows:
        cat = str(row.get(category_key))
        bucket = per_category.setdefault(cat, {m: [] for m in metric_keys})
        for m in metric_keys:
            bucket[m].append(float(row.get(m, 0.0)))

    per_category_means: dict[str, Any] = {}
    for cat, bucket in per_category.items():
        entry = {m: (mean(vals) if vals else 0.0) for m, vals in bucket.items()}
        entry["n"] = len(next(iter(bucket.values()))) if bucket else 0
        per_category_means[cat] = entry

    macro: dict[str, float] = {}
    for m in metric_keys:
        cat_means = [per_category_means[c][m] for c in per_category_means]
        macro[m] = mean(cat_means) if cat_means else 0.0

    micro: dict[str, float] = {}
    for m in metric_keys:
        vals = [float(row.get(m, 0.0)) for row in rows]
        micro[m] = mean(vals) if vals else 0.0

    latencies = [float(row.get(latency_key, 0.0)) for row in rows if latency_key in row]
    latency_stats = {
        "p50": _p(latencies, 50),
        "p95": _p(latencies, 95),
        "mean": mean(latencies) if latencies else 0.0,
    }

    token_vals = [float(row[tokens_key]) for row in rows if row.get(tokens_key) is not None]
    mean_tokens = mean(token_vals) if token_vals else None

    return {
        "n_questions": n,
        "per_category": per_category_means,
        "macro": macro,
        "micro": micro,
        "latency_ms": latency_stats,
        "mean_context_tokens": mean_tokens,
    }


def context_tokens(texts: Sequence[str]) -> float:
    """Cheap token estimate: total chars / 4 across the given texts."""
    return sum(len(t) for t in texts) / 4.0
