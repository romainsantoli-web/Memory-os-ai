"""In-house Okapi BM25 with incremental add/remove and JSON persistence.

Uses dict postings (token -> {doc_id: term_freq}) so a query only touches the
documents containing at least one query token, not a full corpus scan.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


class BM25Index:
    """Okapi BM25 (k1=1.2, b=0.75 by default)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.postings: dict[str, dict[str, int]] = {}
        self.doc_len: dict[str, int] = {}
        # A9 (adversarial review, 2026-09-12): a per-doc term -> term-frequency
        # map (at most one entry per DISTINCT token) instead of the full,
        # possibly highly-redundant token list -- `remove()` only ever needs
        # each token once (to walk `postings`), so persisting every repeated
        # occurrence in bm25.json wasted disk on any doc with repeated words.
        self.doc_term_freqs: dict[str, dict[str, int]] = {}
        self._total_len = 0

    @property
    def n_docs(self) -> int:
        return len(self.doc_len)

    @property
    def avgdl(self) -> float:
        return (self._total_len / self.n_docs) if self.n_docs else 0.0

    def add(self, doc_id: str, tokens: list[str]) -> None:
        """Add or replace a document's tokens."""
        if doc_id in self.doc_len:
            self.remove(doc_id)
        counts = Counter(tokens)
        self.doc_term_freqs[doc_id] = dict(counts)
        self.doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)
        for tok, tf in counts.items():
            self.postings.setdefault(tok, {})[doc_id] = tf

    def remove(self, doc_id: str) -> None:
        if doc_id not in self.doc_len:
            return
        term_freqs = self.doc_term_freqs.pop(doc_id, {})
        self._total_len -= self.doc_len.pop(doc_id, 0)
        for tok in term_freqs:
            bucket = self.postings.get(tok)
            if bucket and doc_id in bucket:
                del bucket[doc_id]
                if not bucket:
                    del self.postings[tok]

    def _idf(self, token: str) -> float:
        df = len(self.postings.get(token, {}))
        if df == 0:
            return 0.0
        n = self.n_docs
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        if not query_tokens or self.n_docs == 0:
            return []
        avgdl = self.avgdl or 1.0
        candidates: set[str] = set()
        for tok in query_tokens:
            candidates.update(self.postings.get(tok, {}).keys())
        if not candidates:
            return []
        scores: dict[str, float] = {doc_id: 0.0 for doc_id in candidates}
        for tok in query_tokens:
            bucket = self.postings.get(tok)
            if not bucket:
                continue
            idf = self._idf(tok)
            if idf <= 0:
                continue
            for doc_id in candidates:
                tf = bucket.get(doc_id)
                if not tf:
                    continue
                dl = self.doc_len.get(doc_id, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                scores[doc_id] += idf * (tf * (self.k1 + 1)) / denom
        # Deterministic total order: `candidates` is a set (PYTHONHASHSEED-
        # dependent iteration order), so ties must break on doc_id, not on
        # the random insertion order into `scores` (adversarial review,
        # 2026-09-12).
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(d, s) for d, s in ranked[:top_k] if s > 0]

    def score_doc(self, query_tokens: list[str], doc_id: str) -> float:
        """True BM25 score of `doc_id` against `query_tokens` — the exact
        same per-term formula `search()` uses, but computed for one
        arbitrary doc regardless of whether it would appear in `search()`'s
        own top-k (used by ``Config.full_candidate_scoring``, see
        retrieval.py). Returns 0.0 for an unknown doc or an empty query,
        never raises.
        """
        if not query_tokens or doc_id not in self.doc_len:
            return 0.0
        avgdl = self.avgdl or 1.0
        dl = self.doc_len[doc_id]
        score = 0.0
        for tok in query_tokens:
            bucket = self.postings.get(tok)
            if not bucket:
                continue
            tf = bucket.get(doc_id)
            if not tf:
                continue
            idf = self._idf(tok)
            if idf <= 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    # ------------------------------------------------------------- persist
    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "postings": self.postings,
            "doc_len": self.doc_len,
            "doc_term_freqs": self.doc_term_freqs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BM25Index":
        idx = cls(k1=d.get("k1", 1.2), b=d.get("b", 0.75))
        idx.postings = {tok: dict(bucket) for tok, bucket in d.get("postings", {}).items()}
        idx.doc_len = dict(d.get("doc_len", {}))
        idx.doc_term_freqs = {k: dict(v) for k, v in d.get("doc_term_freqs", {}).items()}
        idx._total_len = sum(idx.doc_len.values())
        return idx

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
