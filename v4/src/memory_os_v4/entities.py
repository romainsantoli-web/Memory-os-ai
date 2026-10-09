"""Heuristic entity extraction and an inverted-index entity retriever.

Extracts: capitalized multi-word spans (proper nouns, excluding sentence-start
occurrences unless the same span also appears mid-sentence elsewhere),
ALLCAPS tokens (``NQ``, ``ES``, ``MBO``), code-ish identifiers
(``mm_cross_v2``, ``engine.py``, ``xgb_v14``), quoted phrases, numbers with a
unit/currency/percent suffix, and ISO dates. All entities are normalized
(lowercased, accent-stripped) before being used as index keys.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .text import STOPWORDS, split_sentences, strip_accents

# Adversarial review, 2026-09-12: a single-word capitalized span whose
# normalized form is a stopword or interrogative ("What did Caroline
# research?" -> "What" is sentence-start-capitalized) is not an entity — it
# pollutes both indexing and, worse, every query (queries are almost always
# one sentence, so the single-sentence branch below keeps sentence-start
# capitals unconditionally). STOPWORDS already covers what/which/who/whom/
# would/could/should/did/does/is/are/was/were; add the remaining common
# question/interjection words STOPWORDS doesn't carry.
# Public (Palier 2-3 E12 reuses this from retrieval.py for the same
# "not a real entity" filter, applied to seed entities before multi-hop
# expansion -- see retrieval.py `HybridRetriever._rare_entities`).
JUNK_ENTITY_WORDS = STOPWORDS | {"how", "why", "wow", "thanks", "however"}
_JUNK_ENTITY_WORDS = JUNK_ENTITY_WORDS  # back-compat alias, same object

_CAP_SPAN_RE = re.compile(r"\b[A-ZÀ-Þ][a-zà-ÿ'-]+(?:\s+[A-ZÀ-Þ][a-zà-ÿ'-]+)*\b")
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
_IDENTIFIER_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:[_.][a-zA-Z0-9]+)+\b")
_QUOTED_RE = re.compile(r'"([^"]{2,80})"|\'([^\']{2,80})\'|\xab([^\xbb]{2,80})\xbb')
_NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:%|bp|bps|ms|s|h|Go|Mo|Ko|km|kg|g|€|\$|USD|EUR|pts?)(?![a-zA-Z])",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _normalize(s: str) -> str:
    return strip_accents(s.strip().lower())


def _capitalized_spans(text: str) -> list[str]:
    sentences = split_sentences(text)

    if len(sentences) <= 1:
        # A single (often atomic-fact) sentence has nowhere else a subject could
        # recur: applying the start/mid distinction would drop every subject.
        found = []
        for sentence in sentences:
            for m in _CAP_SPAN_RE.finditer(sentence):
                span = m.group(0)
                if not span.isupper():  # ALLCAPS handled by _ALLCAPS_RE
                    found.append(span)
        return found

    sentence_start: list[str] = []
    mid_sentence: list[str] = []
    for sentence in sentences:
        for m in _CAP_SPAN_RE.finditer(sentence):
            span = m.group(0)
            if span.isupper():
                continue  # handled by _ALLCAPS_RE
            if m.start() == 0:
                sentence_start.append(span)
            else:
                mid_sentence.append(span)
    mid_set = set(mid_sentence)
    result = list(mid_sentence)
    for span in sentence_start:
        if span in mid_set:
            result.append(span)
    return result


def extract_entities(text: str) -> list[str]:
    """Extract and normalize entity candidates from free text; de-duplicated."""
    if not text:
        return []
    found: list[str] = []
    found.extend(_capitalized_spans(text))
    found.extend(_ALLCAPS_RE.findall(text))
    found.extend(_IDENTIFIER_RE.findall(text))
    for m in _QUOTED_RE.finditer(text):
        phrase = next(g for g in m.groups() if g)
        found.append(phrase)
    found.extend(m.group(0) for m in _NUMBER_UNIT_RE.finditer(text))
    found.extend(_ISO_DATE_RE.findall(text))

    seen: dict[str, None] = {}
    for raw in found:
        norm = _normalize(raw)
        if not norm or len(norm) < 2:
            continue
        if " " not in norm and norm in _JUNK_ENTITY_WORDS:
            continue  # single-word stopword/question-word, never a real entity
        seen.setdefault(norm, None)
    return list(seen.keys())


class EntityIndex:
    """Inverted index entity -> {doc_id} with inverse-document-frequency weighting."""

    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = {}
        self.doc_entities: dict[str, list[str]] = {}

    @property
    def n_docs(self) -> int:
        return len(self.doc_entities)

    def add(self, doc_id: str, entities: list[str]) -> None:
        if doc_id in self.doc_entities:
            self.remove(doc_id)
        ents = list(dict.fromkeys(entities))
        self.doc_entities[doc_id] = ents
        for e in ents:
            self.postings.setdefault(e, set()).add(doc_id)

    def remove(self, doc_id: str) -> None:
        ents = self.doc_entities.pop(doc_id, None)
        if not ents:
            return
        for e in ents:
            bucket = self.postings.get(e)
            if bucket and doc_id in bucket:
                bucket.discard(doc_id)
                if not bucket:
                    del self.postings[e]

    def search(self, query_entities: list[str], top_k: int) -> list[tuple[str, float]]:
        if not query_entities or self.n_docs == 0:
            return []
        scores: dict[str, float] = {}
        n = self.n_docs
        for e in query_entities:
            bucket = self.postings.get(e)
            if not bucket:
                continue
            df = len(bucket)
            # Adversarial review, 2026-09-12: an entity present in >10% of the
            # corpus (e.g. a dialogue's main participant, mentioned in most
            # turns) is not discriminative — it acts as near-uniform noise
            # across most candidates rather than signal. Skip it entirely
            # rather than let a token-log weight still dominate ties.
            if n > 0 and df / n > 0.1:
                continue
            weight = math.log(1.0 + n / df)
            for doc_id in bucket:
                scores[doc_id] = scores.get(doc_id, 0.0) + weight
        # Deterministic total order: `bucket` is a set (hash-order depends on
        # PYTHONHASHSEED), so ties in score must break on doc_id, not on the
        # random insertion order into `scores` (adversarial review finding:
        # otherwise seed-to-seed jitter of up to 0.007 R@10 on LoCoMo).
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    def score_doc(self, query_entities: list[str], doc_id: str) -> float:
        """True entity-overlap score of `doc_id` against `query_entities` —
        the exact same per-entity IDF-weighted formula `search()` uses
        (including the >10%-of-corpus "not discriminative" skip), but
        computed for one arbitrary doc regardless of whether it would
        appear in `search()`'s own top-k (used by
        ``Config.full_candidate_scoring``, see retrieval.py). Returns 0.0
        for an unknown doc or an empty query, never raises.
        """
        if not query_entities or self.n_docs == 0:
            return 0.0
        doc_ents = self.doc_entities.get(doc_id)
        if not doc_ents:
            return 0.0
        doc_ent_set = set(doc_ents)
        n = self.n_docs
        score = 0.0
        for e in query_entities:
            if e not in doc_ent_set:
                continue
            bucket = self.postings.get(e)
            if not bucket:
                continue
            df = len(bucket)
            if n > 0 and df / n > 0.1:
                continue  # same "not discriminative" skip as search()
            score += math.log(1.0 + n / df)
        return score

    # ------------------------------------------------------------- persist
    def to_dict(self) -> dict:
        return {
            "postings": {e: sorted(ids) for e, ids in self.postings.items()},
            "doc_entities": self.doc_entities,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntityIndex":
        idx = cls()
        idx.postings = {e: set(ids) for e, ids in d.get("postings", {}).items()}
        idx.doc_entities = {k: list(v) for k, v in d.get("doc_entities", {}).items()}
        return idx

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EntityIndex":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
