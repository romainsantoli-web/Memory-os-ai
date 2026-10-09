"""Facet channel: closed-predicate-vocabulary faceted index for multi-hop
retrieval (production port of the LoCoMo-validated bench mechanism -- see
``bench/retrievers.py``'s ``build_facet_index``/``rank_facet_pool`` and
``bench/data/enrich/PREDICATES.md``). Measured on LoCoMo (2026-09-19): fusing
the facet-ranked pool as one more RRF party raised R@10 0.721 -> 0.757; see
``config.py``'s ``facet_channel``/``facet_weight`` for why it ships OFF by
default until a second benchmark confirms.

A facet is a ``(subject, predicate, object)`` triple pulled from a memory
item's own text by the CLIENT LLM -- the server has no LLM of its own, same
division of labour as ``keywords``/``queries``/``hypotheses`` (see
``docs/ARCHITECTURE.md``). ``subject`` + ``predicate`` are indexed for exact
``(subject, predicate)`` lookup at query time; ``object`` is carried along
for display/debugging only and never indexed or fused into ranking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# Closed vocabulary (V5.0), see bench/data/enrich/PREDICATES.md. Grouped by
# family for documentation only -- indexing/matching only ever cares about
# `PREDICATES` (the flat set) and never about which family a predicate
# belongs to.
FACET_FAMILIES: dict[str, tuple[str, ...]] = {
    "identity": (
        "name_alias", "age_birthday", "origin_hometown", "residence",
        "nationality_language", "gender_identity",
    ),
    "people": (
        "family_member", "child", "partner", "friend", "pet", "person_met", "colleague_mentor",
    ),
    "work/school": (
        "job_career", "employer_business", "education_course", "project_work",
        "skill_ability", "achievement_award", "goal_plan",
    ),
    "activities": (
        "hobby_interest", "sport_exercise", "creative_work", "volunteering_cause",
        "event_attended", "trip_location_visited", "outdoor_activity", "game_played",
    ),
    "possessions/media": (
        "possession_purchase", "vehicle", "book_read", "movie_show_music", "food_drink_recipe",
    ),
    "state": (
        "health_condition", "feeling_emotion", "belief_value_opinion", "preference_like",
        "dislike_fear", "habit_routine", "challenge_problem", "life_event_milestone",
        "gift_given_received", "support_advice",
    ),
}

# The 43 indexable predicates (PREDICATES.md's actual count as of V5.0 --
# the task brief that commissioned this port said "45"; the file itself has
# 43, taken here as ground truth). "other" (PREDICATES.md's fallback) is
# deliberately excluded here -- it is a valid INPUT (see `normalize_facet`)
# but is never indexed (same rule `bench.retrievers.build_facet_index`
# already validated), so it must never appear as a member of this set.
PREDICATES: frozenset[str] = frozenset(p for preds in FACET_FAMILIES.values() for p in preds)

_OTHER = "other"
_MAX_SUBJECT_LEN = 80
_MAX_OBJECT_LEN = 200
_MAX_FACETS_PER_ITEM = 8


def normalize_facet(raw: Any) -> dict | None:
    """Validate/normalize one client-supplied facet.

    Accepts ``{"subject": str, "predicate": str, "object": str?}``
    (``object`` optional, used by fact-side facets only -- a question facet
    has no object, see ``bench/enrich.py``'s ``_clean_qfacet_pairs``).
    Never raises: returns ``None`` for a non-dict, or a missing/blank
    ``subject``. ``predicate`` is stripped + lowercased; anything outside
    `PREDICATES` (including missing/blank) normalizes to ``"other"`` --
    accepted as input, never indexed (see `FacetIndex.add`). Lengths are
    capped, never rejected: ``subject`` <= 80 chars, ``object`` <= 200 chars.

    Returns ``{"subject": <display, stripped/capped>, "subject_key": <casefold
    form used for index matching>, "predicate": str, "object": str}``.
    """
    if not isinstance(raw, dict):
        return None
    subject = raw.get("subject")
    if not isinstance(subject, str):
        return None
    subject = subject.strip()[:_MAX_SUBJECT_LEN]
    if not subject:
        return None
    predicate = raw.get("predicate")
    predicate = predicate.strip().lower() if isinstance(predicate, str) else ""
    if predicate not in PREDICATES:
        predicate = _OTHER
    obj = raw.get("object")
    obj = obj.strip()[:_MAX_OBJECT_LEN] if isinstance(obj, str) else ""
    return {
        "subject": subject,
        "subject_key": subject.casefold(),
        "predicate": predicate,
        "object": obj,
    }


def normalize_facets(raw_list: Any) -> list[dict]:
    """Normalize a caller-supplied ``facets`` list (e.g. one message's or
    fact's raw input): drop invalid entries, keep at most
    `_MAX_FACETS_PER_ITEM` (silently clipped -- never an error, matching this
    module's "never crash the caller" convention). A non-list `raw_list`
    (including ``None``) yields ``[]``.
    """
    if not isinstance(raw_list, list):
        return []
    out: list[dict] = []
    for raw in raw_list:
        nf = normalize_facet(raw)
        if nf is None:
            continue
        out.append(nf)
        if len(out) >= _MAX_FACETS_PER_ITEM:
            break
    return out


def _facet_key(qf: Any) -> tuple[str, str] | None:
    """Coerce one query-facet entry -- a ``(subject_key, predicate)`` tuple,
    a raw ``{"subject", "predicate"}`` dict, or an already-`normalize_facet`d
    dict -- into a matching key. ``None`` for anything unusable."""
    if isinstance(qf, dict):
        subject = qf.get("subject_key") or qf.get("subject")
        predicate = qf.get("predicate")
    elif isinstance(qf, (tuple, list)) and len(qf) == 2:
        subject, predicate = qf
    else:
        return None
    if not isinstance(subject, str) or not subject.strip():
        return None
    if not isinstance(predicate, str) or not predicate.strip():
        return None
    return subject.strip().casefold(), predicate.strip().lower()


class FacetIndex:
    """In-memory ``(subject_key, predicate) -> {item_id, ...}`` index.

    Derived state only (production port of `bench.retrievers.
    build_facet_index`): never persisted to its own file. `rebuild()`
    replays every item's already-durable `MemoryItem.facets` (loaded from
    `items.jsonl` by `Store`) -- see docs/ARCHITECTURE.md's on-disk layout,
    which this must not change. Predicate ``"other"`` is never indexed, same
    rule as `bench.retrievers.build_facet_index`.
    """

    def __init__(self) -> None:
        self._index: dict[tuple[str, str], set[str]] = defaultdict(set)
        # item_id -> the (subject_key, predicate) keys it's filed under. Also
        # doubles as the "have we already processed this id" presence set
        # (mirrors bm25.doc_len / entities.doc_entities), so an item with
        # zero indexable facets is still tracked as "present" and never
        # looks "missing" to `memory.MemoryV4._reconcile_indexes`.
        self._by_item: dict[str, set[tuple[str, str]]] = {}

    def add(self, item: Any) -> None:
        """Index `item.id`'s own `item.facets` (already-normalized dicts,
        see `normalize_facet`). Removes any previous entry for this id
        first, so re-adding an updated item (a new `items.jsonl` record for
        the same id) never leaves stale keys behind."""
        self.remove(item.id)
        keys: set[tuple[str, str]] = set()
        for facet in getattr(item, "facets", None) or []:
            if not isinstance(facet, dict):
                continue
            predicate = facet.get("predicate")
            if not predicate or predicate == _OTHER:
                continue
            subject_key = facet.get("subject_key") or facet.get("subject")
            if not subject_key:
                continue
            key = (subject_key, predicate)
            self._index[key].add(item.id)
            keys.add(key)
        self._by_item[item.id] = keys  # always recorded, even if empty

    def remove(self, item_id: str) -> None:
        keys = self._by_item.pop(item_id, None)
        if not keys:
            return
        for key in keys:
            bucket = self._index.get(key)
            if bucket is None:
                continue
            bucket.discard(item_id)
            if not bucket:
                del self._index[key]

    def rebuild(self, items: Any) -> None:
        self._index = defaultdict(set)
        self._by_item = {}
        for item in items:
            self.add(item)

    def ids(self) -> set[str]:
        """Every item id currently tracked (indexed or not -- see `_by_item`
        docstring), for `memory.MemoryV4._reconcile_indexes`'s stale/missing
        diffing, mirroring `BM25Index.doc_len`/`EntityIndex.doc_entities`."""
        return set(self._by_item)

    def pool(self, query_facets: Any) -> set[str]:
        """Union of every indexed item id matching >=1 of `query_facets`.
        Accepts ``(subject_key, predicate)`` tuples or ``{"subject"/
        "subject_key", "predicate"}`` dicts (see `_facet_key`); unusable
        entries are silently skipped, never an error."""
        out: set[str] = set()
        for qf in query_facets or []:
            key = _facet_key(qf)
            if key is None:
                continue
            out |= self._index.get(key, set())
        return out


def rank_facet_pool(pool: set[str], hit_lists: list[list[str]]) -> list[str]:
    """Turn a facet `pool` into a RANKED list for RRF fusion: ordered by the
    best (lowest) 0-based rank a member achieves across `hit_lists` (the
    per-query hit lists already computed for this search, never
    recomputed). A member absent from every list goes last; ties are broken
    by id ascending. Identical semantics to
    `bench.retrievers.rank_facet_pool`.
    """
    best_rank: dict[str, int] = {}
    for ids in hit_lists:
        for rank, doc_id in enumerate(ids):
            if doc_id not in pool:
                continue
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
    return sorted(pool, key=lambda d: (best_rank.get(d, math.inf), d))
