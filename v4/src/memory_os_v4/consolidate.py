"""Sleep-time consolidation: dedup sweep, deterministic supersession sweep,
Ebbinghaus-style importance decay/reinforcement, and summary-cluster proposals.

Nothing is ever hard-deleted: dedup marks the newer near-duplicate with
``deleted=True`` (soft, reversible in the log) and episode TTL decay only
lowers importance.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .config import Config
from .retrieval import cosine_sim
from .store import Store
from .temporal import recency_weight
from .text import tokenize
from .vector_index import VectorIndex


def _jaccard(text_a: str, text_b: str) -> float:
    a, b = set(tokenize(text_a)), set(tokenize(text_b))
    union = a | b
    return len(a & b) / max(1, len(union))


def dedup_sweep(
    store: Store, vector_index: VectorIndex, config: Config, dry_run: bool = False
) -> list[tuple[str, str]]:
    """Find near-duplicate fact-like items (cosine >= dedup_cos and token
    Jaccard >= 0.6); soft-delete the newer of each pair. Returns (kept, removed)."""
    candidates = [it for it in store.all(include_deleted=False) if it.kind in {"fact", "belief", "summary"}]
    candidates.sort(key=lambda it: it.created_at)
    removed: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(candidates):
        if a.id in removed:
            continue
        va = vector_index.get_one(a.id)
        for b in candidates[i + 1 :]:
            if b.id in removed or b.kind != a.kind:
                continue
            if cosine_sim(va, vector_index.get_one(b.id)) < config.dedup_cos:
                continue
            if _jaccard(a.text, b.text) < 0.6:
                continue
            pairs.append((a.id, b.id))
            removed.add(b.id)
            if not dry_run:
                b.deleted = True
                store.append(b)
    return pairs


def supersede_sweep(
    store: Store, vector_index: VectorIndex, config: Config, dry_run: bool = False
) -> list[tuple[str, str]]:
    """Deterministic supersession: for currently-valid facts sharing >=1 entity,
    cosine >= supersede_cos, not a near-duplicate, the later ``event_time`` fact
    supersedes the earlier one (``valid_until``/``invalidated_by`` set; never
    deleted)."""
    facts = [it for it in store.all(include_deleted=False) if it.kind == "fact" and it.valid_until is None]
    facts.sort(key=lambda it: it.event_time)
    pairs: list[tuple[str, str]] = []
    for i, old in enumerate(facts):
        if old.valid_until is not None:
            continue
        entities_old = set(old.entities)
        vo = vector_index.get_one(old.id)
        for new in facts[i + 1 :]:
            if new.event_time <= old.event_time:
                continue
            if not (entities_old & set(new.entities)):
                continue
            cos = cosine_sim(vo, vector_index.get_one(new.id))
            if cos < config.supersede_cos:
                continue
            if cos >= config.dedup_cos and _jaccard(old.text, new.text) >= 0.6:
                continue  # near-duplicate, not a genuine update
            pairs.append((old.id, new.id))
            if not dry_run:
                old.valid_until = new.event_time
                old.invalidated_by = new.id
                store.append(old)
            break
    return pairs


def decay(store: Store, config: Config, now: float, dry_run: bool = False) -> int:
    """Ebbinghaus-style importance decay with access reinforcement; episodes past
    ``episode_ttl_days`` with zero accesses are marked low-importance (never
    deleted). Returns the number of items updated."""
    updated = 0
    for item in store.all(include_deleted=False):
        if item.kind == "episode":
            age_days = (now - item.event_time) / 86400.0
            if age_days > config.episode_ttl_days and item.access_count == 0 and item.importance > 0.1:
                if not dry_run:
                    item.importance = 0.1
                    store.append(item)
                updated += 1
            continue

        half_life = config.half_life_days(item.kind)
        age_days = max(0.0, (now - item.event_time) / 86400.0)
        rec = recency_weight(age_days, half_life)
        new_importance = item.importance * (0.5 + 0.5 * rec) + 0.05 * math.log1p(item.access_count)
        new_importance = max(0.05, min(1.0, new_importance))
        if abs(new_importance - item.importance) > 1e-9:
            if not dry_run:
                item.importance = new_importance
                store.append(item)
            updated += 1
    return updated


def _suggest_title(cluster: list, project: str) -> str:
    counter: Counter = Counter()
    for item in cluster:
        counter.update(item.entities)
    top = [e for e, _ in counter.most_common(3)]
    if top:
        return f"{project}: " + ", ".join(top)
    return f"{project}: {len(cluster)} related episodes"


def summary_proposals(
    store: Store, vector_index: VectorIndex, min_cluster: int = 4, cos_threshold: float = 0.75
) -> list[dict]:
    """Greedy cosine clustering of episodes per project (>=min_cluster members)
    so a client LLM can write a summary via ``add_facts(kind="summary")``."""
    by_project: dict[str, list] = defaultdict(list)
    for item in store.all(include_deleted=False):
        if item.kind == "episode":
            by_project[item.project].append(item)

    proposals: list[dict] = []
    for project, episodes in by_project.items():
        unclustered = list(episodes)
        while unclustered:
            seed = unclustered.pop(0)
            vseed = vector_index.get_one(seed.id)
            cluster = [seed]
            remaining = []
            for ep in unclustered:
                vep = vector_index.get_one(ep.id)
                if vseed is not None and vep is not None and cosine_sim(vseed, vep) >= cos_threshold:
                    cluster.append(ep)
                else:
                    remaining.append(ep)
            unclustered = remaining
            if len(cluster) >= min_cluster:
                proposals.append(
                    {
                        "project": project,
                        "member_ids": [c.id for c in cluster],
                        "suggested_title": _suggest_title(cluster, project),
                    }
                )
    return proposals
