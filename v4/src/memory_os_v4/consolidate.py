"""Sleep-time consolidation: dedup sweep, deterministic supersession sweep,
Ebbinghaus-style importance decay/reinforcement, and summary-cluster proposals.

Nothing is ever hard-deleted: dedup marks the newer normalized-text duplicate with
``deleted=True`` (soft, reversible in the log) and episode TTL decay only
lowers importance.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .config import Config
from .fact_updates import duplicate_compatible, equivalent_assertions, replacement_sweep
from .retrieval import cosine_sim
from .store import Store
from .temporal import recency_weight
from .vector_index import VectorIndex


def dedup_sweep(
    store: Store, vector_index: VectorIndex, config: Config, dry_run: bool = False, project: str | None = None
) -> list[tuple[str, str]]:
    """Dedup compatible active items within each project by conservative text equality.
    Cosine/Jaccard never establish equality, including for summaries. Historical
    validity windows are preserved. Returns (kept, removed)."""
    candidates = [it for it in store.all(include_deleted=False) if it.kind in {"fact", "belief", "summary"}
                  and it.valid_until is None and it.invalidated_by is None
                  and (project is None or it.project == project)]
    candidates.sort(key=lambda it: it.created_at)
    removed: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(candidates):
        if a.id in removed:
            continue
        for b in candidates[i + 1 :]:
            if b.id in removed or not duplicate_compatible(a, b):
                continue
            if not equivalent_assertions(a, b):
                continue
            pairs.append((a.id, b.id))
            removed.add(b.id)
            if not dry_run:
                b.deleted = True
                store.append(b)
    return pairs


def supersede_sweep(
    store: Store, vector_index: VectorIndex, config: Config, dry_run: bool = False, project: str | None = None
) -> list[tuple[str, str]]:
    """Only explicit replacements or caller-declared single-valued identities
    close validity windows. Cosine and shared names cannot prove contradiction."""
    return replacement_sweep(store, project=project, dry_run=dry_run)


def decay(store: Store, config: Config, now: float, dry_run: bool = False, project: str | None = None) -> int:
    """Ebbinghaus-style importance decay with access reinforcement; episodes past
    ``episode_ttl_days`` with zero accesses are marked low-importance (never
    deleted). Returns the number of items updated."""
    updated = 0
    for item in store.all(include_deleted=False):
        if project is not None and item.project != project:
            continue
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
    store: Store, vector_index: VectorIndex, min_cluster: int = 4, cos_threshold: float = 0.75, project: str | None = None
) -> list[dict]:
    """Greedy cosine clustering of episodes per project (>=min_cluster members)
    so a client LLM can write a summary via ``add_facts(kind="summary")``."""
    by_project: dict[str, list] = defaultdict(list)
    for item in store.all(include_deleted=False):
        if item.kind == "episode" and (project is None or item.project == project):
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
