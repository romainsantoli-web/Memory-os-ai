"""Conservative duplicate and replacement rules shared by writes and upkeep.

Embedding similarity is evidence for retrieval, never proof of contradiction.
A single-valued identity is an explicit assertion by the caller that the opaque
subject id and predicate describe one state; multi-valued identities coexist.
"""
from __future__ import annotations

from collections import defaultdict
import unicodedata

from .store import MemoryItem, Store


def start(item: MemoryItem) -> float:
    return item.valid_from if item.valid_from is not None else item.event_time


def duplicate_compatible(a: MemoryItem, b: MemoryItem) -> bool:
    if a.project != b.project or a.kind != b.kind or a.identity != b.identity:
        return False
    if set(a.entities) != set(b.entities):
        return False
    # Explicit updates must survive dedup, including a reversion to old text.
    if b.replaces:
        return False
    if a.invalidated_by is not None or b.invalidated_by is not None:
        return False
    if a.valid_until != b.valid_until:
        return False
    if a.valid_until is not None and start(a) != start(b):
        return False
    # Do not merge an earlier assertion into a not-yet-valid future state.
    if start(b) < start(a):
        return False
    return True


def equivalent_assertions(a: MemoryItem, b: MemoryItem) -> bool:
    """Conservative assertion equality; retrieval tokenization is lossy.

    Preserve stopwords/negation, word order, accents, signs, decimal points,
    single-character values and internal punctuation. Only whitespace, case
    and terminal sentence periods/exclamation marks are ignored.
    """
    def normalized(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(".!")
    return normalized(a.text) == normalized(b.text)


def validate_replacements(item: MemoryItem, store: Store) -> None:
    if (item.identity or item.replaces) and item.valid_until is not None and item.valid_until <= start(item):
        raise ValueError("valid_until must be later than valid_from")
    for old_id in item.replaces:
        old = store.get(old_id)
        if old is None or old.deleted or old.kind != "fact" or old.project != item.project:
            raise ValueError("replacement target must be a live fact in the same project")
        if old.id == item.id or start(item) <= start(old):
            raise ValueError("replacement must become valid strictly after its target")
        if old.valid_until is not None and old.valid_until <= start(item):
            raise ValueError("replacement target is no longer valid at the update time")


def replacement_sweep(store: Store, *, project: str | None = None, dry_run: bool = False,
                      focus: MemoryItem | None = None) -> list[tuple[str, str]]:
    facts = [it for it in store.all(False) if it.kind == "fact" and (project is None or it.project == project)]
    if focus is not None:
        targets = set(focus.replaces) | {focus.id}
        facts = [it for it in facts if it.id in targets or
                 (focus.identity is not None and focus.identity.get("cardinality") == "single"
                  and it.identity == focus.identity)]
    facts.sort(key=lambda it: (start(it), it.created_at, it.id))
    links: dict[str, MemoryItem] = {}
    groups: dict[tuple, list[MemoryItem]] = defaultdict(list)
    by_id = {it.id: it for it in facts}
    for new in facts:
        for old_id in new.replaces:
            old = by_id.get(old_id)
            if old is not None and old.project == new.project and start(new) > start(old):
                links.setdefault(old_id, new)
        if new.identity and new.identity.get("cardinality") == "single":
            key = (new.project, new.identity["subject_id"], new.identity["predicate"])
            groups[key].append(new)
    for group in groups.values():
        # Equal-start assertions coexist; each closes only at the next
        # strictly later state. Walk backwards to avoid an all-pairs scan.
        successor = None
        for i in range(len(group) - 1, -1, -1):
            old = group[i]
            if i + 1 < len(group) and start(group[i + 1]) > start(old):
                successor = group[i + 1]
            if successor is not None:
                explicit = links.get(old.id)
                if explicit is None or start(successor) < start(explicit):
                    links[old.id] = successor
    pairs = []
    for old_id, new in links.items():
        old = store.get(old_id)
        if old.valid_until is not None and old.valid_until <= start(new):
            continue
        pairs.append((old_id, new.id))
        if not dry_run:
            old.valid_until = start(new)
            old.invalidated_by = new.id
            store.append(old)
    return pairs
