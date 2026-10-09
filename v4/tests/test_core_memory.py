"""MemoryV4 façade: writes, dedup, persistence, session_brief, consolidate."""

import json
import time

import numpy as np

from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4, index_text
from memory_os_v4.store import MemoryItem, Store, text_hash
from memory_os_v4.text import tokenize


def _mem(tmp_path, **kwargs) -> MemoryV4:
    cfg = Config(embed_model="hash", data_dir=str(tmp_path), **kwargs)
    return MemoryV4(config=cfg)


def test_add_episode_basic_fields(tmp_path):
    m = _mem(tmp_path)
    item = m.add_episode("Hello there.", role="user", project="p1", importance=0.7)
    assert item.kind == "episode"
    assert item.role == "user"
    assert item.project == "p1"
    assert item.importance == 0.7
    assert item.hash


def test_add_episodes_bulk_equals_repeated_add_episode(tmp_path):
    m1 = _mem(tmp_path / "a")
    m2 = _mem(tmp_path / "b")
    episodes = [
        {"text": "First message.", "role": "user", "project": "p1"},
        {"text": "Second message.", "role": "assistant", "project": "p1"},
        {"text": "Third message.", "role": "user", "project": "p1"},
    ]
    bulk_items = m1.add_episodes_bulk(episodes)
    single_items = [m2.add_episode(**e) for e in episodes]

    assert [i.text for i in bulk_items] == [i.text for i in single_items]
    assert [i.hash for i in bulk_items] == [i.hash for i in single_items]

    hits1 = m1.search("second message", top_k=3)
    hits2 = m2.search("second message", top_k=3)
    # top hit is unambiguous; lower-ranked ties may order differently since ids
    # (and thus set-iteration order) differ between the two independent stores
    assert hits1[0].item.text == hits2[0].item.text == "Second message."
    assert {h.item.text for h in hits1} == {h.item.text for h in hits2}


def test_add_facts_exact_hash_dedup(tmp_path):
    m = _mem(tmp_path)
    r1 = m.add_facts([{"text": "Romain works on GRPO-MM."}], project="p1")
    r2 = m.add_facts([{"text": "Romain works on GRPO-MM."}], project="p1")
    assert len(r1["added"]) == 1
    assert r2["added"] == []
    assert r2["deduped"] == r1["added"]


def test_add_facts_near_duplicate_dedup(tmp_path):
    m = _mem(tmp_path)
    r1 = m.add_facts([{"text": "The project deadline for Falcon is next Friday."}], project="p1")
    r2 = m.add_facts([{"text": "The project deadline for Falcon is next Friday!"}], project="p1")
    assert r2["added"] == []
    assert r2["deduped"] == r1["added"]


def test_add_facts_supersession_sets_valid_until_and_invalidated_by(tmp_path):
    m = _mem(tmp_path, supersede_cos=0.5)
    t0 = time.time() - 200000
    t1 = time.time() - 100000
    r1 = m.add_facts([{"text": "Romain lives in Paris."}], project="p1", event_time=t0)
    r2 = m.add_facts([{"text": "Romain lives in Lyon."}], project="p1", event_time=t1)
    assert r2["superseded"] == [(r1["added"][0], r2["added"][0])]
    old = m.store.get(r1["added"][0])
    assert old.valid_until == t1
    assert old.invalidated_by == r2["added"][0]
    assert old.deleted is False  # never deleted


# ---------------------------------------------------------------------------
# A2: add_facts supersession edge cases (adversarial review, 2026-09-12)
# ---------------------------------------------------------------------------
def test_add_facts_supersession_never_crosses_projects(tmp_path):
    m = _mem(tmp_path, supersede_cos=0.5)
    T = time.time() - 200000
    m.add_facts([{"text": "Romain works at AlphaCorp as a quant."}], project="projA", event_time=T)
    r2 = m.add_facts(
        [{"text": "Romain works at BetaCorp as a trader."}], project="projB", event_time=T + 86400
    )
    assert r2["superseded"] == []
    for it in m.store.all():
        assert it.valid_until is None
        assert it.invalidated_by is None


def test_add_facts_supersession_equal_event_time_does_not_supersede(tmp_path):
    """An equal event_time used to still supersede (`<` was not `<=`),
    setting valid_until == valid_from on the old fact -- a zero-length
    validity window, never retrievable with any at_time."""
    m = _mem(tmp_path, supersede_cos=0.5)
    T = time.time() - 200000
    r1 = m.add_facts([{"text": "Romain lives in Paris now."}], event_time=T)
    r2 = m.add_facts([{"text": "Romain lives in Lyon now."}], event_time=T)
    assert r2["superseded"] == []
    old = m.store.get(r1["added"][0])
    assert old.valid_until is None
    assert old.invalidated_by is None


def test_add_facts_supersession_chain_then_reassert_original(tmp_path):
    """Chain A -> B -> C, then re-assert A with a later date: A must not be
    deduped to its own dead (invalidated) prior record, and by the normal
    supersession rule it can then supersede the currently-valid C."""
    m = _mem(tmp_path, supersede_cos=0.5)
    T = time.time() - 400000
    ra = m.add_facts([{"text": "Romain lives in Paris now."}], event_time=T)
    rb = m.add_facts([{"text": "Romain lives in Lyon now."}], event_time=T + 86400)
    rc = m.add_facts([{"text": "Romain lives in Nice now."}], event_time=T + 2 * 86400)
    assert rb["superseded"] == [(ra["added"][0], rb["added"][0])]
    assert rc["superseded"] == [(rb["added"][0], rc["added"][0])]

    r_reassert = m.add_facts([{"text": "Romain lives in Paris now."}], event_time=T + 3 * 86400)
    assert r_reassert["added"], "re-asserting invalidated text must create a NEW valid fact, not dedupe"
    assert r_reassert["deduped"] == []
    new_a_id = r_reassert["added"][0]
    assert new_a_id != ra["added"][0]
    # by the normal rule: shares entities + cosine >= threshold + later event_time
    # -> supersedes the currently-valid contradicting fact (C, "Nice")
    assert r_reassert["superseded"] == [(rc["added"][0], new_a_id)]

    new_a = m.store.get(new_a_id)
    assert new_a.valid_until is None  # currently valid
    old_a = m.store.get(ra["added"][0])
    assert old_a.valid_until is not None  # still invalidated, untouched by the resurrection
    c = m.store.get(rc["added"][0])
    assert c.invalidated_by == new_a_id


def test_add_facts_dedup_ignores_invalidated_candidates(tmp_path):
    """Exact-hash dedup must ignore an invalidated candidate: re-adding the
    EXACT same text as an already-superseded fact (with a later date) must
    create a fresh valid fact, not dedupe to the dead record."""
    m = _mem(tmp_path, supersede_cos=0.5)
    T = time.time() - 200000
    r1 = m.add_facts([{"text": "Romain lives in Paris."}], event_time=T)
    m.add_facts([{"text": "Romain lives in Lyon."}], event_time=T + 86400)
    old_paris = m.store.get(r1["added"][0])
    assert old_paris.invalidated_by is not None

    r3 = m.add_facts([{"text": "Romain lives in Paris."}], event_time=T + 2 * 86400)
    assert r3["deduped"] == []
    assert len(r3["added"]) == 1
    assert r3["added"][0] != r1["added"][0]


# ---------------------------------------------------------------------------
# A3: add_facts dedup probe must use the SAME header contextual_header(item)
# uses for indexing (adversarial review, 2026-09-12)
# ---------------------------------------------------------------------------
def test_add_facts_probe_vector_matches_indexed_vector(tmp_path):
    """The dedup probe embedded inside add_facts must become bit-identical
    to the vector actually stored in the index (same header, same text,
    embedded exactly once) -- not a second, differently-worded encode()."""
    m = _mem(tmp_path)
    r = m.add_facts([{"text": "Romain lives in Paris."}], project="p1", event_time=1700000000.0)
    item_id = r["added"][0]
    item = m.store.get(item_id)
    from memory_os_v4.text import contextual_header

    expected = m.embedder.encode([f"{contextual_header(item)}\n{item.text}"])[0]
    indexed = m.vector_index.get_one(item_id)
    assert np.array_equal(indexed, expected)


def test_add_facts_dedup_agrees_with_consolidate_dedup_on_same_pair(tmp_path):
    """add_facts's write-time near-duplicate check and consolidate's sweep
    must reach the SAME verdict on the same pair of texts, since both now
    compare the exact same already-indexed, same-header vectors."""
    m = _mem(tmp_path)
    m.add_facts([{"text": "The project deadline for Falcon is next Friday."}], project="p1")
    r2 = m.add_facts([{"text": "The project deadline for Falcon is next Friday, apparently."}], project="p1")

    if r2["deduped"]:
        # add_facts caught it at write time -> consolidate must never have
        # had the chance to see 2 distinct items in the first place.
        assert len(m.store.all(include_deleted=False)) == 1
    else:
        # add_facts did NOT flag it as a near-duplicate -> consolidate's own
        # sweep, using the identical cosine/header formula, must agree and
        # not flag it either (this is the exact discrepancy A3 fixes: a
        # harsher write-time probe than the sweep used to disagree here).
        result = m.consolidate(dry_run=True)
        assert result["deduped"] == 0


# ---------------------------------------------------------------------------
# A4: duplicate item_id WITHIN one batch (adversarial review, 2026-09-12)
# ---------------------------------------------------------------------------
def test_add_episodes_bulk_duplicate_id_in_one_batch_first_wins(tmp_path):
    m = _mem(tmp_path)
    items = m.add_episodes_bulk(
        [
            {"text": "first version of the text about faiss", "item_id": "X"},
            {"text": "second completely different text about tacos", "item_id": "X"},
        ]
    )
    assert len(items) == 1  # the duplicate is skipped, not silently double-indexed
    assert items[0].text == "first version of the text about faiss"  # first wins
    assert len(m.store.items) == 1
    assert m.store.items["X"].text == "first version of the text about faiss"
    assert m.vector_index.ntotal == 1
    assert m.bm25.n_docs == 1

    ids, vecs = m.vector_index.export()
    assert ids == ["X"]
    assert vecs.shape[0] == 1


def test_vector_index_add_deduplicates_ids_within_one_call():
    from memory_os_v4.vector_index import VectorIndex

    idx = VectorIndex(dim=4)
    idx.add(
        ["a", "a", "b"],
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype="float32"),
    )
    assert idx.ntotal == 2
    assert sorted(idx.ids()) == ["a", "b"]
    # first occurrence wins
    assert np.array_equal(idx.get_one("a"), np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"))


# ---------------------------------------------------------------------------
# A5: MemoryV4.forget() -- additive façade method, immediate index removal
# ---------------------------------------------------------------------------
def test_forget_removes_item_from_all_indexes_immediately(tmp_path):
    m = _mem(tmp_path)
    m.add_episode("faiss vector index choice", item_id="A")
    m.add_episode("tacos recipe with cilantro", item_id="B")

    assert m.forget("A", reason="test cleanup") is True

    assert m.store.get("A").deleted is True
    assert m.vector_index.get_one("A") is None
    assert "A" not in m.bm25.doc_len
    assert "A" not in m.entities.doc_entities
    hits = m.search("faiss vector index", top_k=5)
    assert all(h.item.id != "A" for h in hits)
    status = m.status()
    assert status["vector_count"] == status["bm25_docs"] == 1


def test_forget_unknown_or_already_deleted_returns_false(tmp_path):
    m = _mem(tmp_path)
    m.add_episode("something", item_id="A")
    assert m.forget("does-not-exist") is False
    assert m.forget("A") is True
    assert m.forget("A") is False  # already deleted -> idempotent no-op


def test_search_at_time_returns_correct_fact_version(tmp_path):
    m = _mem(tmp_path, supersede_cos=0.5)
    t0 = time.time() - 200000
    t1 = time.time() - 100000
    m.add_facts([{"text": "Romain lives in Paris."}], project="p1", event_time=t0)
    m.add_facts([{"text": "Romain lives in Lyon."}], project="p1", event_time=t1)

    early_hits = m.search("Where does Romain live?", top_k=5, at_time=t0 + 1000)
    assert all("Paris" in h.item.text for h in early_hits)

    late_hits = m.search("Where does Romain live?", top_k=5, at_time=t1 + 1000)
    assert late_hits[0].item.text.find("Lyon") != -1


def test_add_document_chunks_and_is_searchable(tmp_path):
    m = _mem(tmp_path)
    text = " ".join(
        f"Sentence {i} about the Prometheus architecture and its many components." for i in range(40)
    )
    result = m.add_document("architecture.md", text, project="p1")
    assert result["chunks"] > 1
    hits = m.search("Prometheus architecture", top_k=5, kinds=["doc"])
    assert hits


def test_get_context_within_token_budget(tmp_path):
    m = _mem(tmp_path)
    for i in range(20):
        m.add_episode(f"Note number {i} about the XTX gamma dashboard rollout.", project="p1")
    ctx = m.get_context("XTX gamma dashboard", max_tokens=50, project="p1")
    assert len(ctx) <= 50 * 4 + 200  # small slack for the final line


def test_status_reports_counts(tmp_path):
    m = _mem(tmp_path)
    m.add_episode("hi", project="p1")
    m.add_facts([{"text": "Romain uses a Mac Studio M4 for local inference."}], project="p1")
    status = m.status()
    assert status["counts"]["episode"] == 1
    assert status["counts"]["fact"] == 1
    assert status["embed_model"] == "hash"
    assert status["data_dir"] == str(tmp_path)


def test_persistence_roundtrip(tmp_path):
    m1 = _mem(tmp_path)
    m1.add_episode("Persisted episode about mm_cross_v2.", project="p1")
    m1.add_facts([{"text": "Romain works with NQ and ES futures."}], project="p1")
    m1.flush()

    m2 = _mem(tmp_path)
    hits1 = m1.search("mm_cross_v2", top_k=5)
    hits2 = m2.search("mm_cross_v2", top_k=5)
    assert [h.item.id for h in hits1] == [h.item.id for h in hits2]
    assert m2.status()["counts"] == m1.status()["counts"]

    # items.jsonl actually contains the records
    lines = (tmp_path / "items.jsonl").read_text().strip().splitlines()
    assert len(lines) >= 2
    json.loads(lines[0])  # valid json


def test_session_brief_structure_and_budget(tmp_path):
    m = _mem(tmp_path)
    for i in range(10):
        m.add_episode(f"Episode {i} about project Falcon.", project="p1")
    m.add_facts([{"text": "Romain prefers Sonnet for coding tasks."}], project="p1", event_time=time.time())
    m.add_facts([{"text": "TODO: finish the v4 migration script."}], project="p1")
    m.blocks.set("user_profile", "ML/HFT engineer.")

    brief = m.session_brief(focus_query="Falcon", max_tokens=500, project="p1")
    for key in ("blocks", "recent_episodes", "key_facts", "procedures", "focus", "pending", "stats", "text"):
        assert key in brief
    assert len(brief["recent_episodes"]) <= 8
    assert len(brief["key_facts"]) <= 12
    assert len(brief["procedures"]) <= 6
    assert len(brief["focus"]) <= 8
    assert len(brief["pending"]) <= 6
    assert brief["blocks"]["user_profile"] == "ML/HFT engineer."
    assert len(brief["text"]) <= 500 * 4 + 500
    assert any("finish the v4 migration" in p["text"] for p in brief["pending"])


def test_consolidate_dry_run_does_not_mutate(tmp_path):
    m = _mem(tmp_path)
    m.add_episode("Old episode with zero access.", project="p1", event_time=time.time() - 200 * 86400)
    before = m.store.get(list(m.store.items.keys())[0]).importance
    result = m.consolidate(dry_run=True)
    after = m.store.get(list(m.store.items.keys())[0]).importance
    assert before == after
    assert result["decayed"] >= 1


def test_consolidate_decays_stale_episode_importance(tmp_path):
    m = _mem(tmp_path, episode_ttl_days=10)
    item = m.add_episode("Stale episode nobody accessed.", project="p1", event_time=time.time() - 100 * 86400)
    m.consolidate(dry_run=False)
    updated = m.store.get(item.id)
    assert updated.importance == 0.1


def test_consolidate_proposes_summary_clusters(tmp_path):
    m = _mem(tmp_path)
    for i in range(6):
        m.add_episode(
            "We discussed the GRPO-MM cross-domain reward shaping approach.", project="p1", item_id=f"ep{i}"
        )
    result = m.consolidate(dry_run=True)
    assert any(len(p["member_ids"]) >= 4 for p in result["summary_proposals"])


def test_consolidate_supersession_sweep_catches_unlinked_facts(tmp_path):
    m = _mem(tmp_path, supersede_cos=0.5)
    t0 = time.time() - 200000
    t1 = time.time() - 100000
    # add_facts already links these at write time; verify consolidate is idempotent
    m.add_facts([{"text": "Romain lives in Paris."}], project="p1", event_time=t0)
    m.add_facts([{"text": "Romain lives in Lyon."}], project="p1", event_time=t1)
    result = m.consolidate(dry_run=True)
    assert result["superseded"] == 0  # already superseded at write time, sweep finds nothing new


def test_flush_then_new_instance_search_matches(tmp_path):
    m1 = _mem(tmp_path)
    m1.add_episode("Reload check about Kraken 5-minute bars.", project="p1")
    m1.flush()
    m2 = _mem(tmp_path)
    hits = m2.search("Kraken bars", top_k=3)
    assert hits and "Kraken" in hits[0].item.text


# ---------------------------------------------------------------------------
# A1: index/store desync reconciliation on load (_reconcile_indexes)
# ---------------------------------------------------------------------------
def test_reload_reconciles_unflushed_appended_item(tmp_path):
    """An item appended to items.jsonl (via add_episode, which always writes
    through immediately) but never `flush()`-ed still has its embedding
    cache/bm25.json/entities.json left at the OLD state on disk -- a reload
    must notice the live item is unindexed and index it, not silently drop
    it from every retrieval signal."""
    m = _mem(tmp_path)
    m.add_episode("Alpha document about vector indexes.", item_id="A")
    m.flush()
    m.add_episode("Bravo document about gamma walls.", item_id="B")  # appended, never flushed

    reloaded = _mem(tmp_path)
    assert sorted(reloaded.store.items) == ["A", "B"]
    status = reloaded.status()
    assert status["vector_count"] == status["total"] == 2
    assert status["bm25_docs"] == 2
    hits = reloaded.search("gamma walls", top_k=5)
    assert hits and hits[0].item.id == "B"


def test_reload_drops_orphan_index_entries_from_truncated_log(tmp_path):
    """A crash mid-write can truncate the last line of items.jsonl, which
    Store.load() already skips gracefully with a warning -- but the OLD
    embedding cache/bm25/entities on disk still reference that now-gone id.
    A reload must drop those orphan entries, not just tolerate them."""
    m = _mem(tmp_path)
    m.add_episode("first item", item_id="A")
    m.add_episode("second item", item_id="B")
    m.flush()

    items_path = tmp_path / "items.jsonl"
    raw = items_path.read_text()
    items_path.write_text(raw[: -len(raw.splitlines()[-1]) - 1])  # drop B's whole line

    reloaded = _mem(tmp_path)
    assert sorted(reloaded.store.items) == ["A"]
    status = reloaded.status()
    assert status["vector_count"] == status["total"] == 1
    assert status["bm25_docs"] == 1
    assert reloaded.vector_index.get_one("B") is None  # orphan vector dropped
    assert "B" not in reloaded.bm25.doc_len


def test_reload_forces_reembed_instead_of_crashing_on_id_vector_count_mismatch(tmp_path):
    """A hand-corrupted (or partially-written) ids.json shorter than
    embeddings.npy used to reach faiss's add_with_ids as a bare
    AssertionError, bricking the store on every future load."""
    m = _mem(tmp_path)
    m.add_episode("x1", item_id="A")
    m.add_episode("x2", item_id="B")
    m.flush()

    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(["A"]))  # ids shorter than the 2-row vecs cache

    reloaded = _mem(tmp_path)  # must not raise
    status = reloaded.status()
    assert status["vector_count"] == status["total"] == 2
    hits = reloaded.search("x2", top_k=5)
    assert any(h.item.id == "B" for h in hits)


def test_reload_reembeds_when_stored_dim_differs_from_current_embedder(tmp_path):
    """meta.json's recorded `dim` disagreeing with the current embedder's
    dim (e.g. hash_dim changed) must trigger a full re-embed, not a crash or
    a silently-wrong-shape vector index."""
    m = _mem(tmp_path, hash_dim=64)
    m.add_episode("hello world", item_id="A")
    m.flush()

    reloaded = _mem(tmp_path, hash_dim=32)  # same "hash" model name, different dim
    assert reloaded.embedder.dim == 32
    assert reloaded.vector_index.get_one("A").shape == (32,)
    assert reloaded.status()["vector_count"] == 1


def test_reload_reconciles_stale_soft_deleted_index_entries(tmp_path):
    """A soft-deleted item from BEFORE `forget()` existed (or one deleted by
    directly flipping `.deleted` on the stored item) must have its index
    entries dropped on the next reload, even though the item itself stays
    in items.jsonl as a tombstone."""
    m = _mem(tmp_path)
    m.add_episode("faiss vector index choice", item_id="A")
    m.add_episode("tacos recipe with cilantro", item_id="B")
    m.store.items["A"].deleted = True
    m.store.append(m.store.items["A"])  # tombstone written, but indexes untouched (old-style soft delete)
    m.flush()

    reloaded = _mem(tmp_path)
    assert reloaded.vector_index.get_one("A") is None
    assert "A" not in reloaded.bm25.doc_len
    assert "A" not in reloaded.entities.doc_entities
    status = reloaded.status()
    assert status["vector_count"] == status["bm25_docs"] == 1  # only B remains indexed


def test_multi_vector_chunk_map_persists_across_reload(tmp_path):
    """E1 multi_vector: chunk_id -> item_id survives flush()/reload, and dense
    search on the reloaded instance still resolves chunk hits to the item."""
    m1 = _mem(tmp_path, multi_vector=True, chunk_tokens=8, chunk_overlap=2)
    long_text = " ".join(f"sentence number {i} about the gamma rollout." for i in range(20))
    item = m1.add_episode(long_text, project="p1", item_id="long1")
    assert m1.store.chunk_owner  # at least one chunk vector was created
    m1.flush()

    m2 = _mem(tmp_path, multi_vector=True, chunk_tokens=8, chunk_overlap=2)
    assert m2.store.chunk_owner == m1.store.chunk_owner
    hits = m2.search("gamma rollout", top_k=3)
    assert any(h.item.id == item.id for h in hits)


# ---------------------------------------------------------------------------
# Production port A: `MemoryItem.keywords` + `memory.index_text()` (2026-09-14)
# ---------------------------------------------------------------------------
def test_add_episode_keywords_indexed_but_absent_from_text(tmp_path):
    m = _mem(tmp_path)
    item = m.add_episode("A quiet afternoon.", project="p1", keywords=["hiking", "yosemite"])
    assert item.keywords == ["hiking", "yosemite"]
    assert "hiking" not in item.text and "yosemite" not in item.text

    hits = m.search("yosemite hiking", top_k=5, project="p1")
    assert hits and hits[0].item.id == item.id
    # the stored/displayed text is untouched by the keyword augmentation
    assert hits[0].item.text == "A quiet afternoon."


def test_add_episodes_bulk_accepts_keywords_per_episode(tmp_path):
    m = _mem(tmp_path)
    items = m.add_episodes_bulk(
        [
            {"text": "First message.", "project": "p1", "keywords": ["kw1", "kw2"]},
            {"text": "Second message.", "project": "p1"},
        ]
    )
    assert items[0].keywords == ["kw1", "kw2"]
    assert items[1].keywords == []  # default: unaffected caller, backward compatible


def test_keywords_persist_across_reload(tmp_path):
    m1 = _mem(tmp_path)
    item = m1.add_episode("Trip planning notes.", project="p1", keywords=["travel", "budget"])
    m1.flush()

    m2 = _mem(tmp_path)
    reloaded = m2.store.get(item.id)
    assert reloaded.keywords == ["travel", "budget"]


def test_store_loads_old_record_without_keywords_field(tmp_path):
    """Backward compatibility: an `items.jsonl` line written before the
    `keywords` field existed must still load, defaulting to an empty list."""
    item = MemoryItem(id="x1", kind="episode", text="Old record.", hash=text_hash("old record."))
    old_record = item.to_dict()
    del old_record["keywords"]
    (tmp_path / "items.jsonl").write_text(json.dumps(old_record) + "\n", encoding="utf-8")

    store = Store(tmp_path)
    loaded = store.get("x1")
    assert loaded is not None
    assert loaded.keywords == []


def test_index_keywords_config_flag_off_disables_keyword_indexing(tmp_path):
    m = _mem(tmp_path, index_keywords=False)
    m.add_episode("Just a note.", project="p1", keywords=["zzzuniquekeywordzzz"])
    # the keyword never reached BM25 tokenization when the flag is off
    assert m.bm25.search(tokenize("zzzuniquekeywordzzz"), 10) == []


def test_add_facts_keywords_indexed_and_dedup_probe_agrees_with_stored_vector(tmp_path):
    """D: `index_text()` is the ONE helper -- the `add_facts` dedup probe
    must embed the exact same string `_index_items` stores the item's own
    vector under, or dedup silently drifts from what's actually indexed."""
    m = _mem(tmp_path)
    result = m.add_facts(
        [{"text": "The launch date is set.", "keywords": ["falcon", "deadline"]}], project="p1"
    )
    item = m.store.get(result["added"][0])
    assert item.keywords == ["falcon", "deadline"]
    assert "falcon" not in item.text

    probe_vec = m.embedder.encode([index_text(item, m.config)])[0]
    stored_vec = m.vector_index.get_one(item.id)
    assert np.allclose(probe_vec, stored_vec)
