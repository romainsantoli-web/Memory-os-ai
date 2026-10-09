"""Offline regressions for project isolation and conservative fact state changes."""
import json
import time

import pytest

from memory_os_v4 import server
from memory_os_v4.blocks import BlockStore
from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4
from memory_os_v4.temporal import is_valid_at


def mem(tmp_path):
    # Deliberately permissive cosine thresholds expose destructive similarity rules.
    return MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash", dedup_cos=0.0, supersede_cos=0.0))


def identity(subject="person:alice", predicate="residence", cardinality="single"):
    return {"subject_id": subject, "predicate": predicate, "cardinality": cardinality}


def fact(m, text, t, project="A", **metadata):
    r = m.add_facts([{"text": text, "entities": ["Alice"], **metadata}], project=project, event_time=t)
    return r, m.store.get(r["added"][0]) if r["added"] else None


def test_identical_and_near_facts_are_not_deduped_across_projects(tmp_path):
    m = mem(tmp_path)
    first, _ = fact(m, "Alice works at Acme.", 10, project="B")
    second, _ = fact(m, "Alice works at Acme.", 20)
    third, _ = fact(m, "Alice works at Acme!", 30, project="C")
    assert first["added"] and second["added"] and third["added"]
    assert second["deduped"] == third["deduped"] == []
    assert {h.item.project for h in m.search("Acme", project="A")} == {"A"}
    assert {h.item.project for h in m.search("Acme")} == {"A", "B", "C"}
    assert m.consolidate()["deduped"] == 0


def test_scoped_consolidation_brief_stats_and_forget_leave_b_unchanged(tmp_path):
    m = mem(tmp_path)
    stale = time.time() - 300 * 86400
    a = m.add_episode("A stale episode", project="A", event_time=stale)
    b = m.add_episode("B secret episode", project="B", event_time=stale)
    m.blocks.set("project_state", "B secret block", project="B")
    before = b.to_dict()
    stats = m.status(project="B")
    m.consolidate(project="A")
    assert m.store.get(a.id).importance == .1
    assert b.to_dict() == before
    assert m.status(project="B") == stats
    assert not m.forget(b.id, project="A")
    brief = m.session_brief(project="A")
    assert "B secret" not in brief["text"]
    assert brief["stats"]["total"] == 1
    assert m.forget(a.id, project="A")
    assert m.status(project="A")["vector_count"] == 0
    assert b.to_dict() == before


def test_scoped_consolidation_cannot_supersede_matching_identity_in_b(tmp_path):
    m = mem(tmp_path)
    _, b = fact(m, "Alice lives in Paris.", 10, project="B", identity=identity())
    _, a = fact(m, "Alice lives in Lyon.", 20, identity=identity())
    assert m.consolidate(project="A")["superseded"] == 0
    assert m.consolidate()["superseded"] == 0
    assert b.valid_until is a.valid_until is None


def test_blocks_share_only_explicit_globals_and_project_overrides(tmp_path):
    path = tmp_path / "blocks.jsonl"
    # No inferred global sharing, including profile/rules and arbitrary blocks.
    path.write_text(json.dumps({"name": "project_state", "text": "legacy secret", "updated_at": 1}) + "\n")
    b = BlockStore(path)
    assert b.get("project_state", project="default") == "legacy secret"
    assert b.get("project_state", project="A") == ""
    b.set("working_rules", "Shared rule", scope="global")
    b.set("project_state", "A state", project="A")
    b.set("project_state", "B state", project="B")
    assert b.all(project="A")["working_rules"] == "Shared rule"
    assert b.all(project="A")["project_state"] == "A state"
    b.set("working_rules", "A override", project="A")
    b.append("working_rules", "B own rule", project="B")
    assert b.get("working_rules", project="B") == "B own rule"
    reloaded = BlockStore(path)
    assert reloaded.all(project="A") == b.all(project="A")
    assert reloaded.all(project=None)["B/project_state"] == "B state"
    assert reloaded.history("project_state", project="B")[0].text == "B state"


def test_complementary_values_negation_and_order_survive_similarity(tmp_path):
    m = mem(tmp_path)
    texts = ["Alice works at Acme as an engineer.", "Alice works at Acme as a manager.",
             "Alice does not work at Acme as an engineer.", "Alice follows Bob.", "Bob follows Alice."]
    for i, text in enumerate(texts):
        r, _ = fact(m, text, i + 1)
        assert r["added"] and not r["superseded"] and not r["deduped"]
    assert m.consolidate()["deduped"] == 0
    assert m.consolidate()["superseded"] == 0
    assert m.status()["total"] == len(texts)


def test_homonyms_and_multivalued_identity_survive_text_dedup(tmp_path):
    m = mem(tmp_path)
    for subject in ("alice:1", "alice:2"):
        r, _ = fact(m, "Alice lives in Paris.", 10, identity=identity(subject))
        assert r["added"]
    for i, place in enumerate(("Paris", "Lyon")):
        r, _ = fact(m, f"Alice visits {place}.", 20+i, identity=identity(predicate="visits", cardinality="multi"))
        assert r["added"] and not r["superseded"]
    assert m.consolidate()["deduped"] == 0
    assert m.status()["total"] == 4


def test_explicit_replacement_closes_at_valid_from_and_reverts(tmp_path):
    m = mem(tmp_path)
    _, a = fact(m, "Alice lives in Paris.", 100)
    r, b = fact(m, "Alice lives in Lyon.", 200, valid_from=150, replaces=[a.id])
    assert r["superseded"] == [(a.id, b.id)]
    r, c = fact(m, "Alice lives in Paris.", 300, valid_from=250, replaces=[b.id])
    assert r["superseded"] == [(b.id, c.id)]
    assert a.valid_until == 150 and b.valid_until == 250
    for t, expected in ((100, a), (150, b), (249, b), (250, c)):
        hits = m.search("Alice lives", project="A", at_time=t, kinds=["fact"])
        assert [h.item.id for h in hits] == [expected.id]
    assert m.consolidate()["deduped"] == 0
    assert {it.id for it in m.store.all(False)} == {a.id, b.id, c.id}
    assert m.consolidate()["superseded"] == 0


def test_identity_timeline_handles_historical_insertion_and_revert(tmp_path):
    m = mem(tmp_path)
    _, a = fact(m, "Alice lives in Paris.", 100, identity=identity())
    _, c = fact(m, "Alice lives in Nice.", 300, identity=identity())
    _, b = fact(m, "Alice lives in Lyon.", 200, identity=identity())
    _, d = fact(m, "Alice lives in Paris.", 400, identity=identity())
    assert (a.valid_until, b.valid_until, c.valid_until) == (200, 300, 400)
    assert (a.invalidated_by, b.invalidated_by, c.invalidated_by) == (b.id, c.id, d.id)
    for t, expected in ((150, a), (250, b), (350, c), (450, d)):
        assert [it.id for it in m.store.all(False) if is_valid_at(it, t)] == [expected.id]
    m.flush()
    m.close()
    reloaded = mem(tmp_path)
    assert reloaded.store.get(a.id).identity == identity()
    assert reloaded.consolidate()["deduped"] == 0


def test_replacement_rejects_cross_project_and_equal_validity_without_mutation(tmp_path):
    m = mem(tmp_path)
    _, b = fact(m, "Alice lives in Paris.", 100, project="B")
    before = (tmp_path / "items.jsonl").read_bytes()
    with pytest.raises(ValueError, match="same project"):
        fact(m, "Alice lives in Lyon.", 200, replaces=[b.id])
    assert (tmp_path / "items.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="strictly after"):
        fact(m, "Alice lives in Lyon.", 100, project="B", replaces=[b.id])
    assert b.invalidated_by is None


def test_server_scope_defaults_and_explicit_all_projects(tmp_path, monkeypatch):
    m = mem(tmp_path)
    monkeypatch.setattr(server, "get_engine", lambda: m)
    monkeypatch.setenv("MEMORY_V4_PROJECT", "A")
    b = m.add_episode("B private searchable", project="B")
    m.add_episode("A private searchable", project="A")
    assert server.dispatch("memory_status", {})["total"] == 1
    assert server.dispatch("memory_status", {"project": "*"})["total"] == 2
    assert not server.dispatch("memory_forget", {"item_id": b.id, "reason": "test"})["ok"]
    assert not b.deleted
    assert server.dispatch("memory_forget", {"query": "searchable", "confirm": True, "reason": "test"})["forgotten"]
    assert not b.deleted
    server.dispatch("memory_block_set", {"name": "project_state", "text": "A state"})
    assert server.dispatch("memory_block_get", {"project": "B"})["blocks"]["project_state"] == ""
    server.dispatch("memory_block_set", {"name": "working_rules", "text": "Shared", "scope": "global"})
    assert server.dispatch("memory_block_get", {"project": "B"})["blocks"]["working_rules"] == "Shared"
    assert server.dispatch("memory_forget", {"item_id": b.id, "reason": "test", "project": "*"})["forgotten"] == [b.id]


def test_equal_identity_start_never_creates_zero_length_windows(tmp_path):
    m = mem(tmp_path)
    _, a = fact(m, "Alice lives in Paris.", 100, identity=identity())
    _, b = fact(m, "Alice lives in Lyon.", 100, identity=identity())
    assert a.valid_until is b.valid_until is None
    _, c = fact(m, "Alice lives in Nice.", 200, identity=identity())
    assert a.valid_until == b.valid_until == 200
    assert a.invalidated_by == b.invalidated_by == c.id


def test_consolidation_preserves_separate_bounded_assertions(tmp_path):
    m = mem(tmp_path)
    _, a = fact(m, "Alice lives in Paris.", 100, valid_until=200)
    _, b = fact(m, "Alice lives in Paris.", 300, valid_until=400)
    assert a.id != b.id
    before = (tmp_path / "items.jsonl").read_bytes()
    assert m.consolidate(dry_run=True)["deduped"] == 0
    assert (tmp_path / "items.jsonl").read_bytes() == before
    assert m.consolidate()["deduped"] == 0
    assert is_valid_at(a, 150) and not is_valid_at(b, 150)
    assert is_valid_at(b, 350) and not is_valid_at(a, 350)


def test_scoped_consolidation_dedup_drops_vectors_and_history_is_retained(tmp_path):
    m = mem(tmp_path)
    # Bypass write dedup to exercise a pre-existing duplicate store.
    items = m._build_items([
        {"text": "Alice works at Acme.", "project": "A", "event_time": 100, "entities": ["Alice"]},
        {"text": "Alice works at Acme!", "project": "A", "event_time": 200, "entities": ["Alice"]},
        {"text": "Alice works at Acme!", "project": "B", "event_time": 200, "entities": ["Alice"]},
    ], kind="fact")
    m._index_items(items)
    assert m.consolidate(project="A")["deduped"] == 1
    assert m.status(project="A")["total"] == m.status(project="A")["vector_count"] == 1
    assert m.status(project="B")["total"] == m.status(project="B")["vector_count"] == 1
    assert len(m.store.all()) == 3  # soft deletion preserves the log.


def test_replacement_metadata_roundtrip_through_mcp_model(tmp_path, monkeypatch):
    m = mem(tmp_path)
    monkeypatch.setattr(server, "get_engine", lambda: m)
    args = {"project": "A", "facts": [{"text": "Alice lives in Paris.", "identity": identity(), "event_time": "2020-01-01"}]}
    r = server.dispatch("memory_add_facts", args)
    old_id = r["added"][0]
    update = {"text": "Alice lives in Lyon.", "replaces": [old_id], "event_time": "2021-01-01"}
    r = server.dispatch("memory_add_facts", {"project": "A", "facts": [update]})
    assert r["superseded"] == [[old_id, r["added"][0]]]
    assert m.store.get(r["added"][0]).replaces == [old_id]


@pytest.mark.parametrize("positive, changed", [
    ("Alice is authorized to access the production financial reporting service for monthly audit and compliance reviews.",
     "Alice is not authorized to access the production financial reporting service for monthly audit and compliance reviews."),
    ("Alice has access to production service for reporting.", "Alice has no access to production service for reporting."),
    ("Alice possède un accès à la production.", "Alice ne possède pas un accès à la production."),
    ("Alice allows 1 production operator per service account.", "Alice allows 2 production operator per service account."),
    ("Alice's account balance is -5.", "Alice's account balance is +5."),
    ("Alice's account balance is 1.5.", "Alice's account balance is 15."),
    ("Alice parle de la cote.", "Alice parle de la côte."),
])
def test_assertion_dedup_preserves_negation_and_numeric_values(tmp_path, positive, changed):
    m = mem(tmp_path)
    first, _ = fact(m, positive, 100)
    second, _ = fact(m, changed, 200)
    assert first["added"] and second["added"]
    assert second["deduped"] == second["superseded"] == []
    assert m.consolidate()["deduped"] == 0
    assert m.status()["total"] == 2
