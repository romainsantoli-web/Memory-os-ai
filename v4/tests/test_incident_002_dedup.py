"""INC-002: similarity must never erase a complementary relay/hook assertion."""
import numpy as np
import pytest

from memory_os_v4 import Config, MemoryV4, server

RELAY = "On 2026-10-09 Claude configured the Claude desktop relay for memory-os-ai."
HOOK = "On 2026-10-09 Claude configured the Claude Code hook for memory-os-ai."


@pytest.fixture
def engine(tmp_path, monkeypatch):
    with MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash",
                              dedup_cos=0.0, supersede_cos=0.0)) as m:
        # Even cosine=1 cannot prove equality. Avoid expensive Granite dependency.
        vector = np.zeros(m.embedder.dim, dtype=np.float32)
        vector[0] = 1
        monkeypatch.setattr(m.embedder, "encode", lambda texts: np.repeat(vector[None, :], len(texts), axis=0))
        yield m


@pytest.mark.parametrize("kind", ["fact", "belief", "summary", "procedure"])
def test_relay_and_hook_both_persist_at_write_and_consolidation(engine, kind):
    for text in (RELAY, HOOK):
        result = engine.add_facts([{"text": text, "kind": kind, "entities": ["Claude"]}],
                                  project="INC-002", event_time=100)
        assert len(result["added"]) == 1 and result["deduped"] == result["superseded"] == []
    assert engine.consolidate(project="INC-002")["deduped"] == 0
    assert {r.text for r in engine.store.all(False)} == {RELAY, HOOK}


def test_mcp_relay_hook_and_same_identity_different_value_are_not_deduped(engine, monkeypatch):
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    for i, text in enumerate((RELAY, HOOK)):
        result = server.dispatch("memory_add_facts", {"project": "INC-002", "request_id": f"capture-{i}",
                                 "facts": [{"text": text, "entities": ["Claude"]}]})
        assert len(result["added"]) == 1 and result["deduped"] == []
    identity = {"subject_id": "user:synthetic", "predicate": "residence", "cardinality": "single"}
    first = engine.add_facts([{"text": "User lives in Paris.", "identity": identity}],
                            project="INC-002", event_time=100)["added"][0]
    second = engine.add_facts([{"text": "User lives in Lyon.", "identity": identity}],
                             project="INC-002", event_time=200)
    assert second["deduped"] == [] and second["superseded"] == [(first, second["added"][0])]


@pytest.mark.parametrize("kind", ["fact", "summary", "procedure"])
def test_equal_normalized_text_dedup_does_not_depend_on_embedding_threshold(engine, kind):
    first = engine.add_facts([{"text": "Client uses Neptune.", "kind": kind, "entities": []}],
                            project="INC-002", event_time=100)
    engine.config.dedup_cos = 1.01
    second = engine.add_facts([{"text": "  Client   uses Neptune!", "kind": kind, "entities": []}],
                             project="INC-002", event_time=200)
    assert second["deduped"] == first["added"] and second["added"] == []


def test_legacy_complementary_summaries_survive_sweep_but_exact_duplicates_do_not(engine):
    rows = engine._build_items([
        {"text": RELAY, "project": "INC-002", "event_time": 100, "entities": []},
        {"text": HOOK, "project": "INC-002", "event_time": 200, "entities": []},
        {"text": HOOK, "project": "INC-002", "event_time": 300, "entities": []},
    ], kind="summary")
    engine._index_items(rows)
    engine.config.dedup_cos = 1.01
    assert engine.consolidate(project="INC-002")["deduped"] == 1
    assert {r.text for r in engine.store.all(False)} == {RELAY, HOOK}
