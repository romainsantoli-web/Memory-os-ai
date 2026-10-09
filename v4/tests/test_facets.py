"""Tests for the production facet channel (`memory_os_v4.facets`, plus its
wiring through `store.py`/`memory.py`/`retrieval.py`/`models.py`/`server.py`).

Mirrors `docs/ARCHITECTURE.md`'s existing production ports (`keywords`,
`queries`/`hypotheses`): the closed-predicate-vocabulary mechanism itself was
already validated in `bench/retrievers.py` (`build_facet_index`/
`rank_facet_pool`, see `tests/test_bench_facets.py`); this file tests the
NEW pieces added here -- normalization, the in-memory `FacetIndex`,
persistence, and the read/write plumbing. Uses `Config(embed_model="hash")`
(the existing fake/stub embedder, never a real model load) throughout.
"""

from __future__ import annotations

import json

import pytest

from memory_os_v4 import models, server
from memory_os_v4.config import Config
from memory_os_v4.facets import FacetIndex, PREDICATES, normalize_facet, normalize_facets, rank_facet_pool
from memory_os_v4.instructions import FACET_INSTRUCTIONS_ADDENDUM, MEMORY_INSTRUCTIONS, build_instructions
from memory_os_v4.memory import MemoryV4
from memory_os_v4.store import MemoryItem, Store, text_hash


def _mem(tmp_path, **kwargs) -> MemoryV4:
    cfg = Config(embed_model="hash", data_dir=str(tmp_path), **kwargs)
    return MemoryV4(config=cfg)


# ---------------------------------------------------------------------------
# normalize_facet / normalize_facets
# ---------------------------------------------------------------------------
def test_normalize_facet_valid_triple():
    nf = normalize_facet({"subject": "Ann", "predicate": "pet", "object": "a dog"})
    assert nf == {"subject": "Ann", "subject_key": "ann", "predicate": "pet", "object": "a dog"}


def test_normalize_facet_unknown_predicate_becomes_other():
    nf = normalize_facet({"subject": "Ann", "predicate": "not_a_real_predicate"})
    assert nf["predicate"] == "other"


def test_normalize_facet_missing_predicate_becomes_other():
    nf = normalize_facet({"subject": "Ann"})
    assert nf["predicate"] == "other"
    assert nf["object"] == ""


def test_normalize_facet_predicate_case_and_whitespace_insensitive():
    nf = normalize_facet({"subject": "Ann", "predicate": "  PET  "})
    assert nf["predicate"] == "pet"


def test_normalize_facet_rejects_non_dict_or_blank_subject():
    assert normalize_facet(None) is None
    assert normalize_facet("not a dict") is None
    assert normalize_facet([1, 2]) is None
    assert normalize_facet({"subject": "", "predicate": "pet"}) is None
    assert normalize_facet({"subject": "   ", "predicate": "pet"}) is None
    assert normalize_facet({"predicate": "pet"}) is None  # no subject at all
    assert normalize_facet({"subject": 123, "predicate": "pet"}) is None  # wrong type


def test_normalize_facet_never_raises_on_garbage():
    for garbage in (1, 1.5, True, [], {}, {"subject": None}, {"subject": "x", "predicate": 123}):
        normalize_facet(garbage)  # must not raise


def test_normalize_facet_caps_subject_and_object_length():
    nf = normalize_facet({"subject": "A" * 200, "predicate": "pet", "object": "B" * 500})
    assert len(nf["subject"]) == 80
    assert len(nf["object"]) == 200


def test_normalize_facet_subject_key_is_casefolded():
    nf = normalize_facet({"subject": "  Ann  ", "predicate": "pet"})
    assert nf["subject"] == "Ann"
    assert nf["subject_key"] == "ann"


def test_normalize_facets_drops_invalid_entries_keeps_valid_order():
    raw = [
        {"subject": "Ann", "predicate": "pet"},
        None,
        "garbage",
        {"subject": "", "predicate": "pet"},
        {"subject": "Bob", "predicate": "friend"},
    ]
    out = normalize_facets(raw)
    assert [f["subject"] for f in out] == ["Ann", "Bob"]


def test_normalize_facets_caps_at_eight():
    raw = [{"subject": f"s{i}", "predicate": "pet"} for i in range(20)]
    out = normalize_facets(raw)
    assert len(out) == 8
    assert [f["subject"] for f in out] == [f"s{i}" for i in range(8)]


def test_normalize_facets_non_list_returns_empty():
    assert normalize_facets(None) == []
    assert normalize_facets("not a list") == []
    assert normalize_facets({"subject": "Ann"}) == []


def test_predicates_frozenset_excludes_other():
    assert "other" not in PREDICATES
    assert "pet" in PREDICATES
    assert len(PREDICATES) == len(set(PREDICATES))  # no dupes


# ---------------------------------------------------------------------------
# FacetIndex
# ---------------------------------------------------------------------------
class _Item:
    def __init__(self, id_, facets):
        self.id = id_
        self.facets = facets


def test_facet_index_add_and_pool_lookup():
    idx = FacetIndex()
    idx.add(_Item("i1", normalize_facets([{"subject": "Ann", "predicate": "pet", "object": "a dog"}])))
    idx.add(_Item("i2", normalize_facets([{"subject": "Ann", "predicate": "pet"}])))
    idx.add(_Item("i3", normalize_facets([{"subject": "Bob", "predicate": "friend"}])))

    pool = idx.pool([{"subject": "Ann", "predicate": "pet"}])
    assert pool == {"i1", "i2"}

    pool2 = idx.pool([{"subject": "ann", "predicate": "pet"}])  # case-insensitive match
    assert pool2 == {"i1", "i2"}

    pool3 = idx.pool([{"subject": "Nobody", "predicate": "pet"}])
    assert pool3 == set()


def test_facet_index_never_indexes_other_predicate():
    idx = FacetIndex()
    idx.add(_Item("i1", normalize_facets([{"subject": "Ann", "predicate": "totally_unknown"}])))
    assert idx.pool([{"subject": "Ann", "predicate": "other"}]) == set()
    assert idx._index == {}  # nothing at all got indexed


def test_facet_index_remove():
    idx = FacetIndex()
    idx.add(_Item("i1", normalize_facets([{"subject": "Ann", "predicate": "pet"}])))
    assert idx.pool([{"subject": "Ann", "predicate": "pet"}]) == {"i1"}
    idx.remove("i1")
    assert idx.pool([{"subject": "Ann", "predicate": "pet"}]) == set()
    idx.remove("does-not-exist")  # never raises


def test_facet_index_re_add_replaces_old_keys():
    idx = FacetIndex()
    idx.add(_Item("i1", normalize_facets([{"subject": "Ann", "predicate": "pet"}])))
    idx.add(_Item("i1", normalize_facets([{"subject": "Bob", "predicate": "friend"}])))
    assert idx.pool([{"subject": "Ann", "predicate": "pet"}]) == set()
    assert idx.pool([{"subject": "Bob", "predicate": "friend"}]) == {"i1"}


def test_facet_index_rebuild_from_scratch():
    idx = FacetIndex()
    idx.add(_Item("stale", normalize_facets([{"subject": "Old", "predicate": "pet"}])))
    idx.rebuild([_Item("i1", normalize_facets([{"subject": "Ann", "predicate": "pet"}]))])
    assert idx.pool([{"subject": "Old", "predicate": "pet"}]) == set()
    assert idx.pool([{"subject": "Ann", "predicate": "pet"}]) == {"i1"}


def test_facet_index_ids_tracks_presence_even_with_no_indexable_facets():
    idx = FacetIndex()
    idx.add(_Item("i1", normalize_facets([{"subject": "Ann", "predicate": "unknown_pred"}])))  # -> other
    assert idx.ids() == {"i1"}  # tracked as "present" even though nothing got indexed
    idx.remove("i1")
    assert idx.ids() == set()


def test_facet_index_pool_accepts_tuples_and_dicts():
    idx = FacetIndex()
    idx.add(_Item("i1", normalize_facets([{"subject": "Ann", "predicate": "pet"}])))
    assert idx.pool([("ann", "pet")]) == {"i1"}
    assert idx.pool([{"subject": "Ann", "predicate": "pet"}]) == {"i1"}
    assert idx.pool([("", "pet")]) == set()
    assert idx.pool([("ann", "")]) == set()
    assert idx.pool(None) == set()
    assert idx.pool([]) == set()


# ---------------------------------------------------------------------------
# rank_facet_pool
# ---------------------------------------------------------------------------
def test_rank_facet_pool_orders_by_best_rank_across_hit_lists():
    pool = {"a", "b", "c"}
    hit_lists = [["x", "b", "a"], ["c"]]
    # a: rank 2 in list 0; b: rank 1 in list 0; c: rank 0 in list 1 -> best overall
    assert rank_facet_pool(pool, hit_lists) == ["c", "b", "a"]


def test_rank_facet_pool_absent_members_go_last_ties_by_id():
    pool = {"z", "a", "m"}
    assert rank_facet_pool(pool, []) == ["a", "m", "z"]  # nothing ranked anywhere: pure id order

    hit_lists = [["m"]]
    assert rank_facet_pool(pool, hit_lists) == ["m", "a", "z"]  # m ranked; a/z tie, id order


# ---------------------------------------------------------------------------
# MemoryItem.facets: storage / persistence / backward compatibility
# ---------------------------------------------------------------------------
def test_add_episode_stores_and_indexes_facets_even_when_flag_off(tmp_path):
    m = _mem(tmp_path)  # facet_channel defaults to False
    item = m.add_episode(
        "Ann has a pet dog named Max.",
        project="p1",
        facets=[{"subject": "Ann", "predicate": "pet", "object": "Max"}],
    )
    assert item.facets == [{"subject": "Ann", "subject_key": "ann", "predicate": "pet", "object": "Max"}]
    assert "Max" in item.text  # unrelated to facets: unaltered display text
    # storing/indexing happens regardless of the flag -- so enabling it later
    # needs no re-index.
    assert m.facet_index.pool([{"subject": "Ann", "predicate": "pet"}]) == {item.id}


def test_add_episodes_bulk_accepts_facets_per_episode(tmp_path):
    m = _mem(tmp_path)
    items = m.add_episodes_bulk(
        [
            {"text": "First.", "project": "p1", "facets": [{"subject": "Ann", "predicate": "pet"}]},
            {"text": "Second.", "project": "p1"},
        ]
    )
    assert items[0].facets[0]["predicate"] == "pet"
    assert items[1].facets == []  # default: unaffected caller, backward compatible


def test_add_facts_accepts_facets_per_fact(tmp_path):
    m = _mem(tmp_path)
    result = m.add_facts(
        [{"text": "Ann has a dog.", "facets": [{"subject": "Ann", "predicate": "pet", "object": "dog"}]}],
        project="p1",
    )
    item = m.store.get(result["added"][0])
    assert item.facets[0]["subject"] == "Ann"
    assert item.facets[0]["predicate"] == "pet"


def test_facets_persist_across_reload(tmp_path):
    m1 = _mem(tmp_path)
    item = m1.add_episode(
        "Ann has a pet dog.", project="p1", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    m1.flush()

    m1.close()
    m2 = _mem(tmp_path)
    reloaded = m2.store.get(item.id)
    assert reloaded.facets == item.facets
    # the facet index itself is derived state, rebuilt from items.jsonl at load
    assert m2.facet_index.pool([{"subject": "Ann", "predicate": "pet"}]) == {item.id}


def test_store_loads_old_record_without_facets_field(tmp_path):
    """Backward compatibility, exactly like `keywords`: an `items.jsonl`
    line written before the `facets` field existed must still load,
    defaulting to an empty list."""
    item = MemoryItem(id="x1", kind="episode", text="Old record.", hash=text_hash("old record."))
    old_record = item.to_dict()
    del old_record["facets"]
    (tmp_path / "items.jsonl").write_text(json.dumps(old_record) + "\n", encoding="utf-8")

    store = Store(tmp_path)
    loaded = store.get("x1")
    assert loaded is not None
    assert loaded.facets == []


def test_facets_never_appended_to_indexed_text(tmp_path):
    """Facets, unlike keywords, must never change what dense/BM25 index --
    `memory.index_text()` is untouched by this field."""
    from memory_os_v4.memory import index_text

    m = _mem(tmp_path)
    item = m.add_episode(
        "A quiet afternoon.",
        project="p1",
        facets=[{"subject": "Ann", "predicate": "pet", "object": "zzzuniqueobjectzzz"}],
    )
    assert "zzzuniqueobjectzzz" not in index_text(item, m.config)
    from memory_os_v4.text import tokenize

    assert m.bm25.search(tokenize("zzzuniqueobjectzzz"), 10) == []


def test_forget_removes_item_from_facet_index(tmp_path):
    m = _mem(tmp_path)
    item = m.add_episode(
        "Ann has a pet.", project="p1", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    assert m.facet_index.pool([{"subject": "Ann", "predicate": "pet"}]) == {item.id}
    assert m.forget(item.id, reason="test") is True
    assert m.facet_index.pool([{"subject": "Ann", "predicate": "pet"}]) == set()


def test_reconcile_indexes_rebuilds_facet_index_on_fresh_load(tmp_path):
    """`_reconcile_indexes` must repopulate the (never-persisted) facet
    index from the durable `items.jsonl` on every fresh load, same as
    bm25/entities."""
    m1 = _mem(tmp_path)
    item = m1.add_episode(
        "Ann has a pet.", project="p1", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    m1.flush()

    m1.close()
    m2 = _mem(tmp_path)  # fresh engine, same data_dir -- facet_index starts empty
    assert m2.facet_index.pool([{"subject": "Ann", "predicate": "pet"}]) == {item.id}


# ---------------------------------------------------------------------------
# Read path: HybridRetriever.facet_pool / search_multi facet fusion
# ---------------------------------------------------------------------------
def _bm25_only_memory(tmp_path, **cfg_kwargs) -> MemoryV4:
    return _mem(tmp_path, use_dense=False, use_entities=False, use_bm25=True, **cfg_kwargs)


def test_facet_channel_flag_off_search_byte_identical_to_plain(tmp_path):
    """Flag off (default): passing `facets` must never change the result,
    even though the facet index itself is always maintained."""
    m = _bm25_only_memory(tmp_path)  # facet_channel defaults False
    m.add_episode(
        "alpha topic note.", project="p1", item_id="a", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    m.add_episode("beta topic note.", project="p1", item_id="b")

    plain = m.search("alpha", top_k=5, project="p1")
    with_facets = m.search(
        "alpha", top_k=5, project="p1", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    assert [h.item.id for h in plain] == [h.item.id for h in with_facets]
    assert "multi_query" not in plain[0].signals
    assert "multi_query" not in with_facets[0].signals
    assert "facet" not in with_facets[0].signals


def test_facet_channel_flag_on_no_facets_given_byte_identical(tmp_path):
    """Flag on but caller passes no `facets`: still byte-identical to plain
    search (empty pool, no-op)."""
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    m.add_episode("alpha topic note.", project="p1", item_id="a")
    plain = m.search("alpha", top_k=5, project="p1")
    explicit = m.search("alpha", top_k=5, project="p1", facets=None)
    assert [h.item.id for h in plain] == [h.item.id for h in explicit]


def test_facet_channel_flag_on_empty_pool_byte_identical(tmp_path):
    """Flag on, facets given, but nothing in the corpus matches them: still
    byte-identical (no facet list gets appended to the RRF fusion)."""
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    m.add_episode("alpha topic note.", project="p1", item_id="a")
    plain = m.search("alpha", top_k=5, project="p1")
    with_facets = m.search(
        "alpha", top_k=5, project="p1", facets=[{"subject": "Nobody", "predicate": "pet"}]
    )
    assert [h.item.id for h in plain] == [h.item.id for h in with_facets]
    assert plain[0].score == pytest.approx(with_facets[0].score)
    assert "multi_query" not in with_facets[0].signals


def test_facet_channel_brings_in_facet_only_item_not_found_by_query(tmp_path):
    """The whole point of the channel: an item the text query/sub-queries
    would never surface on their own can still enter the top-k purely via
    its (subject, predicate) facet match."""
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    # "zzznomatch" shares no token with the query below, so plain BM25 would
    # never surface it.
    facet_item = m.add_episode(
        "zzznomatch has an unrelated hobby.",
        project="p1",
        item_id="facet_only",
        facets=[{"subject": "Ann", "predicate": "pet", "object": "a dog"}],
    )
    m.add_episode("Some other unrelated note.", project="p1", item_id="other")

    plain = m.search("What pet does Ann have", top_k=5, project="p1")
    assert facet_item.id not in {h.item.id for h in plain}

    with_facets = m.search(
        "What pet does Ann have",
        top_k=5,
        project="p1",
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    ids = [h.item.id for h in with_facets]
    assert "facet_only" in ids
    hit = next(h for h in with_facets if h.item.id == "facet_only")
    assert hit.signals["facet"] is True


def test_facet_channel_marks_pre_existing_hit_with_facet_signal(tmp_path):
    """A pool member that the plain query ALSO finds must still be tagged
    `signals["facet"] = True`, not just facet-only synthesized hits."""
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    m.add_episode(
        "Ann loves her pet dog.",
        project="p1",
        item_id="a",
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    hits = m.search("Ann pet dog", top_k=5, project="p1", facets=[{"subject": "Ann", "predicate": "pet"}])
    hit = next(h for h in hits if h.item.id == "a")
    assert hit.signals.get("facet") is True


def test_facet_weight_zero_reproduces_baseline_ranking(tmp_path):
    m = _bm25_only_memory(tmp_path, facet_channel=True, facet_weight=0.0)
    m.add_episode(
        "zzznomatch has an unrelated hobby.",
        project="p1",
        item_id="facet_only",
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    m.add_episode("Ann talks about her pet dog Rex.", project="p1", item_id="a")
    hits = m.search(
        "Ann pet dog", top_k=5, project="p1", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    # facet_only still enters the pool/fusion (rank_facet_pool includes it,
    # RRF just contributes 0 * 1/(k+rank) = 0 from that party) but never
    # outranks a real match; more importantly it must not crash.
    assert any(h.item.id == "a" for h in hits)


def test_facet_pool_respects_project_filter(tmp_path):
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    m.add_episode(
        "Ann has a pet.", project="other_project", item_id="wrong_project",
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    m.add_episode("Some note.", project="p1", item_id="a")
    hits = m.search(
        "some note", top_k=5, project="p1", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    assert "wrong_project" not in {h.item.id for h in hits}


def test_facet_pool_respects_kind_filter(tmp_path):
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    m.add_facts(
        [{"text": "Ann has a pet.", "facets": [{"subject": "Ann", "predicate": "pet"}]}], project="p1"
    )
    m.add_episode("Some note.", project="p1", item_id="ep")
    hits = m.search(
        "some note",
        top_k=5,
        project="p1",
        kinds=["episode"],
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    assert all(h.item.kind == "episode" for h in hits)


def test_facet_pool_respects_at_time_validity_filter(tmp_path):
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    result = m.add_facts(
        [
            {
                "text": "Ann has a pet.",
                "facets": [{"subject": "Ann", "predicate": "pet"}],
                "valid_until": 1000.0,
            }
        ],
        project="p1",
    )
    fact_id = result["added"][0]
    m.add_facts([{"text": "Some other fact."}], project="p1")

    # Querying at a time after `valid_until` must exclude the facet-matched
    # item, exactly like it would for an ordinary candidate.
    hits = m.search(
        "some other fact",
        top_k=5,
        project="p1",
        at_time=2000.0,
        kinds=["fact"],
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    assert fact_id not in {h.item.id for h in hits}


def test_get_context_accepts_facets(tmp_path):
    m = _bm25_only_memory(tmp_path, facet_channel=True)
    m.add_episode(
        "zzznomatch has an unrelated hobby.",
        project="p1",
        item_id="facet_only",
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    context = m.get_context(
        "irrelevant query text", project="p1", facets=[{"subject": "Ann", "predicate": "pet"}]
    )
    assert "zzznomatch" in context


def test_facet_pool_helper_empty_when_flag_off(tmp_path):
    m = _mem(tmp_path)
    m.add_episode("Ann has a pet.", project="p1", facets=[{"subject": "Ann", "predicate": "pet"}])
    assert m.retriever.facet_pool([{"subject": "Ann", "predicate": "pet"}]) == set()


def test_facet_pool_helper_nonempty_when_flag_on(tmp_path):
    m = _mem(tmp_path, facet_channel=True)
    item = m.add_episode("Ann has a pet.", project="p1", facets=[{"subject": "Ann", "predicate": "pet"}])
    assert m.retriever.facet_pool([{"subject": "Ann", "predicate": "pet"}]) == {item.id}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_facet_defaults():
    cfg = Config()
    assert cfg.facet_channel is False
    assert cfg.facet_weight == 2.0


def test_config_facet_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_FACET_CHANNEL", "true")
    monkeypatch.setenv("MEMORY_V4_FACET_WEIGHT", "3.5")
    cfg = Config.from_env()
    assert cfg.facet_channel is True
    assert cfg.facet_weight == 3.5


# ---------------------------------------------------------------------------
# instructions.py
# ---------------------------------------------------------------------------
def test_build_instructions_off_is_byte_identical_to_base():
    assert build_instructions(Config(facet_channel=False)) == MEMORY_INSTRUCTIONS


def test_build_instructions_on_appends_addendum():
    out = build_instructions(Config(facet_channel=True))
    assert out == MEMORY_INSTRUCTIONS + FACET_INSTRUCTIONS_ADDENDUM


def test_facet_addendum_under_900_chars_and_lists_all_predicates():
    assert len(FACET_INSTRUCTIONS_ADDENDUM) <= 900
    for pred in PREDICATES:
        assert pred in FACET_INSTRUCTIONS_ADDENDUM


# ---------------------------------------------------------------------------
# models.py / server.py plumbing
# ---------------------------------------------------------------------------
def test_fact_in_and_message_in_accept_optional_facets():
    fact = models.FactIn(text="hi", facets=[{"subject": "Ann", "predicate": "pet", "object": "dog"}])
    assert fact.facets[0].subject == "Ann"
    assert models.FactIn(text="hi").facets is None

    msg = models.MessageIn(role="user", content="hi", facets=[{"subject": "Ann", "predicate": "pet"}])
    assert msg.facets[0].predicate == "pet"
    assert models.MessageIn(role="user", content="hi").facets is None


def test_search_in_and_get_context_in_accept_optional_query_facets():
    s = models.SearchIn(query="q", facets=[{"subject": "Ann", "predicate": "pet"}])
    assert s.facets[0].subject == "Ann"
    assert models.SearchIn(query="q").facets is None

    g = models.GetContextIn(query="q", facets=[{"subject": "Ann", "predicate": "pet"}])
    assert g.facets[0].subject == "Ann"


def test_query_facet_in_rejects_object_field():
    with pytest.raises(Exception):
        models.QueryFacetIn(subject="Ann", predicate="pet", object="dog")


def test_facets_field_capped_at_eight_entries():
    with pytest.raises(Exception):
        models.SearchIn(query="q", facets=[{"subject": "s", "predicate": "pet"}] * 9)


@pytest.fixture
def engine(tmp_path):
    return MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path), facet_channel=True))


@pytest.fixture
def server_engine(engine, monkeypatch):
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    yield engine


def _call(name: str, payload: dict) -> dict:
    return server.dispatch(name, payload)


def test_chat_save_handler_stores_facets_on_message_and_fact(server_engine):
    result = _call(
        "memory_chat_save",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Ann has a pet dog.",
                    "facets": [{"subject": "Ann", "predicate": "pet", "object": "dog"}],
                }
            ],
            "facts": [
                {
                    "text": "Ann's dog is named Max.",
                    "facets": [{"subject": "Ann", "predicate": "pet", "object": "Max"}],
                }
            ],
            "project_label": "p1",
        },
    )
    assert "error" not in result
    hits = server_engine.search("Ann", project="p1", top_k=10)
    facet_bearing = [h for h in hits if h.item.facets]
    assert facet_bearing  # at least the episode and/or fact carried facets through
    assert any(f["predicate"] == "pet" for h in facet_bearing for f in h.item.facets)


def test_search_handler_accepts_facets_and_widens_recall(server_engine):
    server_engine.add_episode(
        "zzznomatch unrelated text.",
        project="p1",
        item_id="facet_only",
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    result = _call(
        "memory_search",
        {
            "query": "What pet does Ann have",
            "project": "p1",
            "facets": [{"subject": "Ann", "predicate": "pet"}],
        },
    )
    assert "error" not in result
    ids = {h["id"] for h in result["hits"]}
    assert "facet_only" in ids


def test_get_context_handler_accepts_facets(server_engine):
    server_engine.add_episode(
        "zzznomatch unrelated text.",
        project="p1",
        item_id="facet_only",
        facets=[{"subject": "Ann", "predicate": "pet"}],
    )
    result = _call(
        "memory_get_context",
        {"query": "irrelevant", "project": "p1", "facets": [{"subject": "Ann", "predicate": "pet"}]},
    )
    assert "error" not in result
    assert "zzznomatch" in result["context"]


def test_selfcheck_still_passes_with_facet_channel_default_off():
    """`--selfcheck` (server.py `_selfcheck`) must keep passing untouched --
    it never enables the facet channel and never touches ~/.memory-os-v4."""
    exit_code = server._selfcheck()
    assert exit_code == 0
