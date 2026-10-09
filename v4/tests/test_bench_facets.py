"""Tests for idea 8 (docs/IDEAS.md, 2026-09-19): the facet channel and
implications channel added to `bench.enrich` / `bench.retrievers.V4PlannedRetriever`.

Everything here uses `Config(embed_model="hash")` (never a real model load)
and writes only into pytest `tmp_path` dirs -- the real
`bench/data/enrich/locomo_facets`/`locomo_qfacets` (complete, verified, but
read-only per the task brief) are never read from or depended on.
"""
from __future__ import annotations

import json

from bench.enrich import FactFacetRecord, fid_to_turn_id, load_fact_facets, load_qfacets
from bench.protocol import BenchItem
from bench.retrievers import V4PlannedRetriever, build_facet_index, rank_facet_pool
from bench.run_bench import run_bench


def _write_jsonl(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _item(
    id_: str, text: str, speaker: str = "s", session_id: str = "s1", conv_id: str = "convX"
) -> BenchItem:
    return BenchItem(
        id=id_, text=text, speaker=speaker, event_time=None, session_id=session_id, meta={"conv_id": conv_id}
    )


def _make_retriever(facets_dir=None, qfacets_dir=None, queries_dir=None, **kwargs) -> V4PlannedRetriever:
    r = V4PlannedRetriever(
        config_overrides={"embed_model": "hash"},
        facets_dir=str(facets_dir) if facets_dir is not None else None,
        qfacets_dir=str(qfacets_dir) if qfacets_dir is not None else None,
        queries_dir=str(queries_dir) if queries_dir is not None else None,
        **kwargs,
    )
    r.reset()
    return r


class _FakeItem:
    def __init__(self, id_: str, kind: str = "episode", parent_ids=None):
        self.id = id_
        self.kind = kind
        self.parent_ids = parent_ids or []


class _FakeSearchHit:
    """Drop-in for `memory_os_v4.retrieval.SearchHit` (same pattern as
    `tests/test_bench_plan.py`'s own `_FakeSearchHit`)."""

    def __init__(self, id_: str, score: float = 1.0, kind: str = "episode", parent_ids=None):
        self.score = score
        self.item = _FakeItem(id_, kind=kind, parent_ids=parent_ids)


def _synthetic_locomo_1conv(path):
    data = [
        {
            "sample_id": "convA",
            "conversation": {
                "speaker_a": "Ann",
                "speaker_b": "Bob",
                "session_1_date_time": "1:00 pm on 1 January, 2023",
                "session_1": [
                    {"speaker": "Ann", "dia_id": "D1:1", "text": "I have a pet dog named Max."},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "That sounds fun, I prefer swimming."},
                ],
            },
            "qa": [
                {
                    "question": "What pet does Ann have?",
                    "answer": "a dog",
                    "evidence": ["D1:1"],
                    "category": 4,
                },
            ],
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# bench.enrich.load_fact_facets: tolerance contract
# ---------------------------------------------------------------------------
def test_load_fact_facets_missing_file_returns_empty_dict(tmp_path):
    assert load_fact_facets(tmp_path / "does_not_exist", "convX") == {}


def test_load_fact_facets_parses_valid_lines(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            json.dumps(
                {
                    "fid": "D1:1#0",
                    "facets": [{"subject": "Ann", "predicate": "pet", "object": "a dog"}],
                    "implications": ["Ann likes animals."],
                }
            )
        ],
    )
    records = load_fact_facets(facets_dir, "convX")
    assert set(records.keys()) == {"D1:1#0"}
    assert records["D1:1#0"] == FactFacetRecord(
        facets=[{"subject": "Ann", "predicate": "pet", "object": "a dog"}],
        implications=["Ann likes animals."],
    )


def test_load_fact_facets_skips_malformed_lines(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            "{not valid json",
            json.dumps({"facets": []}),  # missing fid
            json.dumps({"fid": "no-hash-here", "facets": []}),  # fid has no "#"
            json.dumps({"fid": "D1:1#0", "facets": "not-a-list"}),
            json.dumps({"fid": "D1:2#0", "facets": [], "implications": "not-a-list"}),
            json.dumps(
                {"fid": "D1:3#0", "facets": [{"subject": "Ann", "predicate": "pet"}], "implications": []}
            ),
        ],
    )
    records = load_fact_facets(facets_dir, "convX")
    assert set(records.keys()) == {"D1:3#0"}


def test_load_fact_facets_drops_malformed_facet_entries_not_whole_line(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            json.dumps(
                {
                    "fid": "D1:1#0",
                    "facets": [
                        {"subject": "Ann", "predicate": "pet", "object": "dog"},
                        {"subject": "", "predicate": "pet"},  # blank subject dropped
                        {"predicate": "pet"},  # missing subject dropped
                        "not-a-dict",
                    ],
                    "implications": [],
                }
            )
        ],
    )
    records = load_fact_facets(facets_dir, "convX")
    assert records["D1:1#0"].facets == [{"subject": "Ann", "predicate": "pet", "object": "dog"}]


def test_load_fact_facets_duplicate_fid_last_wins(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            json.dumps(
                {"fid": "D1:1#0", "facets": [{"subject": "Ann", "predicate": "pet"}], "implications": []}
            ),
            json.dumps(
                {
                    "fid": "D1:1#0",
                    "facets": [{"subject": "Ann", "predicate": "job_career"}],
                    "implications": ["final"],
                }
            ),
        ],
    )
    records = load_fact_facets(facets_dir, "convX")
    assert records["D1:1#0"].facets == [{"subject": "Ann", "predicate": "job_career", "object": ""}]
    assert records["D1:1#0"].implications == ["final"]


def test_fid_to_turn_id():
    assert fid_to_turn_id("D5:1#0") == "D5:1"
    assert fid_to_turn_id("D5:1#3") == "D5:1"


# ---------------------------------------------------------------------------
# bench.enrich.load_qfacets: tolerance contract
# ---------------------------------------------------------------------------
def test_load_qfacets_missing_file_returns_empty_dict(tmp_path):
    assert load_qfacets(tmp_path / "does_not_exist", "convX") == {}


def test_load_qfacets_parses_valid_lines(tmp_path):
    qfacets_dir = tmp_path / "qfacets"
    _write_jsonl(
        qfacets_dir / "convX.jsonl",
        [
            json.dumps({"qid": "convX:q0", "facets": [{"subject": "Ann", "predicate": "pet"}]}),
            json.dumps({"qid": "convX:q1", "facets": []}),
        ],
    )
    facets = load_qfacets(qfacets_dir, "convX")
    assert facets == {"convX:q0": [("Ann", "pet")], "convX:q1": []}


def test_load_qfacets_skips_malformed_entries_and_lines(tmp_path):
    qfacets_dir = tmp_path / "qfacets"
    _write_jsonl(
        qfacets_dir / "convX.jsonl",
        [
            "{not valid json",
            json.dumps({"facets": []}),  # missing qid
            json.dumps({"qid": "convX:q0", "facets": "not-a-list"}),
            json.dumps(
                {
                    "qid": "convX:q1",
                    "facets": [
                        {"subject": "Ann"},
                        {"predicate": "pet"},
                        {"subject": "Bob", "predicate": "hobby_interest"},
                    ],
                }
            ),
        ],
    )
    facets = load_qfacets(qfacets_dir, "convX")
    assert set(facets.keys()) == {"convX:q1"}
    assert facets["convX:q1"] == [("Bob", "hobby_interest")]


def test_load_qfacets_duplicate_qid_last_wins(tmp_path):
    qfacets_dir = tmp_path / "qfacets"
    _write_jsonl(
        qfacets_dir / "convX.jsonl",
        [
            json.dumps({"qid": "convX:q0", "facets": [{"subject": "Ann", "predicate": "pet"}]}),
            json.dumps({"qid": "convX:q0", "facets": [{"subject": "Bob", "predicate": "hobby_interest"}]}),
        ],
    )
    assert load_qfacets(qfacets_dir, "convX")["convX:q0"] == [("Bob", "hobby_interest")]


# ---------------------------------------------------------------------------
# bench.retrievers.build_facet_index
# ---------------------------------------------------------------------------
def test_build_facet_index_skips_other_predicate():
    fact_facets = {
        "D1:1#0": FactFacetRecord(facets=[{"subject": "Ann", "predicate": "other", "object": "x"}]),
        "D1:2#0": FactFacetRecord(facets=[{"subject": "Ann", "predicate": "pet", "object": "dog"}]),
    }
    index = build_facet_index(fact_facets)
    assert ("Ann", "other") not in index
    assert index[("Ann", "pet")] == {"D1:2"}


def test_build_facet_index_unions_turns_across_facts_and_derives_turn_id_from_fid():
    fact_facets = {
        "D1:1#0": FactFacetRecord(facets=[{"subject": "Ann", "predicate": "pet"}]),
        "D1:1#1": FactFacetRecord(facets=[{"subject": "Bob", "predicate": "pet"}]),
        "D2:3#0": FactFacetRecord(facets=[{"subject": "Ann", "predicate": "pet"}]),
    }
    index = build_facet_index(fact_facets)
    assert index[("Ann", "pet")] == {"D1:1", "D2:3"}
    assert index[("Bob", "pet")] == {"D1:1"}


def test_build_facet_index_is_a_pure_function_no_cross_conversation_leak():
    # "Keyed per conversation": build_facet_index is a pure function of
    # whatever fact_facets dict it's handed -- two different conversations'
    # dicts (even reusing the same (subject, predicate) key) never see each
    # other's turn ids.
    conv_a = build_facet_index({"D1:1#0": FactFacetRecord(facets=[{"subject": "Ann", "predicate": "pet"}])})
    conv_b = build_facet_index({"D9:9#0": FactFacetRecord(facets=[{"subject": "Ann", "predicate": "pet"}])})
    assert conv_a[("Ann", "pet")] == {"D1:1"}
    assert conv_b[("Ann", "pet")] == {"D9:9"}


# ---------------------------------------------------------------------------
# bench.retrievers.rank_facet_pool
# ---------------------------------------------------------------------------
def test_rank_facet_pool_orders_by_best_rank_across_hit_lists():
    pool = {"a", "b", "c"}
    hit_lists = [["x", "c", "b"], ["a"]]  # c rank1, b rank2 in list0; a rank0 in list1
    assert rank_facet_pool(pool, hit_lists) == ["a", "c", "b"]


def test_rank_facet_pool_best_rank_wins_across_multiple_lists():
    pool = {"a"}
    hit_lists = [["z", "a"], ["a"]]  # a is rank1 in list0 but rank0 in list1 -> best=0
    assert rank_facet_pool(pool, hit_lists) == ["a"]


def test_rank_facet_pool_absent_members_go_last_ordered_by_turn_id():
    pool = {"z9", "z1", "found"}
    hit_lists = [["found"]]
    assert rank_facet_pool(pool, hit_lists) == ["found", "z1", "z9"]


def test_rank_facet_pool_empty_pool_returns_empty_list():
    assert rank_facet_pool(set(), [["a", "b"]]) == []


def test_rank_facet_pool_ties_at_same_rank_broken_by_turn_id():
    pool = {"b", "a"}
    hit_lists = [["b"], ["a"]]  # both rank 0 in their own list -> tie -> id ascending
    assert rank_facet_pool(pool, hit_lists) == ["a", "b"]


# ---------------------------------------------------------------------------
# V4PlannedRetriever.search(): facet channel injected into RRF fusion
# ---------------------------------------------------------------------------
def test_facet_channel_off_by_default():
    retriever = _make_retriever()
    assert retriever.facet_channel is False
    assert retriever.implications is False
    assert retriever.facet_weight == 0.5


def test_facet_channel_question_without_facets_is_unchanged(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": [], "hypotheses": []})],
    )
    canned = {"original": [_FakeSearchHit("o1"), _FakeSearchHit("o2")]}

    baseline = _make_retriever(queries_dir=queries_dir)
    baseline.add([_item("D1:1", "Ann: hi.")])
    baseline._memory.search = lambda qtext, **kw: canned[qtext]
    expected = baseline.search("original", top_k=5, qid="convX:q0")

    with_facets = _make_retriever(queries_dir=queries_dir, facet_channel=True)
    with_facets.add([_item("D1:1", "Ann: hi.")])
    with_facets._memory.search = lambda qtext, **kw: canned[qtext]
    with_facets._qfacets = {}  # no facets entry for this qid
    got = with_facets.search("original", top_k=5, qid="convX:q0")
    assert got == expected


def test_facet_channel_no_facet_file_for_conversation_is_unchanged(tmp_path):
    # facets_dir exists but has no <conv_id>.jsonl -> load_fact_facets
    # returns {} -> the facet INDEX is empty even though a qfacets file (a
    # question facet) exists -> pool always empty -> exactly like
    # facet_channel=False.
    facets_dir = tmp_path / "facets"
    facets_dir.mkdir()
    qfacets_dir = tmp_path / "qfacets"
    _write_jsonl(
        qfacets_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "facets": [{"subject": "Ann", "predicate": "pet"}]})],
    )
    canned = {"original": [_FakeSearchHit("o1"), _FakeSearchHit("o2")]}
    retriever = _make_retriever(
        facets_dir=facets_dir, qfacets_dir=qfacets_dir, facet_channel=True, facet_weight=1.0
    )
    retriever.add([_item("D1:1", "Ann: hi.")])
    retriever._memory.search = lambda qtext, **kw: canned[qtext]
    assert retriever._qfacets == {"convX:q0": [("Ann", "pet")]}  # qfacets DID load
    assert retriever._facet_index == {}  # but no facet file for this conversation
    result = retriever.search("original", top_k=5, qid="convX:q0")
    assert result == ["o1", "o2"]


def test_facet_channel_injects_pool_ranked_by_best_hit_rank():
    retriever = _make_retriever(facet_channel=True, facet_weight=1.0)
    retriever.add([_item(f"D1:{i}", f"Ann: filler {i}.") for i in range(1, 6)])
    retriever._facet_index = {("Ann", "pet"): {"D1:5"}}  # D1:5 never hit by the query itself
    retriever._qfacets = {"convX:q0": [("Ann", "pet")]}
    retriever._memory.search = lambda qtext, **kw: [_FakeSearchHit("D1:1"), _FakeSearchHit("D1:2")]
    result = retriever.search("original", top_k=10, qid="convX:q0")
    assert "D1:5" in result  # facet-only candidate, injected purely via the facet channel


def test_facet_channel_weight_zero_reproduces_baseline_ranking_exactly():
    canned = {"original": [_FakeSearchHit("o1"), _FakeSearchHit("o2")]}

    baseline = _make_retriever()
    baseline.add([_item("D1:1", "Ann: hi.")])
    baseline._memory.search = lambda qtext, **kw: canned[qtext]
    expected = baseline.search("original", top_k=10, qid="convX:q0")

    zero_weight = _make_retriever(facet_channel=True, facet_weight=0.0)
    zero_weight.add([_item("D1:1", "Ann: hi.")])
    zero_weight._facet_index = {("Ann", "pet"): {"D1:99"}}  # would otherwise inject a NEW id
    zero_weight._qfacets = {"convX:q0": [("Ann", "pet")]}
    zero_weight._memory.search = lambda qtext, **kw: canned[qtext]
    got = zero_weight.search("original", top_k=10, qid="convX:q0")
    assert got == expected  # byte-for-byte, including length
    assert "D1:99" not in got


def test_facet_channel_exact_predicate_match_only_no_family_widening():
    # "job_career" and "achievement_award" are BOTH in the "work/school"
    # family (bench/data/enrich/PREDICATES.md) -- a family-widening
    # implementation would wrongly match; exact match must not.
    retriever = _make_retriever(facet_channel=True, facet_weight=1.0)
    retriever.add([_item("D1:1", "Ann: hi.")])
    retriever._facet_index = {("Ann", "job_career"): {"D1:9"}}
    retriever._qfacets = {"convX:q0": [("Ann", "achievement_award")]}
    retriever._memory.search = lambda qtext, **kw: [_FakeSearchHit("o1")]
    result = retriever.search("original", top_k=10, qid="convX:q0")
    assert "D1:9" not in result


def test_facet_channel_stats_pool_size_and_new_vs_top50():
    retriever = _make_retriever(facet_channel=True, facet_weight=1.0)
    retriever.add([_item("D1:1", "Ann: hi.")])
    retriever._facet_index = {("Ann", "pet"): {"D1:new1", "D1:new2"}}
    retriever._qfacets = {"convX:q0": [("Ann", "pet")]}
    retriever._memory.search = lambda qtext, **kw: [_FakeSearchHit("D1:already")]
    retriever.search("original", top_k=10, qid="convX:q0")
    assert retriever.facet_stats["n_searches"] == 1
    assert retriever.facet_stats["n_questions_with_facets"] == 1
    assert retriever.facet_stats["total_pool_size"] == 2
    assert retriever.facet_stats["total_new_vs_top50"] == 2  # neither pool member was in the pre-facet top-50


def test_facet_channel_rerank_top_n_applies_after_fusion_sees_injected_candidate():
    class _FakeReranker:
        def __init__(self):
            self.calls: list[list[str]] = []

        def available(self):
            return True

        def score(self, query, texts):
            self.calls.append(list(texts))
            return [float(i) for i in range(len(texts))]

    retriever = _make_retriever(facet_channel=True, facet_weight=1.0, rerank_top_n=10)
    retriever.add([_item("D1:1", "Ann: hi."), _item("D1:2", "Bob: yo.")])
    retriever._facet_index = {("Ann", "pet"): {"D1:2"}}  # "D1:2" only reachable via the facet channel
    retriever._qfacets = {"convX:q0": [("Ann", "pet")]}
    retriever._memory.search = lambda qtext, **kw: [_FakeSearchHit("D1:1")]
    fake = _FakeReranker()
    retriever._reranker = fake
    retriever.search("original", top_k=10, qid="convX:q0")
    assert fake.calls  # reranker actually invoked
    assert any("Bob" in t for t in fake.calls[0])  # facet-injected D1:2's text reached the reranker


# ---------------------------------------------------------------------------
# V4PlannedRetriever.add(): wiring from real facets/qfacets files
# ---------------------------------------------------------------------------
def test_add_builds_facet_index_and_qfacets_from_files(tmp_path):
    facets_dir = tmp_path / "facets"
    qfacets_dir = tmp_path / "qfacets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            json.dumps(
                {
                    "fid": "D1:1#0",
                    "facets": [{"subject": "Ann", "predicate": "pet", "object": "dog"}],
                    "implications": [],
                }
            ),
            json.dumps(
                {
                    "fid": "D1:2#0",
                    "facets": [{"subject": "Ann", "predicate": "other", "object": "x"}],
                    "implications": [],
                }
            ),
        ],
    )
    _write_jsonl(
        qfacets_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "facets": [{"subject": "Ann", "predicate": "pet"}]})],
    )
    retriever = _make_retriever(facets_dir=facets_dir, qfacets_dir=qfacets_dir, facet_channel=True)
    retriever.add([_item("D1:1", "Ann: I have a dog."), _item("D1:2", "Ann: something else.")])
    assert retriever._facet_index == {("Ann", "pet"): {"D1:1"}}  # "other" excluded
    assert retriever._qfacets == {"convX:q0": [("Ann", "pet")]}


def test_add_rebuilds_facet_index_fresh_per_conversation_no_leak(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "conv-a.jsonl",
        [
            json.dumps(
                {"fid": "D1:1#0", "facets": [{"subject": "Ann", "predicate": "pet"}], "implications": []}
            )
        ],
    )
    _write_jsonl(
        facets_dir / "conv-b.jsonl",
        [
            json.dumps(
                {"fid": "D9:9#0", "facets": [{"subject": "Ann", "predicate": "pet"}], "implications": []}
            )
        ],
    )
    retriever = _make_retriever(facets_dir=facets_dir, facet_channel=True)
    retriever.add([_item("D1:1", "Ann: hi.", conv_id="conv-a")])
    assert retriever._facet_index == {("Ann", "pet"): {"D1:1"}}

    retriever.reset()
    retriever.add([_item("D9:9", "Ann: yo.", conv_id="conv-b")])
    assert retriever._facet_index == {("Ann", "pet"): {"D9:9"}}  # conv-a's D1:1 is gone


# ---------------------------------------------------------------------------
# --implications: indexed via add_facts, credited to parent turn
# ---------------------------------------------------------------------------
def test_implications_indexed_and_credited_to_parent_turn(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            json.dumps(
                {
                    "fid": "D1:1#0",
                    "facets": [{"subject": "Ann", "predicate": "pet"}],
                    "implications": ["Ann probably loves animals a lot, zzyzxquibble."],
                }
            )
        ],
    )
    retriever = _make_retriever(facets_dir=facets_dir, implications=True)
    retriever.add([_item("D1:1", "Ann: I have a dog."), _item("D1:2", "Bob: I like swimming.")])
    assert retriever.facet_stats["n_implications_indexed"] == 1

    # top_k=1: the raw turn text never contains "zzyzxquibble" -- only a hit
    # on the indexed implication, mapped back to its parent turn, can win.
    result = retriever.search("zzyzxquibble", top_k=1, qid="convX:q0")
    assert result == ["D1:1"]


def test_implications_independent_of_facet_channel_flag(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            json.dumps(
                {"fid": "D1:1#0", "facets": [{"subject": "Ann", "predicate": "pet"}], "implications": ["one"]}
            )
        ],
    )
    retriever = _make_retriever(facets_dir=facets_dir, implications=True, facet_channel=False)
    retriever.add([_item("D1:1", "Ann: hi.")])
    assert retriever.facet_stats["n_implications_indexed"] == 1
    assert retriever._facet_index == {}  # facet_channel never built the index


def test_implications_no_op_when_fact_has_no_implications(tmp_path):
    facets_dir = tmp_path / "facets"
    _write_jsonl(
        facets_dir / "convX.jsonl",
        [
            json.dumps(
                {"fid": "D1:1#0", "facets": [{"subject": "Ann", "predicate": "pet"}], "implications": []}
            )
        ],
    )
    retriever = _make_retriever(facets_dir=facets_dir, implications=True)
    retriever.add([_item("D1:1", "Ann: hi.")])
    assert retriever.facet_stats["n_implications_indexed"] == 0


# ---------------------------------------------------------------------------
# run_bench end-to-end: metadata["facets"] shape
# ---------------------------------------------------------------------------
def test_run_bench_facet_channel_and_implications_metadata(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_facets.json")
    facets_dir = tmp_path / "facets"
    qfacets_dir = tmp_path / "qfacets"
    _write_jsonl(
        facets_dir / "convA.jsonl",
        [
            json.dumps(
                {
                    "fid": "D1:1#0",
                    "facets": [{"subject": "Ann", "predicate": "pet", "object": "dog"}],
                    "implications": ["Ann is an animal lover."],
                }
            )
        ],
    )
    _write_jsonl(
        qfacets_dir / "convA.jsonl",
        [json.dumps({"qid": "convA:q0", "facets": [{"subject": "Ann", "predicate": "pet"}]})],
    )

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_planned",
        name="synthetic_v5_facets",
        top_k=5,
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        facet_channel=True,
        facet_weight=0.5,
        facets_dir=str(facets_dir),
        qfacets_dir=str(qfacets_dir),
        implications=True,
        allow_partial_plan=True,
        allow_partial_enrich=True,
    )
    facets_meta = result["metadata"]["facets"]
    assert facets_meta["facet_channel"] is True
    assert facets_meta["facet_weight"] == 0.5
    assert facets_meta["implications"] is True
    assert facets_meta["n_questions"] == 1
    assert facets_meta["n_questions_with_facets"] == 1
    assert facets_meta["fraction_questions_with_facets"] == 1.0
    assert facets_meta["mean_pool_size"] >= 0.0
    assert facets_meta["mean_new_vs_fused_top50"] >= 0.0
    assert facets_meta["n_implications_indexed"] == 1


def test_run_bench_facets_metadata_none_for_non_planned_retriever(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_bm25.json")
    result = run_bench(
        dataset="locomo",
        retriever_key="bm25",
        name="synthetic_bm25_no_facets",
        top_k=5,
        locomo_path=str(locomo_path),
    )
    assert result["metadata"]["facets"] is None
