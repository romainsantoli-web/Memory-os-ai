"""Tests for P9 (PLAN.md) query planning, fact links, and reranking:
`bench.enrich.load_query_plans`/`QueryPlan` and
`bench.retrievers.{rrf_fuse, _rank_by_score, V4PlannedRetriever}`.

Everything here uses `Config(embed_model="hash")` (never a real model load)
and writes only into pytest `tmp_path` dirs -- the real
`bench/data/enrich/locomo_queries` (being generated concurrently by other
workers, per the task brief) is never read or depended on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.enrich import (
    CandidateDumpRecord,
    EnrichRecord,
    QueryPlan,
    build_candidate_dump_line,
    load_candidate_dump,
    load_llm_rankings,
    load_query_plans,
)
from bench.protocol import BenchItem
from bench.retrievers import (
    V4LLMRerankedRetriever,
    V4PlannedRetriever,
    _compute_owner_map,
    _is_question_turn,
    _rank_by_score,
    apply_llm_ranking,
    quota_fuse,
    quota_then_rrf_fuse,
    rrf_fuse,
    signal_confidence,
    subquery_mmr_reorder,
)
from bench.run_bench import run_bench, run_bench_from_dump


def _write_jsonl(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _item(id_: str, text: str, speaker: str = "s", session_id: str = "s1", conv_id: str = "convX") -> BenchItem:
    return BenchItem(id=id_, text=text, speaker=speaker, event_time=None, session_id=session_id, meta={"conv_id": conv_id})


def _make_retriever(enrich_dir=None, queries_dir=None, **kwargs) -> V4PlannedRetriever:
    r = V4PlannedRetriever(
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir) if enrich_dir is not None else None,
        queries_dir=str(queries_dir) if queries_dir is not None else None,
        **kwargs,
    )
    r.reset()
    return r


class _FakeReranker:
    """Drop-in for CrossEncoderReranker (see tests/test_core_retrieval.py's
    own `_FakeReranker`): records the exact texts it was asked to score,
    returns fixed scores keyed by text."""

    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def available(self) -> bool:
        return True

    def score(self, query: str, texts: list[str]):
        self.calls.append((query, list(texts)))
        return [self.scores_by_text.get(t, 0.0) for t in texts]


class _FakeItem:
    def __init__(self, id_: str, kind: str = "episode", parent_ids=None):
        self.id = id_
        self.kind = kind
        self.parent_ids = parent_ids or []


class _FakeSearchHit:
    """Drop-in for `memory_os_v4.retrieval.SearchHit`: just `.item`
    (id/kind/parent_ids) and `.score`, enough for
    `bench.retrievers._map_hits_to_turn_ids`/`signal_confidence` and for
    monkeypatching `retriever._memory.search` with fully-canned results in
    the idées-inter-disciplines integration tests below."""

    def __init__(self, id_: str, score: float = 1.0, kind: str = "episode", parent_ids=None):
        self.score = score
        self.item = _FakeItem(id_, kind=kind, parent_ids=parent_ids)


# ---------------------------------------------------------------------------
# bench.enrich.load_query_plans: tolerance contract (mirrors load_enrichment)
# ---------------------------------------------------------------------------
def test_load_query_plans_missing_file_returns_empty_dict(tmp_path):
    assert load_query_plans(tmp_path / "does_not_exist", "convX") == {}


def test_load_query_plans_parses_valid_lines(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [
            json.dumps({"qid": "convX:q0", "subqueries": ["sq1", "sq2"], "hypotheses": ["hyp1"]}),
            json.dumps({"qid": "convX:q1", "subqueries": [], "hypotheses": []}),
        ],
    )
    plans = load_query_plans(queries_dir, "convX")
    assert set(plans.keys()) == {"convX:q0", "convX:q1"}
    assert plans["convX:q0"] == QueryPlan(subqueries=["sq1", "sq2"], hypotheses=["hyp1"])
    assert plans["convX:q1"] == QueryPlan(subqueries=[], hypotheses=[])


def test_load_query_plans_skips_malformed_lines(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [
            "{not valid json",
            json.dumps({"subqueries": ["missing qid"]}),
            json.dumps({"qid": "", "subqueries": []}),
            json.dumps({"qid": "convX:q9", "subqueries": "not-a-list"}),
            json.dumps({"qid": "convX:q0", "subqueries": ["good"], "hypotheses": []}),
        ],
    )
    plans = load_query_plans(queries_dir, "convX")
    assert plans == {"convX:q0": QueryPlan(subqueries=["good"], hypotheses=[])}


def test_load_query_plans_duplicate_qid_last_wins(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [
            json.dumps({"qid": "convX:q0", "subqueries": ["first"], "hypotheses": []}),
            json.dumps({"qid": "convX:q0", "subqueries": ["final"], "hypotheses": ["h"]}),
        ],
    )
    plans = load_query_plans(queries_dir, "convX")
    assert plans["convX:q0"].subqueries == ["final"]
    assert plans["convX:q0"].hypotheses == ["h"]


# ---------------------------------------------------------------------------
# rrf_fuse / _rank_by_score
# ---------------------------------------------------------------------------
def test_rrf_fuse_basic_reciprocal_rank_scores():
    scores = rrf_fuse([(["a", "b", "c"], 1.0)], k=60)
    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["b"] == pytest.approx(1.0 / 62)
    assert scores["c"] == pytest.approx(1.0 / 63)


def test_rrf_fuse_sums_contributions_across_lists():
    # "b" appears at rank 0 in list 2 and rank 1 in list 1 -> higher fused
    # score than "a" (rank 0 in list 1 only) once list 2 contributes enough.
    scores = rrf_fuse([(["a", "b"], 1.0), (["b"], 1.0)], k=60)
    assert scores["b"] == pytest.approx(1.0 / 62 + 1.0 / 61)
    assert scores["a"] == pytest.approx(1.0 / 61)
    assert scores["b"] > scores["a"]


def test_rrf_fuse_weight_scales_contribution_and_can_flip_order():
    # Without weighting, "a" (rank 0 in list 1) would outrank "b" (rank 0 in
    # list 2); down-weighting list 1 to 0.1 flips it.
    scores = rrf_fuse([(["a"], 0.1), (["b"], 1.0)], k=60)
    assert scores["b"] > scores["a"]


def test_rrf_fuse_absent_id_contributes_nothing_not_a_penalty():
    scores = rrf_fuse([(["a"], 1.0), ([], 1.0)], k=60)
    assert scores["a"] == pytest.approx(1.0 / 61)


def test_rank_by_score_orders_descending_ties_by_id_ascending():
    ranked = _rank_by_score({"z": 1.0, "b": 2.0, "a": 2.0})
    assert ranked == ["a", "b", "z"]  # a,b tie at 2.0 -> id-ascending; z last


# ---------------------------------------------------------------------------
# V4PlannedRetriever: construction / plan_mode validation
# ---------------------------------------------------------------------------
def test_v4_planned_rejects_unknown_plan_mode(tmp_path):
    with pytest.raises(ValueError):
        V4PlannedRetriever(config_overrides={"embed_model": "hash"}, plan_mode="bogus")


# ---------------------------------------------------------------------------
# V4PlannedRetriever: plan lookup / fallback
# ---------------------------------------------------------------------------
def test_v4_planned_fallback_when_qid_missing_from_plans(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": ["sq"], "hypotheses": ["hyp"]})],
    )
    retriever = _make_retriever(queries_dir=queries_dir)
    items = [_item("D1:1", "Ann: I love hiking."), _item("D1:2", "Bob: nice.")]
    retriever.add(items)

    calls = []
    orig_search = retriever._memory.search

    def spy(query, **kwargs):
        calls.append((query, kwargs))
        return orig_search(query, **kwargs)

    retriever._memory.search = spy
    retriever.search("hiking", top_k=5, qid="convX:q999")  # no plan for this qid

    assert retriever.plan_stats["n_fallback_missing_plan"] == 1
    assert retriever.plan_stats["n_plan_used"] == 0
    assert len(calls) == 1  # only the original question was searched
    assert calls[0][0] == "hiking"
    assert calls[0][1]["rerank"] is False


def test_v4_planned_uses_plan_subqueries_and_hypotheses_when_present(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": ["sq1", "sq2"], "hypotheses": ["hyp1"]})],
    )
    retriever = _make_retriever(queries_dir=queries_dir, plan_mode="sub+hyp")
    items = [_item("D1:1", "Ann: I love hiking."), _item("D1:2", "Bob: nice.")]
    retriever.add(items)

    calls = []
    orig_search = retriever._memory.search

    def spy(query, **kwargs):
        calls.append(query)
        return orig_search(query, **kwargs)

    retriever._memory.search = spy
    retriever.search("original question", top_k=5, qid="convX:q0")

    assert retriever.plan_stats["n_plan_used"] == 1
    assert retriever.plan_stats["n_fallback_missing_plan"] == 0
    assert calls == ["original question", "sq1", "sq2", "hyp1"]


def test_v4_planned_plan_mode_sub_excludes_hypotheses(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": ["sq1"], "hypotheses": ["hyp1"]})],
    )
    retriever = _make_retriever(queries_dir=queries_dir, plan_mode="sub")
    retriever.add([_item("D1:1", "Ann: hi.")])
    calls = []
    orig_search = retriever._memory.search
    retriever._memory.search = lambda q, **kw: (calls.append(q), orig_search(q, **kw))[1]
    retriever.search("q", top_k=5, qid="convX:q0")
    assert calls == ["q", "sq1"]


def test_v4_planned_plan_mode_hyp_excludes_subqueries(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": ["sq1"], "hypotheses": ["hyp1"]})],
    )
    retriever = _make_retriever(queries_dir=queries_dir, plan_mode="hyp")
    retriever.add([_item("D1:1", "Ann: hi.")])
    calls = []
    orig_search = retriever._memory.search
    retriever._memory.search = lambda q, **kw: (calls.append(q), orig_search(q, **kw))[1]
    retriever.search("q", top_k=5, qid="convX:q0")
    assert calls == ["q", "hyp1"]


def test_v4_planned_no_queries_dir_always_falls_back(tmp_path):
    retriever = _make_retriever(queries_dir=None)
    retriever.add([_item("D1:1", "Ann: hi.")])
    retriever.search("q", top_k=5, qid="convX:q0")
    assert retriever.plan_stats["n_fallback_missing_plan"] == 1


# ---------------------------------------------------------------------------
# V4PlannedRetriever: search() fuses via RRF and maps fact hits to turns
# ---------------------------------------------------------------------------
def test_v4_planned_search_maps_fact_hit_to_parent_turn(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [json.dumps({"turn_id": "D1:1", "facts": ["Ann's favorite word is zzyzxquibble."], "keywords": [], "entities": []})],
    )
    retriever = _make_retriever(enrich_dir=enrich_dir)
    retriever.add([_item("D1:1", "Ann: I love hiking."), _item("D1:2", "Bob: swimming.")])
    # top_k=1 (not 5): with only 2 turns total in this tiny index, a top_k
    # bigger than the corpus would return both regardless of relevance --
    # top_k=1 isolates the actual claim (the fact hit maps to its parent
    # turn, ranked first), matching V4EnrichedRetriever's analogous test.
    result = retriever.search("zzyzxquibble", top_k=1, qid="convX:q0")
    assert result == ["D1:1"]  # the raw turn text never contains this word


# ---------------------------------------------------------------------------
# V4PlannedRetriever: §B fact link graph construction
# ---------------------------------------------------------------------------
def _filler_items(n, session_id="session_1", conv_id="convX"):
    return [_item(f"D1:{i}", f"Ann: filler turn number {i}.", session_id=session_id, conv_id=conv_id) for i in range(n)]


def test_build_link_graph_shared_rare_entity_links_facts():
    # 40 turns: "zzyzx" appears in exactly turns 0 and 39 -> df=2/40=5% (rare).
    items = _filler_items(40)
    enrichment = {
        "D1:0": EnrichRecord(facts=["fact zero"], keywords=[], entities=["Zzyzx"]),
        "D1:39": EnrichRecord(facts=["fact last"], keywords=[], entities=["Zzyzx"]),
    }
    retriever = _make_retriever()
    retriever._build_fact_index(items, enrichment)
    retriever._build_link_graph(items, enrichment)
    assert "D1:39#f0" in retriever._fact_links_graph["D1:0#f0"]
    assert "D1:0#f0" in retriever._fact_links_graph["D1:39#f0"]


def test_build_link_graph_common_entity_not_rare_enough_to_link():
    # "common" appears in 10/40=25% of turns -> NOT rare (>5%), so two facts
    # far apart in different sessions sharing ONLY "common" must not link.
    items = _filler_items(40)
    enrichment = {}
    for i in range(10):
        enrichment[f"D1:{i}"] = EnrichRecord(facts=[f"fact {i}"], keywords=[], entities=["Common"])
    retriever = _make_retriever()
    retriever._build_fact_index(items, enrichment)
    retriever._build_link_graph(items, enrichment)
    # D1:0 and D1:9 are >2 turns apart (same session doesn't save them) and
    # share only the non-rare "common" entity -> not linked.
    assert "D1:9#f0" not in retriever._fact_links_graph.get("D1:0#f0", set())


def test_build_link_graph_same_session_within_2_turns_links_even_without_shared_entity():
    items = _filler_items(10)
    enrichment = {
        "D1:2": EnrichRecord(facts=["fact a"], keywords=[], entities=[]),
        "D1:4": EnrichRecord(facts=["fact b"], keywords=[], entities=[]),  # gap = 2
    }
    retriever = _make_retriever()
    retriever._build_fact_index(items, enrichment)
    retriever._build_link_graph(items, enrichment)
    assert "D1:4#f0" in retriever._fact_links_graph["D1:2#f0"]


def test_build_link_graph_same_session_beyond_gap_not_linked():
    items = _filler_items(10)
    enrichment = {
        "D1:2": EnrichRecord(facts=["fact a"], keywords=[], entities=[]),
        "D1:6": EnrichRecord(facts=["fact b"], keywords=[], entities=[]),  # gap = 4 > 2
    }
    retriever = _make_retriever()
    retriever._build_fact_index(items, enrichment)
    retriever._build_link_graph(items, enrichment)
    assert "D1:6#f0" not in retriever._fact_links_graph.get("D1:2#f0", set())


def test_build_link_graph_different_session_not_linked_by_adjacency():
    items = _filler_items(5, session_id="session_1") + _filler_items(5, session_id="session_2")
    # NOTE: _filler_items always starts ids at D1:0 -- rename second batch's
    # ids to avoid collision.
    for idx, it in enumerate(items[5:]):
        it.id = f"D2:{idx}"
    enrichment = {
        "D1:4": EnrichRecord(facts=["fact a"], keywords=[], entities=[]),
        "D2:0": EnrichRecord(facts=["fact b"], keywords=[], entities=[]),
    }
    retriever = _make_retriever()
    retriever._build_fact_index(items, enrichment)
    retriever._build_link_graph(items, enrichment)
    assert "D2:0#f0" not in retriever._fact_links_graph.get("D1:4#f0", set())


def test_build_link_graph_speaker_name_excluded_from_rare_entities():
    # "Ann" is a speaker in this conversation; even at rare df it must never
    # be treated as a linking entity.
    items = _filler_items(40, session_id="session_1")
    for it in items:
        it.speaker = "Ann" if int(it.id.split(":")[1]) % 2 == 0 else "Bob"
    enrichment = {
        "D1:0": EnrichRecord(facts=["fact zero"], keywords=[], entities=["Ann"]),
        "D1:38": EnrichRecord(facts=["fact far"], keywords=[], entities=["Ann"]),
    }
    retriever = _make_retriever()
    retriever._build_fact_index(items, enrichment)
    retriever._build_link_graph(items, enrichment)
    assert "D1:38#f0" not in retriever._fact_links_graph.get("D1:0#f0", set())


def test_build_link_graph_entity_shorter_than_min_len_excluded():
    items = _filler_items(40)
    enrichment = {
        "D1:0": EnrichRecord(facts=["fact zero"], keywords=[], entities=["Ab"]),  # 2 chars < min 3
        "D1:39": EnrichRecord(facts=["fact far"], keywords=[], entities=["Ab"]),
    }
    retriever = _make_retriever()
    retriever._build_fact_index(items, enrichment)
    retriever._build_link_graph(items, enrichment)
    assert "D1:39#f0" not in retriever._fact_links_graph.get("D1:0#f0", set())


# ---------------------------------------------------------------------------
# V4PlannedRetriever: §B link expansion adds decayed-score turns
# ---------------------------------------------------------------------------
def test_expand_fact_links_adds_linked_turn_with_decayed_score():
    retriever = _make_retriever(link_seed=8, link_decay=0.5)
    retriever._facts_by_turn = {"seed_turn": ["seed_turn#f0"], "linked_turn": ["linked_turn#f0"]}
    retriever._fact_parent_turn = {"seed_turn#f0": "seed_turn", "linked_turn#f0": "linked_turn"}
    retriever._fact_links_graph = {"seed_turn#f0": {"linked_turn#f0"}}

    ranked = ["seed_turn"]
    scores = {"seed_turn": 0.8}
    new_ranked, new_scores = retriever._expand_fact_links(ranked, scores)

    assert "linked_turn" in new_ranked
    assert new_scores["linked_turn"] == pytest.approx(0.8 * 0.5)
    assert new_scores["seed_turn"] == 0.8  # untouched


def test_expand_fact_links_never_overrides_an_already_scored_turn():
    retriever = _make_retriever(link_decay=0.5)
    retriever._facts_by_turn = {"seed_turn": ["seed_turn#f0"], "other_turn": ["other_turn#f0"]}
    retriever._fact_parent_turn = {"seed_turn#f0": "seed_turn", "other_turn#f0": "other_turn"}
    retriever._fact_links_graph = {"seed_turn#f0": {"other_turn#f0"}}

    ranked = ["seed_turn", "other_turn"]
    scores = {"seed_turn": 0.9, "other_turn": 0.1}  # already present with its OWN fused score
    new_ranked, new_scores = retriever._expand_fact_links(ranked, scores)
    assert new_scores["other_turn"] == 0.1  # not overwritten by 0.9*0.5=0.45
    assert new_ranked == ranked  # nothing added, order unchanged


def test_expand_fact_links_no_op_when_seed_has_no_facts():
    retriever = _make_retriever()
    retriever._facts_by_turn = {}
    retriever._fact_links_graph = {}
    ranked = ["seed_turn"]
    scores = {"seed_turn": 0.5}
    new_ranked, new_scores = retriever._expand_fact_links(ranked, scores)
    assert new_ranked == ranked
    assert new_scores == scores


# ---------------------------------------------------------------------------
# V4PlannedRetriever: §C reranker -- blend vs cross-encoder-only
# ---------------------------------------------------------------------------
def test_apply_rerank_blend_path_uses_symmetric_weight(tmp_path):
    retriever = _make_retriever(plan_rerank_blend=True, rerank_top_n=10)
    retriever.add(
        [
            _item("a", "Kraken exchange API rate limit.", speaker=""),
            _item("b", "A different note about deposits.", speaker=""),
        ]
    )
    fake = _FakeReranker({"Kraken exchange API rate limit.": -5.0, "A different note about deposits.": 5.0})
    retriever._reranker = fake
    ranked = ["a", "b"]
    scores = {"a": 1.0, "b": 0.0}  # pre-rerank fused favors "a"
    result = retriever._apply_rerank("query", ranked, scores)
    assert fake.calls  # reranker actually invoked
    query, texts = fake.calls[0]
    assert query == "query"
    # blend: base_norm favors a (1.0/0.0), ce_norm favors b (0.0/1.0) -> 0.5/0.5 tie -> id order
    assert result == ["a", "b"]


def test_apply_rerank_ce_only_path_ignores_fused_score():
    retriever = _make_retriever(plan_rerank_blend=False, rerank_top_n=10)
    retriever.add([_item("a", "aaa", speaker=""), _item("b", "bbb", speaker="")])
    fake = _FakeReranker({"aaa": 0.0, "bbb": 1.0})
    retriever._reranker = fake
    ranked = ["a", "b"]
    scores = {"a": 10.0, "b": 0.0}  # fused score strongly favors "a"
    result = retriever._apply_rerank("q", ranked, scores)
    assert result == ["b", "a"]  # cross-encoder score alone decides -> b wins


def test_apply_rerank_keeps_ids_beyond_rerank_top_n_untouched():
    retriever = _make_retriever(plan_rerank_blend=False, rerank_top_n=1)
    retriever.add([_item(i, i, speaker="") for i in ("a", "b", "c")])
    fake = _FakeReranker({"a": 1.0})
    retriever._reranker = fake
    result = retriever._apply_rerank("q", ["a", "b", "c"], {"a": 1.0, "b": 0.5, "c": 0.1})
    assert result == ["a", "b", "c"]  # only "a" (top_n=1) is rescored; rest untouched
    assert fake.calls[0][1] == ["a"]


def test_apply_rerank_returns_unchanged_when_no_reranker_configured():
    retriever = _make_retriever()
    assert retriever._reranker is None
    assert retriever._apply_rerank("q", ["a", "b"], {"a": 1.0, "b": 0.5}) == ["a", "b"]


# ---------------------------------------------------------------------------
# run_bench end-to-end with --retriever v4_planned (synthetic, tmp dirs)
# ---------------------------------------------------------------------------
def _synthetic_locomo_1conv(path):
    data = [
        {
            "sample_id": "convA",
            "conversation": {
                "speaker_a": "Ann",
                "speaker_b": "Bob",
                "session_1_date_time": "1:00 pm on 1 January, 2023",
                "session_1": [
                    {"speaker": "Ann", "dia_id": "D1:1", "text": "I love hiking in the mountains."},
                    {"speaker": "Bob", "dia_id": "D1:2", "text": "That sounds fun, I prefer swimming."},
                ],
            },
            "qa": [
                {"question": "What does Ann love?", "answer": "hiking", "evidence": ["D1:1"], "category": 4},
            ],
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_full_enrich_and_queries(enrich_dir, queries_dir):
    _write_jsonl(
        enrich_dir / "convA.jsonl",
        [
            json.dumps({"turn_id": "D1:1", "facts": ["Ann loves hiking."], "keywords": ["hiking"], "entities": ["Ann"]}),
            json.dumps({"turn_id": "D1:2", "facts": [], "keywords": [], "entities": ["Bob"]}),
        ],
    )
    _write_jsonl(
        queries_dir / "convA.jsonl",
        [json.dumps({"qid": "convA:q0", "subqueries": ["Ann hiking mountains"], "hypotheses": ["I love hiking."]})],
    )


def test_run_bench_v4_planned_end_to_end(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo.json")
    enrich_dir = tmp_path / "enrich"
    queries_dir = tmp_path / "queries"
    _write_full_enrich_and_queries(enrich_dir, queries_dir)

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_planned",
        name="synthetic_v4_planned",
        top_k=5,
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        enrich_mode="facts+keywords",
        queries_dir=str(queries_dir),
        plan_mode="sub+hyp",
    )

    assert result["retriever"] == "v4_planned"
    assert result["n_conversations"] == 1
    assert result["n_questions"] == 1
    planning_meta = result["metadata"]["planning"]
    assert planning_meta["n_plan_used"] == 1
    assert planning_meta["n_fallback_missing_plan"] == 0
    assert planning_meta["fraction_questions_with_plan"] == 1.0
    assert result["metadata"]["enrichment"]["facts_added"] == 1


def test_run_bench_v4_planned_fails_fast_on_missing_queries_dir(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo2.json")
    enrich_dir = tmp_path / "enrich"
    queries_dir = tmp_path / "empty_queries"
    queries_dir.mkdir()
    _write_jsonl(enrich_dir / "convA.jsonl", [json.dumps({"turn_id": "D1:1", "facts": [], "keywords": [], "entities": []})])

    with pytest.raises(RuntimeError, match="convA"):
        run_bench(
            dataset="locomo",
            retriever_key="v4_planned",
            name="should_fail",
            locomo_path=str(locomo_path),
            config_overrides={"embed_model": "hash"},
            enrich_dir=str(enrich_dir),
            queries_dir=str(queries_dir),
        )


def test_run_bench_v4_planned_allow_partial_plan_falls_back(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo3.json")
    enrich_dir = tmp_path / "enrich"
    queries_dir = tmp_path / "empty_queries"
    queries_dir.mkdir()
    _write_jsonl(enrich_dir / "convA.jsonl", [json.dumps({"turn_id": "D1:1", "facts": [], "keywords": [], "entities": []})])

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_planned",
        name="partial_plan_ok",
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        queries_dir=str(queries_dir),
        allow_partial_plan=True,
    )
    assert result["n_questions"] == 1
    planning_meta = result["metadata"]["planning"]
    assert planning_meta["n_fallback_missing_plan"] == 1
    assert planning_meta["n_plan_used"] == 0


# ---------------------------------------------------------------------------
# P9 step D: bench.enrich.build_candidate_dump_line / load_candidate_dump /
# load_llm_rankings, bench.retrievers.apply_llm_ranking/V4LLMRerankedRetriever,
# bench.run_bench --dump-candidates / run_bench_from_dump.
# ---------------------------------------------------------------------------
def test_build_candidate_dump_line_fields_labels_and_truncation():
    long_text = "x" * 400
    long_fact = "y" * 200
    line = build_candidate_dump_line(
        qid="convX:q0",
        question="What did Ann do?",
        ranked_turn_ids=["D1:1", "D1:2", "D1:3"],
        text_by_id={"D1:1": long_text, "D1:2": "short", "D1:3": "third"},
        date_by_id={"D1:1": "2023-01-01", "D1:2": None, "D1:3": "2023-01-02"},
        speaker_by_id={"D1:1": "Ann", "D1:2": "Bob", "D1:3": "Ann"},
        facts_by_id={"D1:1": [long_fact, "fact2", "fact3 (should be dropped, max 2)"]},
    )
    assert line["qid"] == "convX:q0"
    assert line["question"] == "What did Ann do?"
    assert [c["label"] for c in line["candidates"]] == ["c01", "c02", "c03"]
    assert line["label_map"] == {"c01": "D1:1", "c02": "D1:2", "c03": "D1:3"}

    c1 = line["candidates"][0]
    assert set(c1.keys()) == {"label", "turn_id", "date", "speaker", "text", "facts"}
    assert c1["turn_id"] == "D1:1"
    assert c1["date"] == "2023-01-01"
    assert c1["speaker"] == "Ann"
    assert len(c1["text"]) == 320  # truncated from 400
    assert len(c1["facts"]) == 2  # truncated from 3 (max 2 facts/turn)
    assert len(c1["facts"][0]) == 160  # truncated from 200

    c2 = line["candidates"][1]
    assert c2["date"] is None
    assert c2["facts"] == []


def test_build_candidate_dump_line_caps_at_20_candidates():
    ranked = [f"D1:{i}" for i in range(25)]
    line = build_candidate_dump_line(
        qid="convX:q0",
        question="q",
        ranked_turn_ids=ranked,
        text_by_id={tid: "t" for tid in ranked},
        date_by_id={tid: None for tid in ranked},
        speaker_by_id={tid: "Ann" for tid in ranked},
        facts_by_id={},
    )
    assert len(line["candidates"]) == 20
    assert line["candidates"][-1]["label"] == "c20"
    assert line["candidates"][-1]["turn_id"] == "D1:19"


def test_build_candidate_dump_line_fewer_than_20_turns():
    line = build_candidate_dump_line(
        qid="convX:q0",
        question="q",
        ranked_turn_ids=["D1:1", "D1:2"],
        text_by_id={"D1:1": "a", "D1:2": "b"},
        date_by_id={"D1:1": None, "D1:2": None},
        speaker_by_id={"D1:1": "Ann", "D1:2": "Bob"},
        facts_by_id={},
    )
    assert len(line["candidates"]) == 2


def test_candidate_dump_record_defaults():
    record = CandidateDumpRecord(qid="convX:q0")
    assert record.question == ""
    assert record.turn_ids == []
    assert record.label_map == {}


def test_load_candidate_dump_reads_label_map_and_turn_ids(tmp_path):
    dump_dir = tmp_path / "dump"
    line = build_candidate_dump_line(
        qid="convX:q0",
        question="q",
        ranked_turn_ids=["D1:2", "D1:1"],
        text_by_id={"D1:1": "a", "D1:2": "b"},
        date_by_id={"D1:1": None, "D1:2": None},
        speaker_by_id={"D1:1": "Ann", "D1:2": "Bob"},
        facts_by_id={},
    )
    _write_jsonl(dump_dir / "convX.jsonl", [json.dumps(line)])

    records = load_candidate_dump(dump_dir, "convX")
    assert set(records.keys()) == {"convX:q0"}
    record = records["convX:q0"]
    assert record.turn_ids == ["D1:2", "D1:1"]
    assert record.label_map == {"c01": "D1:2", "c02": "D1:1"}
    assert record.question == "q"


def test_load_candidate_dump_rebuilds_label_map_when_top_level_missing(tmp_path):
    dump_dir = tmp_path / "dump"
    # A hand-written line predating the top-level "label_map" field.
    raw = {
        "qid": "convX:q0",
        "question": "q",
        "candidates": [
            {"label": "c01", "turn_id": "D1:5", "date": None, "speaker": "Ann", "text": "t", "facts": []},
        ],
    }
    _write_jsonl(dump_dir / "convX.jsonl", [json.dumps(raw)])
    records = load_candidate_dump(dump_dir, "convX")
    assert records["convX:q0"].label_map == {"c01": "D1:5"}
    assert records["convX:q0"].turn_ids == ["D1:5"]


def test_load_candidate_dump_missing_file_returns_empty_dict(tmp_path):
    assert load_candidate_dump(tmp_path / "does_not_exist", "convX") == {}


def test_load_candidate_dump_skips_malformed_lines(tmp_path):
    dump_dir = tmp_path / "dump"
    good = build_candidate_dump_line(
        qid="convX:q1",
        question="q",
        ranked_turn_ids=["D1:1"],
        text_by_id={"D1:1": "a"},
        date_by_id={"D1:1": None},
        speaker_by_id={"D1:1": "Ann"},
        facts_by_id={},
    )
    _write_jsonl(
        dump_dir / "convX.jsonl",
        [
            "{not valid json",
            json.dumps({"question": "missing qid", "candidates": []}),
            json.dumps({"qid": "convX:q0", "candidates": "not-a-list"}),
            json.dumps(good),
        ],
    )
    records = load_candidate_dump(dump_dir, "convX")
    assert set(records.keys()) == {"convX:q1"}


def test_load_llm_rankings_parses_valid_lines(tmp_path):
    llmrank_dir = tmp_path / "llmrank"
    _write_jsonl(
        llmrank_dir / "convX.jsonl",
        [
            json.dumps({"qid": "convX:q0", "ranked": ["c07", "c02", "c01"]}),
            json.dumps({"qid": "convX:q1", "ranked": []}),
        ],
    )
    rankings = load_llm_rankings(llmrank_dir, "convX")
    assert rankings == {"convX:q0": ["c07", "c02", "c01"], "convX:q1": []}


def test_load_llm_rankings_missing_file_returns_empty_dict(tmp_path):
    assert load_llm_rankings(tmp_path / "does_not_exist", "convX") == {}


def test_load_llm_rankings_skips_malformed_lines_duplicate_qid_last_wins(tmp_path):
    llmrank_dir = tmp_path / "llmrank"
    _write_jsonl(
        llmrank_dir / "convX.jsonl",
        [
            "{not valid json",
            json.dumps({"ranked": ["c01"]}),  # missing qid
            json.dumps({"qid": "convX:q0", "ranked": "not-a-list"}),
            json.dumps({"qid": "convX:q0", "ranked": ["c01"]}),
            json.dumps({"qid": "convX:q0", "ranked": ["c02"]}),  # duplicate qid: last wins
        ],
    )
    rankings = load_llm_rankings(llmrank_dir, "convX")
    assert rankings == {"convX:q0": ["c02"]}


# ---------------------------------------------------------------------------
# bench.retrievers.apply_llm_ranking
# ---------------------------------------------------------------------------
def test_apply_llm_ranking_llm_ids_first_then_remaining_candidates():
    result = apply_llm_ranking(["a", "b", "c", "d"], ["c", "a"], top_k=10)
    assert result == ["c", "a", "b", "d"]


def test_apply_llm_ranking_dedups_llm_ids():
    result = apply_llm_ranking(["a", "b", "c"], ["b", "b", "a"], top_k=10)
    assert result == ["b", "a", "c"]


def test_apply_llm_ranking_cuts_to_top_k():
    result = apply_llm_ranking(["a", "b", "c", "d"], ["c"], top_k=2)
    assert result == ["c", "a"]


def test_apply_llm_ranking_fallback_when_no_llm_ranking():
    assert apply_llm_ranking(["a", "b", "c"], None, top_k=2) == ["a", "b"]
    assert apply_llm_ranking(["a", "b", "c"], [], top_k=2) == ["a", "b"]


# ---------------------------------------------------------------------------
# V4LLMRerankedRetriever
# ---------------------------------------------------------------------------
def _make_llmrank_retriever(enrich_dir=None, queries_dir=None, llmrank_dir=None, candidates_dir=None, **kwargs) -> V4LLMRerankedRetriever:
    r = V4LLMRerankedRetriever(
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir) if enrich_dir is not None else None,
        queries_dir=str(queries_dir) if queries_dir is not None else None,
        llmrank_dir=str(llmrank_dir) if llmrank_dir is not None else None,
        candidates_dir=str(candidates_dir) if candidates_dir is not None else None,
        **kwargs,
    )
    r.reset()
    return r


def test_v4_llmrank_falls_back_to_parent_order_without_llmrank_dir():
    retriever = _make_llmrank_retriever()
    items = [_item(f"D1:{i}", f"Ann: filler turn number {i}.") for i in range(5)]
    retriever.add(items)
    plain = V4PlannedRetriever(config_overrides={"embed_model": "hash"})
    plain.reset()
    plain.add(items)

    got = retriever.search("filler", top_k=3, qid="convX:q0")
    expected = plain.search("filler", top_k=3, qid="convX:q0")
    assert got == expected
    assert retriever.llm_stats == {"n_searches": 1, "llm_ranked_questions": 0, "n_fallback_no_llm_ranking": 1}


def test_v4_llmrank_applies_ranking_via_dump_label_map(tmp_path):
    items = [_item(f"D1:{i}", f"Ann: filler turn number {i}.") for i in range(5)]
    plain = V4PlannedRetriever(config_overrides={"embed_model": "hash"})
    plain.reset()
    plain.add(items)
    parent_order = plain.search("filler", top_k=20, qid="convX:q0")
    assert len(parent_order) >= 2

    dump_dir = tmp_path / "dump"
    line = build_candidate_dump_line(
        qid="convX:q0",
        question="filler",
        ranked_turn_ids=parent_order,
        text_by_id={it.id: it.text for it in items},
        date_by_id={it.id: None for it in items},
        speaker_by_id={it.id: it.speaker for it in items},
        facts_by_id={},
    )
    _write_jsonl(dump_dir / "convX.jsonl", [json.dumps(line)])

    # LLM promotes the LAST parent-order candidate to first place.
    last_label = line["candidates"][-1]["label"]
    llmrank_dir = tmp_path / "llmrank"
    _write_jsonl(llmrank_dir / "convX.jsonl", [json.dumps({"qid": "convX:q0", "ranked": [last_label]})])

    retriever = _make_llmrank_retriever(llmrank_dir=llmrank_dir, candidates_dir=dump_dir)
    retriever.add(items)
    result = retriever.search("filler", top_k=3, qid="convX:q0")

    assert result[0] == line["candidates"][-1]["turn_id"]
    assert retriever.llm_stats["llm_ranked_questions"] == 1
    assert retriever.llm_stats["n_fallback_no_llm_ranking"] == 0


def test_v4_llmrank_unknown_labels_fall_back_to_parent_order(tmp_path):
    items = [_item(f"D1:{i}", f"Ann: filler turn number {i}.") for i in range(3)]
    dump_dir = tmp_path / "dump"
    llmrank_dir = tmp_path / "llmrank"
    # Dump exists but the LLM ranking references labels not in its map.
    line = build_candidate_dump_line(
        qid="convX:q0",
        question="filler",
        ranked_turn_ids=["D1:0", "D1:1", "D1:2"],
        text_by_id={it.id: it.text for it in items},
        date_by_id={it.id: None for it in items},
        speaker_by_id={it.id: it.speaker for it in items},
        facts_by_id={},
    )
    _write_jsonl(dump_dir / "convX.jsonl", [json.dumps(line)])
    _write_jsonl(llmrank_dir / "convX.jsonl", [json.dumps({"qid": "convX:q0", "ranked": ["c99"]})])

    retriever = _make_llmrank_retriever(llmrank_dir=llmrank_dir, candidates_dir=dump_dir)
    retriever.add(items)
    result = retriever.search("filler", top_k=3, qid="convX:q0")
    plain = V4PlannedRetriever(config_overrides={"embed_model": "hash"})
    plain.reset()
    plain.add(items)
    assert result == plain.search("filler", top_k=3, qid="convX:q0")
    assert retriever.llm_stats["n_fallback_no_llm_ranking"] == 1


# ---------------------------------------------------------------------------
# run_bench --dump-candidates (P9 step D, Part 1)
# ---------------------------------------------------------------------------
def test_run_bench_dump_candidates_writes_expected_files_and_line_counts(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_dump.json")
    enrich_dir = tmp_path / "enrich"
    queries_dir = tmp_path / "queries"
    _write_full_enrich_and_queries(enrich_dir, queries_dir)
    dump_dir = tmp_path / "dump"

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_planned",
        name="dump_test",
        top_k=1,
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        enrich_mode="facts+keywords",
        queries_dir=str(queries_dir),
        dump_candidates_dir=str(dump_dir),
    )

    dump_meta = result["metadata"]["dump_candidates"]
    assert dump_meta["n_files"] == 1
    assert dump_meta["n_lines"] == 1  # _synthetic_locomo_1conv has exactly 1 question
    assert dump_meta["max_candidates"] == 20

    lines = (dump_dir / "convA.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    # _synthetic_locomo_1conv's convA has only 2 turns -> fewer than 20 candidates.
    assert len(obj["candidates"]) == 2
    for c in obj["candidates"]:
        assert set(c.keys()) == {"label", "turn_id", "date", "speaker", "text", "facts"}
    for forbidden in ("evidence_ids", "evidence", "answer", "category"):
        assert forbidden not in obj
        for c in obj["candidates"]:
            assert forbidden not in c

    # metrics unaffected by dumping: same top_k=1 metric shape as a plain run.
    assert result["n_questions"] == 1
    assert "r5" in result["per_question"][0]


# ---------------------------------------------------------------------------
# run_bench_from_dump (P9 step D, Part 2)
# ---------------------------------------------------------------------------
def _write_dump_from_run(dump_dir: Path, conv_id: str, qid: str, question: str, ranked_turn_ids, items) -> dict:
    line = build_candidate_dump_line(
        qid=qid,
        question=question,
        ranked_turn_ids=ranked_turn_ids,
        text_by_id={it.id: it.text for it in items},
        date_by_id={it.id: None for it in items},
        speaker_by_id={it.id: it.speaker for it in items},
        facts_by_id={},
    )
    _write_jsonl(dump_dir / f"{conv_id}.jsonl", [json.dumps(line)])
    return line


def test_run_bench_from_dump_reproduces_ceiling_without_llm_ranking(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_fd.json")
    items = [
        BenchItem(id="D1:1", text="Ann: I love hiking in the mountains.", speaker="Ann", event_time=None, session_id="session_1", meta={}),
        BenchItem(id="D1:2", text="Bob: That sounds fun, I prefer swimming.", speaker="Bob", event_time=None, session_id="session_1", meta={}),
    ]
    dump_dir = tmp_path / "dump"
    # Gold evidence for q0 is D1:1 (see _synthetic_locomo_1conv) -- dump it first.
    _write_dump_from_run(dump_dir, "convA", "convA:q0", "What does Ann love?", ["D1:1", "D1:2"], items)

    result = run_bench_from_dump(
        dataset="locomo",
        name="fromdump_ceiling",
        candidates_dir=str(dump_dir),
        top_k=5,
        locomo_path=str(locomo_path),
    )
    assert result["retriever"] == "v4_llmrank_from_dump"
    assert result["n_questions"] == 1
    row = result["per_question"][0]
    assert row["latency_ms"] == 0.0
    assert row["r5"] == 1.0
    assert row["mrr"] == 1.0
    assert result["metadata"]["from_dump"]["n_missing_dump"] == 0
    assert result["metadata"]["from_dump"]["n_llm_ranked"] == 0


def test_run_bench_from_dump_applies_llm_ranking(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_fd2.json")
    items = [
        BenchItem(id="D1:1", text="Ann: I love hiking in the mountains.", speaker="Ann", event_time=None, session_id="session_1", meta={}),
        BenchItem(id="D1:2", text="Bob: That sounds fun, I prefer swimming.", speaker="Bob", event_time=None, session_id="session_1", meta={}),
    ]
    dump_dir = tmp_path / "dump"
    # Dump order puts the gold (D1:1) SECOND -- MRR should be 0.5 without any LLM ranking.
    _write_dump_from_run(dump_dir, "convA", "convA:q0", "What does Ann love?", ["D1:2", "D1:1"], items)

    llmrank_dir = tmp_path / "llmrank"
    # LLM promotes D1:1 (label c02 in this dump) to first place.
    _write_jsonl(llmrank_dir / "convA.jsonl", [json.dumps({"qid": "convA:q0", "ranked": ["c02"]})])

    without_llm = run_bench_from_dump(
        dataset="locomo", name="fromdump_no_llm", candidates_dir=str(dump_dir), top_k=5, locomo_path=str(locomo_path)
    )
    assert without_llm["per_question"][0]["mrr"] == 0.5

    with_llm = run_bench_from_dump(
        dataset="locomo",
        name="fromdump_with_llm",
        candidates_dir=str(dump_dir),
        llmrank_dir=str(llmrank_dir),
        top_k=5,
        locomo_path=str(locomo_path),
    )
    assert with_llm["per_question"][0]["mrr"] == 1.0
    assert with_llm["metadata"]["from_dump"]["n_llm_ranked"] == 1


def test_run_bench_from_dump_missing_dump_record_counts_and_scores_zero(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_fd3.json")
    dump_dir = tmp_path / "empty_dump"
    dump_dir.mkdir()  # exists but has no convA.jsonl at all

    result = run_bench_from_dump(
        dataset="locomo", name="fromdump_missing", candidates_dir=str(dump_dir), top_k=5, locomo_path=str(locomo_path)
    )
    assert result["metadata"]["from_dump"]["n_missing_dump"] == 1
    row = result["per_question"][0]
    assert row["r5"] == 0.0
    assert row["mrr"] == 0.0
    assert row["latency_ms"] == 0.0


# ---------------------------------------------------------------------------
# Idées inter-disciplines (docs/IDEAS.md, 2026-09-14)
# ---------------------------------------------------------------------------

# ---- Idea 1: Sainte-Laguë quota fusion ----
def test_quota_fuse_sainte_lague_hand_example():
    # Hand-computed (see docs/ABLATIONS.md "Idées inter-disciplines"): 3
    # parties, weights 1.0/1.0/0.7, seats=5.
    # round1: quotients 1.0/1.0/0.7 -> tie A/B -> A (first in list order)
    # round2: A=1/3, B=1.0, C=0.7 -> B
    # round3: A=1/3, B=1/3, C=0.7 -> C
    # round4: A=1/3, B=1/3, C=0.7/3=0.233 -> tie A/B -> A
    # round5: A=1/5, B=1/3, C=0.233 -> B
    ranked_lists = [
        (["a1", "a2", "a3", "a4"], 1.0),
        (["b1", "b2"], 1.0),
        (["c1", "c2", "c3"], 0.7),
    ]
    assert quota_fuse(ranked_lists, seats=5) == ["a1", "b1", "c1", "a2", "b2"]


def test_quota_fuse_stops_when_all_parties_exhausted():
    ranked_lists = [(["x1"], 1.0), (["y1"], 1.0)]
    assert quota_fuse(ranked_lists, seats=10) == ["x1", "y1"]


def test_quota_fuse_excludes_zero_weight_party():
    ranked_lists = [(["a1"], 1.0), (["b1"], 0.0)]
    assert quota_fuse(ranked_lists, seats=2) == ["a1"]


def test_quota_fuse_never_double_selects_id_shared_by_two_parties():
    # "shared" appears in both parties' own lists -- once selected by A it
    # must never be re-selected (and re-counted) for B.
    ranked_lists = [(["shared", "a2"], 1.0), (["shared", "b2"], 1.0)]
    result = quota_fuse(ranked_lists, seats=3)
    assert result.count("shared") == 1
    assert set(result) == {"shared", "a2", "b2"}


def test_quota_then_rrf_fuse_quota_head_then_rrf_tail():
    ranked_lists = [(["a1", "a2"], 1.0), (["b1"], 1.0)]
    fused_scores = rrf_fuse(ranked_lists, k=60)
    result = quota_then_rrf_fuse(ranked_lists, fused_scores, quota_seats=2, total_seats=10)
    # quota(seats=2): round1 tie a1/b1 -> a1 (first party); round2 a=1/3,b=1.0 -> b1
    assert result[:2] == ["a1", "b1"]
    assert result[2:] == ["a2"]  # RRF tail: only id left, appended after the quota head


# ---- Idea 2: subquery MMR ----
def test_compute_owner_map_best_rank_wins():
    labeled = [("orig", ["x", "y"]), ("sub0", ["y", "x"])]
    owners = _compute_owner_map(labeled)
    assert owners["x"] == "orig"  # rank0 in orig vs rank1 in sub0
    assert owners["y"] == "sub0"  # rank0 in sub0 vs rank1 in orig


def test_compute_owner_map_tie_prefers_first_list():
    labeled = [("orig", ["x"]), ("sub0", ["x"])]
    assert _compute_owner_map(labeled)["x"] == "orig"


def test_subquery_mmr_reorder_penalizes_repeated_owner():
    ranked = ["a", "b", "c"]
    scores = {"a": 1.0, "b": 0.9, "c": 0.5}
    owner_of = {"a": "sub0", "b": "sub0", "c": "sub1"}
    order, _ = subquery_mmr_reorder(ranked, scores, owner_of, lam=0.5)
    # round1: a=1.0, b=0.9, c=0.5 -> a (owner sub0 n=0->1)
    # round2: b=0.9*0.5=0.45 (sub0 n=1), c=0.5 (sub1 n=0) -> c beats penalized b
    # round3: b remains
    assert order == ["a", "c", "b"]


def test_subquery_mmr_reorder_missing_owner_gets_singleton_group():
    # No entry in owner_of at all -> every id is its own singleton owner
    # group -> no penalty ever applies -> plain (score desc, id asc) order.
    ranked = ["a", "b", "c"]
    scores = {"a": 1.0, "b": 0.9, "c": 0.9}
    order, _ = subquery_mmr_reorder(ranked, scores, {}, lam=0.1)
    assert order == ["a", "b", "c"]


def test_subquery_mmr_reorder_tie_broken_by_higher_base_score():
    ranked = ["a", "b", "c"]
    scores = {"a": 2.0, "b": 1.0, "c": 0.5}
    owner_of = {"a": "g", "b": "g", "c": "h"}
    order, _ = subquery_mmr_reorder(ranked, scores, owner_of, lam=0.5)
    # round1: a=2.0 (max) -> select a, counts[g]=1
    # round2: b=1.0*0.5=0.5 (owner g, n=1); c=0.5*1=0.5 (owner h, n=0) -> TIE
    #   ties broken by (unpenalised) base score: b's base 1.0 > c's base 0.5 -> b wins
    assert order == ["a", "b", "c"]


def test_subquery_mmr_reorder_new_scores_reflect_penalised_value():
    ranked = ["a", "b"]
    scores = {"a": 1.0, "b": 1.0}
    owner_of = {"a": "g", "b": "g"}
    order, new_scores = subquery_mmr_reorder(ranked, scores, owner_of, lam=0.5)
    assert order == ["a", "b"]  # tie -> id ascending
    assert new_scores["a"] == pytest.approx(1.0)  # n=0 at selection
    assert new_scores["b"] == pytest.approx(0.5)  # n=1 (same owner "g") at selection


# ---- Idea 3: signal confidence ----
def test_signal_confidence_high_margin_clips_to_one():
    assert signal_confidence([10.0, 0.0]) == 1.0


def test_signal_confidence_flat_scores_floor_at_point_three():
    assert signal_confidence([1.0, 0.999]) == pytest.approx(0.3)


def test_signal_confidence_single_hit_gets_full_confidence():
    assert signal_confidence([5.0]) == 1.0


def test_signal_confidence_no_hits_gets_floor():
    assert signal_confidence([]) == 0.3


def test_signal_confidence_moderate_margin_is_the_relative_margin_itself():
    assert signal_confidence([1.0, 0.5]) == pytest.approx(0.5)


def test_signal_confidence_never_exceeds_one_even_with_negative_top2():
    # top1=1.0, top2=-5.0 -> raw margin 6.0 -> clipped to the 1.0 ceiling
    # (down-weight only, never boost a list beyond its base weight).
    assert signal_confidence([1.0, -5.0]) == 1.0


# ---- Idea 4: adjacency-pair detection ----
def test_is_question_turn_ends_with_question_mark():
    assert _is_question_turn("Ann: What time is it?", "Ann") is True


def test_is_question_turn_starts_with_question_word_no_mark():
    assert _is_question_turn("Bob: Why did you leave", "Bob") is True


def test_is_question_turn_statement_is_false():
    assert _is_question_turn("Ann: I love hiking.", "Ann") is False


def test_is_question_turn_strips_speaker_prefix_before_checking_first_word():
    # Without stripping "Ann:" the first token, "Ann:", isn't a question
    # word; the check must look past the speaker prefix.
    assert _is_question_turn("Ann: Is this correct?", "Ann") is True


def test_is_question_turn_empty_after_prefix_strip_is_false():
    assert _is_question_turn("Ann: ", "Ann") is False


# ---- V4PlannedRetriever construction: new flags ----
def test_v4_planned_rejects_unknown_fusion_mode():
    with pytest.raises(ValueError):
        V4PlannedRetriever(config_overrides={"embed_model": "hash"}, fusion_mode="bogus")


# ---- Idea 4: _adjacency_pair_expand ----
def test_adjacency_pair_expand_adds_followup_when_speaker_differs():
    retriever = _make_retriever(adjacency_pair=True)
    retriever._turns_by_session = {"s1": ["seed", "next"]}
    retriever._turn_order_in_session = {"seed": 0, "next": 1}
    retriever._turn_session = {"seed": "s1", "next": "s1"}
    retriever._turn_text_raw = {"seed": "Ann: What time is it?", "next": "Bob: It's noon."}
    retriever._turn_speaker = {"seed": "Ann", "next": "Bob"}

    ranked = ["seed"]
    scores = {"seed": 0.8}
    new_ranked, new_scores = retriever._adjacency_pair_expand(ranked, scores)
    assert "next" in new_ranked
    assert new_scores["next"] == pytest.approx(0.8 * 0.8)
    assert new_scores["seed"] == 0.8


def test_adjacency_pair_expand_skips_when_same_speaker_follows():
    retriever = _make_retriever(adjacency_pair=True)
    retriever._turns_by_session = {"s1": ["seed", "next"]}
    retriever._turn_order_in_session = {"seed": 0, "next": 1}
    retriever._turn_session = {"seed": "s1", "next": "s1"}
    retriever._turn_text_raw = {"seed": "Ann: What time is it?", "next": "Ann: Never mind."}
    retriever._turn_speaker = {"seed": "Ann", "next": "Ann"}  # same speaker -> not a partner's answer

    ranked = ["seed"]
    scores = {"seed": 0.8}
    new_ranked, new_scores = retriever._adjacency_pair_expand(ranked, scores)
    assert new_ranked == ranked
    assert new_scores == scores


def test_adjacency_pair_expand_skips_non_question_seed_turn():
    retriever = _make_retriever(adjacency_pair=True)
    retriever._turns_by_session = {"s1": ["seed", "next"]}
    retriever._turn_order_in_session = {"seed": 0, "next": 1}
    retriever._turn_session = {"seed": "s1", "next": "s1"}
    retriever._turn_text_raw = {"seed": "Ann: I love hiking.", "next": "Bob: Nice."}
    retriever._turn_speaker = {"seed": "Ann", "next": "Bob"}

    ranked = ["seed"]
    scores = {"seed": 0.8}
    new_ranked, new_scores = retriever._adjacency_pair_expand(ranked, scores)
    assert new_ranked == ranked
    assert new_scores == scores


def test_adjacency_pair_expand_never_overrides_already_present_turn():
    retriever = _make_retriever(adjacency_pair=True)
    retriever._turns_by_session = {"s1": ["seed", "next"]}
    retriever._turn_order_in_session = {"seed": 0, "next": 1}
    retriever._turn_session = {"seed": "s1", "next": "s1"}
    retriever._turn_text_raw = {"seed": "Ann: What time is it?", "next": "Bob: Noon."}
    retriever._turn_speaker = {"seed": "Ann", "next": "Bob"}

    ranked = ["seed", "next"]
    scores = {"seed": 0.8, "next": 0.1}  # "next" already present with its OWN fused score
    new_ranked, new_scores = retriever._adjacency_pair_expand(ranked, scores)
    assert new_scores["next"] == 0.1  # not overwritten by 0.8*0.8=0.64
    assert new_ranked == ranked


def test_adjacency_pair_expand_no_op_at_end_of_session():
    retriever = _make_retriever(adjacency_pair=True)
    retriever._turns_by_session = {"s1": ["seed"]}  # seed is the LAST turn -> no follow-up
    retriever._turn_order_in_session = {"seed": 0}
    retriever._turn_session = {"seed": "s1"}
    retriever._turn_text_raw = {"seed": "Ann: What time is it?"}
    retriever._turn_speaker = {"seed": "Ann"}

    ranked = ["seed"]
    scores = {"seed": 0.8}
    new_ranked, new_scores = retriever._adjacency_pair_expand(ranked, scores)
    assert new_ranked == ranked
    assert new_scores == scores


# ---- Idea 5: _iterative_forage ----
def test_iterative_forage_merges_top5_after_head_when_subquery_starved():
    retriever = _make_retriever(iterative_budget=True)
    retriever._sub_query_text_by_label = {"sub0": "the starved subquery"}
    calls = []

    def fake_search(qtext, **kw):
        calls.append(qtext)
        return [_FakeSearchHit("c"), _FakeSearchHit("d")]

    retriever._memory.search = fake_search
    plan_ranked_lists = [("orig", ["a", "b"], 1.0), ("sub0", [], 1.0)]  # sub0: zero hits -> starved
    fused_ranked = ["a", "b"]
    fused_scores = {"a": 1.0, "b": 0.5}
    new_ranked, new_scores = retriever._iterative_forage(plan_ranked_lists, fused_ranked, fused_scores, top_k=5, at_time=None)
    assert new_ranked == ["a", "b", "c", "d"]
    assert calls == ["the starved subquery"]
    assert "c" in new_scores and "d" in new_scores


def test_iterative_forage_no_op_when_subquery_not_starved():
    retriever = _make_retriever(iterative_budget=True)
    calls = []
    retriever._memory.search = lambda qtext, **kw: calls.append(qtext) or []
    # sub0's own hit "a" IS already in the fused top-20 -> not starved.
    plan_ranked_lists = [("orig", ["a"], 1.0), ("sub0", ["a"], 1.0)]
    fused_ranked = ["a"]
    fused_scores = {"a": 1.0}
    new_ranked, new_scores = retriever._iterative_forage(plan_ranked_lists, fused_ranked, fused_scores, top_k=5, at_time=None)
    assert new_ranked == fused_ranked
    assert new_scores == fused_scores
    assert calls == []  # no extra search performed


def test_iterative_forage_caps_extra_searches_at_two():
    retriever = _make_retriever(iterative_budget=True)
    retriever._sub_query_text_by_label = {"sub0": "q0", "sub1": "q1", "sub2": "q2"}
    calls = []
    retriever._memory.search = lambda qtext, **kw: (calls.append(qtext), [])[1]
    plan_ranked_lists = [
        ("orig", ["a"], 1.0),
        ("sub0", [], 1.0),
        ("sub1", [], 1.0),
        ("sub2", [], 1.0),
    ]
    retriever._iterative_forage(plan_ranked_lists, ["a"], {"a": 1.0}, top_k=5, at_time=None)
    assert calls == ["q0", "q1"]  # sub2 never searched -- cap of 2 extra searches/question


def test_iterative_forage_deduplicates_against_existing_tail():
    retriever = _make_retriever(iterative_budget=True)
    retriever._sub_query_text_by_label = {"sub0": "q0"}
    retriever._memory.search = lambda qtext, **kw: [_FakeSearchHit("z"), _FakeSearchHit("y")]
    # "z" already sits at position 11 (tail[0]) -- must not be duplicated.
    plan_ranked_lists = [("orig", ["a"], 1.0), ("sub0", [], 1.0)]
    fused_ranked = ["a"] + [f"pad{i}" for i in range(9)] + ["z", "w"]  # 10 head + tail=[z,w]
    fused_scores = {tid: 1.0 - i * 0.01 for i, tid in enumerate(fused_ranked)}
    new_ranked, _ = retriever._iterative_forage(plan_ranked_lists, fused_ranked, fused_scores, top_k=5, at_time=None)
    assert new_ranked.count("z") == 1
    assert new_ranked[10] == "y"  # the genuinely new one is inserted right after the head
    assert new_ranked[11] == "z"  # the pre-existing tail id keeps its (deduped) place


# ---------------------------------------------------------------------------
# Idées inter-disciplines: V4PlannedRetriever.search() integration
# ---------------------------------------------------------------------------
def test_search_fusion_mode_quota_matches_sainte_lague_allocation(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": ["sq1", "sq2"], "hypotheses": []})],
    )
    retriever = _make_retriever(queries_dir=queries_dir, plan_mode="sub", fusion_mode="quota", rerank_top_n=4)
    retriever.add([_item("D1:1", "Ann: hi.")])

    canned = {
        "original": [_FakeSearchHit(f"o{i}") for i in range(1, 5)],
        "sq1": [_FakeSearchHit("s1a"), _FakeSearchHit("s1b")],
        "sq2": [_FakeSearchHit("s2a"), _FakeSearchHit("s2b")],
    }
    retriever._memory.search = lambda qtext, **kw: canned[qtext]
    result = retriever.search("original", top_k=4, qid="convX:q0")
    # round1: tie orig/sq1/sq2=1.0 -> orig (o1); round2: orig=1/3,sub=1.0 tie sq1/sq2 -> sq1 (s1a)
    # round3: orig=1/3,sq1=1/3,sq2=1.0 -> sq2 (s2a); round4: all=1/3 tie -> orig (o2)
    assert result == ["o1", "s1a", "s2a", "o2"]


def test_search_signal_confidence_downweights_flat_subquery_list(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": ["sq1"], "hypotheses": []})],
    )
    canned = {
        "original": [_FakeSearchHit("zz", score=5.0)],  # single hit -> full confidence regardless
        "sq1": [_FakeSearchHit("aa", score=1.0), _FakeSearchHit("ab", score=0.999)],  # flat -> confidence floor
    }

    baseline = _make_retriever(queries_dir=queries_dir, plan_mode="sub")
    baseline.add([_item("D1:1", "Ann: hi.")])
    baseline._memory.search = lambda qtext, **kw: canned[qtext]
    # off: orig/sq1 equal weight -> "zz"/"aa" tie in fused score -> id-ascending "aa" wins
    assert baseline.search("original", top_k=3, qid="convX:q0")[0] == "aa"

    confident = _make_retriever(queries_dir=queries_dir, plan_mode="sub", signal_confidence=True)
    confident.add([_item("D1:1", "Ann: hi.")])
    confident._memory.search = lambda qtext, **kw: canned[qtext]
    # on: sq1's flat top1/top2 down-weights it to the 0.3 floor -> "zz" (orig) now wins outright
    assert confident.search("original", top_k=3, qid="convX:q0")[0] == "zz"


def test_search_subquery_mmr_diversifies_final_order(tmp_path):
    queries_dir = tmp_path / "queries"
    _write_jsonl(
        queries_dir / "convX.jsonl",
        [json.dumps({"qid": "convX:q0", "subqueries": ["sq1"], "hypotheses": []})],
    )
    canned = {
        "original": [_FakeSearchHit("o1")],
        "sq1": [_FakeSearchHit("s0a"), _FakeSearchHit("s0b")],
    }

    baseline = _make_retriever(queries_dir=queries_dir, plan_mode="sub", weight_sub=1.5)
    baseline.add([_item("D1:1", "Ann: hi.")])
    baseline._memory.search = lambda qtext, **kw: canned[qtext]
    assert baseline.search("original", top_k=3, qid="convX:q0") == ["s0a", "s0b", "o1"]

    diversified = _make_retriever(
        queries_dir=queries_dir, plan_mode="sub", weight_sub=1.5, subquery_mmr=True, mmr_lambda=0.2
    )
    diversified.add([_item("D1:1", "Ann: hi.")])
    diversified._memory.search = lambda qtext, **kw: canned[qtext]
    assert diversified.search("original", top_k=3, qid="convX:q0") == ["s0a", "o1", "s0b"]


# ---------------------------------------------------------------------------
# Idées inter-disciplines: run_bench end-to-end (synthetic, tmp dirs)
# ---------------------------------------------------------------------------
def test_run_bench_report_ceilings_shape(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_ceilings.json")
    enrich_dir = tmp_path / "enrich"
    queries_dir = tmp_path / "queries"
    _write_full_enrich_and_queries(enrich_dir, queries_dir)

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_planned",
        name="synthetic_ceilings",
        top_k=5,
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        enrich_mode="facts+keywords",
        queries_dir=str(queries_dir),
        plan_mode="sub+hyp",
        report_ceilings=True,
    )
    ceilings = result["metadata"]["ceilings"]
    assert ceilings is not None
    assert set(ceilings["macro"].keys()) == {"r10", "r20", "r30", "r50"}
    assert ceilings["macro"]["r10"] <= ceilings["macro"]["r50"] + 1e-9
    for cat_stats in ceilings["per_category"].values():
        assert set(cat_stats.keys()) == {"r10", "r20", "r30", "r50", "n"}


def test_run_bench_no_report_ceilings_means_no_ceilings_key(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_noceil.json")
    enrich_dir = tmp_path / "enrich"
    queries_dir = tmp_path / "queries"
    _write_full_enrich_and_queries(enrich_dir, queries_dir)

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_planned",
        name="synthetic_noceil",
        top_k=5,
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        enrich_mode="facts+keywords",
        queries_dir=str(queries_dir),
        plan_mode="sub+hyp",
    )
    assert result["metadata"]["ceilings"] is None
    assert "ceiling_r10" not in result["per_question"][0]


def test_run_bench_fusion_mode_quota_end_to_end(tmp_path):
    locomo_path = _synthetic_locomo_1conv(tmp_path / "synthetic_locomo_quota.json")
    enrich_dir = tmp_path / "enrich"
    queries_dir = tmp_path / "queries"
    _write_full_enrich_and_queries(enrich_dir, queries_dir)

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_planned",
        name="synthetic_quota",
        top_k=5,
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        enrich_mode="facts+keywords",
        queries_dir=str(queries_dir),
        plan_mode="sub+hyp",
        fusion_mode="quota",
    )
    assert result["metadata"]["planning"]["fusion_mode"] == "quota"
    assert result["n_questions"] == 1
