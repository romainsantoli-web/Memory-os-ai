"""Tests for bench.enrich and bench.retrievers.V4EnrichedRetriever.

Everything here uses `Config(embed_model="hash")` (never a real model load)
and writes only into pytest `tmp_path` dirs -- the real
`bench/data/enrich/locomo_facts` (being generated concurrently by other
workers, per the task brief) is never read or depended on.
"""
from __future__ import annotations

import json

import pytest

from bench.enrich import EnrichRecord, coverage_report, load_enrichment
from bench.protocol import BenchItem
from bench.retrievers import V4EnrichedRetriever
from bench.run_bench import run_bench


def _write_jsonl(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _item(id_: str, text: str, conv_id: str = "convX", event_time=None) -> BenchItem:
    return BenchItem(
        id=id_, text=text, speaker="s", event_time=event_time, session_id="s1", meta={"conv_id": conv_id}
    )


# ---------------------------------------------------------------------------
# load_enrichment: tolerance contract
# ---------------------------------------------------------------------------
def test_load_enrichment_missing_file_returns_empty_dict(tmp_path):
    assert load_enrichment(tmp_path / "does_not_exist", "convX") == {}


def test_load_enrichment_parses_valid_lines(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            json.dumps({"turn_id": "D1:1", "facts": ["fact one"], "keywords": ["kw1"], "entities": ["Ann"]}),
            json.dumps({"turn_id": "D1:2", "facts": [], "keywords": [], "entities": []}),
        ],
    )
    records = load_enrichment(enrich_dir, "convX")
    assert set(records.keys()) == {"D1:1", "D1:2"}
    assert records["D1:1"] == EnrichRecord(facts=["fact one"], keywords=["kw1"], entities=["Ann"])
    assert records["D1:2"] == EnrichRecord(facts=[], keywords=[], entities=[])


def test_load_enrichment_skips_malformed_lines(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            "{not valid json",  # broken JSON (e.g. truncated mid-write)
            json.dumps(["not", "an", "object"]),  # valid JSON, wrong shape
            json.dumps({"facts": ["missing turn_id"]}),  # no turn_id
            json.dumps({"turn_id": "", "facts": []}),  # blank turn_id
            json.dumps({"turn_id": "D1:9", "facts": "not-a-list"}),  # facts not list-shaped
            json.dumps({"turn_id": "D1:1", "facts": ["good"], "keywords": [], "entities": []}),  # valid
        ],
    )
    records = load_enrichment(enrich_dir, "convX")
    assert records == {"D1:1": EnrichRecord(facts=["good"], keywords=[], entities=[])}


def test_load_enrichment_duplicate_turn_id_last_wins(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            json.dumps({"turn_id": "D1:1", "facts": ["first partial write"], "keywords": [], "entities": []}),
            json.dumps(
                {
                    "turn_id": "D1:1",
                    "facts": ["final, more complete write"],
                    "keywords": ["kw"],
                    "entities": ["X"],
                }
            ),
        ],
    )
    records = load_enrichment(enrich_dir, "convX")
    assert records["D1:1"].facts == ["final, more complete write"]
    assert records["D1:1"].keywords == ["kw"]


# ---------------------------------------------------------------------------
# coverage_report
# ---------------------------------------------------------------------------
def test_coverage_report_basic(tmp_path):
    enrich_dir = tmp_path / "enrich" / "locomo_facts"
    turns_dir = tmp_path / "enrich" / "locomo_turns"
    turns_dir.mkdir(parents=True)
    (turns_dir / "convX.json").write_text(
        json.dumps([{"id": "D1:1"}, {"id": "D1:2"}, {"id": "D1:3"}]), encoding="utf-8"
    )
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            json.dumps({"turn_id": "D1:1", "facts": ["f1", "f2"], "keywords": [], "entities": []}),
            json.dumps({"turn_id": "D1:2", "facts": [], "keywords": [], "entities": []}),
            # D1:3 not present at all yet (worker still running)
        ],
    )
    report = coverage_report(enrich_dir, turns_dir)
    assert report["n_conversations"] == 1
    conv = report["per_conversation"]["convX"]
    assert conv["n_turns"] == 3
    assert conv["n_turns_with_facts"] == 1  # only D1:1 has non-empty facts
    assert conv["n_facts"] == 2
    assert conv["file_found"] is True
    assert report["total_turns"] == 3
    assert report["overall_coverage"] == pytest.approx(1 / 3)


def test_coverage_report_conversation_with_no_enrich_file_yet(tmp_path):
    """A conversation whose enrichment file hasn't been written at all yet
    still gets a per_conversation entry (file_found=False, 0 coverage), not
    an exception or a silently dropped row."""
    enrich_dir = tmp_path / "enrich" / "locomo_facts"  # never created
    turns_dir = tmp_path / "enrich" / "locomo_turns"
    turns_dir.mkdir(parents=True)
    (turns_dir / "convY.json").write_text(json.dumps([{"id": "D1:1"}]), encoding="utf-8")

    report = coverage_report(enrich_dir, turns_dir)
    conv = report["per_conversation"]["convY"]
    assert conv["file_found"] is False
    assert conv["n_turns_with_facts"] == 0
    assert conv["coverage"] == 0.0


def test_coverage_report_skips_malformed_turns_snapshot(tmp_path):
    turns_dir = tmp_path / "turns"
    turns_dir.mkdir()
    (turns_dir / "convBad.json").write_text("not json at all", encoding="utf-8")
    report = coverage_report(tmp_path / "enrich", turns_dir)
    assert report["n_conversations"] == 0
    assert report["n_conversations_skipped"] == 1
    assert "convBad" in report["skipped_conversations"]


# ---------------------------------------------------------------------------
# V4EnrichedRetriever
# ---------------------------------------------------------------------------
def _make_retriever(enrich_dir, mode="facts"):
    r = V4EnrichedRetriever(config_overrides={"embed_model": "hash"}, mode=mode, enrich_dir=str(enrich_dir))
    r.reset()
    return r


def test_v4_enriched_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        V4EnrichedRetriever(config_overrides={"embed_model": "hash"}, mode="bogus", enrich_dir=str(tmp_path))


def test_v4_enriched_search_maps_fact_hit_to_parent_turn(tmp_path):
    """The core enrichment payoff: a fact whose wording matches the query
    while the RAW turn's own wording does not must still surface the turn."""
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            json.dumps(
                {
                    "turn_id": "D1:1",
                    "facts": ["Ann's favorite made-up word is zzyzxquibble."],
                    "keywords": [],
                    "entities": ["Ann"],
                }
            ),
            json.dumps({"turn_id": "D1:2", "facts": [], "keywords": [], "entities": ["Bob"]}),
        ],
    )
    items = [
        _item("D1:1", "Ann: I love hiking in the mountains."),
        _item("D1:2", "Bob: That sounds fun, I prefer swimming."),
    ]
    retriever = _make_retriever(enrich_dir)
    retriever.add(items)

    result = retriever.search("zzyzxquibble", top_k=1)
    assert result == ["D1:1"]  # the raw turn text never contains this word


def test_v4_enriched_search_dedupes_and_respects_top_k(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            json.dumps(
                {"turn_id": "D1:1", "facts": ["quibblewomp fact about D1:1"], "keywords": [], "entities": []}
            ),
        ],
    )
    items = [_item("D1:1", "Ann: something unrelated."), _item("D1:2", "Bob: also unrelated.")]
    retriever = _make_retriever(enrich_dir)
    retriever.add(items)
    result = retriever.search("quibblewomp", top_k=5)
    assert len(result) == len(set(result))  # no duplicate turn ids even if multiple hits map to D1:1


def test_v4_enriched_coverage_tracks_add(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            json.dumps({"turn_id": "D1:1", "facts": ["one fact"], "keywords": [], "entities": []}),
            json.dumps({"turn_id": "D1:2", "facts": [], "keywords": [], "entities": []}),
        ],
    )
    items = [_item("D1:1", "Ann: a."), _item("D1:2", "Bob: b.")]
    retriever = _make_retriever(enrich_dir)
    retriever.add(items)
    assert retriever.coverage == {
        "conv_id": "convX",
        "n_turns": 2,
        "n_turns_with_facts": 1,
        "n_facts_added": 1,
        "enrich_file_found": True,
    }


def test_v4_enriched_missing_enrich_file_is_not_an_error(tmp_path):
    enrich_dir = tmp_path / "enrich"  # no files written at all
    items = [_item("D1:1", "Ann: a.", conv_id="convNoFile")]
    retriever = _make_retriever(enrich_dir)
    retriever.add(items)  # must not raise
    assert retriever.coverage["enrich_file_found"] is False
    assert retriever.coverage["n_turns_with_facts"] == 0
    assert retriever.search("a", top_k=5)  # raw turn is still indexed and findable


def test_v4_enriched_keywords_mode_changes_indexed_text(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [
            json.dumps(
                {"turn_id": "D1:1", "facts": [], "keywords": ["unobtainium77", "widget"], "entities": []}
            )
        ],
    )
    items = [_item("D1:1", "Ann: I had a normal day.")]

    facts_retriever = _make_retriever(enrich_dir, mode="facts")
    facts_retriever.add(items)
    stored_text_facts_mode = facts_retriever._memory.store.get("D1:1").text
    assert "unobtainium77" not in stored_text_facts_mode  # facts mode: raw text untouched

    kw_retriever = _make_retriever(enrich_dir, mode="facts+keywords")
    kw_retriever.add(items)
    stored_text_kw_mode = kw_retriever._memory.store.get("D1:1").text
    assert "unobtainium77" in stored_text_kw_mode
    assert "widget" in stored_text_kw_mode

    # and it's actually searchable via that appended keyword in kw mode only
    assert "D1:1" in kw_retriever.search("unobtainium77", top_k=5)


def test_v4_enriched_search_stats_counts_top_hits(tmp_path):
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convX.jsonl",
        [json.dumps({"turn_id": "D1:1", "facts": ["thermolumin fact"], "keywords": [], "entities": []})],
    )
    items = [_item("D1:1", "Ann: something else entirely.")]
    retriever = _make_retriever(enrich_dir)
    retriever.add(items)

    assert retriever.search_stats["n_searches"] == 0
    retriever.search("thermolumin", top_k=3)
    assert retriever.search_stats["n_searches"] == 1
    # the top-ranked raw hit for this query is the fact item itself
    assert retriever.search_stats["top_hit_facts"] == 1
    assert retriever.search_stats["top_hit_episodes"] == 0


def test_v4_enriched_search_stats_accumulate_across_conversations(tmp_path):
    """search_stats is cumulative for the whole run (reset() must not clear
    it), since run_bench.py reports one final total in the run's metadata."""
    enrich_dir = tmp_path / "enrich"
    _write_jsonl(
        enrich_dir / "convA.jsonl",
        [json.dumps({"turn_id": "D1:1", "facts": [], "keywords": [], "entities": []})],
    )
    retriever = V4EnrichedRetriever(config_overrides={"embed_model": "hash"}, enrich_dir=str(enrich_dir))
    retriever.reset()
    retriever.add([_item("D1:1", "Ann: hello.", conv_id="convA")])
    retriever.search("hello", top_k=3)
    retriever.reset()  # new conversation
    retriever.add([_item("D2:1", "Bob: hi there.", conv_id="convB")])
    retriever.search("hi", top_k=3)
    assert retriever.search_stats["n_searches"] == 2


# ---------------------------------------------------------------------------
# run_bench end-to-end with --retriever v4_enriched
# ---------------------------------------------------------------------------
def _synthetic_locomo_2convs(path):
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
        },
        {
            "sample_id": "convB",
            "conversation": {
                "speaker_a": "Cara",
                "speaker_b": "Dave",
                "session_1_date_time": "2:00 pm on 2 January, 2023",
                "session_1": [
                    {"speaker": "Cara", "dia_id": "D1:1", "text": "I adopted a puppy named Rex."},
                    {"speaker": "Dave", "dia_id": "D1:2", "text": "What breed is Rex?"},
                ],
            },
            "qa": [
                {
                    "question": "What did Cara adopt?",
                    "answer": "a puppy",
                    "evidence": ["D1:1"],
                    "category": 4,
                },
            ],
        },
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_full_enrich_dir(enrich_dir):
    _write_jsonl(
        enrich_dir / "convA.jsonl",
        [
            json.dumps(
                {
                    "turn_id": "D1:1",
                    "facts": ["Ann loves hiking."],
                    "keywords": ["hiking"],
                    "entities": ["Ann"],
                }
            ),
            json.dumps({"turn_id": "D1:2", "facts": [], "keywords": [], "entities": ["Bob"]}),
        ],
    )
    _write_jsonl(
        enrich_dir / "convB.jsonl",
        [
            json.dumps(
                {
                    "turn_id": "D1:1",
                    "facts": ["Cara adopted a puppy named Rex."],
                    "keywords": ["puppy"],
                    "entities": ["Cara", "Rex"],
                }
            ),
            json.dumps({"turn_id": "D1:2", "facts": [], "keywords": [], "entities": ["Dave"]}),
        ],
    )


def test_run_bench_v4_enriched_end_to_end(tmp_path):
    locomo_path = _synthetic_locomo_2convs(tmp_path / "synthetic_locomo.json")
    enrich_dir = tmp_path / "enrich"
    _write_full_enrich_dir(enrich_dir)

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_enriched",
        name="synthetic_v4_enriched",
        top_k=5,
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        enrich_mode="facts",
    )

    assert result["retriever"] == "v4_enriched"
    assert result["n_conversations"] == 2
    assert result["n_questions"] == 2
    enrichment_meta = result["metadata"]["enrichment"]
    assert enrichment_meta["n_conversations_missing_file"] == 0
    assert enrichment_meta["turns_total"] == 4
    assert enrichment_meta["turns_with_facts"] == 2
    assert enrichment_meta["coverage_turns"] == pytest.approx(0.5)
    assert enrichment_meta["facts_added"] == 2
    assert result["metadata"]["search_stats"]["n_searches"] == 2


def test_run_bench_v4_enriched_fails_fast_on_missing_enrich_files(tmp_path):
    locomo_path = _synthetic_locomo_2convs(tmp_path / "synthetic_locomo2.json")
    enrich_dir = tmp_path / "empty_enrich"  # no files at all
    enrich_dir.mkdir()

    with pytest.raises(RuntimeError, match="convA|convB"):
        run_bench(
            dataset="locomo",
            retriever_key="v4_enriched",
            name="should_fail",
            locomo_path=str(locomo_path),
            config_overrides={"embed_model": "hash"},
            enrich_dir=str(enrich_dir),
        )


def test_run_bench_v4_enriched_allow_partial_enrich_runs_and_reports_coverage(tmp_path):
    locomo_path = _synthetic_locomo_2convs(tmp_path / "synthetic_locomo3.json")
    enrich_dir = tmp_path / "partial_enrich"
    enrich_dir.mkdir()
    # Only convA's file exists; convB's worker hasn't produced one yet.
    _write_jsonl(
        enrich_dir / "convA.jsonl",
        [
            json.dumps(
                {"turn_id": "D1:1", "facts": ["Ann loves hiking."], "keywords": [], "entities": ["Ann"]}
            )
        ],
    )

    result = run_bench(
        dataset="locomo",
        retriever_key="v4_enriched",
        name="partial_ok",
        locomo_path=str(locomo_path),
        config_overrides={"embed_model": "hash"},
        enrich_dir=str(enrich_dir),
        allow_partial_enrich=True,
    )
    assert result["n_conversations"] == 2
    enrichment_meta = result["metadata"]["enrichment"]
    assert enrichment_meta["n_conversations_missing_file"] == 1
    assert enrichment_meta["allow_partial_enrich"] is True
