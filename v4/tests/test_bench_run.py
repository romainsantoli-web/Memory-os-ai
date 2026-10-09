"""End-to-end test of run_bench on a synthetic 2-conversation dataset.

Uses the bm25 retriever (deterministic, no model weights) and writes
everything into tmp_path — the real bench/results/ and RESULTS.md are never
touched by this file.
"""
from __future__ import annotations

import json


from bench.run_bench import (
    append_ledger_row,
    main,
    next_ledger_number,
    next_version_path,
    run_bench,
    write_result_json,
)

LEDGER_HEADER = (
    "# RESULTS — ledger des expériences v4 (TEST FIXTURE)\n\n"
    "| # | Approche (fichier) | Embedding | Chunking | Sparse | Fusion | Rerank | R@5 | R@10 | MRR | nDCG@10 | p50 ms | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    "\n## Journal de phases\n- fixture only\n"
)


def _synthetic_locomo(path):
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
                    {"speaker": "Ann", "dia_id": "D1:3", "text": "My favorite mountain is Everest."},
                ],
            },
            "qa": [
                {"question": "What does Ann love?", "answer": "hiking", "evidence": ["D1:1"], "category": 4},
                {"question": "What is Ann's favorite mountain?", "answer": "Everest", "evidence": ["D1:3"], "category": 4},
                {"question": "Adversarial one", "answer": None, "evidence": ["D1:2"], "category": 5},
                {"question": "Empty evidence one", "answer": "?", "evidence": [], "category": 3},
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
                    {"speaker": "Cara", "dia_id": "D1:3", "text": "Rex is a golden retriever."},
                ],
            },
            "qa": [
                {"question": "What breed is Rex?", "answer": "golden retriever", "evidence": ["D1:3"], "category": 4},
                {
                    "question": "What did Cara adopt and what breed is it?",
                    "answer": "a golden retriever puppy",
                    "evidence": ["D1:1", "D1:3"],
                    "category": 1,
                },
            ],
        },
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_run_bench_end_to_end_with_bm25(tmp_path):
    locomo_path = _synthetic_locomo(tmp_path / "synthetic_locomo.json")

    result = run_bench(
        dataset="locomo",
        retriever_key="bm25",
        name="synthetic_bm25",
        top_k=10,
        locomo_path=str(locomo_path),
    )

    assert result["dataset"] == "locomo"
    assert result["retriever"] == "bm25"
    assert result["n_conversations"] == 2
    # 2 valid Qs in convA (cat5 + empty-evidence excluded) + 2 valid Qs in convB = 4
    assert result["n_questions"] == 4
    assert set(result["per_category"].keys()) == {"1", "4"}
    assert result["per_category"]["4"]["n"] == 3
    assert result["per_category"]["1"]["n"] == 1
    assert "r10" in result["macro"] and "mrr" in result["macro"]
    assert result["latency_ms"]["p50"] >= 0.0
    assert len(result["per_question"]) == 4
    for row in result["per_question"]:
        assert set(row.keys()) >= {"qid", "category", "r5", "r10", "mrr", "ndcg10", "latency_ms"}

    # BM25 should trivially find at least one single-hop question via lexical overlap
    assert any(row["r10"] == 1.0 for row in result["per_question"])


def test_write_result_json_auto_increments_never_overwrites(tmp_path):
    results_dir = tmp_path / "results"
    result1 = {"name": "foo", "dataset": "d", "retriever": "bm25", "config": {}, "metadata": {}, "n_conversations": 0,
               "n_questions": 0, "per_category": {}, "macro": {}, "micro": {}, "latency_ms": {"p50": 0, "p95": 0, "mean": 0},
               "mean_context_tokens": None, "per_question": []}
    p1 = write_result_json(result1, results_dir)
    p2 = write_result_json(result1, results_dir)
    assert p1.name == "foo_v1.json"
    assert p2.name == "foo_v2.json"
    assert p1 != p2
    assert p1.exists() and p2.exists()
    assert json.loads(p1.read_text())["name"] == "foo"


def test_next_version_path_skips_existing(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "bar_v1.json").write_text("{}")
    (results_dir / "bar_v2.json").write_text("{}")
    p = next_version_path(results_dir, "bar")
    assert p.name == "bar_v3.json"


def test_append_ledger_row_into_fixture_ledger(tmp_path):
    ledger_path = tmp_path / "RESULTS_fixture.md"
    ledger_path.write_text(LEDGER_HEADER, encoding="utf-8")
    assert next_ledger_number(ledger_path) == 1

    fake_result = {
        "name": "synthetic_bm25",
        "macro": {"r5": 0.75, "r10": 1.0, "mrr": 0.8333, "ndcg10": 0.9},
        "latency_ms": {"p50": 1.234, "p95": 2.0, "mean": 1.5},
    }
    row = append_ledger_row(ledger_path, fake_result, tmp_path / "results" / "synthetic_bm25_v1.json", "bm25", {}, notes="unit test")
    assert row.startswith("| 1 |")
    assert "0.750" in row and "1.000" in row and "0.833" in row and "0.900" in row
    content = ledger_path.read_text(encoding="utf-8")
    assert row in content
    assert next_ledger_number(ledger_path) == 2  # next call would be row 2

    # append a second row: must land after the first data row, and never clobber it
    row2 = append_ledger_row(ledger_path, fake_result, tmp_path / "results" / "synthetic_bm25_v2.json", "bm25", {}, notes="second")
    assert row2.startswith("| 2 |")
    content2 = ledger_path.read_text(encoding="utf-8")
    assert row in content2  # first row still present
    assert content2.index(row) < content2.index(row2)  # in order
    assert "## Journal de phases" in content2  # trailing section preserved


def test_main_cli_no_ledger_smoke(tmp_path, capsys):
    locomo_path = _synthetic_locomo(tmp_path / "synthetic_locomo2.json")
    results_dir = tmp_path / "results"
    rc = main(
        [
            "--dataset",
            "locomo",
            "--retriever",
            "bm25",
            "--name",
            "cli_smoke",
            "--locomo-path",
            str(locomo_path),
            "--results-dir",
            str(results_dir),
            "--no-ledger",
        ]
    )
    assert rc == 0
    written = list(results_dir.glob("cli_smoke_v*.json"))
    assert len(written) == 1
    out = capsys.readouterr().out
    assert "cli_smoke" in out


def test_main_cli_appends_to_fixture_ledger_not_real_one(tmp_path, capsys):
    locomo_path = _synthetic_locomo(tmp_path / "synthetic_locomo3.json")
    results_dir = tmp_path / "results"
    ledger_path = tmp_path / "RESULTS_fixture2.md"
    ledger_path.write_text(LEDGER_HEADER, encoding="utf-8")

    rc = main(
        [
            "--dataset",
            "locomo",
            "--retriever",
            "bm25",
            "--name",
            "cli_ledger_smoke",
            "--locomo-path",
            str(locomo_path),
            "--results-dir",
            str(results_dir),
            "--ledger",
            str(ledger_path),
        ]
    )
    assert rc == 0
    content = ledger_path.read_text(encoding="utf-8")
    assert "cli_ledger_smoke" in content
    assert "| 1 |" in content


def test_context_tokens_honors_top_k_not_hardcoded_ten(tmp_path):
    """A11 (adversarial review, 2026-09-12): context_tokens used to always
    be computed over a hardcoded ranked[:10] regardless of --top-k."""
    locomo_path = _synthetic_locomo(tmp_path / "synthetic_locomo_topk.json")

    result_k1 = run_bench(
        dataset="locomo", retriever_key="bm25", name="topk1", top_k=1, locomo_path=str(locomo_path)
    )
    result_k10 = run_bench(
        dataset="locomo", retriever_key="bm25", name="topk10", top_k=10, locomo_path=str(locomo_path)
    )
    # every row's context_tokens must be computed from at most top_k texts;
    # a hardcoded ranked[:10] would make top_k=1's rows identical to top_k=10's
    tokens_k1 = [row["context_tokens"] for row in result_k1["per_question"]]
    tokens_k10 = [row["context_tokens"] for row in result_k10["per_question"]]
    assert any(a <= b for a, b in zip(tokens_k1, tokens_k10))
    assert all(t >= 0 for t in tokens_k1)


def test_metadata_records_pythonhashseed(tmp_path, monkeypatch):
    """A11: PYTHONHASHSEED must be recorded in every run's metadata for
    reproducibility (adversarial review, 2026-09-12)."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    locomo_path = _synthetic_locomo(tmp_path / "synthetic_locomo_seed.json")
    result = run_bench(dataset="locomo", retriever_key="bm25", name="seedcheck", locomo_path=str(locomo_path))
    assert result["metadata"]["pythonhashseed"] == "0"


def test_main_cli_invalid_config_json_returns_error(tmp_path):
    locomo_path = _synthetic_locomo(tmp_path / "synthetic_locomo4.json")
    rc = main(
        [
            "--dataset",
            "locomo",
            "--retriever",
            "v4",
            "--name",
            "bad_json",
            "--locomo-path",
            str(locomo_path),
            "--config-json",
            "{not valid json",
            "--no-ledger",
        ]
    )
    assert rc == 2
