"""Unit tests for memory_os_v4.server: handlers called directly against an
engine bound to tmp_path via the ``server_engine`` fixture (monkeypatches the
lazy ``get_engine`` getter). Covers every tool's happy path plus validation
errors, the ingest traversal guard, and the forget confirmation guard.
"""

from __future__ import annotations

import json
import time

import pytest
from pydantic import ValidationError

from memory_os_v4 import models, server
from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4


@pytest.fixture
def engine(tmp_path):
    eng = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path)))
    yield eng


@pytest.fixture
def server_engine(engine, monkeypatch):
    """Monkeypatch the server's lazy engine getter to return ``engine``."""
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    yield engine


def call(name: str, payload: dict) -> dict:
    return server.dispatch(name, payload)


# --------------------------------------------------------------- dispatch --
def test_dispatch_unknown_tool_returns_error(server_engine):
    result = call("memory_does_not_exist", {})
    assert result == {"ok": False, "error": "unknown tool: 'memory_does_not_exist'"}


def test_dispatch_never_raises_on_bad_input(server_engine):
    result = call("memory_search", {"query": ""})  # min_length=1 violated
    assert result["ok"] is False
    assert "error" in result


def test_all_tools_registered_consistently():
    assert set(server.TOOL_MODELS) == set(server.TOOL_HANDLERS) == set(server.TOOL_DESCRIPTIONS)
    assert len(server.TOOL_MODELS) >= 15


# --------------------------------------------------------- memory_status ---
def test_status_happy_path(server_engine):
    result = call("memory_status", {})
    assert "error" not in result
    assert result["total"] == 0
    assert result["embed_model"] == "hash"


def test_status_rejects_unknown_field(server_engine):
    with pytest.raises(ValidationError):
        models.StatusIn(bogus=1)


# ------------------------------------------------ models: keywords/queries --
def test_message_in_and_fact_in_accept_optional_keywords():
    msg = models.MessageIn(role="user", content="hi", keywords=["a", "b"])
    assert msg.keywords == ["a", "b"]
    assert models.MessageIn(role="user", content="hi").keywords is None

    fact = models.FactIn(text="hi", keywords=["falcon"])
    assert fact.keywords == ["falcon"]
    assert models.FactIn(text="hi").keywords is None


def test_search_in_guards_queries_and_hypotheses_length_and_count():
    models.SearchIn(query="q", queries=["a"] * 8, hypotheses=["b"] * 8)  # exactly at the cap: ok
    with pytest.raises(ValidationError):
        models.SearchIn(query="q", queries=["a"] * 9)
    with pytest.raises(ValidationError):
        models.SearchIn(query="q", hypotheses=["a" * 301])


def test_instructions_stay_under_2600_chars():
    from memory_os_v4.instructions import MEMORY_INSTRUCTIONS

    assert len(MEMORY_INSTRUCTIONS) < 2600


# --------------------------------------------------- memory_session_brief --
def test_session_brief_happy_path(server_engine):
    server_engine.add_episode("Hello world.", project="p1")
    result = call("memory_session_brief", {"project": "p1"})
    assert "blocks" in result
    assert result["recent_episodes"][0]["text"] == "Hello world."
    # datetimes must be ISO strings, not raw floats
    assert isinstance(result["recent_episodes"][0]["event_time"], str)


def test_session_brief_max_tokens_bounds():
    with pytest.raises(ValidationError):
        models.SessionBriefIn(max_tokens=10)  # below ge=200


# -------------------------------------------------------- memory_chat_save-
def test_chat_save_happy_path_with_facts_and_summary(server_engine):
    payload = {
        "messages": [
            {"role": "user", "content": "We picked FAISS FlatL2."},
            {"role": "assistant", "content": "Noted."},
        ],
        "summary": "Decision: FAISS FlatL2 for the vector index.",
        "project_label": "prometheus",
        "facts": [{"text": "Romain lives in Paris."}],
    }
    result = call("memory_chat_save", payload)
    assert result["episodes"] == 2
    assert result["summary_id"] is not None
    assert result["facts"]["added"] == [result["facts"]["added"][0]]
    assert result["facts"]["deduped"] == []

    # episodes actually persisted under the right project
    hits = server_engine.search("FAISS FlatL2", top_k=5, project="prometheus")
    assert any(h.item.kind == "episode" for h in hits)


def test_chat_save_dedupes_facts_on_repeat(server_engine):
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "facts": [{"text": "Romain works on GRPO-MM."}],
    }
    r1 = call("memory_chat_save", payload)
    r2 = call("memory_chat_save", payload)
    assert len(r1["facts"]["added"]) == 1
    assert r2["facts"]["added"] == []
    assert r2["facts"]["deduped"] == r1["facts"]["added"]


def test_chat_save_supersedes_with_new_dated_fact(tmp_path, monkeypatch):
    # HashEmbedder has no semantic generalization (see v4/README.md "Other known
    # limitations"): use a lower supersede_cos, as the project's own tests do,
    # so token-overlapping-but-reworded facts still trigger supersession.
    eng = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path), supersede_cos=0.3))
    monkeypatch.setattr(server, "get_engine", lambda: eng)

    call(
        "memory_chat_save",
        {
            "messages": [{"role": "user", "content": "job update"}],
            "facts": [
                {
                    "text": "Romain is a software engineer at Acme.",
                    "entities": ["Romain", "Acme"],
                    "event_time": "2026-01-01",
                }
            ],
        },
    )
    r2 = call(
        "memory_chat_save",
        {
            "messages": [{"role": "user", "content": "job update 2"}],
            "facts": [
                {
                    "text": "Romain is a senior engineer at Acme now.",
                    "entities": ["Romain", "Acme"],
                    "event_time": "2026-06-01",
                }
            ],
        },
    )
    assert len(r2["facts"]["added"]) == 1
    assert len(r2["facts"]["superseded"]) == 1


def test_chat_save_message_keywords_indexed_but_not_in_text(server_engine):
    result = call(
        "memory_chat_save",
        {"messages": [{"role": "user", "content": "A quiet afternoon.", "keywords": ["hiking", "yosemite"]}]},
    )
    assert result["episodes"] == 1
    hits = server_engine.search("yosemite hiking", top_k=5)
    assert hits and hits[0].item.text == "A quiet afternoon."
    assert hits[0].item.keywords == ["hiking", "yosemite"]


def test_chat_save_no_messages_rejected(server_engine):
    result = call("memory_chat_save", {"messages": []})
    assert result["ok"] is False


def test_chat_save_bad_role_rejected(server_engine):
    result = call("memory_chat_save", {"messages": [{"role": "narrator", "content": "x"}]})
    assert result["ok"] is False


def test_chat_save_event_time_iso_applied_to_episodes(server_engine):
    result = call(
        "memory_chat_save",
        {"messages": [{"role": "user", "content": "dated message"}], "event_time": "2020-03-04"},
    )
    assert result["episodes"] == 1
    hits = server_engine.search("dated message", top_k=1)
    ev = hits[0].item.event_time
    # 2020-03-04 local midnight, well before "now"
    assert ev < time.time() - 1000


# -------------------------------------------------------- memory_add_facts-
def test_add_facts_happy_path(server_engine):
    result = call("memory_add_facts", {"facts": [{"text": "Nisha's dad likes chess."}], "project": "p1"})
    assert len(result["added"]) == 1
    assert result["deduped"] == []
    assert result["superseded"] == []


def test_add_facts_handler_accepts_keywords(server_engine):
    result = call(
        "memory_add_facts",
        {"facts": [{"text": "The launch date is set.", "keywords": ["falcon", "deadline"]}], "project": "p1"},
    )
    assert len(result["added"]) == 1
    item = server_engine.store.get(result["added"][0])
    assert item.keywords == ["falcon", "deadline"]
    assert "falcon" not in item.text


def test_add_facts_rejects_extra_field(server_engine):
    result = call("memory_add_facts", {"facts": [{"text": "x", "unexpected": 1}]})
    assert result["ok"] is False


def test_add_facts_invalid_kind_rejected(server_engine):
    result = call("memory_add_facts", {"facts": [{"text": "x", "kind": "opinion"}]})
    assert result["ok"] is False


# ------------------------------------------------------------ memory_search
def test_search_happy_path(server_engine):
    server_engine.add_facts([{"text": "The vector index uses FAISS FlatIP."}], project="p1")
    result = call("memory_search", {"query": "FAISS vector index", "project": "p1"})
    assert result["hits"]
    hit = result["hits"][0]
    assert set(["id", "kind", "text", "score", "event_time", "project", "signals"]).issubset(hit)
    assert isinstance(hit["event_time"], str)


def test_search_top_k_bounds_rejected(server_engine):
    result = call("memory_search", {"query": "x", "top_k": 0})
    assert result["ok"] is False
    result = call("memory_search", {"query": "x", "top_k": 500})
    assert result["ok"] is False


def test_search_at_time_filters_superseded_facts(server_engine):
    call("memory_add_facts", {"facts": [{"text": "Romain is a junior dev.", "event_time": "2020-01-01"}]})
    call(
        "memory_add_facts",
        {"facts": [{"text": "Romain is a senior dev.", "entities": ["Romain"], "event_time": "2025-01-01"}]},
    )
    past = call("memory_search", {"query": "Romain dev", "at_time": "2020-06-01", "kinds": ["fact"]})
    now_ = call("memory_search", {"query": "Romain dev", "kinds": ["fact"]})
    assert any("junior" in h["text"] for h in past["hits"])
    assert any("senior" in h["text"] for h in now_["hits"])


def test_search_invalid_at_time_rejected(server_engine):
    result = call("memory_search", {"query": "x", "at_time": "not-a-date"})
    assert result["ok"] is False


def test_search_handler_wires_queries_and_hypotheses(server_engine):
    server_engine.add_episode("queryhit unique term.", project="p1", item_id="qh")
    server_engine.add_episode("hypothesishit unique term.", project="p1", item_id="hh")
    result = call(
        "memory_search",
        {
            "query": "nomatchatall000",
            "queries": ["queryhit"],
            "hypotheses": ["hypothesishit"],
            "project": "p1",
        },
    )
    ids = {h["id"] for h in result["hits"]}
    assert ids == {"qh", "hh"}
    assert all("multi_query" in h["signals"] for h in result["hits"])


def test_search_rejects_too_many_or_too_long_queries(server_engine):
    result = call("memory_search", {"query": "x", "queries": ["a"] * 9})
    assert result["ok"] is False
    result = call("memory_search", {"query": "x", "hypotheses": ["a" * 301]})
    assert result["ok"] is False


# ------------------------------------------------------- memory_get_context
def test_get_context_happy_path(server_engine):
    server_engine.add_episode("Paris is the capital of France.", project="p1")
    result = call("memory_get_context", {"query": "capital of France", "project": "p1"})
    assert "Paris" in result["context"]
    assert result["n_items"] >= 1


def test_get_context_max_tokens_bounds_rejected(server_engine):
    result = call("memory_get_context", {"query": "x", "max_tokens": 100})
    assert result["ok"] is False


def test_get_context_wires_queries_and_hypotheses(server_engine):
    server_engine.add_episode("Paris is the capital of France.", project="p1", item_id="paris")
    server_engine.add_episode("Lyon has great food.", project="p1", item_id="lyon")
    result = call(
        "memory_get_context",
        {"query": "nomatchatall000", "queries": ["capital of France"], "project": "p1"},
    )
    assert "Paris" in result["context"]


# --------------------------------------------------------- memory_timeline-
def test_timeline_sorted_by_event_time(server_engine):
    call(
        "memory_add_facts",
        {"facts": [{"text": "Project X kicked off.", "entities": ["Project X"], "event_time": "2025-01-01"}]},
    )
    call(
        "memory_add_facts",
        {"facts": [{"text": "Project X shipped v1.", "entities": ["Project X"], "event_time": "2025-06-01"}]},
    )
    result = call("memory_timeline", {"query": "Project X"})
    times = [it["event_time"] for it in result["items"]]
    assert times == sorted(times)
    assert len(result["items"]) == 2


def test_timeline_limit_bounds_rejected(server_engine):
    result = call("memory_timeline", {"query": "x", "limit": 0})
    assert result["ok"] is False


# ------------------------------------------------------- memory_block_get/set
def test_block_set_replace_then_get(server_engine):
    call("memory_block_set", {"name": "user_profile", "text": "Romain, ML engineer."})
    result = call("memory_block_get", {"name": "user_profile"})
    assert result["blocks"]["user_profile"] == "Romain, ML engineer."


def test_block_get_all_when_no_name(server_engine):
    call("memory_block_set", {"name": "working_rules", "text": "Be concise."})
    result = call("memory_block_get", {})
    assert "working_rules" in result["blocks"]
    assert "user_profile" in result["blocks"]


def test_block_set_append_mode(server_engine):
    call("memory_block_set", {"name": "project_state", "text": "Phase 1 done.", "mode": "replace"})
    call("memory_block_set", {"name": "project_state", "text": "Phase 2 started.", "mode": "append"})
    result = call("memory_block_get", {"name": "project_state"})
    assert "Phase 1 done." in result["blocks"]["project_state"]
    assert "Phase 2 started." in result["blocks"]["project_state"]


def test_block_set_reports_truncation(server_engine):
    server_engine.blocks.char_limit = 20
    result = call("memory_block_set", {"name": "user_profile", "text": "x" * 50})
    assert result["truncated"] is True
    assert result["chars"] == 20


def test_block_set_missing_name_rejected(server_engine):
    result = call("memory_block_set", {"text": "no name"})
    assert result["ok"] is False


# ----------------------------------------------- memory_consolidate/compact
def test_consolidate_dry_run_happy_path(server_engine):
    server_engine.add_facts([{"text": "Some fact about GRPO-MM."}])
    result = call("memory_consolidate", {"dry_run": True})
    assert "deduped" in result and "summary_proposals" in result


def test_compact_is_alias_of_consolidate(server_engine):
    result = call("memory_compact", {"dry_run": True})
    assert "deduped" in result and "summary_proposals" in result
    assert server.TOOL_HANDLERS["memory_compact"] is server.TOOL_HANDLERS["memory_consolidate"]


# --------------------------------------------------------- memory_forget --
def test_forget_by_item_id(server_engine):
    item = server_engine.add_episode("Delete me.", project="p1")
    result = call("memory_forget", {"item_id": item.id, "reason": "test cleanup"})
    assert result["forgotten"] == [item.id]
    assert server_engine.store.get(item.id).deleted is True
    # soft-deleted items are excluded from search
    hits = server_engine.search("Delete me", top_k=5, project="p1")
    assert item.id not in {h.item.id for h in hits}


def test_forget_removes_from_bm25_and_vector_index_immediately(server_engine):
    """A5 (adversarial review, 2026-09-12): the server handler now goes
    through MemoryV4.forget() instead of flipping `.deleted` directly on the
    stored item -- the vector/BM25/entity index entries must be gone right
    away, not just after the next consolidate() rebuild."""
    item = server_engine.add_episode("Delete me from every index.", project="p1")
    call("memory_forget", {"item_id": item.id, "reason": "test cleanup"})
    assert server_engine.vector_index.get_one(item.id) is None
    assert item.id not in server_engine.bm25.doc_len
    assert item.id not in server_engine.entities.doc_entities


def test_forget_unknown_item_id_errors(server_engine):
    result = call("memory_forget", {"item_id": "does-not-exist", "reason": "x"})
    assert result["ok"] is False


def test_forget_by_query_requires_confirm(server_engine):
    server_engine.add_episode("Forget-by-query target.", project="p1")
    result = call("memory_forget", {"query": "Forget-by-query target", "reason": "x"})
    assert result["ok"] is False
    assert "confirm" in result["error"]


def test_forget_by_query_with_confirm_succeeds(server_engine):
    item = server_engine.add_episode("Forget-by-query target two.", project="p1")
    result = call(
        "memory_forget", {"query": "Forget-by-query target two", "confirm": True, "reason": "test cleanup"}
    )
    assert result["forgotten"] == [item.id]


def test_forget_requires_item_id_or_query(server_engine):
    result = call("memory_forget", {"reason": "no target given"})
    assert result["ok"] is False


def test_forget_missing_reason_rejected(server_engine):
    item = server_engine.add_episode("x", project="p1")
    result = call("memory_forget", {"item_id": item.id})
    assert result["ok"] is False


# --------------------------------------------------------- memory_ingest --
def test_ingest_single_file(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path))
    doc = tmp_path / "notes.txt"
    doc.write_text("Line one about FAISS.\nLine two about BM25.\n" * 5, encoding="utf-8")
    result = call("memory_ingest", {"path": str(doc), "project": "p1"})
    assert result["documents"] == 1
    assert result["chunks"] >= 1


def test_ingest_folder(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path))
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("Alpha document text here for ingest.", encoding="utf-8")
    (folder / "b.md").write_text("# Beta\nBeta document text here for ingest.", encoding="utf-8")
    (folder / "c.bin").write_bytes(b"\x00\x01\x02")
    result = call("memory_ingest", {"path": str(folder), "project": "p1"})
    assert result["documents"] == 2  # c.bin skipped (not in default extensions)


def test_ingest_missing_path_errors(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path))
    result = call("memory_ingest", {"path": str(tmp_path / "nope.txt")})
    assert result["ok"] is False


def test_ingest_blocks_path_traversal_outside_home(server_engine, tmp_path, monkeypatch):
    outside = tmp_path.parent / f"outside_{tmp_path.name}"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("should not be ingestible", encoding="utf-8")
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.delenv("MEMORY_WORKSPACE", raising=False)

    traversal_path = str(tmp_path / "home" / ".." / f"outside_{tmp_path.name}" / "secret.txt")
    result = call("memory_ingest", {"path": traversal_path})
    assert result["ok"] is False
    assert "traversal" in result["error"]


def test_ingest_allows_memory_workspace_override(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path / "home_unused"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "w.txt").write_text("workspace document content for ingest test.", encoding="utf-8")
    monkeypatch.setenv("MEMORY_WORKSPACE", str(workspace))
    result = call("memory_ingest", {"path": str(workspace / "w.txt")})
    assert result["documents"] == 1


def test_ingest_pdf_skipped_with_note_when_fitz_missing(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path))
    doc = tmp_path / "file.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "fitz":
            raise ImportError("no fitz")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = call("memory_ingest", {"path": str(doc)})
    assert result["documents"] == 0
    assert any("pdf skipped" in s for s in result["skipped"])


# --------------------------------------------------- memory_list_documents-
def test_list_documents_groups_by_doc_ref(server_engine):
    server_engine.add_document("report.md", "para one. " * 40 + "para two. " * 40, project="p1")
    result = call("memory_list_documents", {"project": "p1"})
    assert len(result["documents"]) == 1
    assert result["documents"][0]["chunks"] >= 1
    assert result["documents"][0]["name"] == "report.md"


def test_list_documents_empty_when_no_docs(server_engine):
    result = call("memory_list_documents", {"project": "p1"})
    assert result["documents"] == []


# ----------------------------------------------- memory_search_occurrences-
def test_search_occurrences_counts_and_snippets(server_engine):
    server_engine.add_episode("The quick brown fox jumps.", project="p1")
    server_engine.add_episode("A slow brown turtle walks.", project="p1")
    result = call("memory_search_occurrences", {"keyword": "brown", "project": "p1"})
    assert result["count"] == 2
    assert len(result["snippets"]) == 2


def test_search_occurrences_no_match(server_engine):
    server_engine.add_episode("Nothing relevant here.", project="p1")
    result = call("memory_search_occurrences", {"keyword": "zzznomatch", "project": "p1"})
    assert result["count"] == 0
    assert result["snippets"] == []


# -------------------------------------------------------- memory_migrate_v3
def test_migrate_v3_happy_path(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path))
    log = tmp_path / "_conversation_log.jsonl"
    rec = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "project": "p1",
        "summary": "Migrated summary.",
        "message_count": 1,
        "messages": [{"role": "user", "content": "migrated message"}],
    }
    log.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    result = call("memory_migrate_v3", {"log_path": str(log)})
    assert result["episodes"] == 1
    assert result["summaries"] == 1


def test_migrate_v3_missing_log_returns_zero_counts(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path))
    result = call("memory_migrate_v3", {"log_path": str(tmp_path / "nope.jsonl")})
    assert result["episodes"] == 0
    assert result["summaries"] == 0


def test_migrate_v3_blocks_traversal(server_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(server.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.delenv("MEMORY_WORKSPACE", raising=False)
    result = call("memory_migrate_v3", {"log_path": "/etc/passwd"})
    assert result["ok"] is False
    assert "traversal" in result["error"]


# --------------------------------------------------------------- selfcheck-
def test_selfcheck_exits_zero(capsys):
    rc = server._selfcheck()
    captured = capsys.readouterr()
    assert rc == 0
    assert "RESULT: PASS" in captured.out
    for name in server.TOOL_MODELS:
        assert name in captured.out


def test_iso_parsing_date_only_and_full_datetime():
    ts_date = server.parse_iso("2026-09-12")
    ts_dt = server.parse_iso("2026-09-12T10:30:00+02:00")
    assert ts_date is not None and ts_dt is not None
    assert ts_date < ts_dt or ts_date <= ts_dt  # both parse without error, same day
    assert server.parse_iso(None) is None
    with pytest.raises(ValueError):
        server.parse_iso("not-a-date")
