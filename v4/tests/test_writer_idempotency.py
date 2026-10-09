"""Offline tests for the single writer lease and durable MCP retry contract."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from memory_os_v4 import memory, server
from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4
from memory_os_v4.writer_lock import WriterConflictError


def _engine(path):
    return MemoryV4(config=Config(data_dir=str(path), embed_model="hash"))


def _child(path, hold=False):
    src = Path(__file__).resolve().parents[1] / "src"
    code = """
import sys
from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4
try:
    with MemoryV4(config=Config(data_dir=sys.argv[1], embed_model='hash')):
        print('acquired', flush=True)
        if sys.argv[2] == 'hold':
            input()
except Exception as exc:
    print(str(exc), flush=True)
    sys.exit(3)
"""
    return subprocess.Popen([sys.executable, "-c", code, str(path), "hold" if hold else "exit"],
                            env={**os.environ, "PYTHONPATH": str(src), "HF_HUB_OFFLINE": "1"},
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_same_process_refusal_close_and_closed_handle(tmp_path):
    first = _engine(tmp_path)
    with pytest.raises(WriterConflictError, match="shared HTTP"):
        _engine(tmp_path / ".")
    first.close()
    first.close()
    with _engine(tmp_path) as second:
        second.add_episode("New owner")
        with pytest.raises(RuntimeError, match="closed"):
            first.flush()
        with pytest.raises(RuntimeError, match="closed"):
            first.add_episode("Stale writer")
    assert (tmp_path / "writer.lock").exists()


def test_subprocess_refusal_then_acquisition_after_release(tmp_path):
    with _engine(tmp_path):
        child = _child(tmp_path)
        out, err = child.communicate(timeout=20)
        assert child.returncode == 3, err
        assert "shared HTTP" in out
    child = _child(tmp_path)
    out, err = child.communicate(timeout=20)
    assert child.returncode == 0, err
    assert "acquired" in out


def test_process_exit_releases_lock_without_deleting_file(tmp_path):
    child = _child(tmp_path, hold=True)
    try:
        assert child.stdout.readline().strip() == "acquired"
        with pytest.raises(WriterConflictError):
            _engine(tmp_path)
    finally:
        child.kill()
        child.communicate(timeout=20)
    with _engine(tmp_path):
        pass


def test_failed_initialization_releases_lease(tmp_path, monkeypatch):
    original = memory._make_embedder
    def fail(config):
        raise RuntimeError("injected init failure")
    monkeypatch.setattr(memory, "_make_embedder", fail)
    with pytest.raises(RuntimeError, match="injected"):
        _engine(tmp_path)
    monkeypatch.setattr(memory, "_make_embedder", original)
    with _engine(tmp_path):
        pass


def test_close_releases_lease_even_when_flush_fails(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(engine, "flush", lambda: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError, match="disk error"):
        engine.close()
    with _engine(tmp_path):
        pass


@pytest.fixture
def bound_engine(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setenv("MEMORY_V4_PROJECT", "alpha")
    yield engine
    engine.close()


def test_concurrent_chat_retries_and_restart_replay_exact_result(bound_engine, tmp_path, monkeypatch):
    payload = {"request_id": "turn-1", "messages": [{"role": "user", "content": "Saved turn"}],
               "facts": [{"text": "Alice uses Python."}]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: server.dispatch("memory_chat_save", payload), range(2)))
    assert "error" not in responses[0], responses
    assert responses[0] == responses[1]
    assert len(bound_engine.store.all()) == 2
    bound_engine.close()
    with _engine(tmp_path) as reloaded:
        monkeypatch.setattr(server, "get_engine", lambda: reloaded)
        assert server.dispatch("memory_chat_save", payload) == responses[0]
        assert len(reloaded.store.all()) == 2
        changed = {**payload, "summary": "Different payload"}
        assert "different operation or payload" in server.dispatch("memory_chat_save", changed)["error"]


def test_request_id_is_scoped_and_implicit_project_matches_explicit(bound_engine):
    payload = {"request_id": "same", "facts": [{"text": "Alice likes tea."}]}
    first = server.dispatch("memory_add_facts", payload)
    assert server.dispatch("memory_add_facts", {**payload, "project": "alpha"}) == first
    other = server.dispatch("memory_add_facts", {**payload, "project": "beta",
                                                "facts": [{"text": "Bob repairs bicycles."}]})
    assert other["added"] != first["added"]
    assert bound_engine.store.get(other["added"][0]).project == "beta"
    conflict = server.dispatch("memory_chat_save", {"request_id": "same", "project_label": "alpha",
                                                    "messages": [{"role": "user", "content": "new"}]})
    assert "different operation or payload" in conflict["error"]


def test_block_append_and_forget_replay_original_result(bound_engine):
    payload = {"request_id": "append", "name": "project_state", "text": "one", "mode": "append"}
    first = server.dispatch("memory_block_set", payload)
    assert server.dispatch("memory_block_set", payload) == first
    assert bound_engine.blocks.get("project_state") == "one"
    item = bound_engine.add_episode("Please forget me", project="alpha")
    forget = {"request_id": "forget", "item_id": item.id, "reason": "test"}
    first = server.dispatch("memory_forget", forget)
    assert first["forgotten"] == [item.id]
    assert server.dispatch("memory_forget", forget) == first


def test_pending_partial_mutation_is_explicit_and_never_automatically_retried(bound_engine, tmp_path, monkeypatch):
    def fail(engine, args):
        engine.add_episode("Partially applied", project="alpha")
        raise OSError("injected failure after mutation")
    monkeypatch.setitem(server.TOOL_HANDLERS, "memory_chat_save", fail)
    payload = {"request_id": "partial", "messages": [{"role": "user", "content": "partial"}]}
    assert "injected failure" in server.dispatch("memory_chat_save", payload)["error"]
    assert "incomplete outcome" in server.dispatch("memory_chat_save", payload)["error"]
    assert len(bound_engine.store.all()) == 1
    bound_engine.close()
    with _engine(tmp_path) as reloaded:
        monkeypatch.setattr(server, "get_engine", lambda: reloaded)
        assert "incomplete outcome" in server.dispatch("memory_chat_save", payload)["error"]
        assert len(reloaded.store.all()) == 1


def test_damaged_journal_fails_closed_and_releases_lease(tmp_path):
    (tmp_path / "requests.jsonl").write_text('{"scope":', encoding="utf-8")
    with pytest.raises(RuntimeError, match="journal is damaged"):
        _engine(tmp_path)
    (tmp_path / "requests.jsonl").unlink()
    with _engine(tmp_path):
        pass


def test_idempotency_covers_all_mutating_tool_schemas():
    names = {"memory_chat_save", "memory_add_facts", "memory_block_set", "memory_consolidate",
             "memory_compact", "memory_forget", "memory_ingest", "memory_migrate_v3"}
    for name in names:
        assert "request_id" in server.TOOL_MODELS[name].model_json_schema()["properties"]


def test_wildcard_consolidation_has_a_distinct_scope(bound_engine):
    wildcard = {"project": "*", "dry_run": False}
    concrete = {"project": "alpha", "dry_run": False}
    assert server._request_scope("memory_consolidate", wildcard) != server._request_scope("memory_consolidate", concrete)
    assert wildcard["project"] == "*"
    # These represent different scopes, and must not replay each other's result.
    assert bound_engine.execute_request("memory_consolidate", "*", wildcard, "same",
                                        lambda: {"target": "*"}) == {"target": "*"}
    assert bound_engine.execute_request("memory_consolidate", "alpha", concrete, "same",
                                        lambda: {"target": "alpha"}) == {"target": "alpha"}


def test_global_block_scope_does_not_depend_on_server_default(bound_engine, monkeypatch):
    first = {"scope": "global", "project": None, "name": "user_profile", "text": "one"}
    scope1 = server._request_scope("memory_block_set", first)
    monkeypatch.setenv("MEMORY_V4_PROJECT", "beta")
    second = {"scope": "global", "project": "beta", "name": "user_profile", "text": "one"}
    assert server._request_scope("memory_block_set", second) == scope1
    assert first == second


def test_response_mutation_cannot_change_cached_receipt(bound_engine):
    payload = {"request_id": "receipt-copy", "facts": [{"text": "Dana paints."}]}
    result = server.dispatch("memory_add_facts", payload)
    ids = list(result["added"])
    result["added"].clear()
    assert server.dispatch("memory_add_facts", payload)["added"] == ids


def test_failed_journal_sync_prevents_mutation_and_further_requests(bound_engine, monkeypatch):
    def fail_sync():
        raise OSError("injected journal sync error")
    monkeypatch.setattr(bound_engine.requests, "_sync_directory", fail_sync)
    payload = {"request_id": "unsynced", "messages": [{"role": "user", "content": "not applied"}]}
    assert "injected journal sync error" in server.dispatch("memory_chat_save", payload)["error"]
    assert bound_engine.store.all() == []
    assert "journal write failed" in server.dispatch("memory_chat_save", {**payload, "request_id": "other"})["error"]
