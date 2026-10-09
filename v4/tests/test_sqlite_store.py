"""ADR-001 S1: transactional Store compatibility on fresh synthetic stores."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from memory_os_v4 import Config, MemoryItem, MemoryV4
from memory_os_v4.readiness import engine_readiness
from memory_os_v4.sqlite_store import APPLICATION_ID, MIN_SQLITE, SQLiteStore

pytestmark = pytest.mark.skipif(sqlite3.sqlite_version_info < MIN_SQLITE,
                               reason="SQLite S1 requires >= 3.51.3; this runtime is not qualified")


def item(item_id="one", text="Synthetic fact", **kwargs):
    return MemoryItem(id=item_id, kind="fact", text=text, **{"project": "A", **kwargs})


@pytest.fixture
def store(tmp_path):
    value = SQLiteStore(tmp_path)
    try:
        yield value
    finally:
        value.close()


def test_interface_persistence_history_metadata_and_required_settings(store, tmp_path):
    store.append(item())
    store.append(item(text="Updated synthetic fact"))
    store.save_meta({"profile": "synthetic"})
    store.save_chunk_map({"chunk-1": "one"})
    store.touch("one", t=200)
    store.flush_access()
    assert store.get("one").access_count == 1
    assert len(store.history("one")) == 3
    assert not (tmp_path / "items.jsonl").exists() and not (tmp_path / "meta.json").exists()
    assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert store._connection.execute("PRAGMA fullfsync").fetchone()[0] == 1


def test_reopen_full_records_and_metadata(store, tmp_path):
    store.append(item(source="client:test", parent_ids=["external-provenance"], tags=["test"],
                      identity={"subject_id": "synthetic", "predicate": "state", "cardinality": "single"}))
    store.save_meta({"profile": "synthetic"})
    store.save_chunk_map({"c": "one"})
    reopened = SQLiteStore(tmp_path)
    try:
        assert reopened.get("one").to_dict() == store.get("one").to_dict()
        assert reopened.meta == store.meta and reopened.chunk_owner == store.chunk_owner
        assert reopened.check_integrity()
    finally:
        reopened.close()


def test_bulk_append_rolls_back_all_rows_and_memory_on_late_failure(store):
    with pytest.raises(ValueError):
        store.append_many([item("valid"), item("bad", project="*")])
    assert store.all() == []
    assert store._connection.execute("SELECT count(*) FROM item_versions").fetchone()[0] == 0


def test_explicit_transaction_groups_history_heads_metadata_and_chunkmap(store):
    store.append(item())
    with pytest.raises(RuntimeError, match="injected"):
        with store.transaction():
            store.get("one").text = "Must roll back"
            store.append(store.get("one"))
            store.append(item("two"))
            store.save_meta({"must": "roll back"})
            store.save_chunk_map({"orphan": "two"})
            raise RuntimeError("injected failure")
    assert store.get("one").text == "Synthetic fact" and store.get("two") is None
    assert len(store.history("one")) == 1 and store.meta == {} and store.chunk_owner == {}
    assert store.check_integrity()


def test_inner_savepoint_failure_preserves_outer_changes(store):
    with store.transaction():
        store.append(item("outer"))
        with pytest.raises(ValueError):
            with store.transaction():
                store.append(item("inner"))
                raise ValueError("injected inner failure")
        store.save_meta({"committed": True})
    assert store.get("outer") and store.get("inner") is None
    assert store.meta == {"committed": True}


def test_parallel_worker_threads_serialize_one_connection(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: store.append(item(str(i))), range(12)))
    assert len(store.all()) == 12 and store.check_integrity()


def test_direct_sql_corruption_detected(store):
    store.append(item())
    store._connection.execute("DROP TRIGGER item_versions_no_update")
    store._connection.execute("UPDATE item_versions SET record_json=?", ('{}',))
    assert not store.check_integrity()


def test_history_is_immutable_and_database_failure_rolls_back_batch(store):
    store.append(item())
    for sql in ("UPDATE item_versions SET recorded_at=0", "DELETE FROM item_versions"):
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute(sql)
    store._connection.execute("CREATE TRIGGER inject_failure BEFORE INSERT ON item_versions "
                              "WHEN NEW.item_id='bad' BEGIN SELECT RAISE(ABORT,'injected SQL failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.append_many([item("valid"), item("bad")])
    assert store.get("valid") is None and store.get("bad") is None
    assert len(store.all()) == 1 and store.check_integrity()


def test_ambiguous_commit_blocks_further_writes_and_readiness_until_reopen(store, tmp_path):
    class AmbiguousCommit:
        def __init__(self, connection):
            self.connection = connection
            self.inject = True

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *args):
            result = self.connection.execute(sql, *args)
            if sql == "COMMIT" and self.inject:
                self.inject = False
                raise sqlite3.OperationalError("injected acknowledgement lost after commit")
            return result

    store._connection = AmbiguousCommit(store._connection)
    with pytest.raises(sqlite3.OperationalError, match="acknowledgement"):
        store.append(item())
    assert not store.check_integrity()
    with pytest.raises(RuntimeError, match="uncertain"):
        store.append(item("blind-retry"))
    reopened = SQLiteStore(tmp_path)
    try:
        assert reopened.get("one") and reopened.get("blind-retry") is None
        assert reopened.check_integrity()
    finally:
        reopened.close()


def test_future_schema_and_foreign_database_refused_without_downgrade(tmp_path):
    db = tmp_path / "store.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
    conn.execute("PRAGMA user_version=2")
    conn.close()
    before = db.read_bytes()
    with pytest.raises(ValueError, match="future"):
        SQLiteStore(tmp_path)
    assert db.read_bytes() == before


def test_jsonl_source_and_production_paths_refused_before_writes(tmp_path, monkeypatch):
    (tmp_path / "items.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="implicit"):
        MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash", store_backend="sqlite_s1"))
    assert not (tmp_path / "writer.lock").exists() and not (tmp_path / "store.sqlite3").exists()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    live = tmp_path / ".memory-os-v4"
    with pytest.raises(ValueError, match="forbidden"):
        MemoryV4(config=Config(data_dir=str(live), embed_model="hash", store_backend="sqlite_s1"))
    assert not live.exists()


def test_explicit_flag_config_and_refusal_of_silent_backend_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_V4_STORE_BACKEND", "sqlite_s1")
    assert Config.from_env().store_backend == "sqlite_s1" and Config().store_backend == "jsonl"
    with MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash", store_backend="sqlite_s1")) as engine:
        engine.add_episode("Synthetic episode", project="A")
        assert isinstance(engine.store, SQLiteStore)
        assert engine_readiness(engine) == (True, "ready")
    with pytest.raises(ValueError, match="explicit backend"):
        MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash"))
    with MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash", store_backend="sqlite_s1")) as engine:
        assert engine.status(project="A")["total"] == 1


def test_inc001_and_inc002_regressions_on_sqlite_engine(tmp_path):
    relay = "On 2026-10-09 Claude configured the Claude desktop relay for memory-os-ai."
    hook = "On 2026-10-09 Claude configured the Claude Code hook for memory-os-ai."
    with MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash", store_backend="sqlite_s1",
                              dedup_cos=0, supersede_cos=0)) as engine:
        for text in (relay, hook, "BTC A5 completed.", "ETH A5 completed."):
            result = engine.add_facts([{"text": text, "entities": ["Claude", "2026-10-09"]}],
                                      project="A", event_time=100)
            assert result["added"] and not result["deduped"] and not result["superseded"]
        assert engine.consolidate(project="A")["superseded"] == 0
        assert len(engine.store.all(False)) == 4
        identity = {"subject_id": "user:synthetic", "predicate": "residence", "cardinality": "single"}
        old = engine.add_facts([{"text": "User lives in Paris.", "identity": identity}],
                               project="A", event_time=100)["added"][0]
        new = engine.add_facts([{"text": "User lives in Lyon.", "identity": identity}],
                               project="A", event_time=200)["added"][0]
        assert engine.store.get(old).invalidated_by == new
        assert len(engine.store.history(old)) >= 2


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_killed_process_retains_whole_transaction_or_none(tmp_path, phase):
    code = '''
import os, sys
from pathlib import Path
from memory_os_v4.sqlite_store import SQLiteStore
from memory_os_v4.store import MemoryItem
s=SQLiteStore(Path(sys.argv[1]))
with s.transaction():
    s.append(MemoryItem(id='one',kind='fact',text='Synthetic one'))
    s.append(MemoryItem(id='two',kind='fact',text='Synthetic two'))
    if sys.argv[2]=='before_commit': os._exit(9)
print('committed',flush=True)
os._exit(9)
'''
    src = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), phase],
                            env={**os.environ, "PYTHONPATH": src, "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 9, result.stderr
    reopened = SQLiteStore(tmp_path)
    try:
        assert {r.id for r in reopened.all()} == ({"one", "two"} if phase == "after_commit" else set())
        assert reopened.check_integrity()
    finally:
        reopened.close()
