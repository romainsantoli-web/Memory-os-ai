"""ADR-001 S1: experimental SQLite implementation of the existing Store API.

Fresh stores only; no migration. Blocks/receipts and whole engine transactions
are NOT integrated yet (S2). Never enable on the live ALPHA store.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
import time

from .store import MemoryItem, Store, VALID_KINDS

MIN_SQLITE = (3, 51, 3)
APPLICATION_ID = 0x4D4F5334
SCHEMA_VERSION = 1

SCHEMA = (
    "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, checksum TEXT NOT NULL, applied_at REAL NOT NULL)",
    "CREATE TABLE store_metadata(key TEXT PRIMARY KEY, value_json TEXT NOT NULL)",
    """CREATE TABLE items(
        id TEXT PRIMARY KEY CHECK(length(id)>0), project TEXT NOT NULL CHECK(length(project)>0 AND project<>'*'),
        kind TEXT NOT NULL, current_revision INTEGER NOT NULL CHECK(current_revision>0),
        FOREIGN KEY(id,current_revision) REFERENCES item_versions(item_id,revision) DEFERRABLE INITIALLY DEFERRED)""",
    """CREATE TABLE item_versions(
        item_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>0), record_json TEXT NOT NULL,
        record_sha256 TEXT NOT NULL, recorded_at REAL NOT NULL, PRIMARY KEY(item_id,revision),
        FOREIGN KEY(item_id) REFERENCES items(id) DEFERRABLE INITIALLY DEFERRED)""",
    "CREATE INDEX items_project_kind ON items(project,kind)",
    "CREATE TRIGGER item_versions_no_update BEFORE UPDATE ON item_versions BEGIN SELECT RAISE(ABORT,'immutable version'); END",
    "CREATE TRIGGER item_versions_no_delete BEFORE DELETE ON item_versions BEGIN SELECT RAISE(ABORT,'immutable version'); END",
)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_sqlite_target(directory: Path) -> None:
    directory = directory.expanduser().resolve()
    for forbidden in (Path.home() / ".memory-os-v4", Path.home() / "Memory-os-ai"):
        forbidden = forbidden.resolve()
        if directory == forbidden or forbidden in directory.parents:
            raise ValueError("Experimental SQLite S1 is forbidden on production/original-repository paths")
    db = directory / "store.sqlite3"
    if (directory / "items.jsonl").exists():
        raise ValueError("No implicit JSONL migration or dual item authority; use a fresh test store")
    if not db.exists() and any((directory / name).exists()
                              for name in ("blocks.jsonl", "requests.jsonl", "meta.json")):
        raise ValueError("SQLite S1 requires a fresh test store; migration is not implemented")
    for name in ("store.sqlite3", "store.sqlite3-wal", "store.sqlite3-shm"):
        path = directory / name
        if path.is_symlink() or (path.exists() and path.stat().st_nlink != 1):
            raise ValueError("Linked SQLite files are forbidden")


class SQLiteStore(Store):
    """Store-compatible item/history/metadata layer with nested transactions.

    Item mutations and metadata inside transaction() commit as one unit. On
    rollback, readers must refetch items (old object references may be stale).
    Embedding files are derived and explicitly outside these transactions.
    """

    def __init__(self, data_dir: Path) -> None:
        if sqlite3.sqlite_version_info < MIN_SQLITE:
            raise RuntimeError("SQLite S1 requires runtime >= 3.51.3 (WAL-reset fix)")
        data_dir = Path(data_dir).expanduser().resolve()
        validate_sqlite_target(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "store.sqlite3"
        fresh = not self.db_path.exists()
        if fresh:
            fd = os.open(self.db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        self._sql_lock = RLock()
        self._depth = 0
        self._poisoned = False
        self._connection = sqlite3.connect(self.db_path, isolation_level=None, timeout=5,
                                           check_same_thread=False)
        try:
            if not fresh:
                app_id = self._connection.execute("PRAGMA application_id").fetchone()[0]
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                if app_id != APPLICATION_ID or version != SCHEMA_VERSION:
                    raise ValueError("Unknown/future SQLite schema; no implicit downgrade")
            self._connection.execute("PRAGMA foreign_keys=ON")
            mode = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA fullfsync=ON")
            self._connection.execute("PRAGMA checkpoint_fullfsync=ON")
            self._connection.execute("PRAGMA busy_timeout=5000")
            settings = {key: self._connection.execute(f"PRAGMA {key}").fetchone()[0]
                        for key in ("foreign_keys", "synchronous", "fullfsync", "checkpoint_fullfsync")}
            if mode != "wal" or settings != {"foreign_keys": 1, "synchronous": 2,
                                            "fullfsync": 1, "checkpoint_fullfsync": 1}:
                raise RuntimeError("Required SQLite WAL/durability settings unavailable")
            if fresh:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    for statement in SCHEMA:
                        self._connection.execute(statement)
                    checksum = hashlib.sha256("\n".join(SCHEMA).encode()).hexdigest()
                    self._connection.execute("INSERT INTO schema_migrations VALUES(?,?,?)",
                                             (SCHEMA_VERSION, checksum, time.time()))
                    self._connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
                    self._connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                    self._connection.execute("COMMIT")
                except BaseException:
                    self._connection.execute("ROLLBACK")
                    raise
            super().__init__(data_dir)
            if not self.check_integrity():
                raise RuntimeError("SQLite source integrity failed")
        except BaseException:
            self._connection.close()
            raise

    @contextmanager
    def transaction(self):
        """BEGIN IMMEDIATE / savepoints, serializing use across worker threads."""
        with self._sql_lock:
            if self._poisoned:
                raise RuntimeError("SQLite commit outcome uncertain; inspect/reopen before further writes")
            level = self._depth
            savepoint = f"store_{level}"
            self._connection.execute("BEGIN IMMEDIATE" if level == 0 else f"SAVEPOINT {savepoint}")
            self._depth += 1
            committing = False
            try:
                yield self
                committing = True
                self._connection.execute("COMMIT" if level == 0 else f"RELEASE SAVEPOINT {savepoint}")
            except BaseException:
                if committing:
                    self._poisoned = True  # No blind retry after an ambiguous commit exception.
                if self._connection.in_transaction:
                    if level == 0:
                        self._connection.execute("ROLLBACK")
                    else:
                        self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                self.load()
                self.load_chunk_map()
                self._dirty_access.clear()
                raise
            finally:
                self._depth -= 1

    def load(self) -> None:
        with self._sql_lock:
            self.items.clear()
            for item_id, project, kind, record_json in self._connection.execute(
                "SELECT i.id,i.project,i.kind,v.record_json FROM items i JOIN item_versions v "
                "ON v.item_id=i.id AND v.revision=i.current_revision"
            ):
                item = MemoryItem.from_dict(json.loads(record_json))
                if (item.id, item.project, item.kind) != (item_id, project, kind):
                    raise RuntimeError("SQLite head fields disagree with their record")
                self.items[item.id] = item
            row = self._connection.execute("SELECT value_json FROM store_metadata WHERE key='metadata'").fetchone()
            self.meta = json.loads(row[0]) if row else {}

    def append_many(self, items: list[MemoryItem]) -> None:
        if not items:
            return
        with self.transaction():
            for item in items:
                if (not item.id or not isinstance(item.text, str) or item.kind not in VALID_KINDS
                        or not isinstance(item.project, str) or not item.project or item.project == "*"):
                    raise ValueError("Invalid SQLite item identity/kind/project/text")
                old = self._connection.execute("SELECT project,kind,current_revision FROM items WHERE id=?",
                                               (item.id,)).fetchone()
                if old and (item.project, item.kind) != old[:2]:
                    raise ValueError("An item cannot change project/kind")
                revision = old[2] + 1 if old else 1
                record = _json(item.to_dict())
                self._connection.execute("INSERT INTO item_versions VALUES(?,?,?,?,?)",
                                         (item.id, revision, record, hashlib.sha256(record.encode()).hexdigest(), time.time()))
                self._connection.execute("INSERT INTO items VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE "
                                         "SET current_revision=excluded.current_revision",
                                         (item.id, item.project, item.kind, revision))
                self.items[item.id] = item

    def save_meta(self, meta: dict) -> None:
        with self.transaction():
            self._connection.execute("INSERT INTO store_metadata VALUES('metadata',?) "
                                     "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json", (_json(meta),))
            self.meta = dict(meta)

    def load_chunk_map(self) -> None:
        with self._sql_lock:
            row = self._connection.execute("SELECT value_json FROM store_metadata WHERE key='chunk_map'").fetchone()
            self.chunk_owner = json.loads(row[0]) if row else {}

    def save_chunk_map(self, chunk_owner: dict[str, str] | None = None) -> None:
        with self.transaction():
            value = self.chunk_owner if chunk_owner is None else chunk_owner
            self._connection.execute("INSERT INTO store_metadata VALUES('chunk_map',?) "
                                     "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json", (_json(value),))
            self.chunk_owner = dict(value)

    def flush_access(self) -> None:
        with self.transaction():
            super().flush_access()

    def history(self, item_id: str) -> list[MemoryItem]:
        with self._sql_lock:
            return [MemoryItem.from_dict(json.loads(row[0])) for row in self._connection.execute(
                "SELECT record_json FROM item_versions WHERE item_id=? ORDER BY revision", (item_id,))]

    def check_integrity(self) -> bool:
        with self._sql_lock:
            try:
                if self._poisoned or not self.db_path.is_file():
                    return False
                if (self._connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                        or self._connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION):
                    return False
                if self._connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    return False
                if self._connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    return False
                expected = hashlib.sha256("\n".join(SCHEMA).encode()).hexdigest()
                if self._connection.execute("SELECT version,checksum FROM schema_migrations").fetchall() != [
                    (SCHEMA_VERSION, expected)
                ]:
                    return False
                for item_id, record, checksum in self._connection.execute(
                    "SELECT item_id,record_json,record_sha256 FROM item_versions"
                ):
                    if hashlib.sha256(record.encode()).hexdigest() != checksum:
                        return False
                    item = MemoryItem.from_dict(json.loads(record))
                    if (item.id != item_id or not isinstance(item.text, str) or item.kind not in VALID_KINDS
                            or not isinstance(item.project, str) or not item.project or item.project == "*"):
                        return False
                if self._connection.execute(
                    "SELECT i.id FROM items i WHERE i.current_revision <> "
                    "(SELECT max(v.revision) FROM item_versions v WHERE v.item_id=i.id) LIMIT 1"
                ).fetchone() is not None:
                    return False
                heads = {}
                for item_id, project, kind, record in self._connection.execute(
                    "SELECT i.id,i.project,i.kind,v.record_json FROM items i JOIN item_versions v "
                    "ON v.item_id=i.id AND v.revision=i.current_revision"
                ):
                    heads[item_id] = json.loads(record)
                    if (heads[item_id]["id"], heads[item_id]["project"], heads[item_id]["kind"]) != (item_id, project, kind):
                        return False
                if set(heads) != set(self.items):
                    return False
                for item_id, record in heads.items():
                    current = self.items[item_id].to_dict()
                    if any(current[key] != value for key, value in record.items()
                           if key in current and key not in {"access_count", "last_accessed"}):
                        return False
                for row in self._connection.execute("SELECT value_json FROM store_metadata"):
                    if not isinstance(json.loads(row[0]), dict):
                        return False
                return True
            except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
                return False

    def close(self) -> None:
        with self._sql_lock:
            self._connection.close()
