"""Durable request receipts, serialized by MemoryV4's lifetime lease and RLock.

Pending reservations deliberately survive exceptions and crashes. JSONL has no
transaction spanning all mutations: an interrupted request may have partly
applied. Refuse its retry explicitly instead of silently applying it twice.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable


class RequestConflictError(ValueError):
    pass


class RequestIncompleteError(RuntimeError):
    pass


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RequestJournal:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "requests.jsonl"
        self.records: dict[tuple[str, str], dict] = {}
        self._poisoned = False
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = (record["scope"], record["request_id"])
                    if record["state"] not in {"pending", "complete"}:
                        raise ValueError("unknown request state")
                    if not isinstance(record["fingerprint"], str):
                        raise ValueError("invalid fingerprint")
                    if record["state"] == "complete" and "result" not in record:
                        raise ValueError("missing result")
                    self.records[key] = record
                except (ValueError, KeyError, TypeError) as exc:
                    raise RuntimeError(
                        "Request journal is damaged; restore or inspect requests.jsonl before retrying writes."
                    ) from exc

    def _append(self, record: dict) -> None:
        line = _json(record) + "\n"
        if record["state"] == "pending":
            # Even a failed fsync has an uncertain durable outcome; never rerun
            # that request automatically in this process.
            self.records[(record["scope"], record["request_id"])] = record
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._sync_directory()
        except BaseException:
            # A partial append must not be followed by more journal records.
            self._poisoned = True
            raise
        self.records[(record["scope"], record["request_id"])] = json.loads(line)

    def _sync_directory(self) -> None:
        if os.name != "nt":
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    def execute(self, scope: str, request_id: str, operation: str, payload: dict,
                action: Callable[[], dict], sync: Callable[[], None]) -> dict:
        if self._poisoned:
            raise RuntimeError("Request journal write failed; inspect requests.jsonl and restart before retrying writes")
        fingerprint = hashlib.sha256(_json({"operation": operation, "payload": payload}).encode()).hexdigest()
        key = (scope, request_id)
        previous = self.records.get(key)
        if previous is not None:
            if previous["fingerprint"] != fingerprint:
                raise RequestConflictError("request_id already used with a different operation or payload in this scope")
            if previous["state"] != "complete":
                raise RequestIncompleteError(
                    "request_id has an incomplete outcome; inspect the store before issuing a new request. "
                    "The previous attempt may have partially applied and will not be replayed automatically."
                )
            # A fresh copy keeps callers from modifying the durable response in memory.
            return json.loads(_json(previous["result"]))
        record = {"scope": scope, "request_id": request_id, "fingerprint": fingerprint, "state": "pending"}
        self._append(record)
        result = action()
        sync()
        self._append({**record, "state": "complete", "result": result})
        return result


def sync_sources(data_dir: Path) -> None:
    """Make authoritative mutations durable before a completed receipt is written.

    Embedding/BM25/entity files are disposable caches reconciled on startup.
    """
    for name in ("items.jsonl", "blocks.jsonl", "meta.json"):
        path = data_dir / name
        if path.exists():
            with path.open("rb") as fh:
                os.fsync(fh.fileno())
    if os.name != "nt":
        fd = os.open(data_dir, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
