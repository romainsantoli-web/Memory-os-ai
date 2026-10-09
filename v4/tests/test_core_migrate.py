"""migrate_v3: reading the v3 conversation log, idempotence, path guard (A6),
in-batch duplicate-line idempotence (A4)."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4
from memory_os_v4.migrate import _parse_v3_timestamp, resolve_under_home


@pytest.fixture(autouse=True)
def _workspace(tmp_path, monkeypatch):
    """A6 (adversarial review, 2026-09-12): migrate_v3's log_path must now
    resolve under $HOME or $MEMORY_WORKSPACE (see resolve_under_home). Every
    test in this file uses pytest's tmp_path, which lives under the system
    temp dir, NOT $HOME -- so point MEMORY_WORKSPACE at it (mirrors the
    pattern test_server_handlers.py uses for the same guard at the server
    layer)."""
    monkeypatch.setenv("MEMORY_WORKSPACE", str(tmp_path))


def _write_v3_log(path, records):
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _sample_records():
    return [
        {
            "timestamp": "2026-06-21T17:54:00.433364+00:00",
            "project": "WaterSense",
            "summary": "WaterSense v0.1.0 created from scratch.",
            "message_count": 2,
            "messages": [
                {"role": "user", "content": "New project: detect water infiltration."},
                {"role": "assistant", "content": "Implemented WaterSense v0.1.0."},
            ],
        },
        {
            "timestamp": "2026-06-22T09:00:00.000000+00:00",
            "project": "WaterSense",
            "summary": "Added spectrogram and transformer model.",
            "message_count": 2,
            "messages": [
                {"role": "user", "content": "Add spectrogram support."},
                {"role": "assistant", "content": "Added STFT/CWT and a Transformer model."},
            ],
        },
    ]


def test_migrate_v3_creates_episodes_and_summaries(tmp_path):
    log_path = tmp_path / "v3.jsonl"
    _write_v3_log(log_path, _sample_records())
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    result = m.migrate_v3(str(log_path))
    assert result["episodes"] == 4
    assert result["summaries"] == 2
    assert result["skipped"] == 0

    summaries = [it for it in m.store.all(include_deleted=False) if it.kind == "summary"]
    assert len(summaries) == 2
    assert all(it.importance == 0.8 for it in summaries)
    assert all("v3" in it.tags for it in summaries)

    episodes = [it for it in m.store.all(include_deleted=False) if it.kind == "episode"]
    assert len(episodes) == 4
    assert all(it.source == "migrate_v3" for it in episodes)
    assert all(it.project == "WaterSense" for it in episodes)


def test_migrate_v3_is_idempotent(tmp_path):
    log_path = tmp_path / "v3.jsonl"
    _write_v3_log(log_path, _sample_records())
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    first = m.migrate_v3(str(log_path))
    second = m.migrate_v3(str(log_path))
    assert second["episodes"] == 0
    assert second["summaries"] == 0
    assert second["skipped"] == first["episodes"] + first["summaries"]
    assert m.status()["counts"]["episode"] == 4
    assert m.status()["counts"]["summary"] == 2


def test_migrate_v3_missing_file_returns_zero_counts(tmp_path):
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    result = m.migrate_v3(str(tmp_path / "does-not-exist.jsonl"))
    assert result == {"episodes": 0, "summaries": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# A6: log_path path-traversal guard, naive timestamps treated as UTC
# ---------------------------------------------------------------------------
def test_migrate_v3_blocks_path_outside_home_and_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_WORKSPACE", raising=False)  # override the autouse fixture
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    with pytest.raises(ValueError, match="path traversal blocked"):
        m.migrate_v3("/etc/hosts")


def test_resolve_under_home_allows_memory_workspace_override(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORY_WORKSPACE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home_unused"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("MEMORY_WORKSPACE", str(workspace))
    target = workspace / "log.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    assert resolve_under_home(str(target)) == target.resolve()


def test_parse_v3_timestamp_naive_iso_treated_as_utc():
    """A6 (adversarial review, 2026-09-12): datetime.fromisoformat on a naive
    (no tzinfo) string, then .timestamp(), silently assumes the LOCAL system
    timezone -- must be pinned to UTC instead, or migrated event_times shift
    by the host's UTC offset."""
    naive = "2026-01-02T03:04:05"
    expected = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()
    assert _parse_v3_timestamp(naive) == expected

    # an explicit offset must still be honoured, not overridden to UTC
    offset = "2026-01-02T03:04:05+02:00"
    expected_offset = datetime.fromisoformat(offset).timestamp()
    assert _parse_v3_timestamp(offset) == expected_offset


def test_migrate_v3_naive_timestamp_end_to_end_matches_utc(tmp_path):
    log_path = tmp_path / "v3_naive.jsonl"
    rec = {
        "timestamp": "2026-01-02T03:04:05",
        "project": "p",
        "summary": None,
        "messages": [{"role": "user", "content": "naive timestamp message"}],
    }
    _write_v3_log(log_path, [rec])
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    m.migrate_v3(str(log_path))
    episode = next(it for it in m.store.all() if it.kind == "episode")
    assert episode.event_time == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# A4: duplicate line WITHIN one migration run (idempotence within one file)
# ---------------------------------------------------------------------------
def test_migrate_v3_duplicated_line_within_one_file_stores_once(tmp_path):
    """A duplicated log line (a replayed/duplicated line, not a re-run of the
    whole migration) used to build 2 MemoryItems sharing the same
    deterministic id -- the store kept only 1, but both got embedded and
    indexed, so `episodes` over-reported vs what was actually stored and
    VectorIndex.ntotal drifted above the true item count."""
    log_path = tmp_path / "v3_dup.jsonl"
    rec = {
        "timestamp": "2026-01-02T03:04:05",
        "project": "p",
        "summary": "S",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    _write_v3_log(log_path, [rec, rec])  # same record twice in ONE file
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    result = m.migrate_v3(str(log_path))

    assert result["episodes"] == 1  # not 2: the duplicate line stores once
    assert result["summaries"] == 1
    assert result["skipped"] == 2  # duplicate episode and summary are both accounted for

    episodes = [it for it in m.store.all(include_deleted=False) if it.kind == "episode"]
    assert len(episodes) == 1
    assert m.status()["vector_count"] == m.status()["total"]  # 1 vector per stored item, no orphans

    # re-running the same (duplicated-line) file again is still fully idempotent
    second = m.migrate_v3(str(log_path))
    assert second["episodes"] == 0
    assert second["summaries"] == 0
    # both (duplicate) lines now match an already-migrated id, each
    # contributing 2 skips (1 summary-check + 1 message-check) -- read_v3_log
    # counts skips per log LINE, not per distinct id, same convention as
    # test_migrate_v3_is_idempotent above.
    assert second["skipped"] == 4
    assert m.status()["counts"]["episode"] == 1


def test_migrate_v3_time_travels_naive_utc(tmp_path):
    """Sanity check that migrated episodes land in the past relative to now
    when given a plausible naive UTC timestamp (guards against an inverted
    UTC/local-time sign error slipping back in)."""
    log_path = tmp_path / "v3_past.jsonl"
    rec = {
        "timestamp": "2020-01-01T00:00:00",
        "project": "p",
        "summary": None,
        "messages": [{"role": "user", "content": "old message"}],
    }
    _write_v3_log(log_path, [rec])
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    m.migrate_v3(str(log_path))
    episode = next(it for it in m.store.all() if it.kind == "episode")
    assert episode.event_time < time.time()


def test_migrate_v3_summary_only_duplicate_records_are_counted_once(tmp_path):
    log_path = tmp_path / "v3_summary_dup.jsonl"
    rec = {"timestamp": "2020-01-01T00:00:00", "project": "p", "summary": "Session summary", "messages": []}
    _write_v3_log(log_path, [rec, rec, rec])
    m = MemoryV4(config=Config(embed_model="hash", data_dir=str(tmp_path / "store")))
    assert m.migrate_v3(str(log_path)) == {"episodes": 0, "summaries": 1, "skipped": 2}
    assert m.status()["vector_count"] == m.status()["total"] == 1
    assert m.migrate_v3(str(log_path)) == {"episodes": 0, "summaries": 0, "skipped": 3}
    summary = next(it for it in m.store.all(False) if it.kind == "summary")
    before = (m.data_dir / "items.jsonl").read_bytes()
    with pytest.raises(ValueError, match="duplicate item id"):
        m.add_facts([{"item_id": summary.id, "kind": "summary", "text": "Unrelated conflicting summary"}], project="p")
    assert (m.data_dir / "items.jsonl").read_bytes() == before
