"""Safety and invariants of the offline INC-001 repair (synthetic stores)."""
import copy
import fcntl
import json
import os
from pathlib import Path

import pytest

from tools import repair_incident_0941 as tool


@pytest.fixture
def incident(tmp_path):
    rows = [{"id": tool.CULPRIT, "kind": "fact", "project": tool.PROJECT, "event_time": 200}]
    for prefix in tool.TARGETS:
        before = {"id": prefix + "0" * 20, "kind": "fact", "project": tool.PROJECT,
                  "text": "Synthetic evidence", "source": "client", "parent_ids": ["episode-1"],
                  "event_time": 100, "valid_from": 100, "valid_until": None,
                  "invalidated_by": None, "deleted": False, "unknown_metadata": {"keep": True}}
        after = {**before, "valid_until": 200, "invalidated_by": tool.CULPRIT, "access_count": 3}
        rows.extend([before, after])
    rows.append({"id": "unrelated", "kind": "episode", "text": "Unchanged"})
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "writer.lock").touch()
    (tmp_path / "cache").write_bytes(b"unchanged cache")
    return tmp_path


def test_simulation_writes_nothing(incident):
    before = {p.name: p.read_bytes() for p in incident.iterdir()}
    result = tool.repair(incident)
    assert result["mode"] == "simulation" and result["count"] == 9
    assert before == {p.name: p.read_bytes() for p in incident.iterdir()}


def test_only_nine_validity_pairs_change_with_history_and_provenance_intact(incident):
    before = (incident / "items.jsonl").read_bytes()
    latest = {r["id"]: r for r in map(json.loads, before.splitlines())}
    assert tool.repair(incident, apply=True, offline=True)["count"] == 9
    after = (incident / "items.jsonl").read_bytes()
    assert after.startswith(before)
    suffix = list(map(json.loads, after[len(before):].splitlines()))
    assert len(suffix) == 9
    for restored in suffix:
        original = latest[restored["id"]]
        assert {k for k in original if original[k] != restored[k]} == {"valid_until", "invalidated_by"}
        assert restored["valid_until"] is None and restored["invalidated_by"] is None
    assert (incident / "cache").read_bytes() == b"unchanged cache"
    assert (incident / "writer.lock").read_bytes() == b""
    trace = json.loads((incident / tool.TRACE).read_text())
    assert trace["status"] == "complete" and trace["after_sha256"] == tool.digest(after)
    assert set(p.name for p in incident.iterdir()) == {"items.jsonl", "cache", "writer.lock", tool.TRACE}
    with pytest.raises(ValueError, match="already exists"):
        tool.repair(incident, apply=True, offline=True)
    assert (incident / "items.jsonl").read_bytes() == after


@pytest.mark.parametrize("change", ["missing", "ambiguous", "deleted", "other_culprit", "other_project",
                                   "other_time", "previously_invalid", "truncated", "malformed"])
def test_unexpected_history_refused_before_any_write(incident, change):
    path = incident / "items.jsonl"
    rows = list(map(json.loads, path.read_bytes().splitlines()))
    if change == "missing":
        rows = [r for r in rows if not r["id"].startswith(tool.TARGETS[0])]
    elif change == "ambiguous":
        other = copy.deepcopy(rows[1])
        other["id"] = tool.TARGETS[0] + "1" * 20
        rows.append(other)
    elif change == "previously_invalid":
        rows[1]["valid_until"] = 150
    elif change not in {"truncated", "malformed"}:
        key, value = {"deleted": ("deleted", True), "other_culprit": ("invalidated_by", "other"),
                      "other_project": ("project", "other"), "other_time": ("valid_until", 201)}[change]
        rows[2][key] = value
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    if change == "truncated":
        data = data[:-1]
    if change == "malformed":
        data += b"not json\n"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        tool.repair(incident, apply=True, offline=True)
    assert path.read_bytes() == data and not (incident / tool.TRACE).exists()


def test_offline_acknowledgement_and_writer_lease_required(incident):
    with pytest.raises(ValueError, match="offline"):
        tool.repair(incident, apply=True)
    with (incident / "writer.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            tool.repair(incident, apply=True, offline=True)
    assert not (incident / tool.TRACE).exists()


@pytest.mark.parametrize("name", [".memory-os-v4", "Memory-os-ai"])
def test_live_paths_always_forbidden(tmp_path, monkeypatch, name):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    forbidden = tmp_path / name
    forbidden.mkdir()
    with pytest.raises(ValueError, match="forbidden"):
        tool.repair(forbidden)


@pytest.mark.parametrize("link", ["symbolic", "hard"])
def test_linked_log_forbidden(incident, tmp_path, link):
    path = incident / "items.jsonl"
    original = tmp_path / "original"
    path.rename(original)
    if link == "symbolic":
        path.symlink_to(original)
    else:
        os.link(original, path)
    with pytest.raises(ValueError, match="links"):
        tool.repair(incident, apply=True, offline=True)


def test_failure_after_append_is_explicit_and_not_replayed(incident, monkeypatch):
    original = tool.write_trace

    def fail_completion(directory, trace, *, create=False):
        if not create:
            raise OSError("Injected trace completion failure")
        original(directory, trace, create=create)

    monkeypatch.setattr(tool, "write_trace", fail_completion)
    with pytest.raises(OSError):
        tool.repair(incident, apply=True, offline=True)
    data = (incident / "items.jsonl").read_bytes()
    assert json.loads((incident / tool.TRACE).read_text())["status"] == "pending"
    with pytest.raises(ValueError, match="reconcile"):
        tool.repair(incident, apply=True, offline=True)
    assert (incident / "items.jsonl").read_bytes() == data
