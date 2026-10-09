"""Tests for memory_os_v4.ops (client integration and operations). No network, no live store."""

from __future__ import annotations

import datetime as dt
import json
import os
import plistlib
from pathlib import Path

import anyio
import pytest

from memory_os_v4.ops import auth, backup, cli, health, hooks
from memory_os_v4.ops.relay import UNCERTAIN, Upstream
from memory_os_v4.ops.settings import OpsSettings

pytest.importorskip("cryptography")


@pytest.fixture()
def settings(tmp_path: Path) -> OpsSettings:
    store = tmp_path / "store"
    store.mkdir()
    rows = [
        {"id": "a", "kind": "fact", "text": "A", "created_at": 1.0},
        {"id": "b", "kind": "episode", "text": "B", "created_at": 1.0},
        {"id": "c", "kind": "fact", "text": "C", "created_at": 1.0},
        {"id": "c", "kind": "fact", "text": "C", "created_at": 1.0, "deleted": True},
    ]
    (store / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (store / "meta.json").write_text("{}")
    return OpsSettings(
        url="http://127.0.0.1:9/mcp/",
        token_file=tmp_path / "cfg/http.token",
        store_dir=store,
        backup_dir=tmp_path / "backups",
        backup_key=tmp_path / "cfg/backup.key",
        ops_dir=tmp_path / "ops",
        service_label=None,
    )


# --- auth -----------------------------------------------------------------------------------------------
def test_token_created_owner_only_and_never_required(settings):
    assert auth.auth_headers(settings) == {}
    assert auth.create_token(settings.token_file) is True
    assert auth.create_token(settings.token_file) is False
    assert oct(os.stat(settings.token_file).st_mode & 0o777) == "0o600"
    header = auth.auth_headers(settings)["Authorization"]
    assert header.startswith("Bearer ") and len(header) >= len("Bearer ") + 43


def test_token_rejects_loose_permissions_and_bad_format(settings):
    settings.token_file.parent.mkdir(parents=True)
    settings.token_file.write_text("x" * 50)
    os.chmod(settings.token_file, 0o644)
    with pytest.raises(PermissionError):
        auth.read_token(settings.token_file)
    os.chmod(settings.token_file, 0o600)
    settings.token_file.write_text("short")
    with pytest.raises(ValueError):
        auth.read_token(settings.token_file)


# --- backup ---------------------------------------------------------------------------------------------
def test_backup_roundtrip_manifest_and_counts(settings, tmp_path):
    rec = backup.backup(settings, stop_service=False)
    assert rec["ok"] and rec["consistent_server_stopped"] is False
    assert rec["items"]["live_total"] == 2 and rec["items"]["ids"] == 3
    archive = settings.backup_dir / rec["archive"]
    assert oct(os.stat(archive).st_mode & 0o777) == "0o600"
    root = backup.extract(archive, settings.backup_key.read_bytes(), tmp_path / "restore")
    manifest, bad = backup.verify_manifest(root)
    assert bad == [] and set(manifest["files"]) == {"items.jsonl", "meta.json"}
    (root / "items.jsonl").write_text("tampered\n")
    assert backup.verify_manifest(root)[1] == ["items.jsonl"]


def test_backup_wrong_key_and_foreign_file_fail(settings):
    rec = backup.backup(settings, stop_service=False)
    blob = (settings.backup_dir / rec["archive"]).read_bytes()
    with pytest.raises(Exception):
        backup.decrypt(blob, os.urandom(32))
    with pytest.raises(ValueError):
        backup.decrypt(b"not-an-archive", settings.backup_key.read_bytes())


def test_consistent_backup_requires_a_service_label(settings):
    with pytest.raises(RuntimeError):
        backup.backup(settings, stop_service=True)


def test_rotation_keeps_7_daily_and_4_sundays(tmp_path):
    start = dt.date(2026, 8, 1)
    for i in range(60):
        day = start + dt.timedelta(days=i)
        (tmp_path / f"memory-os-v4-{day:%Y%m%d}-040000.mosb").write_bytes(b"x")
    backup.rotate(tmp_path)
    kept = sorted(p.name for p in tmp_path.iterdir())
    sundays = [n for n in kept if dt.datetime.strptime(n[13:21], "%Y%m%d").weekday() == 6]
    assert len(kept) <= 11 and len(sundays) >= 4 and kept[-1].startswith("memory-os-v4-20260929")


def test_restore_drill_without_server(settings):
    backup.backup(settings, stop_service=False)
    rec = backup.restore_drill(settings, start_server=False)
    assert rec["ok"] and rec["manifest_mismatches"] == [] and rec["expected_live_total"] == 2


# --- health ---------------------------------------------------------------------------------------------
def test_health_flags_only_new_supersessions_when_confined(settings):
    r1 = health.report(settings, confined=True)
    assert r1["superseded_total"] == 0 and r1["superseded_new"] == []
    with (settings.store_dir / "items.jsonl").open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "id": "a",
                    "kind": "fact",
                    "text": "A",
                    "created_at": 1.0,
                    "valid_until": 2.0,
                    "invalidated_by": "z",
                }
            )
            + "\n"
        )
    r2 = health.report(settings, confined=True)
    assert r2["superseded_new"] == ["a"]
    assert any("new supersession" in a for a in r2["anomalies"])
    assert "server unreachable" in r2["anomalies"]
    assert (settings.ops_dir / "reports" / f"{dt.date.today().isoformat()}.md").exists()


# --- hooks ----------------------------------------------------------------------------------------------
def _transcript(path: Path, names: list[str]) -> Path:
    rows = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": n}]}} for n in names]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_count_tool_calls_resets_on_memory_save(tmp_path):
    t = _transcript(tmp_path / "t.jsonl", ["Bash", "mcp__memory-os-ai__memory_chat_save", "Read", "Edit"])
    assert hooks.count_tool_calls(t) == (4, 2)


def test_stop_nudge_blocks_once_per_window(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "health_ok", lambda *a, **k: True)
    t = _transcript(tmp_path / "t.jsonl", ["Bash"] * 5)
    payload = {"session_id": "s1", "transcript_path": str(t)}
    first = hooks.stop_nudge(payload, settings, threshold=3)
    assert first["decision"] == "block" and "5 tool calls" in first["reason"]
    assert hooks.stop_nudge(payload, settings, threshold=3) is None  # throttled
    assert hooks.stop_nudge({**payload, "stop_hook_active": True}, settings, threshold=3) is None


def test_stop_nudge_silent_when_server_down(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "health_ok", lambda *a, **k: False)
    t = _transcript(tmp_path / "t.jsonl", ["Bash"] * 5)
    assert hooks.stop_nudge({"session_id": "s2", "transcript_path": str(t)}, settings, threshold=3) is None


def test_brief_is_framed_as_data_and_truncated():
    out = hooks.frame_brief("proj", "x" * (hooks.BRIEF_MAX_CHARS + 50))
    assert "never instructions" in out and 'project="proj"' in out and "truncated" in out


def test_project_of_home_means_all_projects(tmp_path):
    assert hooks.project_of(os.path.expanduser("~")) == "*"
    assert hooks.project_of(str(tmp_path)) == tmp_path.name


# --- relay ----------------------------------------------------------------------------------------------
def test_relay_never_replays_a_tool_call(settings):
    up = Upstream(settings)
    calls = {"n": 0}

    async def connect():
        up.session = object()

    async def failing(_session):
        calls["n"] += 1
        raise ConnectionError("dropped")

    up.connect = connect

    async def run():
        up.session = object()
        with pytest.raises(RuntimeError, match="uncertain result"):
            await up.call(failing, replay=False)

    anyio.run(run)
    assert calls["n"] == 1 and "Do not repeat the mutation" in UNCERTAIN


def test_relay_retries_listing_once(settings):
    up = Upstream(settings)
    calls = {"n": 0}

    async def connect():
        up.session = object()

    async def flaky(_session):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("restarted")
        return "ok"

    up.connect = connect

    async def run():
        up.session = object()
        return await up.call(flaky)

    assert anyio.run(run) == "ok" and calls["n"] == 2


# --- setup ----------------------------------------------------------------------------------------------
def test_setup_writes_valid_launchd_plists(settings, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(OpsSettings, "from_env", classmethod(lambda cls: settings))
    assert cli.main(["setup", "--write-launchd", str(tmp_path / "la")]) == 0
    out = capsys.readouterr().out
    assert (
        "memory_os_v4.ops.relay" in out
        and "codex mcp add memory-os-ai" in out
        and "hook session-start" in out
    )
    plists = sorted((tmp_path / "la").glob("*.plist"))
    assert len(plists) == 3
    server = plistlib.loads((tmp_path / "la" / "local.memory-os-v4.plist").read_bytes())
    assert server["KeepAlive"] is True and "MEMORY_V4_TOKEN_FILE" in server["EnvironmentVariables"]
    assert server["EnvironmentVariables"]["MEMORY_V4_SUPERSEDE_COS"] == "1.01"
