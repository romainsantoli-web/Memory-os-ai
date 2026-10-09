"""Reproduce A6 review blockers on synthetic files; no network or real launchctl.

Exit 1 means publication must stay blocked. This is an adverse review probe,
not a test accepting defective behaviour as a specification.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from memory_os_v4.ops import auth, backup
from memory_os_v4.ops.settings import OpsSettings


def main() -> int:
    defects = []
    with tempfile.TemporaryDirectory(prefix="a6-adverse-") as directory:
        root = Path(directory)
        store = root / "store"
        store.mkdir()
        (store / "items.jsonl").write_text('{"id":"test","kind":"fact","text":"Synthetic"}\n')
        settings = OpsSettings("http://127.0.0.1:9/mcp/", root / "token", store,
                               root / "backups", root / "key", root / "ops", "synthetic.test")
        with patch.object(backup, "_launchctl"), patch.object(backup.time, "sleep"), \
                patch.object(backup, "health_ok", return_value=True):
            record = backup.backup(settings)
        if record["ok"] and record["consistent_server_stopped"]:
            defects.append("R1: unverified stop certified consistent")
        extracted = backup.extract(settings.backup_dir / record["archive"],
                                   settings.backup_key.read_bytes(), root / "restore")
        (extracted / "unmanifested.txt").write_text("Synthetic extra file")
        if backup.verify_manifest(extracted)[1] == []:
            defects.append("R3: unmanifested file accepted")
        linked_key = root / "linked-key"
        linked_key.symlink_to(settings.backup_key)
        if backup.load_or_create_key(linked_key) == settings.backup_key.read_bytes():
            defects.append("R2: symlink backup key accepted")
        (store / "items.jsonl").write_text('{"id":"test","kind":"fact"}\nBROKEN\n')
        offline = OpsSettings(**{**settings.__dict__, "service_label": None})
        bad = backup.backup(offline, stop_service=False)
        drill = backup.restore_drill(offline, start_server=False)
        if bad["ok"] and bad["items"]["malformed_lines"] and drill["ok"]:
            defects.append("R4: malformed inconsistent snapshot certified/restored OK")
        settings.token_file.write_text("invalid")
        os.chmod(settings.token_file, 0o644)
        if not auth.create_token(settings.token_file):
            defects.append("R2: existing unsafe token reported already exists without validation")
    print(json.dumps({"release_ready": not defects, "observed_blockers": defects}, indent=2))
    return int(bool(defects))


if __name__ == "__main__":
    raise SystemExit(main())
