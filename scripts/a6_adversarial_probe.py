"""Adverse A6 review on synthetic files; no network or real launchctl.

Run against the reviewed package via PYTHONPATH. Exit 1 is a release refusal,
not acceptance of defective behaviour. No production data or secrets are read.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import plistlib
import tempfile
from unittest.mock import patch

from memory_os_v4.ops import auth, backup, cli
from memory_os_v4.ops.settings import OpsSettings


def main() -> int:
    defects, passed = [], []
    with tempfile.TemporaryDirectory(prefix="a6-adverse-v2-") as directory:
        root = Path(directory)
        store = root / "store"
        store.mkdir()
        valid = b'{"id":"synthetic","kind":"fact","text":"Synthetic fact","project":"test"}\n'
        (store / "items.jsonl").write_bytes(valid)
        settings = OpsSettings("http://127.0.0.1:9/mcp/", root / "token", store,
                               root / "backups", root / "key", root / "ops", "synthetic.test")
        with patch.object(backup, "_service_loaded", return_value=True), \
                patch.object(backup, "_launchctl", return_value=1), \
                patch.object(backup, "_wait_healthy", return_value=False):
            refused = backup.backup(settings)
        if not refused["ok"] and not refused.get("archive"):
            passed.append("R1: failed bootout refused")
        else:
            defects.append("R1: failed bootout certified")

        # A2 maintenance creates writer.lock; the live 4.0.0 engine ignores it.
        (store / "writer.lock").touch()
        with patch.object(backup, "_service_loaded", return_value=True), \
                patch.object(backup, "_writer_processes", return_value=["123 memory_os_v4.server"]), \
                patch.object(backup, "_port_open", return_value=True), \
                patch.object(backup, "_launchctl", return_value=1) as launchctl:
            record = backup.backup(settings, now=dt.datetime(2026, 10, 9, 1))
        if record["ok"] and record["qualified"] and launchctl.call_count == 0:
            defects.append("R1b: free lock certifies backup despite active legacy writer; no stop attempted")
        if not record.get("archive"):
            record = backup.backup(settings, stop_service=False, now=dt.datetime(2026, 10, 9, 1))
        archive = settings.backup_dir / record["archive"]
        extracted = backup.extract(archive, backup.load_key(settings.backup_key), root / "restore")
        (extracted / "unmanifested.txt").write_text("Synthetic extra")
        if any(p.startswith("extra:") for p in backup.verify_manifest(extracted)[1]):
            passed.append("R3: unmanifested file refused")
        else:
            defects.append("R3: unmanifested file accepted")
        (extracted / "unmanifested.txt").unlink()
        outside = extracted.parent / "outside.txt"
        outside.write_bytes(b"Synthetic outside-root file")
        manifest_file = extracted / "MANIFEST.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["files"]["../outside.txt"] = {"sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
                                               "bytes": outside.stat().st_size}
        manifest_file.write_text(json.dumps(manifest))
        if not backup.verify_manifest(extracted)[1]:
            defects.append("R3b: manifest ../outside.txt accepted and checked outside extraction root")

        for name, data in (("invalid-fields", b'{"id":"x","kind":"unknown","text":123,"project":"*"}\n'),
                           ("invalid-blocks", valid)):
            (store / "items.jsonl").write_bytes(data)
            if name == "invalid-blocks":
                (store / "blocks.jsonl").write_bytes(b"BROKEN\n")
            with patch.object(backup, "_service_loaded", return_value=True), \
                    patch.object(backup, "_launchctl", return_value=0), \
                    patch.object(backup, "_wait_stopped", return_value=True), \
                    patch.object(backup, "_wait_healthy", return_value=True):
                checked = backup.backup(settings, now=dt.datetime(2026, 10, 9, 2 if name == "invalid-fields" else 3))
            if checked["ok"] and checked["qualified"] and checked["verified"]:
                defects.append("R4b: " + name + " certified qualified/verified")
        (store / "blocks.jsonl").unlink()
        (store / "items.jsonl").write_bytes(valid)

        linked_key = root / "linked-key"
        linked_key.symlink_to(settings.backup_key)
        try:
            backup.load_key(linked_key)
        except (ValueError, OSError):
            passed.append("R2: symlink key refused")
        else:
            defects.append("R2: symlink key accepted")
        settings.token_file.write_text("invalid")
        settings.token_file.chmod(0o644)
        try:
            auth.create_token(settings.token_file)
        except (ValueError, OSError):
            passed.append("R2: unsafe existing token refused")
        else:
            defects.append("R2: unsafe token accepted")

        # An unverified future filename must not evict the backup just verified.
        for index in range(8):
            day = dt.datetime(2090, 1, 1) + dt.timedelta(days=7 * index)
            (settings.backup_dir / f"memory-os-v4-{day:%Y%m%d}-010000.mosb").write_bytes(b"corrupt")
        with patch.object(backup, "_service_loaded", return_value=True), \
                patch.object(backup, "_launchctl", return_value=0), \
                patch.object(backup, "_wait_stopped", return_value=True), \
                patch.object(backup, "_wait_healthy", return_value=True):
            rotated = backup.backup(settings, now=dt.datetime(2026, 10, 9, 4))
        if rotated["ok"] and rotated.get("archive") and not (settings.backup_dir / rotated["archive"]).exists():
            defects.append("R3c: rotation deletes just-verified backup and returns ok=True")

        output = io.StringIO()
        with patch.object(OpsSettings, "from_env", classmethod(lambda cls: settings)), \
                patch.dict(os.environ, {"MEMORY_V4_BUILD_COMMIT": "a" * 40}), redirect_stdout(output):
            cli.main(["setup", "--write-launchd", str(root / "plists")])
        job = plistlib.loads((root / "plists/synthetic.test.plist").read_bytes())
        if "MEMORY_V4_BUILD_COMMIT" not in job["EnvironmentVariables"] and "WARNING" not in output.getvalue():
            defects.append("R5b: attested build commit omitted from launchd config without warning")

    print(json.dumps({"reviewed_ops": backup.__file__, "release_ready": not defects,
                      "passed_controls": passed, "observed_blockers": defects}, indent=2))
    return int(bool(defects))


if __name__ == "__main__":
    raise SystemExit(main())
