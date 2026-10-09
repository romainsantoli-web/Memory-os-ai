import json

import pytest
from cryptography.exceptions import InvalidTag

from tools import offline_store_backup as tool


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "items.jsonl").write_text(json.dumps({"id": "one", "kind": "fact", "text": "Synthetic"}) + "\n")
    (source / "cache").write_bytes(b"cache")
    key = tmp_path / "key"
    key.write_bytes(b"k" * 32)
    key.chmod(0o600)
    return source, tmp_path / "snapshot.mosb", key


def test_encrypted_offline_backup_verified_without_mutating_source(inputs):
    source, archive, key = inputs
    before = tool.inventory(tool.snapshot(source))
    result = tool.backup(source, archive, key, offline=True)
    assert result["verified"] and result["consistent_server_stopped"]
    assert result["items"]["live_total"] == 1
    assert tool.inventory(tool.snapshot(source)) == before
    assert archive.stat().st_mode & 0o777 == 0o600
    assert b"Synthetic" not in archive.read_bytes()
    assert tool.verify(archive, key, source=source) == result
    with pytest.raises(ValueError, match="new"):
        tool.backup(source, archive, key, offline=True)


def test_offline_acknowledgement_required(inputs):
    with pytest.raises(ValueError, match="offline"):
        tool.backup(*inputs)
    assert not inputs[1].exists()


def test_ciphertext_tampering_rejected(inputs):
    source, archive, key = inputs
    tool.backup(source, archive, key, offline=True)
    blob = bytearray(archive.read_bytes())
    blob[-1] ^= 1
    archive.write_bytes(blob)
    with pytest.raises(InvalidTag):
        tool.verify(archive, key)


def test_changed_source_or_truncated_log_refused(inputs):
    source, archive, key = inputs
    tool.backup(source, archive, key, offline=True)
    (source / "cache").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        tool.verify(archive, key, source=source)
    (source / "items.jsonl").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Truncated"):
        tool.backup(source, archive.with_name("other.mosb"), key, offline=True)


def test_unsafe_key_mode_and_symlink_refused(inputs):
    source, archive, key = inputs
    key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        tool.backup(source, archive, key, offline=True)
    key.chmod(0o600)
    (source / "linked").symlink_to(key)
    with pytest.raises(ValueError, match="Linked"):
        tool.backup(source, archive, key, offline=True)
