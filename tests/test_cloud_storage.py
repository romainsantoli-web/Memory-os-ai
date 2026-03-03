"""Tests for cloud_storage.py and storage_router.py — backends + routing logic."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from memory_os_ai.cloud_storage import (
    CloudFile,
    StorageQuota,
    SyncResult,
    StorageBackend,
    GoogleDriveBackend,
    ICloudBackend,
    DropboxBackend,
    OneDriveBackend,
    S3Backend,
    AzureBlobBackend,
    BoxBackend,
    B2Backend,
    PROVIDERS,
    PROVIDER_NAMES,
    get_backend,
)
from memory_os_ai.storage_router import StorageRouter, MEMORY_FILE_PATTERNS, MEMORY_FILE_EXTENSIONS
from memory_os_ai.models import CloudConfigureInput, CloudStatusInput, CloudSyncInput


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class TestCloudFile:
    def test_basic(self):
        cf = CloudFile(name="test.npy", path="data/test.npy", size_bytes=1024, modified_at=1000.0)
        assert cf.name == "test.npy"
        assert cf.size_bytes == 1024
        assert cf.checksum == ""

    def test_with_checksum(self):
        cf = CloudFile(name="f.json", path="f.json", size_bytes=0, modified_at=0, checksum="abc123")
        assert cf.checksum == "abc123"


class TestStorageQuota:
    def test_usage_percent(self):
        q = StorageQuota(total_bytes=1000, used_bytes=300, free_bytes=700, provider="test")
        assert q.usage_percent == 30.0

    def test_zero_total(self):
        q = StorageQuota(total_bytes=0, used_bytes=0, free_bytes=0, provider="test")
        assert q.usage_percent == 100.0

    def test_full_disk(self):
        q = StorageQuota(total_bytes=1000, used_bytes=1000, free_bytes=0, provider="test")
        assert q.usage_percent == 100.0


class TestSyncResult:
    def test_defaults(self):
        r = SyncResult()
        assert r.uploaded == []
        assert r.downloaded == []
        assert r.deleted == []
        assert r.errors == []
        assert r.elapsed_seconds == 0.0

    def test_with_data(self):
        r = SyncResult(uploaded=["a.npy"], errors=["failed"], elapsed_seconds=1.5)
        assert len(r.uploaded) == 1
        assert len(r.errors) == 1


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------
class TestProviderRegistry:
    def test_all_providers_registered(self):
        expected = {"google-drive", "icloud", "dropbox", "onedrive", "s3", "azure-blob", "box", "b2"}
        assert set(PROVIDERS.keys()) == expected

    def test_provider_names_sorted(self):
        assert PROVIDER_NAMES == sorted(PROVIDER_NAMES)

    def test_get_backend_valid(self):
        for name in PROVIDER_NAMES:
            backend = get_backend(name)
            assert isinstance(backend, StorageBackend)
            assert backend.provider_name == name

    def test_get_backend_invalid(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_backend("ftp")

    def test_get_backend_returns_fresh_instance(self):
        b1 = get_backend("icloud")
        b2 = get_backend("icloud")
        assert b1 is not b2


# ---------------------------------------------------------------------------
# Abstract Backend — file_checksum
# ---------------------------------------------------------------------------
class TestStorageBackendChecksum:
    def test_file_checksum(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        # ICloudBackend inherits file_checksum from StorageBackend
        backend = ICloudBackend()
        checksum = backend.file_checksum(str(f))
        assert len(checksum) == 64  # sha256 hex
        # Deterministic
        assert checksum == backend.file_checksum(str(f))


# ---------------------------------------------------------------------------
# iCloud Backend (file-system based — fully testable)
# ---------------------------------------------------------------------------
class TestICloudBackend:
    def test_configure_missing_icloud_dir(self):
        backend = ICloudBackend()
        result = backend.configure({"container": "test"})
        # On CI or non-macOS, iCloud dir won't exist
        if not os.path.isdir(os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")):
            assert result["ok"] is False
            assert "not found" in result["error"]
        else:
            assert result["ok"] is True

    def test_not_configured_operations(self):
        backend = ICloudBackend()
        assert backend.is_configured() is False
        assert backend.upload("/tmp/x", "x") == {"ok": False, "error": "iCloud not configured"}
        assert backend.download("x", "/tmp/x") == {"ok": False, "error": "iCloud not configured"}
        assert backend.delete("x") == {"ok": False, "error": "iCloud not configured"}
        assert backend.list_files() == []
        q = backend.get_quota()
        assert q.total_bytes == 0

    def test_icloud_with_fake_mount(self, tmp_path):
        """Simulate iCloud by patching the root path."""
        backend = ICloudBackend()
        # Manually set up as if configured
        container_dir = tmp_path / "memory-test"
        container_dir.mkdir()
        backend._base_path = str(container_dir)
        backend._configured = True

        # Upload
        src = tmp_path / "source.txt"
        src.write_text("test content 123")
        result = backend.upload(str(src), "source.txt")
        assert result["ok"] is True
        assert result["size_bytes"] > 0
        assert (container_dir / "source.txt").exists()

        # List
        files = backend.list_files()
        assert len(files) == 1
        assert files[0].name == "source.txt"

        # Download
        dest = tmp_path / "downloaded.txt"
        result = backend.download("source.txt", str(dest))
        assert result["ok"] is True
        assert dest.read_text() == "test content 123"

        # Download missing
        result = backend.download("missing.txt", str(tmp_path / "nope.txt"))
        assert result["ok"] is False

        # Delete
        result = backend.delete("source.txt")
        assert result["ok"] is True
        assert not (container_dir / "source.txt").exists()

        # Quota
        q = backend.get_quota()
        assert q.total_bytes > 0
        assert q.provider == "icloud"


# ---------------------------------------------------------------------------
# OneDrive Backend (mount mode — file-system based, testable)
# ---------------------------------------------------------------------------
class TestOneDriveBackend:
    def test_configure_with_mount(self, tmp_path):
        mount = tmp_path / "OneDrive"
        mount.mkdir()
        backend = OneDriveBackend()
        result = backend.configure({"mount_path": str(mount), "folder": "memory-test"})
        assert result["ok"] is True
        assert result["mode"] == "mount"
        assert backend.is_configured()

    def test_configure_no_mount_no_token(self):
        backend = OneDriveBackend()
        result = backend.configure({"mount_path": "/nonexistent/path"})
        # No mount, no access_token
        assert result["ok"] is False

    def test_configure_with_token(self):
        backend = OneDriveBackend()
        result = backend.configure({"access_token": "fake-token"})
        assert result["ok"] is True
        assert result["mode"] == "api"

    def test_mount_upload_download(self, tmp_path):
        mount = tmp_path / "OneDrive"
        mount.mkdir()
        backend = OneDriveBackend()
        backend.configure({"mount_path": str(mount), "folder": "mem"})

        src = tmp_path / "data.json"
        src.write_text('{"key": "value"}')
        result = backend.upload(str(src), "data.json")
        assert result["ok"] is True

        dest = tmp_path / "out.json"
        result = backend.download("data.json", str(dest))
        assert result["ok"] is True
        assert json.loads(dest.read_text()) == {"key": "value"}

    def test_mount_list_and_delete(self, tmp_path):
        mount = tmp_path / "OneDrive"
        mount.mkdir()
        backend = OneDriveBackend()
        backend.configure({"mount_path": str(mount), "folder": "mem"})

        src = tmp_path / "test.npy"
        src.write_bytes(b"\x00" * 100)
        backend.upload(str(src), "test.npy")

        files = backend.list_files()
        assert len(files) == 1

        result = backend.delete("test.npy")
        assert result["ok"] is True

    def test_not_configured(self):
        backend = OneDriveBackend()
        assert backend.upload("/x", "x")["ok"] is False
        assert backend.download("x", "/x")["ok"] is False
        assert backend.delete("x")["ok"] is False
        assert backend.list_files() == []
        q = backend.get_quota()
        assert q.total_bytes == 0


# ---------------------------------------------------------------------------
# Google Drive Backend — mock-based
# ---------------------------------------------------------------------------
class TestGoogleDriveBackend:
    def test_not_configured(self):
        backend = GoogleDriveBackend()
        assert backend.is_configured() is False
        assert backend.upload("/x", "x")["ok"] is False
        assert backend.download("x", "/x")["ok"] is False
        assert backend.delete("x")["ok"] is False
        assert backend.list_files() == []
        q = backend.get_quota()
        assert q.total_bytes == 0

    def test_configure_missing_creds(self):
        backend = GoogleDriveBackend()
        # Mock the google imports to avoid ImportError
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.oauth2": MagicMock(),
            "google.oauth2.credentials": MagicMock(),
            "google.oauth2.service_account": MagicMock(),
            "googleapiclient": MagicMock(),
            "googleapiclient.discovery": MagicMock(),
        }):
            result = backend.configure({})
            assert result["ok"] is False
            assert "credentials_json" in result["error"] or "token_json" in result["error"]


# ---------------------------------------------------------------------------
# Dropbox Backend — mock-based
# ---------------------------------------------------------------------------
class TestDropboxBackend:
    def test_not_configured(self):
        backend = DropboxBackend()
        assert backend.is_configured() is False
        assert backend.upload("/x", "x")["ok"] is False
        assert backend.download("x", "/x")["ok"] is False
        assert backend.delete("x")["ok"] is False
        assert backend.list_files() == []
        q = backend.get_quota()
        assert q.total_bytes == 0

    def test_configure_no_token(self):
        backend = DropboxBackend()
        with patch.dict("sys.modules", {"dropbox": MagicMock()}):
            result = backend.configure({})
            assert result["ok"] is False
            assert "access_token" in result["error"]


# ---------------------------------------------------------------------------
# S3 Backend — mock-based
# ---------------------------------------------------------------------------
class TestS3Backend:
    def test_not_configured(self):
        backend = S3Backend()
        assert backend.is_configured() is False
        assert backend.upload("/x", "x")["ok"] is False
        assert backend.download("x", "/x")["ok"] is False
        assert backend.delete("x")["ok"] is False
        assert backend.list_files() == []

    def test_configure_no_bucket(self):
        backend = S3Backend()
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            result = backend.configure({})
            assert result["ok"] is False
            assert "bucket" in result["error"].lower()

    def test_quota_unlimited(self):
        backend = S3Backend()
        q = backend.get_quota()
        assert q.total_bytes == 10 * 1024**4


# ---------------------------------------------------------------------------
# Azure Blob Backend — mock-based
# ---------------------------------------------------------------------------
class TestAzureBlobBackend:
    def test_not_configured(self):
        backend = AzureBlobBackend()
        assert backend.is_configured() is False
        assert backend.upload("/x", "x")["ok"] is False
        assert backend.download("x", "/x")["ok"] is False
        assert backend.delete("x")["ok"] is False
        assert backend.list_files() == []

    def test_configure_no_connection_string(self):
        backend = AzureBlobBackend()
        with patch.dict("sys.modules", {
            "azure": MagicMock(),
            "azure.storage": MagicMock(),
            "azure.storage.blob": MagicMock(),
        }):
            result = backend.configure({})
            assert result["ok"] is False
            assert "connection_string" in result["error"]

    def test_quota(self):
        backend = AzureBlobBackend()
        q = backend.get_quota()
        assert q.total_bytes == 5 * 1024**4


# ---------------------------------------------------------------------------
# Box Backend — mock-based
# ---------------------------------------------------------------------------
class TestBoxBackend:
    def test_not_configured(self):
        backend = BoxBackend()
        assert backend.is_configured() is False
        assert backend.upload("/x", "x")["ok"] is False
        assert backend.download("x", "/x")["ok"] is False
        assert backend.delete("x")["ok"] is False
        assert backend.list_files() == []
        q = backend.get_quota()
        assert q.total_bytes == 0

    def test_configure_no_token(self):
        backend = BoxBackend()
        with patch.dict("sys.modules", {
            "boxsdk": MagicMock(),
        }):
            result = backend.configure({})
            assert result["ok"] is False
            assert "access_token" in result["error"]


# ---------------------------------------------------------------------------
# B2 Backend — mock-based
# ---------------------------------------------------------------------------
class TestB2Backend:
    def test_not_configured(self):
        backend = B2Backend()
        assert backend.is_configured() is False
        assert backend.upload("/x", "x")["ok"] is False
        assert backend.download("x", "/x")["ok"] is False
        assert backend.delete("x")["ok"] is False
        assert backend.list_files() == []

    def test_configure_no_keys(self):
        backend = B2Backend()
        with patch.dict("sys.modules", {
            "b2sdk": MagicMock(),
            "b2sdk.v2": MagicMock(),
        }):
            result = backend.configure({})
            assert result["ok"] is False
            assert "application_key_id" in result["error"]

    def test_quota_unlimited(self):
        backend = B2Backend()
        q = backend.get_quota()
        assert q.total_bytes == 10 * 1024**4


# ---------------------------------------------------------------------------
# StorageRouter
# ---------------------------------------------------------------------------
class TestStorageRouter:
    def test_init_creates_local_dir(self, tmp_path):
        d = tmp_path / "cache"
        router = StorageRouter(local_dir=str(d))
        assert d.is_dir()

    def test_no_cloud_by_default(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path))
        assert router.has_cloud is False
        assert router.cloud_provider == "none"

    def test_configure_invalid_provider(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path))
        result = router.configure_cloud("ftp", {})
        assert result["ok"] is False
        assert "Unknown provider" in result["error"]

    def test_local_disk_free(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path))
        free = router.local_disk_free()
        assert free > 0

    def test_is_disk_low_threshold(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path), disk_threshold=10**18)
        assert router.is_disk_low() is True

        router2 = StorageRouter(local_dir=str(tmp_path), disk_threshold=1)
        assert router2.is_disk_low() is False

    def test_memory_files_detection(self, tmp_path):
        # Create various files
        (tmp_path / "embeddings_cache.npy").write_bytes(b"\x00")
        (tmp_path / "_project_links.json").write_text("{}")
        (tmp_path / "data.jsonl").write_text("{}\n")
        (tmp_path / "index.faiss").write_bytes(b"\x00")
        (tmp_path / "readme.txt").write_text("not a memory file")
        (tmp_path / "_cloud_config.json").write_text("{}")  # should be skipped

        router = StorageRouter(local_dir=str(tmp_path))
        files = router._memory_files()
        assert "embeddings_cache.npy" in files
        assert "_project_links.json" in files
        assert "data.jsonl" in files
        assert "index.faiss" in files
        assert "readme.txt" not in files
        assert "_cloud_config.json" not in files

    def test_check_and_offload_disk_ok(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path), disk_threshold=1)
        result = router.check_and_offload()
        assert result["status"] == "ok"
        assert result["disk_low"] is False

    def test_check_and_offload_no_cloud(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path), disk_threshold=10**18)
        result = router.check_and_offload()
        assert result["status"] == "warning"
        assert "Configure a cloud backend" in result["message"]

    def test_sync_to_cloud_no_backend(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path))
        result = router.sync_to_cloud()
        assert len(result.errors) == 1
        assert "No cloud backend" in result.errors[0]

    def test_sync_from_cloud_no_backend(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path))
        result = router.sync_from_cloud()
        assert len(result.errors) == 1
        assert "No cloud backend" in result.errors[0]

    def test_status_no_cloud(self, tmp_path):
        (tmp_path / "data.jsonl").write_text("{}\n")
        router = StorageRouter(local_dir=str(tmp_path))
        status = router.status()
        assert status["local"]["files"] == 1
        assert status["cloud"]["configured"] is False
        assert "available_providers" in status
        assert len(status["available_providers"]) == 8

    def test_configure_persist_and_load(self, tmp_path):
        """Test that config is persisted and can be reloaded."""
        router = StorageRouter(local_dir=str(tmp_path))

        # Manually configure with a fake icloud-like backend
        icloud_dir = tmp_path / "icloud"
        icloud_dir.mkdir()
        backend = ICloudBackend()
        backend._base_path = str(icloud_dir)
        backend._configured = True
        router._backends.append(backend)
        router._active_backend = backend
        router._persist_config("icloud", {"container": "test"})

        config_path = tmp_path / "_cloud_config.json"
        assert config_path.exists()
        saved = json.loads(config_path.read_text())
        assert saved["provider"] == "icloud"

    def test_sync_to_cloud_with_mock_backend(self, tmp_path):
        """Full sync cycle with a mock backend."""
        (tmp_path / "embeddings_cache.npy").write_bytes(b"\x00" * 100)
        (tmp_path / "_chat_state.json").write_text('{"ok": true}')

        router = StorageRouter(local_dir=str(tmp_path))

        # Create a mock backend
        mock_backend = MagicMock(spec=StorageBackend)
        mock_backend.is_configured.return_value = True
        mock_backend.provider_name = "mock"
        mock_backend.upload.return_value = {"ok": True}
        mock_backend.list_files.return_value = []
        mock_backend.get_quota.return_value = StorageQuota(1000, 200, 800, "mock")

        router._active_backend = mock_backend
        router._backends = [mock_backend]

        # Sync to cloud
        result = router.sync_to_cloud()
        assert len(result.errors) == 0
        assert len(result.uploaded) >= 2
        assert mock_backend.upload.call_count >= 2

    def test_sync_from_cloud_with_mock_backend(self, tmp_path):
        router = StorageRouter(local_dir=str(tmp_path))

        mock_backend = MagicMock(spec=StorageBackend)
        mock_backend.is_configured.return_value = True
        mock_backend.provider_name = "mock"
        mock_backend.list_files.return_value = [
            CloudFile(name="remote.npy", path="remote.npy", size_bytes=50, modified_at=1000),
        ]
        mock_backend.download.return_value = {"ok": True}

        router._active_backend = mock_backend
        router._backends = [mock_backend]

        result = router.sync_from_cloud()
        assert len(result.errors) == 0
        assert "remote.npy" in result.downloaded

    def test_check_and_offload_with_cloud(self, tmp_path):
        (tmp_path / "data.jsonl").write_text("{}\n")
        router = StorageRouter(local_dir=str(tmp_path), disk_threshold=10**18)

        mock_backend = MagicMock(spec=StorageBackend)
        mock_backend.is_configured.return_value = True
        mock_backend.provider_name = "mock"
        mock_backend.upload.return_value = {"ok": True}

        router._active_backend = mock_backend
        router._backends = [mock_backend]

        result = router.check_and_offload()
        assert result["status"] == "offloaded"
        assert len(result["uploaded"]) >= 1

    def test_status_with_cloud(self, tmp_path):
        (tmp_path / "data.jsonl").write_text("{}\n")
        router = StorageRouter(local_dir=str(tmp_path))

        mock_backend = MagicMock(spec=StorageBackend)
        mock_backend.is_configured.return_value = True
        mock_backend.provider_name = "mock-cloud"
        mock_backend.list_files.return_value = [
            CloudFile(name="x.npy", path="x.npy", size_bytes=500, modified_at=0),
        ]
        mock_backend.get_quota.return_value = StorageQuota(10000, 500, 9500, "mock-cloud")

        router._active_backend = mock_backend
        router._backends = [mock_backend]

        status = router.status()
        assert status["cloud"]["configured"] is True
        assert status["cloud"]["files"] == 1
        assert status["cloud"]["size_bytes"] == 500

    def test_env_auto_configure(self, tmp_path):
        """Test auto-configuration from env vars."""
        with patch.dict(os.environ, {
            "MEMORY_CLOUD_PROVIDER": "icloud",
            "MEMORY_CLOUD_CONFIG": json.dumps({"container": "test"}),
            "MEMORY_CACHE_DIR": str(tmp_path),
        }):
            router = StorageRouter(local_dir=str(tmp_path))
            # iCloud won't work on CI but the configure path is exercised


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class TestCloudModels:
    def test_configure_input_valid(self):
        inp = CloudConfigureInput(provider="icloud", credentials={"container": "test"})
        assert inp.provider == "icloud"

    def test_configure_input_invalid_provider(self):
        with pytest.raises(Exception):
            CloudConfigureInput(provider="ftp", credentials={})

    def test_status_input(self):
        inp = CloudStatusInput()
        assert inp is not None

    def test_sync_input_valid(self):
        for d in ("push", "pull", "auto"):
            inp = CloudSyncInput(direction=d)
            assert inp.direction == d

    def test_sync_input_invalid_direction(self):
        with pytest.raises(Exception):
            CloudSyncInput(direction="sideways")


# ---------------------------------------------------------------------------
# Server dispatch — cloud tools
# ---------------------------------------------------------------------------
class TestServerCloudDispatch:
    def test_dispatch_cloud_configure(self, tmp_path):
        import memory_os_ai.server as srv

        orig_router = srv._storage_router
        srv._storage_router = StorageRouter(local_dir=str(tmp_path))
        try:
            result = srv._dispatch("memory_cloud_configure", {
                "provider": "icloud",
                "credentials": {"container": "test"},
            })
            # May fail (no iCloud on CI) but should return a dict
            assert isinstance(result, dict)
            assert "ok" in result
        finally:
            srv._storage_router = orig_router

    def test_dispatch_cloud_status(self, tmp_path):
        import memory_os_ai.server as srv

        orig_router = srv._storage_router
        srv._storage_router = StorageRouter(local_dir=str(tmp_path))
        try:
            result = srv._dispatch("memory_cloud_status", {})
            assert "local" in result
            assert "cloud" in result
            assert "available_providers" in result
        finally:
            srv._storage_router = orig_router

    def test_dispatch_cloud_sync_push_no_backend(self, tmp_path):
        import memory_os_ai.server as srv

        orig_router = srv._storage_router
        srv._storage_router = StorageRouter(local_dir=str(tmp_path))
        try:
            result = srv._dispatch("memory_cloud_sync", {"direction": "push"})
            assert result["ok"] is False
            assert "No cloud backend" in result["errors"][0]
        finally:
            srv._storage_router = orig_router

    def test_dispatch_cloud_sync_pull_no_backend(self, tmp_path):
        import memory_os_ai.server as srv

        orig_router = srv._storage_router
        srv._storage_router = StorageRouter(local_dir=str(tmp_path))
        try:
            result = srv._dispatch("memory_cloud_sync", {"direction": "pull"})
            assert result["ok"] is False
        finally:
            srv._storage_router = orig_router

    def test_dispatch_cloud_sync_auto(self, tmp_path):
        import memory_os_ai.server as srv

        orig_router = srv._storage_router
        srv._storage_router = StorageRouter(local_dir=str(tmp_path), disk_threshold=1)
        try:
            result = srv._dispatch("memory_cloud_sync", {"direction": "auto"})
            assert result["status"] == "ok"
        finally:
            srv._storage_router = orig_router

    def test_cloud_tools_in_tools_list(self):
        import memory_os_ai.server as srv
        tool_names = [t["name"] for t in srv.TOOLS]
        assert "memory_cloud_configure" in tool_names
        assert "memory_cloud_status" in tool_names
        assert "memory_cloud_sync" in tool_names
