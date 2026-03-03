"""Storage Router for Memory OS AI.

Smart routing layer that writes to local disk first and automatically
overflows to a configured cloud backend when local disk space runs low.

Features:
- Auto-detect low disk space and offload to cloud
- Configurable threshold (default: 500 MB free)
- Transparent upload/download — engine doesn't need to know
- Sync: push local → cloud, pull cloud → local
- Multi-provider: configure multiple backends, use the first available

Environment variables:
    MEMORY_CLOUD_PROVIDER   — provider name (google-drive, icloud, dropbox, etc.)
    MEMORY_CLOUD_CONFIG     — JSON string or path to JSON config file
    MEMORY_DISK_THRESHOLD   — minimum free bytes before overflow (default: 524288000 = 500MB)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from .cloud_storage import (
    CloudFile,
    StorageBackend,
    StorageQuota,
    SyncResult,
    get_backend,
    PROVIDER_NAMES,
)

logger = logging.getLogger("memory_os_ai.storage")

# Default: overflow when less than 500 MB free
DEFAULT_DISK_THRESHOLD = 500 * 1024 * 1024  # 500 MB

# Files to sync to cloud (memory data files)
MEMORY_FILE_PATTERNS = {
    "embeddings_cache.npy",
    "_project_links.json",
    "_chat_state.json",
}
MEMORY_FILE_EXTENSIONS = {".npy", ".json", ".jsonl", ".faiss"}


class StorageRouter:
    """Routes storage operations between local disk and cloud backends.

    Usage:
        router = StorageRouter(local_dir="~/.memory-os-ai")
        router.configure_cloud("icloud", {"container": "memory-os-ai"})

        # Write locally; if disk low, also push to cloud
        router.write("data.jsonl", content_bytes)

        # Auto-sync on low disk
        router.check_and_offload()
    """

    def __init__(
        self,
        local_dir: str = "",
        disk_threshold: int = DEFAULT_DISK_THRESHOLD,
    ) -> None:
        self.local_dir = os.path.expanduser(local_dir or os.environ.get("MEMORY_CACHE_DIR", "~/.memory-os-ai"))
        self.disk_threshold = int(os.environ.get("MEMORY_DISK_THRESHOLD", str(disk_threshold)))
        self._backends: list[StorageBackend] = []
        self._active_backend: Optional[StorageBackend] = None
        self._config_path = os.path.join(self.local_dir, "_cloud_config.json")

        os.makedirs(self.local_dir, exist_ok=True)

        # Auto-configure from env
        self._auto_configure_from_env()
        # Load persisted config
        self._load_persisted_config()

    def _auto_configure_from_env(self) -> None:
        """Configure cloud backend from environment variables."""
        provider = os.environ.get("MEMORY_CLOUD_PROVIDER", "")
        config_raw = os.environ.get("MEMORY_CLOUD_CONFIG", "")

        if not provider:
            return

        creds: dict[str, Any] = {}
        if config_raw:
            if os.path.isfile(config_raw):
                with open(config_raw, "r") as f:
                    creds = json.load(f)
            else:
                try:
                    creds = json.loads(config_raw)
                except json.JSONDecodeError:
                    logger.warning("MEMORY_CLOUD_CONFIG is not valid JSON: %s", config_raw[:100])
                    return

        self.configure_cloud(provider, creds)

    def _load_persisted_config(self) -> None:
        """Load previously saved cloud config."""
        if os.path.isfile(self._config_path) and not self._active_backend:
            try:
                with open(self._config_path, "r") as f:
                    saved = json.load(f)
                provider = saved.get("provider", "")
                creds = saved.get("credentials", {})
                if provider:
                    self.configure_cloud(provider, creds, persist=False)
            except Exception as e:
                logger.debug("Failed to load persisted cloud config: %s", e)

    def _persist_config(self, provider: str, credentials: dict) -> None:
        """Save cloud config for next session (secrets masked)."""
        # Mask sensitive values
        safe_creds = {}
        sensitive_keys = {"access_token", "aws_secret_access_key", "application_key",
                          "connection_string", "credentials_json", "token_json"}
        for k, v in credentials.items():
            if k in sensitive_keys and isinstance(v, str) and len(v) > 8:
                safe_creds[k] = v[:4] + "****" + v[-4:]
            else:
                safe_creds[k] = v
        try:
            with open(self._config_path, "w") as f:
                json.dump({"provider": provider, "credentials": safe_creds}, f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist cloud config: %s", e)

    def configure_cloud(
        self,
        provider: str,
        credentials: dict[str, Any],
        persist: bool = True,
    ) -> dict:
        """Configure a cloud storage backend.

        Args:
            provider: One of PROVIDER_NAMES (google-drive, icloud, dropbox, etc.)
            credentials: Provider-specific config dict.
            persist: Save config for next session.

        Returns: {"ok": True/False, ...}
        """
        try:
            backend = get_backend(provider)
        except ValueError as e:
            return {"ok": False, "error": str(e)}

        result = backend.configure(credentials)
        if result.get("ok"):
            self._backends.append(backend)
            self._active_backend = backend
            if persist:
                self._persist_config(provider, credentials)
            logger.info("Cloud storage configured: %s", provider)
        return result

    @property
    def has_cloud(self) -> bool:
        """Whether a cloud backend is configured and ready."""
        return self._active_backend is not None and self._active_backend.is_configured()

    @property
    def cloud_provider(self) -> str:
        """Name of the active cloud provider."""
        if self._active_backend:
            return self._active_backend.provider_name
        return "none"

    def local_disk_free(self) -> int:
        """Free bytes on the local disk."""
        try:
            stat = shutil.disk_usage(self.local_dir)
            return stat.free
        except Exception:
            return 0

    def is_disk_low(self) -> bool:
        """Check if local disk is below threshold."""
        return self.local_disk_free() < self.disk_threshold

    def _memory_files(self) -> list[str]:
        """List all memory data files in local_dir."""
        files = []
        if not os.path.isdir(self.local_dir):
            return files
        for f in os.listdir(self.local_dir):
            full = os.path.join(self.local_dir, f)
            if not os.path.isfile(full):
                continue
            if f.startswith("_cloud_"):
                continue  # skip our own config
            ext = os.path.splitext(f)[1].lower()
            if f in MEMORY_FILE_PATTERNS or ext in MEMORY_FILE_EXTENSIONS:
                files.append(f)
        return sorted(files)

    def check_and_offload(self) -> dict:
        """Check disk space and offload to cloud if needed.

        Returns status dict with actions taken.
        """
        free = self.local_disk_free()
        threshold = self.disk_threshold

        result = {
            "disk_free_bytes": free,
            "disk_free_mb": round(free / (1024 * 1024), 1),
            "threshold_mb": round(threshold / (1024 * 1024), 1),
            "disk_low": free < threshold,
            "cloud_configured": self.has_cloud,
            "cloud_provider": self.cloud_provider,
            "actions": [],
        }

        if free >= threshold:
            result["status"] = "ok"
            return result

        if not self.has_cloud:
            result["status"] = "warning"
            result["message"] = (
                f"Local disk low ({result['disk_free_mb']} MB free). "
                "Configure a cloud backend with memory_cloud_configure to enable overflow."
            )
            return result

        # Offload: upload all memory files to cloud
        uploaded = []
        errors = []
        for filename in self._memory_files():
            local_path = os.path.join(self.local_dir, filename)
            res = self._active_backend.upload(local_path, filename)
            if res.get("ok"):
                uploaded.append(filename)
            else:
                errors.append(f"{filename}: {res.get('error', 'unknown')}")

        result["status"] = "offloaded" if uploaded else "error"
        result["uploaded"] = uploaded
        result["errors"] = errors
        result["actions"].append(f"Uploaded {len(uploaded)} files to {self.cloud_provider}")

        return result

    def sync_to_cloud(self) -> SyncResult:
        """Push all local memory files to cloud."""
        start = time.time()
        result = SyncResult()

        if not self.has_cloud:
            result.errors.append("No cloud backend configured")
            result.elapsed_seconds = round(time.time() - start, 2)
            return result

        for filename in self._memory_files():
            local_path = os.path.join(self.local_dir, filename)
            res = self._active_backend.upload(local_path, filename)
            if res.get("ok"):
                result.uploaded.append(filename)
            else:
                result.errors.append(f"upload {filename}: {res.get('error')}")

        result.elapsed_seconds = round(time.time() - start, 2)
        return result

    def sync_from_cloud(self) -> SyncResult:
        """Pull cloud files that don't exist locally."""
        start = time.time()
        result = SyncResult()

        if not self.has_cloud:
            result.errors.append("No cloud backend configured")
            result.elapsed_seconds = round(time.time() - start, 2)
            return result

        local_files = set(self._memory_files())
        cloud_files = self._active_backend.list_files()

        for cf in cloud_files:
            if cf.name not in local_files:
                local_path = os.path.join(self.local_dir, cf.path)
                res = self._active_backend.download(cf.path, local_path)
                if res.get("ok"):
                    result.downloaded.append(cf.path)
                else:
                    result.errors.append(f"download {cf.path}: {res.get('error')}")

        result.elapsed_seconds = round(time.time() - start, 2)
        return result

    def status(self) -> dict:
        """Full status of local + cloud storage."""
        local_files = self._memory_files()
        local_size = sum(
            os.path.getsize(os.path.join(self.local_dir, f))
            for f in local_files
            if os.path.isfile(os.path.join(self.local_dir, f))
        )
        disk_stat = shutil.disk_usage(self.local_dir) if os.path.isdir(self.local_dir) else None

        result: dict[str, Any] = {
            "local": {
                "dir": self.local_dir,
                "files": len(local_files),
                "size_bytes": local_size,
                "size_mb": round(local_size / (1024 * 1024), 2),
                "disk_free_mb": round(disk_stat.free / (1024 * 1024), 1) if disk_stat else 0,
                "disk_total_mb": round(disk_stat.total / (1024 * 1024), 1) if disk_stat else 0,
                "disk_low": self.is_disk_low(),
            },
            "cloud": {
                "configured": self.has_cloud,
                "provider": self.cloud_provider,
            },
            "threshold_mb": round(self.disk_threshold / (1024 * 1024), 1),
            "available_providers": PROVIDER_NAMES,
        }

        if self.has_cloud:
            cloud_files = self._active_backend.list_files()
            cloud_size = sum(f.size_bytes for f in cloud_files)
            quota = self._active_backend.get_quota()
            result["cloud"].update({
                "files": len(cloud_files),
                "size_bytes": cloud_size,
                "size_mb": round(cloud_size / (1024 * 1024), 2),
                "quota_total_mb": round(quota.total_bytes / (1024 * 1024), 1),
                "quota_used_mb": round(quota.used_bytes / (1024 * 1024), 1),
                "quota_free_mb": round(quota.free_bytes / (1024 * 1024), 1),
                "quota_usage_percent": quota.usage_percent,
            })

        return result
