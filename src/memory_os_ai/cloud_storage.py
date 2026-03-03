"""Cloud storage backends for Memory OS AI.

When local disk runs low, memory data (FAISS indexes, JSONL, embeddings cache)
can overflow to cloud storage. Supported providers:

- Google Drive          (google-drive)
- iCloud Drive          (icloud)
- Dropbox               (dropbox)
- Microsoft OneDrive    (onedrive)
- Amazon S3             (s3)
- Azure Blob Storage    (azure-blob)
- Box                   (box)
- Backblaze B2          (b2)

Each backend implements the same StorageBackend interface so the StorageRouter
can swap between them transparently.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("memory_os_ai.cloud")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CloudFile:
    """Metadata about a file stored in the cloud."""
    name: str
    path: str           # cloud path (relative key)
    size_bytes: int
    modified_at: float  # epoch seconds
    checksum: str = ""  # sha256 hex


@dataclass
class StorageQuota:
    """Disk / cloud usage information."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    provider: str

    @property
    def usage_percent(self) -> float:
        if self.total_bytes == 0:
            return 100.0
        return round(self.used_bytes / self.total_bytes * 100, 2)


@dataclass
class SyncResult:
    """Result of a sync operation."""
    uploaded: list[str] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Abstract Backend
# ---------------------------------------------------------------------------
class StorageBackend(ABC):
    """Abstract interface for cloud storage providers."""

    provider_name: str = "unknown"

    @abstractmethod
    def configure(self, credentials: dict[str, Any]) -> dict:
        """Configure the backend with provider-specific credentials.

        Returns {"ok": True/False, ...}.
        """

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if backend is ready to use."""

    @abstractmethod
    def upload(self, local_path: str, remote_key: str) -> dict:
        """Upload a local file to the cloud.

        Args:
            local_path: Absolute path to the local file.
            remote_key: Relative path/key in the cloud container.

        Returns {"ok": True/False, "size_bytes": ..., ...}.
        """

    @abstractmethod
    def download(self, remote_key: str, local_path: str) -> dict:
        """Download a file from the cloud to a local path.

        Returns {"ok": True/False, ...}.
        """

    @abstractmethod
    def delete(self, remote_key: str) -> dict:
        """Delete a file from the cloud.

        Returns {"ok": True/False, ...}.
        """

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[CloudFile]:
        """List files in the cloud storage under given prefix."""

    @abstractmethod
    def get_quota(self) -> StorageQuota:
        """Get storage quota/usage information."""

    def file_checksum(self, local_path: str) -> str:
        """Compute SHA-256 checksum of a local file."""
        h = hashlib.sha256()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Google Drive Backend
# ---------------------------------------------------------------------------
class GoogleDriveBackend(StorageBackend):
    """Google Drive storage backend.

    Credentials: {"credentials_json": "/path/to/credentials.json", "folder_id": "..."}
    Or: {"token_json": "/path/to/token.json", "folder_id": "..."}
    """

    provider_name = "google-drive"

    def __init__(self) -> None:
        self._service = None
        self._folder_id: str = ""
        self._configured = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        try:
            from google.oauth2.credentials import Credentials
            from google.oauth2.service_account import Credentials as ServiceCredentials
            from googleapiclient.discovery import build

            self._folder_id = credentials.get("folder_id", "root")

            # Service account or OAuth token
            creds_path = credentials.get("credentials_json", "")
            token_path = credentials.get("token_json", "")

            if creds_path and os.path.isfile(creds_path):
                creds = ServiceCredentials.from_service_account_file(
                    creds_path, scopes=["https://www.googleapis.com/auth/drive"]
                )
            elif token_path and os.path.isfile(token_path):
                creds = Credentials.from_authorized_user_file(
                    token_path, scopes=["https://www.googleapis.com/auth/drive"]
                )
            else:
                return {"ok": False, "error": "Provide 'credentials_json' or 'token_json'"}

            self._service = build("drive", "v3", credentials=creds)
            self._configured = True
            return {"ok": True, "provider": self.provider_name, "folder_id": self._folder_id}
        except ImportError:
            return {"ok": False, "error": "Install: pip install google-api-python-client google-auth"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def is_configured(self) -> bool:
        return self._configured and self._service is not None

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Google Drive not configured"}
        try:
            from googleapiclient.http import MediaFileUpload

            file_metadata = {
                "name": os.path.basename(remote_key),
                "parents": [self._folder_id],
                "description": f"memory-os-ai:{remote_key}",
            }
            media = MediaFileUpload(local_path, resumable=True)
            result = self._service.files().create(
                body=file_metadata, media_body=media, fields="id,size"
            ).execute()

            return {
                "ok": True,
                "provider": self.provider_name,
                "remote_key": remote_key,
                "file_id": result.get("id"),
                "size_bytes": int(result.get("size", 0)),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Google Drive not configured"}
        try:
            from googleapiclient.http import MediaIoBaseDownload

            # Find file by description tag
            query = f"description contains 'memory-os-ai:{remote_key}' and '{self._folder_id}' in parents and trashed=false"
            results = self._service.files().list(q=query, fields="files(id,name,size)").execute()
            files = results.get("files", [])
            if not files:
                return {"ok": False, "error": f"File not found in Drive: {remote_key}"}

            file_id = files[0]["id"]
            request = self._service.files().get_media(fileId=file_id)
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            with open(local_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Google Drive not configured"}
        try:
            query = f"description contains 'memory-os-ai:{remote_key}' and '{self._folder_id}' in parents and trashed=false"
            results = self._service.files().list(q=query, fields="files(id)").execute()
            for f in results.get("files", []):
                self._service.files().delete(fileId=f["id"]).execute()
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        try:
            query = f"'{self._folder_id}' in parents and trashed=false"
            if prefix:
                query += f" and description contains 'memory-os-ai:{prefix}'"
            results = self._service.files().list(
                q=query, fields="files(id,name,size,modifiedTime,description)", pageSize=1000
            ).execute()
            cloud_files = []
            for f in results.get("files", []):
                desc = f.get("description", "")
                key = desc.replace("memory-os-ai:", "") if desc.startswith("memory-os-ai:") else f["name"]
                cloud_files.append(CloudFile(
                    name=f["name"],
                    path=key,
                    size_bytes=int(f.get("size", 0)),
                    modified_at=0,
                ))
            return cloud_files
        except Exception:
            return []

    def get_quota(self) -> StorageQuota:
        if not self.is_configured():
            return StorageQuota(0, 0, 0, self.provider_name)
        try:
            about = self._service.about().get(fields="storageQuota").execute()
            q = about.get("storageQuota", {})
            total = int(q.get("limit", 0))
            used = int(q.get("usage", 0))
            return StorageQuota(total, used, max(0, total - used), self.provider_name)
        except Exception:
            return StorageQuota(0, 0, 0, self.provider_name)


# ---------------------------------------------------------------------------
# iCloud Drive Backend (macOS native via file system mount)
# ---------------------------------------------------------------------------
class ICloudBackend(StorageBackend):
    """Apple iCloud Drive backend.

    Uses the macOS native iCloud Drive mount at ~/Library/Mobile Documents/.
    Credentials: {"container": "memory-os-ai"} (subfolder name)
    """

    provider_name = "icloud"

    def __init__(self) -> None:
        self._base_path: str = ""
        self._configured = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        container = credentials.get("container", "memory-os-ai")
        # macOS iCloud Drive path
        icloud_root = os.path.expanduser(
            "~/Library/Mobile Documents/com~apple~CloudDocs"
        )
        if not os.path.isdir(icloud_root):
            return {"ok": False, "error": f"iCloud Drive not found at {icloud_root}"}

        self._base_path = os.path.join(icloud_root, container)
        os.makedirs(self._base_path, exist_ok=True)
        self._configured = True
        return {"ok": True, "provider": self.provider_name, "path": self._base_path}

    def is_configured(self) -> bool:
        return self._configured and os.path.isdir(self._base_path)

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "iCloud not configured"}
        try:
            dest = os.path.join(self._base_path, remote_key)
            os.makedirs(os.path.dirname(dest) or self._base_path, exist_ok=True)
            shutil.copy2(local_path, dest)
            size = os.path.getsize(dest)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key, "size_bytes": size}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "iCloud not configured"}
        try:
            src = os.path.join(self._base_path, remote_key)
            if not os.path.isfile(src):
                return {"ok": False, "error": f"Not found in iCloud: {remote_key}"}
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            shutil.copy2(src, local_path)
            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "iCloud not configured"}
        try:
            path = os.path.join(self._base_path, remote_key)
            if os.path.isfile(path):
                os.remove(path)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        result = []
        search_dir = os.path.join(self._base_path, prefix) if prefix else self._base_path
        if not os.path.isdir(search_dir):
            return []
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.startswith("."):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, self._base_path)
                st = os.stat(full)
                result.append(CloudFile(
                    name=f, path=rel, size_bytes=st.st_size, modified_at=st.st_mtime
                ))
        return result

    def get_quota(self) -> StorageQuota:
        if not self.is_configured():
            return StorageQuota(0, 0, 0, self.provider_name)
        try:
            stat = shutil.disk_usage(self._base_path)
            return StorageQuota(stat.total, stat.used, stat.free, self.provider_name)
        except Exception:
            return StorageQuota(0, 0, 0, self.provider_name)


# ---------------------------------------------------------------------------
# Dropbox Backend
# ---------------------------------------------------------------------------
class DropboxBackend(StorageBackend):
    """Dropbox storage backend.

    Credentials: {"access_token": "...", "folder": "/memory-os-ai"}
    """

    provider_name = "dropbox"

    def __init__(self) -> None:
        self._dbx = None
        self._folder: str = "/memory-os-ai"
        self._configured = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        try:
            import dropbox

            token = credentials.get("access_token", "")
            if not token:
                return {"ok": False, "error": "Provide 'access_token'"}

            self._folder = credentials.get("folder", "/memory-os-ai")
            self._dbx = dropbox.Dropbox(token)
            # Verify connection
            self._dbx.users_get_current_account()
            self._configured = True
            return {"ok": True, "provider": self.provider_name, "folder": self._folder}
        except ImportError:
            return {"ok": False, "error": "Install: pip install dropbox"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def is_configured(self) -> bool:
        return self._configured and self._dbx is not None

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Dropbox not configured"}
        try:
            import dropbox

            dest = f"{self._folder}/{remote_key}"
            with open(local_path, "rb") as f:
                meta = self._dbx.files_upload(
                    f.read(), dest, mode=dropbox.files.WriteMode.overwrite
                )
            return {
                "ok": True, "provider": self.provider_name,
                "remote_key": remote_key, "size_bytes": meta.size,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Dropbox not configured"}
        try:
            src = f"{self._folder}/{remote_key}"
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            self._dbx.files_download_to_file(local_path, src)
            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Dropbox not configured"}
        try:
            self._dbx.files_delete_v2(f"{self._folder}/{remote_key}")
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        try:
            import dropbox

            folder = f"{self._folder}/{prefix}" if prefix else self._folder
            result = self._dbx.files_list_folder(folder, recursive=True)
            files = []
            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        files.append(CloudFile(
                            name=entry.name,
                            path=entry.path_display.replace(f"{self._folder}/", ""),
                            size_bytes=entry.size,
                            modified_at=entry.server_modified.timestamp(),
                        ))
                if not result.has_more:
                    break
                result = self._dbx.files_list_folder_continue(result.cursor)
            return files
        except Exception:
            return []

    def get_quota(self) -> StorageQuota:
        if not self.is_configured():
            return StorageQuota(0, 0, 0, self.provider_name)
        try:
            usage = self._dbx.users_get_space_usage()
            used = usage.used
            total = 0
            if hasattr(usage.allocation, "allocated"):
                total = usage.allocation.get_individual().allocated
            return StorageQuota(total, used, max(0, total - used), self.provider_name)
        except Exception:
            return StorageQuota(0, 0, 0, self.provider_name)


# ---------------------------------------------------------------------------
# OneDrive Backend
# ---------------------------------------------------------------------------
class OneDriveBackend(StorageBackend):
    """Microsoft OneDrive storage backend.

    Credentials: {"access_token": "...", "folder": "/memory-os-ai"}
    Or local mount: {"mount_path": "/Users/.../OneDrive"}
    """

    provider_name = "onedrive"

    def __init__(self) -> None:
        self._mount_path: str = ""
        self._access_token: str = ""
        self._folder: str = "memory-os-ai"
        self._configured = False
        self._use_api = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        self._folder = credentials.get("folder", "memory-os-ai")

        # Prefer local mount (simpler, no API deps)
        mount = credentials.get("mount_path", "")
        if not mount:
            # Auto-detect common OneDrive paths
            for candidate in [
                os.path.expanduser("~/OneDrive"),
                os.path.expanduser("~/Library/CloudStorage/OneDrive-Personal"),
                os.path.expanduser("~/Library/CloudStorage/OneDrive"),
            ]:
                if os.path.isdir(candidate):
                    mount = candidate
                    break

        if mount and os.path.isdir(mount):
            self._mount_path = os.path.join(mount, self._folder)
            os.makedirs(self._mount_path, exist_ok=True)
            self._configured = True
            self._use_api = False
            return {"ok": True, "provider": self.provider_name, "mode": "mount", "path": self._mount_path}

        # Fall back to Graph API
        token = credentials.get("access_token", "")
        if token:
            self._access_token = token
            self._use_api = True
            self._configured = True
            return {"ok": True, "provider": self.provider_name, "mode": "api"}

        return {"ok": False, "error": "Provide 'mount_path' or 'access_token'"}

    def is_configured(self) -> bool:
        return self._configured

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "OneDrive not configured"}
        if not self._use_api:
            return self._fs_upload(local_path, remote_key)
        return self._api_upload(local_path, remote_key)

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "OneDrive not configured"}
        if not self._use_api:
            return self._fs_download(remote_key, local_path)
        return self._api_download(remote_key, local_path)

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "OneDrive not configured"}
        if not self._use_api:
            try:
                path = os.path.join(self._mount_path, remote_key)
                if os.path.isfile(path):
                    os.remove(path)
                return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return self._api_delete(remote_key)

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        if not self._use_api:
            return self._fs_list(prefix)
        return []  # API listing is complex — mount mode preferred

    def get_quota(self) -> StorageQuota:
        if not self.is_configured():
            return StorageQuota(0, 0, 0, self.provider_name)
        if not self._use_api and self._mount_path:
            stat = shutil.disk_usage(self._mount_path)
            return StorageQuota(stat.total, stat.used, stat.free, self.provider_name)
        return StorageQuota(0, 0, 0, self.provider_name)

    # --- Filesystem methods (mount mode) ---
    def _fs_upload(self, local_path: str, remote_key: str) -> dict:
        try:
            dest = os.path.join(self._mount_path, remote_key)
            os.makedirs(os.path.dirname(dest) or self._mount_path, exist_ok=True)
            shutil.copy2(local_path, dest)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key, "size_bytes": os.path.getsize(dest)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _fs_download(self, remote_key: str, local_path: str) -> dict:
        try:
            src = os.path.join(self._mount_path, remote_key)
            if not os.path.isfile(src):
                return {"ok": False, "error": f"Not found: {remote_key}"}
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            shutil.copy2(src, local_path)
            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _fs_list(self, prefix: str = "") -> list[CloudFile]:
        search_dir = os.path.join(self._mount_path, prefix) if prefix else self._mount_path
        if not os.path.isdir(search_dir):
            return []
        result = []
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.startswith("."):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, self._mount_path)
                st = os.stat(full)
                result.append(CloudFile(name=f, path=rel, size_bytes=st.st_size, modified_at=st.st_mtime))
        return result

    # --- API methods (Graph API) ---
    def _api_upload(self, local_path: str, remote_key: str) -> dict:
        try:
            import requests

            size = os.path.getsize(local_path)
            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{self._folder}/{remote_key}:/content"
            with open(local_path, "rb") as f:
                resp = requests.put(url, data=f, headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/octet-stream",
                })
            if resp.status_code in (200, 201):
                return {"ok": True, "provider": self.provider_name, "remote_key": remote_key, "size_bytes": size}
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _api_download(self, remote_key: str, local_path: str) -> dict:
        try:
            import requests

            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{self._folder}/{remote_key}:/content"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._access_token}"}, stream=True)
            if resp.status_code != 200:
                return {"ok": False, "error": f"HTTP {resp.status_code}"}
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _api_delete(self, remote_key: str) -> dict:
        try:
            import requests

            url = f"https://graph.microsoft.com/v1.0/me/drive/root:/{self._folder}/{remote_key}"
            resp = requests.delete(url, headers={"Authorization": f"Bearer {self._access_token}"})
            if resp.status_code in (200, 204):
                return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
            return {"ok": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Amazon S3 Backend
# ---------------------------------------------------------------------------
class S3Backend(StorageBackend):
    """Amazon S3 (or S3-compatible) storage backend.

    Credentials: {
        "bucket": "my-memory-bucket",
        "prefix": "memory-os-ai/",
        "aws_access_key_id": "...",
        "aws_secret_access_key": "...",
        "region": "us-east-1",
        "endpoint_url": "..."  # optional, for S3-compatible (MinIO, Wasabi, etc.)
    }
    """

    provider_name = "s3"

    def __init__(self) -> None:
        self._client = None
        self._bucket: str = ""
        self._prefix: str = "memory-os-ai/"
        self._configured = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        try:
            import boto3

            self._bucket = credentials.get("bucket", "")
            self._prefix = credentials.get("prefix", "memory-os-ai/")
            if not self._bucket:
                return {"ok": False, "error": "Provide 'bucket'"}

            kwargs: dict[str, Any] = {
                "service_name": "s3",
                "region_name": credentials.get("region", "us-east-1"),
            }
            if credentials.get("aws_access_key_id"):
                kwargs["aws_access_key_id"] = credentials["aws_access_key_id"]
                kwargs["aws_secret_access_key"] = credentials.get("aws_secret_access_key", "")
            if credentials.get("endpoint_url"):
                kwargs["endpoint_url"] = credentials["endpoint_url"]

            self._client = boto3.client(**kwargs)
            # Verify bucket access
            self._client.head_bucket(Bucket=self._bucket)
            self._configured = True
            return {"ok": True, "provider": self.provider_name, "bucket": self._bucket}
        except ImportError:
            return {"ok": False, "error": "Install: pip install boto3"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def is_configured(self) -> bool:
        return self._configured and self._client is not None

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "S3 not configured"}
        try:
            key = f"{self._prefix}{remote_key}"
            self._client.upload_file(local_path, self._bucket, key)
            size = os.path.getsize(local_path)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key, "size_bytes": size}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "S3 not configured"}
        try:
            key = f"{self._prefix}{remote_key}"
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            self._client.download_file(self._bucket, key, local_path)
            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "S3 not configured"}
        try:
            key = f"{self._prefix}{remote_key}"
            self._client.delete_object(Bucket=self._bucket, Key=key)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        try:
            full_prefix = f"{self._prefix}{prefix}"
            paginator = self._client.get_paginator("list_objects_v2")
            files = []
            for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key.replace(self._prefix, "", 1)
                    files.append(CloudFile(
                        name=os.path.basename(key),
                        path=rel,
                        size_bytes=obj["Size"],
                        modified_at=obj["LastModified"].timestamp(),
                    ))
            return files
        except Exception:
            return []

    def get_quota(self) -> StorageQuota:
        # S3 has no quota concept — virtually unlimited
        return StorageQuota(
            total_bytes=10 * 1024**4,  # 10 TB virtual
            used_bytes=0,
            free_bytes=10 * 1024**4,
            provider=self.provider_name,
        )


# ---------------------------------------------------------------------------
# Azure Blob Storage Backend
# ---------------------------------------------------------------------------
class AzureBlobBackend(StorageBackend):
    """Azure Blob Storage backend.

    Credentials: {
        "connection_string": "...",
        "container": "memory-os-ai"
    }
    """

    provider_name = "azure-blob"

    def __init__(self) -> None:
        self._client = None
        self._container_name: str = "memory-os-ai"
        self._configured = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        try:
            from azure.storage.blob import BlobServiceClient

            conn_str = credentials.get("connection_string", "")
            if not conn_str:
                return {"ok": False, "error": "Provide 'connection_string'"}

            self._container_name = credentials.get("container", "memory-os-ai")
            svc = BlobServiceClient.from_connection_string(conn_str)
            self._client = svc.get_container_client(self._container_name)

            # Create container if not exists
            try:
                self._client.get_container_properties()
            except Exception:
                self._client.create_container()

            self._configured = True
            return {"ok": True, "provider": self.provider_name, "container": self._container_name}
        except ImportError:
            return {"ok": False, "error": "Install: pip install azure-storage-blob"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def is_configured(self) -> bool:
        return self._configured and self._client is not None

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Azure Blob not configured"}
        try:
            blob = self._client.get_blob_client(remote_key)
            with open(local_path, "rb") as f:
                blob.upload_blob(f, overwrite=True)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key, "size_bytes": os.path.getsize(local_path)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Azure Blob not configured"}
        try:
            blob = self._client.get_blob_client(remote_key)
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            with open(local_path, "wb") as f:
                data = blob.download_blob()
                data.readinto(f)
            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Azure Blob not configured"}
        try:
            blob = self._client.get_blob_client(remote_key)
            blob.delete_blob()
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        try:
            blobs = self._client.list_blobs(name_starts_with=prefix)
            return [
                CloudFile(
                    name=b.name.split("/")[-1],
                    path=b.name,
                    size_bytes=b.size,
                    modified_at=b.last_modified.timestamp() if b.last_modified else 0,
                )
                for b in blobs
            ]
        except Exception:
            return []

    def get_quota(self) -> StorageQuota:
        return StorageQuota(5 * 1024**4, 0, 5 * 1024**4, self.provider_name)


# ---------------------------------------------------------------------------
# Box Backend
# ---------------------------------------------------------------------------
class BoxBackend(StorageBackend):
    """Box.com storage backend.

    Credentials: {"access_token": "...", "folder_id": "0"}
    """

    provider_name = "box"

    def __init__(self) -> None:
        self._client = None
        self._folder_id: str = "0"
        self._configured = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        try:
            from boxsdk import OAuth2, Client

            token = credentials.get("access_token", "")
            if not token:
                return {"ok": False, "error": "Provide 'access_token'"}

            self._folder_id = credentials.get("folder_id", "0")
            auth = OAuth2(
                client_id="",
                client_secret="",
                access_token=token,
            )
            self._client = Client(auth)
            self._client.user().get()  # verify
            self._configured = True
            return {"ok": True, "provider": self.provider_name, "folder_id": self._folder_id}
        except ImportError:
            return {"ok": False, "error": "Install: pip install boxsdk"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def is_configured(self) -> bool:
        return self._configured and self._client is not None

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Box not configured"}
        try:
            folder = self._client.folder(self._folder_id)
            name = os.path.basename(remote_key)
            with open(local_path, "rb") as f:
                uploaded = folder.upload_stream(f, name)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key, "file_id": uploaded.id}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Box not configured"}
        return {"ok": False, "error": "Box download requires file_id — use list_files first"}

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "Box not configured"}
        return {"ok": False, "error": "Box delete requires file_id — use list_files first"}

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        try:
            items = self._client.folder(self._folder_id).get_items(limit=1000)
            return [
                CloudFile(
                    name=item.name,
                    path=item.name,
                    size_bytes=getattr(item, "size", 0) or 0,
                    modified_at=0,
                )
                for item in items if item.type == "file"
            ]
        except Exception:
            return []

    def get_quota(self) -> StorageQuota:
        if not self.is_configured():
            return StorageQuota(0, 0, 0, self.provider_name)
        try:
            user = self._client.user().get()
            total = user.space_amount
            used = user.space_used
            return StorageQuota(total, used, max(0, total - used), self.provider_name)
        except Exception:
            return StorageQuota(0, 0, 0, self.provider_name)


# ---------------------------------------------------------------------------
# Backblaze B2 Backend
# ---------------------------------------------------------------------------
class B2Backend(StorageBackend):
    """Backblaze B2 storage backend (S3-compatible).

    Credentials: {
        "application_key_id": "...",
        "application_key": "...",
        "bucket_name": "memory-os-ai"
    }
    """

    provider_name = "b2"

    def __init__(self) -> None:
        self._bucket = None
        self._api = None
        self._configured = False

    def configure(self, credentials: dict[str, Any]) -> dict:
        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo

            key_id = credentials.get("application_key_id", "")
            key = credentials.get("application_key", "")
            bucket_name = credentials.get("bucket_name", "memory-os-ai")

            if not key_id or not key:
                return {"ok": False, "error": "Provide 'application_key_id' and 'application_key'"}

            info = InMemoryAccountInfo()
            self._api = B2Api(info)
            self._api.authorize_account("production", key_id, key)
            self._bucket = self._api.get_bucket_by_name(bucket_name)
            self._configured = True
            return {"ok": True, "provider": self.provider_name, "bucket": bucket_name}
        except ImportError:
            return {"ok": False, "error": "Install: pip install b2sdk"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def is_configured(self) -> bool:
        return self._configured and self._bucket is not None

    def upload(self, local_path: str, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "B2 not configured"}
        try:
            result = self._bucket.upload_local_file(local_file=local_path, file_name=remote_key)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key, "file_id": result.id_}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def download(self, remote_key: str, local_path: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "B2 not configured"}
        try:
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            dl = self._bucket.download_file_by_name(remote_key)
            dl.save_to(local_path)
            return {"ok": True, "provider": self.provider_name, "local_path": local_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete(self, remote_key: str) -> dict:
        if not self.is_configured():
            return {"ok": False, "error": "B2 not configured"}
        try:
            file_version = self._bucket.get_file_info_by_name(remote_key)
            self._bucket.delete_file_version(file_version.id_, remote_key)
            return {"ok": True, "provider": self.provider_name, "remote_key": remote_key}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def list_files(self, prefix: str = "") -> list[CloudFile]:
        if not self.is_configured():
            return []
        try:
            files = []
            for fv, _ in self._bucket.ls(folder_to_list=prefix):
                files.append(CloudFile(
                    name=fv.file_name.split("/")[-1],
                    path=fv.file_name,
                    size_bytes=fv.size,
                    modified_at=fv.upload_timestamp / 1000,
                ))
            return files
        except Exception:
            return []

    def get_quota(self) -> StorageQuota:
        return StorageQuota(10 * 1024**4, 0, 10 * 1024**4, self.provider_name)


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------
PROVIDERS: dict[str, type[StorageBackend]] = {
    "google-drive": GoogleDriveBackend,
    "icloud": ICloudBackend,
    "dropbox": DropboxBackend,
    "onedrive": OneDriveBackend,
    "s3": S3Backend,
    "azure-blob": AzureBlobBackend,
    "box": BoxBackend,
    "b2": B2Backend,
}

PROVIDER_NAMES = sorted(PROVIDERS.keys())


def get_backend(provider: str) -> StorageBackend:
    """Instantiate a storage backend by provider name."""
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider}. Available: {PROVIDER_NAMES}")
    return cls()
