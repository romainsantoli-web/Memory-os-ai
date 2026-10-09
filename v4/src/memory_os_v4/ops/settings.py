"""Environment-driven settings shared by the ops tools (no machine-specific defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _path(name: str, default: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(name, default)))


def validate_url(url: str) -> str:
    """The bearer token may only ever be sent to the local server: plain HTTP on a loopback host, explicit port,
    path exactly ``/mcp/``, no userinfo, query or fragment."""
    parts = urlsplit(url)
    if (
        parts.scheme != "http"
        or parts.hostname not in LOOPBACK_HOSTS
        or parts.port is None
        or parts.username is not None
        or parts.password is not None
        or parts.path != "/mcp/"
        or parts.query
        or parts.fragment
    ):
        raise ValueError("MEMORY_V4_URL must be http://<loopback>:<port>/mcp/ (no userinfo, query or fragment)")
    return url


@dataclass(frozen=True)
class OpsSettings:
    url: str
    token_file: Path
    store_dir: Path
    backup_dir: Path
    backup_key: Path
    ops_dir: Path
    service_label: str | None

    def __post_init__(self) -> None:
        validate_url(self.url)

    @property
    def base_url(self) -> str:
        return self.url.rsplit("/mcp/", 1)[0]

    @property
    def health_url(self) -> str:
        return self.base_url + "/health"

    @property
    def ready_url(self) -> str:
        return self.base_url + "/ready"

    @property
    def port(self) -> int:
        return urlsplit(self.url).port or 0

    @property
    def launchd_plist(self) -> Path | None:
        if not self.service_label:
            return None
        return Path.home() / "Library/LaunchAgents" / f"{self.service_label}.plist"

    @classmethod
    def from_env(cls) -> "OpsSettings":
        return cls(
            url=os.environ.get("MEMORY_V4_URL", "http://127.0.0.1:8765/mcp/"),
            token_file=_path("MEMORY_V4_TOKEN_FILE", "~/.config/memory-os/http.token"),
            store_dir=_path("MEMORY_V4_DIR", "~/.memory-os-v4"),
            backup_dir=_path("MEMORY_V4_BACKUP_DIR", "~/memory-os-backups"),
            backup_key=_path("MEMORY_V4_BACKUP_KEY", "~/.config/memory-os/backup.key"),
            ops_dir=_path("MEMORY_V4_OPS_DIR", "~/.local/state/memory-os"),
            service_label=os.environ.get("MEMORY_V4_SERVICE_LABEL") or None,
        )
