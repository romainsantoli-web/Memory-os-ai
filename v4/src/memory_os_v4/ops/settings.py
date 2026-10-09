"""Environment-driven settings shared by the ops tools (no machine-specific defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(name: str, default: str) -> Path:
    return Path(os.path.expanduser(os.environ.get(name, default)))


@dataclass(frozen=True)
class OpsSettings:
    url: str
    token_file: Path
    store_dir: Path
    backup_dir: Path
    backup_key: Path
    ops_dir: Path
    service_label: str | None

    @property
    def health_url(self) -> str:
        return self.url.rsplit("/mcp", 1)[0] + "/health"

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
