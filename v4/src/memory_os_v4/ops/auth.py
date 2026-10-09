"""Client side of the SEC-1 HTTP contract (docs/HTTP_CLIENT_CONTRACT.md).

The bearer token lives in an owner-only (0600) regular file. It is never printed, never passed on a command
line or in a URL. When the file is absent, requests are sent without the header (useful against a server that
predates SEC-1); a server with SEC-1 then answers 401.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import urllib.request
from pathlib import Path

from .settings import OpsSettings

_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43,256}")


def read_token(path: Path) -> str | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
        ):
            raise PermissionError(f"{path} must be a regular file owned by this user with mode 0600")
        raw = fh.read(258).rstrip(b"\n")
    token = raw.decode("ascii", errors="replace")
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError(f"{path}: invalid token format")
    return token


def create_token(path: Path) -> bool:
    """Create a 256-bit URL-safe token file (0600) if missing. Returns True when created. Never prints it."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(secrets.token_urlsafe(32) + "\n")
    return True


def auth_headers(settings: OpsSettings | None = None) -> dict[str, str]:
    settings = settings or OpsSettings.from_env()
    token = read_token(settings.token_file)
    return {"Authorization": f"Bearer {token}"} if token else {}


def http_client(settings: OpsSettings | None = None):
    """An ``httpx.AsyncClient`` carrying the auth header; the caller closes it."""
    import httpx

    return httpx.AsyncClient(headers=auth_headers(settings), timeout=httpx.Timeout(30.0, read=300.0))


def health_ok(settings: OpsSettings | None = None, timeout: float = 2.0) -> bool:
    settings = settings or OpsSettings.from_env()
    req = urllib.request.Request(settings.health_url, headers=auth_headers(settings))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


async def call_tool(name: str, arguments: dict, settings: OpsSettings | None = None) -> dict:
    """One authenticated MCP tool call; returns the decoded JSON payload."""
    import json

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    settings = settings or OpsSettings.from_env()
    async with http_client(settings) as hc, streamable_http_client(settings.url, http_client=hc) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            res = await session.call_tool(name, arguments)
            if res.isError:
                raise RuntimeError(res.content[0].text if res.content else f"{name} failed")
            return json.loads(res.content[0].text)
