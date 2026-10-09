"""Client side of the SEC-1 HTTP contract (docs/HTTP_CLIENT_CONTRACT.md) and secure secret files.

Secrets (bearer token, backup key) live in owner-only files: regular, not a symlink, a single hard link, owned
by the current user, mode exactly 0600, bounded size. They are never printed, never passed on a command line or in
a URL, and only ever sent to the validated loopback URL — no proxy, no redirect.
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import urllib.error
import urllib.request
from pathlib import Path

from .settings import OpsSettings

_TOKEN_RE = re.compile(rb"[A-Za-z0-9_-]{43,256}")


class SecretFileError(ValueError):
    """A secret file is missing, unsafe or malformed (message never contains the secret)."""


def read_secret_file(path: Path, *, max_bytes: int, exact: int | None = None) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise
    except OSError as exc:  # ELOOP for a symlink, EACCES…
        raise SecretFileError(f"{path}: cannot be opened safely ({type(exc).__name__})") from None
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
        ):
            raise SecretFileError(f"{path}: must be a regular file, owned by this user, mode 0600, one link")
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes or (exact is not None and len(data) != exact):
        raise SecretFileError(f"{path}: unexpected size")
    return data


def write_secret_file(path: Path, data: bytes) -> None:
    """Create exclusively (never overwrite) with mode 0600, fsync the file and its directory."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def read_token(path: Path) -> str | None:
    """The bearer token, or None when the file does not exist. Unsafe or malformed files raise."""
    try:
        raw = read_secret_file(path, max_bytes=257)
    except FileNotFoundError:
        return None
    raw = raw[:-1] if raw.endswith(b"\n") else raw
    if not _TOKEN_RE.fullmatch(raw):
        raise SecretFileError(f"{path}: invalid token format")
    return raw.decode("ascii")


def create_token(path: Path) -> bool:
    """Create a 256-bit token file if missing (True). An existing file is validated, never overwritten (False);
    an unsafe or malformed existing file raises."""
    if path.exists() or path.is_symlink():
        read_token(path)
        return False
    write_secret_file(path, (secrets.token_urlsafe(32) + "\n").encode("ascii"))
    return True


def auth_headers(settings: OpsSettings | None = None) -> dict[str, str]:
    settings = settings or OpsSettings.from_env()
    token = read_token(settings.token_file)
    return {"Authorization": f"Bearer {token}"} if token else {}


def http_client(settings: OpsSettings | None = None):
    """``httpx.AsyncClient`` with the auth header, ignoring proxy environment, never following redirects."""
    import httpx

    return httpx.AsyncClient(
        headers=auth_headers(settings),
        timeout=httpx.Timeout(30.0, read=300.0),
        trust_env=False,
        follow_redirects=False,
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # never forward the bearer token elsewhere
        return None


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def http_get(url: str, settings: OpsSettings | None = None, timeout: float = 3.0) -> tuple[int, bytes]:
    """GET on the local server with auth; returns (status, body), status -1 when unreachable."""
    settings = settings or OpsSettings.from_env()
    req = urllib.request.Request(url, headers=auth_headers(settings))
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, resp.read(65536)
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception:
        return -1, b""


def health_ok(settings: OpsSettings | None = None, timeout: float = 2.0) -> bool:
    settings = settings or OpsSettings.from_env()
    try:
        return http_get(settings.health_url, settings, timeout)[0] == 200
    except SecretFileError:
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
