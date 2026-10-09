"""Fail-closed security for the local HTTP transport (stdio is unchanged)."""
from __future__ import annotations

import hmac
import os
from pathlib import Path
import re
import stat

from mcp.server.transport_security import TransportSecurityMiddleware, TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse


def load_http_token(path: str | Path | None) -> bytes:
    """Read an owner-only, regular token file without following a symlink.

    Tokens are URL-safe ASCII, 43-256 characters (secrets.token_urlsafe(32)
    generates the recommended 256-bit token). Never include credentials in errors.
    """
    if not path:
        raise ValueError("HTTP requires MEMORY_V4_TOKEN_FILE (owner-only mode 0600)")
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.geteuid() or info.st_nlink != 1):
            raise ValueError("HTTP token file must be regular, owned by this user, and mode 0600")
        with os.fdopen(fd, "rb") as fh:
            fd = None
            token = fh.read(258)
        if token.endswith(b"\n"):
            token = token[:-1]
        if re.fullmatch(rb"[A-Za-z0-9_-]{43,256}", token) is None:
            raise ValueError("HTTP token must be 43-256 URL-safe ASCII characters")
        return token
    except OSError:
        raise ValueError("HTTP token file cannot be read safely") from None
    finally:
        if fd is not None:
            os.close(fd)


def local_security_settings(host: str, port: int) -> TransportSecuritySettings:
    if host not in {"127.0.0.1", "localhost", "::1"} or not 1 <= port <= 65535:
        raise ValueError("HTTP ALPHA is restricted to loopback and a valid TCP port")
    authorities = [f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=authorities,
                                     allowed_origins=[f"http://{value}" for value in authorities])


class LocalHTTPGuard:
    """Protect all HTTP routes, including /health, before MCP/engine dispatch."""

    def __init__(self, app, *, settings: TransportSecuritySettings, token: bytes):
        self.app = app
        self.security = TransportSecurityMiddleware(settings)
        self._token = token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        headers = scope.get("headers", [])
        if any(sum(key.lower() == name for key, _ in headers) > 1
               for name in (b"host", b"origin", b"authorization")):
            await JSONResponse({"error": "Ambiguous security headers"}, status_code=400)(scope, receive, send)
            return
        response = await self.security.validate_request(request, is_post=request.method == "POST")
        if response is not None:
            await response(scope, receive, send)
            return
        authorization = request.headers.get("authorization", "")
        scheme, separator, value = authorization.partition(" ")
        if (not separator or scheme.lower() != "bearer"
                or not hmac.compare_digest(value.encode("utf-8"), self._token)):
            await JSONResponse({"error": "Unauthorized"}, status_code=401,
                               headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
            return
        await self.app(scope, receive, send)
