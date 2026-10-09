"""HTTP transport guards: credentials, origin/host and no unauthorized dispatch."""
from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from memory_os_v4 import server
from memory_os_v4.http_security import LocalHTTPGuard, load_http_token, local_security_settings

TOKEN = b"synthetic-test-token-" + b"x" * 32


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "token"
    path.write_bytes(TOKEN + b"\n")
    path.chmod(0o600)
    return path


def test_read_regular_owner_only_file(token_file):
    assert load_http_token(token_file) == TOKEN


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o400, 0o660])
def test_unsafe_modes_rejected(token_file, mode):
    token_file.chmod(mode)
    with pytest.raises(ValueError, match="0600"):
        load_http_token(token_file)


@pytest.mark.parametrize("value", [b"", b"short", b"x" * 257, b"x" * 43 + b"\n\n",
                                 b" " + b"x" * 43, b"x" * 43 + b"\x00", b"\xff" * 43])
def test_malformed_tokens_rejected_without_disclosure(token_file, value):
    token_file.write_bytes(value)
    with pytest.raises(ValueError) as exc:
        load_http_token(token_file)
    assert "synthetic-test-token" not in str(exc.value)


def test_missing_symlink_hardlink_directory_fifo_and_other_owner_rejected(token_file, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="requires"):
        load_http_token(None)
    for path in [tmp_path / "missing", tmp_path]:
        with pytest.raises(ValueError):
            load_http_token(path)
    linked = tmp_path / "link"
    linked.symlink_to(token_file)
    with pytest.raises(ValueError):
        load_http_token(linked)
    linked.unlink()
    os.link(token_file, linked)
    with pytest.raises(ValueError):
        load_http_token(token_file)
    linked.unlink()
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(ValueError):
        load_http_token(fifo)
    monkeypatch.setattr(os, "geteuid", lambda: token_file.stat().st_uid + 1)
    with pytest.raises(ValueError, match="owned"):
        load_http_token(token_file)


@pytest.mark.parametrize("host,port", [("0.0.0.0", 8765), ("192.168.1.1", 8765),
                                       ("example.com", 8765), ("127.0.0.1", 0), ("::1", 65536)])
def test_remote_bind_and_invalid_ports_refused(host, port):
    with pytest.raises(ValueError, match="loopback"):
        local_security_settings(host, port)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_settings_use_exact_local_authorities(host):
    settings = local_security_settings(host, 8765)
    assert settings.enable_dns_rebinding_protection is True
    assert settings.allowed_hosts == ["127.0.0.1:8765", "localhost:8765", "[::1]:8765"]
    assert all("*" not in value for value in settings.allowed_hosts + settings.allowed_origins)


@pytest.fixture
def guard():
    calls = []

    async def app(scope, receive, send):
        calls.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"authorized"})

    return LocalHTTPGuard(app, settings=local_security_settings("127.0.0.1", 8765), token=TOKEN), calls


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [None, "http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"])
async def test_valid_requests_and_missing_origin_allowed(guard, origin):
    middleware, calls = guard
    headers = {"Authorization": "Bearer " + TOKEN.decode()}
    if origin is not None:
        headers["Origin"] = origin
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware)) as client:
        response = await client.get("http://127.0.0.1:8765/health", headers=headers)
    assert response.status_code == 200 and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides,status", [
    ({"authorization": ""}, 401), ({"authorization": "Bearer wrong"}, 401),
    ({"authorization": TOKEN.decode()}, 401), ({"authorization": "Basic abc"}, 401),
    ({"host": "attacker.invalid:8765"}, 421), ({"host": "127.0.0.1:8766"}, 421),
    ({"host": "127.0.0.1.attacker.invalid:8765"}, 421),
    ({"origin": "https://attacker.invalid"}, 403), ({"origin": "null"}, 403),
    ({"origin": "http://localhost:8766"}, 403), ({"origin": "http://localhost:8765.attacker.invalid"}, 403),
])
async def test_invalid_requests_never_dispatch(guard, overrides, status):
    middleware, calls = guard
    headers = {"authorization": "Bearer " + TOKEN.decode(), **overrides}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware)) as client:
        response = await client.get("http://127.0.0.1:8765/health", headers=headers)
    assert response.status_code == status and calls == []
    assert TOKEN.decode() not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate", ["host", "origin", "authorization"])
async def test_duplicate_security_headers_rejected(guard, duplicate):
    middleware, calls = guard
    headers = [("authorization", "Bearer " + TOKEN.decode()), ("host", "127.0.0.1:8765"),
               ("origin", "http://127.0.0.1:8765"), (duplicate, "ambiguous")]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware)) as client:
        response = await client.get("http://127.0.0.1:8765/health", headers=headers)
    assert response.status_code == 400 and calls == []


@pytest.mark.asyncio
async def test_missing_host_and_invalid_post_content_type_rejected(guard):
    middleware, calls = guard
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware)) as client:
        request = client.build_request("GET", "http://127.0.0.1:8765/health")
        del request.headers["host"]
        assert (await client.send(request)).status_code == 421
        response = await client.post("http://127.0.0.1:8765/mcp/", content="bad",
                                     headers={"Authorization": "Bearer " + TOKEN.decode(),
                                              "Content-Type": "text/plain"})
    assert response.status_code == 400 and calls == []


def test_http_startup_fails_before_loading_engine_without_credentials(monkeypatch):
    monkeypatch.delenv("MEMORY_V4_TOKEN_FILE", raising=False)
    monkeypatch.setattr(server, "get_engine", lambda: pytest.fail("Must not load engine"))
    with pytest.raises(ValueError, match="requires"):
        server._run_http("127.0.0.1", 8765, warm=True)


def test_manager_receives_sdk_security_settings(token_file, monkeypatch):
    from mcp.server import streamable_http_manager
    real_manager = streamable_http_manager.StreamableHTTPSessionManager
    captured = []

    def manager(**kwargs):
        captured.append(kwargs["security_settings"])
        return real_manager(**kwargs)

    monkeypatch.setenv("MEMORY_V4_TOKEN_FILE", str(token_file))
    monkeypatch.setattr(streamable_http_manager, "StreamableHTTPSessionManager", manager)
    app = server._build_http_app("127.0.0.1", 8765)
    assert app.security.settings is captured[0]
    assert captured[0].enable_dns_rebinding_protection is True


@pytest.mark.asyncio
async def test_rotation_requires_new_application_and_revokes_old_token(token_file, monkeypatch):
    monkeypatch.setenv("MEMORY_V4_TOKEN_FILE", str(token_file))
    original = server._build_http_app("127.0.0.1", 8765)
    new_token = b"replacement-test-token-" + b"y" * 32
    token_file.write_bytes(new_token)
    replacement = server._build_http_app("127.0.0.1", 8765)
    assert original._token == TOKEN
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=replacement)) as client:
        assert (await client.get("http://127.0.0.1:8765/health", headers={
            "Authorization": "Bearer " + TOKEN.decode()})).status_code == 401
        assert (await client.get("http://127.0.0.1:8765/health", headers={
            "Authorization": "Bearer " + new_token.decode()})).status_code == 200


@asynccontextmanager
async def _application_client(tmp_path, token_file, monkeypatch):
    from memory_os_v4.config import Config
    from memory_os_v4.memory import MemoryV4

    engine = MemoryV4(config=Config(data_dir=str(tmp_path / "store"), embed_model="hash"))
    monkeypatch.setattr(server, "_engine", engine)
    monkeypatch.setenv("MEMORY_V4_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("MEMORY_V4_PROJECT", "security-test")
    app = server._build_http_app("127.0.0.1", 8765)
    try:
        async with app.app.router.lifespan_context(app.app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://127.0.0.1:8765") as client:
                yield client, engine
    finally:
        engine.close()


async def _rpc(client, method, params, *, headers=None):
    response = await client.post("/mcp/", json={"jsonrpc": "2.0", "id": 1,
                                              "method": method, "params": params},
                                 headers={"Authorization": "Bearer " + TOKEN.decode(),
                                          "Accept": "application/json, text/event-stream", **(headers or {})})
    assert response.status_code == 200, response.text
    payloads = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    result = json.loads(payloads[-1]) if payloads else response.json()
    assert "error" not in result, result
    return result["result"]


@pytest.mark.asyncio
async def test_real_mcp_application_initialize_save_search_and_retry(tmp_path, token_file, monkeypatch):
    async with _application_client(tmp_path, token_file, monkeypatch) as (client, engine):
        initialized = await _rpc(client, "initialize", {"protocolVersion": "2025-03-26",
                                 "capabilities": {}, "clientInfo": {"name": "security-test", "version": "1"}})
        assert initialized["serverInfo"]["name"] == "memory-os-v4"
        payload = {"name": "memory_add_facts", "arguments": {"request_id": "authorized-1",
                   "project": "security-test", "facts": [{"text": "Synthetic client uses Neptune."}]}}
        saved = await _rpc(client, "tools/call", payload)
        assert saved.get("isError", False) is False
        assert await _rpc(client, "tools/call", payload) == saved
        found = await _rpc(client, "tools/call", {"name": "memory_search", "arguments": {
                           "query": "Neptune", "project": "security-test"}})
        assert "Neptune" in json.dumps(found)
        assert len(engine.store.all()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,status", [
    ({"Authorization": ""}, 401), ({"Authorization": "Bearer wrong"}, 401),
    ({"Host": "attacker.invalid:8765"}, 421), ({"Origin": "https://attacker.invalid"}, 403),
])
async def test_real_mcp_application_rejects_before_mutation(tmp_path, token_file, monkeypatch, headers, status):
    async with _application_client(tmp_path, token_file, monkeypatch) as (client, engine):
        before = {p.name: p.read_bytes() for p in engine.store.data_dir.iterdir() if p.is_file()}
        response = await client.post("/mcp/", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                     "params": {"name": "memory_add_facts", "arguments": {
                                         "facts": [{"text": "Must not persist"}]}}},
                                     headers={"Authorization": "Bearer " + TOKEN.decode(),
                                              "Accept": "application/json, text/event-stream", **headers})
        assert response.status_code == status
        assert engine.store.all() == []
        assert {p.name: p.read_bytes() for p in engine.store.data_dir.iterdir() if p.is_file()} == before
