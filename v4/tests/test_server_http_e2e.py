"""End-to-end test: one shared ``--transport http`` server, two concurrent
MCP clients. What one client saves, the other must find."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

pytestmark = pytest.mark.asyncio
TEST_TOKEN = "synthetic-test-token-" + "x" * 32


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(tmp_path):
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ)
    token_file = tmp_path / "http-token"
    token_file.write_text(TEST_TOKEN + "\n")
    token_file.chmod(0o600)
    env.update(PYTHONPATH=src_dir, MEMORY_V4_DIR=str(tmp_path), MEMORY_V4_EMBED_MODEL="hash",
               HF_HUB_OFFLINE="1", MEMORY_V4_PROJECT="e2e", MEMORY_V4_TOKEN_FILE=str(token_file),
               MEMORY_V4_BUILD_COMMIT="a" * 40)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "memory_os_v4.server", "--transport", "http", "--port", str(port), "--warm"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while True:
        try:
            request = urllib.request.Request(
                base + "/health", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
            with urllib.request.urlopen(request, timeout=1) as r:
                assert json.loads(r.read())["engine_loaded"] is True
                break
        except Exception:
            if proc.poll() is not None or time.time() > deadline:
                proc.kill()
                pytest.fail("server did not start: " + proc.stderr.read().decode()[-2000:])
            time.sleep(0.2)
    try:
        yield base + "/mcp/", tmp_path
    finally:
        proc.terminate()
        proc.wait(timeout=30)


async def _call(url, name, args):
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as client:
        async with streamable_http_client(url, http_client=client) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(name, args)
                return json.loads(res.content[0].text)


async def test_two_clients_share_one_engine(http_server):
    url, data_dir = http_server
    save, status = await asyncio.gather(
        _call(url, "memory_chat_save", {
            "messages": [{"role": "user", "content": "Le VPS de Londres est chez OVH, utilisateur ubuntu."}],
            "summary": "VPS Londres OVH ubuntu",
        }),
        _call(url, "memory_status", {}),
    )
    assert "error" not in save, save
    assert "error" not in status, status

    found = await _call(url, "memory_search", {"query": "VPS Londres OVH"})
    assert "OVH" in json.dumps(found, ensure_ascii=False)
    # single writer, persisted where the shared server was told to write
    assert (data_dir / "items.jsonl").exists()


async def test_two_http_clients_retry_one_mutation(http_server):
    url, data_dir = http_server
    payload = {"request_id": "shared-turn-1", "messages": [{"role": "user", "content": "HTTP retry turn"}]}
    first, second = await asyncio.gather(_call(url, "memory_chat_save", payload),
                                         _call(url, "memory_chat_save", payload))
    assert "error" not in first, first
    assert second == first
    rows = [json.loads(line) for line in (data_dir / "items.jsonl").read_text().splitlines()]
    assert len({row["id"] for row in rows}) == 1
    conflict = await _call(url, "memory_chat_save", {**payload, "summary": "conflicting reuse"})
    assert conflict["ok"] is False
    assert "different operation or payload" in conflict["error"]


@pytest.mark.parametrize("headers,status", [
    ({}, 401),
    ({"Authorization": "Bearer wrong"}, 401),
    ({"Authorization": f"Bearer {TEST_TOKEN}", "Host": "attacker.invalid"}, 421),
    ({"Authorization": f"Bearer {TEST_TOKEN}", "Origin": "https://attacker.invalid"}, 403),
])
async def test_rejected_network_requests_leave_store_unchanged(http_server, headers, status):
    url, data_dir = http_server
    before = {p.name: p.read_bytes() for p in data_dir.iterdir() if p.is_file()}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json={"jsonrpc": "2.0", "id": 1,
                                     "method": "tools/call", "params": {"name": "memory_add_facts",
                                     "arguments": {"facts": [{"text": "Must never persist"}]}}})
    assert response.status_code == status
    assert {p.name: p.read_bytes() for p in data_dir.iterdir() if p.is_file()} == before


async def test_authenticated_ready_reports_engine_and_build_identity(http_server):
    url, _ = http_server
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TEST_TOKEN}"}) as client:
        ready = await client.get(url.replace("/mcp/", "/ready"))
        health = await client.get(url.replace("/mcp/", "/health"))
    assert ready.status_code == health.status_code == 200
    assert ready.json()["ready"] is True and ready.json()["commit"] == "a" * 40
    assert ready.json()["version"] == health.json()["version"] == "4.0.1rc1"


async def test_ready_requires_authorization_over_tcp(http_server):
    url, _ = http_server
    async with httpx.AsyncClient() as client:
        assert (await client.get(url.replace("/mcp/", "/ready"))).status_code == 401
