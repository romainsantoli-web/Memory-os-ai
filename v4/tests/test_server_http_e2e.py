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

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

pytestmark = pytest.mark.asyncio


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def http_server(tmp_path):
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ)
    env.update(PYTHONPATH=src_dir, MEMORY_V4_DIR=str(tmp_path), MEMORY_V4_EMBED_MODEL="hash",
               HF_HUB_OFFLINE="1", MEMORY_V4_PROJECT="e2e")
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "memory_os_v4.server", "--transport", "http", "--port", str(port), "--warm"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=1) as r:
                assert json.loads(r.read())["engine_loaded"] is True
                break
        except Exception:
            if proc.poll() is not None or time.time() > deadline:
                proc.kill()
                pytest.fail("server did not start: " + proc.stderr.read().decode()[-2000:])
            time.sleep(0.2)
    yield base + "/mcp/", tmp_path
    proc.terminate()
    proc.wait(timeout=30)


async def _call(url, name, args):
    async with streamable_http_client(url) as (r, w, _):
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
