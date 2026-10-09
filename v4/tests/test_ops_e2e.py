"""End-to-end: packaged relay (``python -m memory_os_v4.ops.relay``) → SEC-1 HTTP server with a bearer token.

Real Uvicorn server on a free loopback port, temporary store, ``hash`` embedder (no model download).
Needs loopback TCP (skipped where a sandbox forbids binding).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from memory_os_v4.ops import auth

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _free_port() -> int:
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except PermissionError:
        pytest.skip("loopback bind forbidden in this sandbox")


def _status(url: str, headers: dict) -> int:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=3) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def test_relay_with_token_against_secured_server(tmp_path):
    port = _free_port()
    token = tmp_path / "http.token"
    auth.create_token(token)
    env = dict(
        os.environ,
        MEMORY_V4_DIR=str(tmp_path / "store"),
        MEMORY_V4_EMBED_MODEL="hash",
        MEMORY_V4_PROJECT="*",
        MEMORY_V4_TOKEN_FILE=str(token),
        MEMORY_V4_URL=f"http://127.0.0.1:{port}/mcp/",
        PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    server = subprocess.Popen(
        [sys.executable, "-m", "memory_os_v4.server", "--transport", "http", "--host", "127.0.0.1",
         "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    health = f"http://127.0.0.1:{port}/health"
    headers = {"Authorization": f"Bearer {auth.read_token(token)}"}
    try:
        for _ in range(60):
            if _status(health, headers) == 200:
                break
            time.sleep(0.5)
        assert _status(health, headers) == 200
        assert _status(health, {}) == 401

        async def roundtrip():
            params = StdioServerParameters(command=sys.executable, args=["-m", "memory_os_v4.ops.relay"], env=env)
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    tools = {t.name for t in (await s.list_tools()).tools}
                    add = await s.call_tool(
                        "memory_add_facts",
                        {"project": "e2e", "facts": [{"text": "The e2e relay test stores one fact."}]},
                    )
                    hits = await s.call_tool(
                        "memory_search", {"query": "e2e relay test fact", "project": "e2e", "top_k": 1}
                    )
                    return tools, json.loads(add.content[0].text), json.loads(hits.content[0].text)

        tools, added, hits = anyio.run(roundtrip)
        assert {"memory_add_facts", "memory_search", "memory_session_brief"} <= tools
        assert len(added["added"]) == 1 and len(hits["hits"]) == 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
