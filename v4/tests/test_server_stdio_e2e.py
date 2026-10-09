"""End-to-end test: launches ``python -m memory_os_v4.server`` as a real
subprocess over stdio and drives it with the official MCP client."""

from __future__ import annotations

import json
import os
import sys

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

pytestmark = pytest.mark.asyncio


def _server_params(tmp_path) -> StdioServerParameters:
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_dir
    env["MEMORY_V4_DIR"] = str(tmp_path)
    env["MEMORY_V4_EMBED_MODEL"] = "hash"
    env["HF_HUB_OFFLINE"] = "1"
    env["MEMORY_V4_PROJECT"] = "e2e"
    return StdioServerParameters(
        command=sys.executable, args=["-m", "memory_os_v4.server"], env=env
    )


async def test_stdio_list_tools_and_status_and_chat_save(tmp_path):
    params = _server_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            names = {t.name for t in tools_result.tools}
            assert "memory_status" in names
            assert "memory_chat_save" in names
            assert "memory_search" in names
            assert "memory_forget" in names

            status_result = await session.call_tool("memory_status", {})
            status_text = status_result.content[0].text
            status = json.loads(status_text)
            assert status["embed_model"] == "hash"
            assert status["total"] == 0

            save_result = await session.call_tool(
                "memory_chat_save",
                {
                    "messages": [{"role": "user", "content": "Hello from the e2e stdio test."}],
                    "summary": "e2e stdio smoke test ran.",
                    "facts": [{"text": "The e2e stdio test ran successfully."}],
                },
            )
            save_payload = json.loads(save_result.content[0].text)
            assert save_payload["episodes"] == 1
            assert len(save_payload["facts"]["added"]) == 1

            status_after = await session.call_tool("memory_status", {})
            status_after_payload = json.loads(status_after.content[0].text)
            assert status_after_payload["total"] >= 3  # episode + summary + fact
