"""Run outside the checkout after installing a built wheel or sdist.

Exercises real installed console scripts through MCP; never downloads weights.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def smoke(command: Path, data_dir: str) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(
        MEMORY_V4_DIR=data_dir,
        MEMORY_V4_EMBED_MODEL="hash",
        MEMORY_CACHE_DIR=data_dir,
        MEMORY_REPLAY_LOG="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    params = StdioServerParameters(command=str(command), env=env)
    async with asyncio.timeout(45):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert {"memory_status", "memory_search", "memory_chat_save"} <= names


def main() -> None:
    distribution = importlib.metadata.distribution("memory-os-ai")
    assert distribution.metadata["License-Expression"] == "LGPL-3.0-or-later"
    files = {str(path): path for path in distribution.files or ()}
    for name, header in (("LICENSE", "GNU LESSER GENERAL PUBLIC LICENSE"),
                         ("LICENSE.GPL-3.0", "GNU GENERAL PUBLIC LICENSE")):
        candidates = [path for path in files.values() if path.name == name]
        assert len(candidates) == 1, f"packaged license missing/ambiguous: {name}"
        assert header in Path(distribution.locate_file(candidates[0])).read_text()
    expected = {
        "memory-os-ai": "memory_os_v4.server:run",
        "memory-os-v4": "memory_os_v4.server:run",
        "memory-os-ai-v3": "memory_os_ai.server:run",
    }
    entry_points = {ep.name: ep.value for ep in distribution.entry_points if ep.group == "console_scripts"}
    assert expected.items() <= entry_points.items(), entry_points
    package_root = Path(distribution.locate_file("")).resolve()
    assert "site-packages" in package_root.parts, package_root
    for module in ("memory_os_ai", "memory_os_v4"):
        assert (package_root / module / "__init__.py").is_file(), module
    for name in expected:
        command = Path(sys.executable).parent / name
        assert command.is_file(), command
        with tempfile.TemporaryDirectory(prefix="memory-package-smoke-") as data_dir:
            asyncio.run(smoke(command, data_dir))
        if name != "memory-os-ai-v3":
            subprocess.run([str(command), "--help"], check=True, timeout=15)
            subprocess.run([str(command), "--selfcheck"], check=True, timeout=45)
    print("Built distribution: all three installed entry points passed MCP startup.")


if __name__ == "__main__":
    main()
