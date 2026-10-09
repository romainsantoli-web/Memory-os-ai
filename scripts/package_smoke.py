"""Run outside the checkout after installing a built wheel or sdist.

Exercises real installed console scripts through MCP; never downloads weights.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
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
    assert distribution.version == "4.0.1", distribution.version
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
        "memory-os-relay": "memory_os_v4.ops.relay:main",
        "memory-os-admin": "memory_os_v4.ops.cli:main",
    }
    entry_points = {ep.name: ep.value for ep in distribution.entry_points if ep.group == "console_scripts"}
    assert expected.items() <= entry_points.items(), entry_points
    package_root = Path(distribution.locate_file("")).resolve()
    assert "site-packages" in package_root.parts, package_root
    for module in ("memory_os_ai", "memory_os_v4"):
        assert (package_root / module / "__init__.py").is_file(), module
    for module in ("memory_os_ai", "memory_os_v4", "memory_os_v4.ops"):
        imported = __import__(module, fromlist=["__file__"])
        assert package_root in Path(imported.__file__).resolve().parents, imported.__file__
    docs = Path(sys.executable).parent.parent / "share/doc/memory-os-ai"
    for document in ("AGENTS.md", "CLAUDE.md", "CHANGELOG.md", "v4.0.1.md", "v4/OPERATIONS.md",
                     "v4/A6_REVIEW_966eca5.md"):
        assert (docs / document).is_file(), document
    for name in ("memory-os-ai", "memory-os-v4", "memory-os-ai-v3"):
        command = Path(sys.executable).parent / name
        assert command.is_file(), command
        with tempfile.TemporaryDirectory(prefix="memory-package-smoke-") as data_dir:
            asyncio.run(smoke(command, data_dir))
        if name != "memory-os-ai-v3":
            subprocess.run([str(command), "--help"], check=True, timeout=15)
            subprocess.run([str(command), "--selfcheck"], check=True, timeout=45)
    admin = Path(sys.executable).parent / "memory-os-admin"
    relay = Path(sys.executable).parent / "memory-os-relay"
    assert admin.is_file() and relay.is_file()
    with tempfile.TemporaryDirectory(prefix="memory-ops-smoke-") as directory:
        root = Path(directory)
        env = {**os.environ, "MEMORY_V4_TOKEN_FILE": str(root / "http.token"),
               "MEMORY_V4_DIR": str(root / "store"), "MEMORY_V4_OPS_DIR": str(root / "ops"),
               "MEMORY_V4_BACKUP_DIR": str(root / "backups"), "MEMORY_V4_BACKUP_KEY": str(root / "backup.key"),
               "MEMORY_V4_URL": "http://127.0.0.1:9/mcp/", "MEMORY_V4_EMBED_MODEL": "hash",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        env.pop("PYTHONPATH", None)
        created = subprocess.run([str(admin), "token", "create"], env=env, check=True,
                                 capture_output=True, text=True, timeout=10)
        token = (root / "http.token").read_text().strip()
        assert token not in created.stdout + created.stderr
        assert (root / "http.token").stat().st_mode & 0o777 == 0o600
        setup = subprocess.run([str(admin), "setup", "--write-launchd", str(root / "plists")],
                               env=env, check=True, capture_output=True, text=True, timeout=10)
        assert token not in setup.stdout + setup.stderr and len(list((root / "plists").glob("*.plist"))) == 3
        hook = subprocess.run([str(admin), "hook", "stop-nudge"], input=json.dumps({}), env=env,
                              check=True, capture_output=True, text=True, timeout=10)
        assert not hook.stdout
        (root / "store").mkdir()
        (root / "store/items.jsonl").write_text('{"id":"smoke","kind":"fact","text":"Synthetic"}\n')
        result = subprocess.run([str(admin), "backup", "--no-stop"], env=env, check=True,
                                capture_output=True, text=True, timeout=15)
        snapshot = json.loads(result.stdout)
        assert snapshot["ok"] and snapshot["verified"] and not snapshot["qualified"]
        for arguments in (("restore-drill", "--no-server", "--archive",
                           str(root / "backups" / snapshot["archive"])),):
            result = subprocess.run([str(admin), *arguments], env=env, check=True,
                                    capture_output=True, text=True, timeout=15)
            drill = json.loads(result.stdout)
            assert drill["ok"] and drill["level"] == "files" and not drill["qualified"]
        unhealthy = subprocess.run([str(admin), "health"], env=env, capture_output=True,
                                   text=True, timeout=10)
        assert unhealthy.returncode == 1 and "server /health status" in unhealthy.stdout
        unavailable = subprocess.run([str(relay)], input="", env=env, capture_output=True,
                                     text=True, timeout=20)
        assert unavailable.returncode == 1, unavailable.returncode
        assert "cannot reach" in unavailable.stderr and token not in unavailable.stderr + unavailable.stdout
    print("4.0.1: three installed servers passed MCP startup; six admin commands and relay-unavailable passed.")
    print("Five entry points and installed docs/licenses verified; relay TCP/reconnection NOT qualified here.")
    print("Backup/drill CLI smoke is diagnostic only, not A6 durability/consistency qualification.")


if __name__ == "__main__":
    main()
