"""stdio → streamable-HTTP relay to the shared memory-os-v4 server (``memory-os-relay``).

For stdio-only MCP clients (Claude Desktop) and for clients that would otherwise need the bearer token in a
plain-text config (Claude Code ``.mcp.json``, Codex): the relay reads the token file on every (re)connection.

Mutations are never replayed: if the connection drops during a tool call, the server may already have applied
it, so the relay reports an *uncertain result* instead of risking a duplicate write. Read-only listing calls are
retried once after reconnecting.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack

import anyio
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from .auth import http_client
from .settings import OpsSettings

UNCERTAIN = (
    "memory-os relay: uncertain result, the connection was lost during the call. Do not repeat the "
    "mutation until its status is resolved: an empty search does not prove it was not applied."
)


class Upstream:
    def __init__(self, settings: OpsSettings) -> None:
        self.settings = settings
        self.stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.init: types.InitializeResult | None = None

    async def connect(self) -> None:
        await self.close()
        stack = AsyncExitStack()
        client = await stack.enter_async_context(http_client(self.settings))
        read, write, _ = await stack.enter_async_context(
            streamable_http_client(self.settings.url, http_client=client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        self.init = await session.initialize()
        self.stack, self.session = stack, session

    async def close(self) -> None:
        if self.stack is not None:
            try:
                await self.stack.aclose()
            except Exception:
                pass
        self.stack = self.session = None

    async def call(self, fn, replay: bool = True):
        if self.session is None:
            await self.connect()
        try:
            return await fn(self.session)
        except Exception as exc:
            try:
                await self.connect()
            except Exception:
                pass
            if not replay:
                raise RuntimeError(UNCERTAIN) from exc
            return await fn(self.session)


async def serve(settings: OpsSettings) -> None:
    up = Upstream(settings)
    try:
        await up.connect()
    except Exception as exc:
        print(f"memory-os relay: cannot reach {settings.url}: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)

    server = Server(
        up.init.serverInfo.name if up.init else "memory-os-v4",
        instructions=up.init.instructions if up.init else None,
    )

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return (await up.call(lambda s: s.list_tools())).tools

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict):
        res = await up.call(lambda s: s.call_tool(name, arguments), replay=False)
        if res.isError:
            text = "\n".join(c.text for c in res.content if isinstance(c, types.TextContent))
            raise RuntimeError(text or f"{name} failed")
        if res.structuredContent is not None:
            return res.content, res.structuredContent
        return res.content

    try:
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())
    finally:
        await up.close()


def main() -> None:
    anyio.run(serve, OpsSettings.from_env())


if __name__ == "__main__":
    main()
