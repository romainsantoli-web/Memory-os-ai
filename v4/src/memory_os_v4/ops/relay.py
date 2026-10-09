"""stdio → streamable-HTTP relay to the shared memory-os-v4 server (``memory-os-relay``).

For stdio-only MCP clients (Claude Desktop) and for clients that would otherwise need the bearer token in a
plain-text config (Claude Code ``.mcp.json``, Codex): the relay reads the token file on every (re)connection.

Lifecycle: ONE owner task opens, uses and closes the upstream connection (AnyIO task groups and cancel scopes
must be exited by the task that entered them). Handlers send requests to the owner through a channel; requests
are serialized. On a transport failure the owner closes the connection in its own task and reconnects for the next
request.

Mutations are never replayed: if the connection drops during a tool call, the server may already have applied it,
so the relay reports an *uncertain result* instead of risking a duplicate write. Tool listing (read-only) is
retried once after reconnecting.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import anyio
import mcp.types as types
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
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
UNAVAILABLE = "memory-os relay: the memory server is unreachable; nothing was sent."

Operation = Callable[[ClientSession], Awaitable[Any]]


@dataclass
class _Request:
    op: Operation
    reply: MemoryObjectSendStream = field(repr=False)


class _Failed:
    def __init__(self, exc: BaseException, sent: bool) -> None:
        self.exc, self.sent = exc, sent


async def _port_alive(settings: OpsSettings) -> bool:
    """TCP-level liveness: a busy server still completes the handshake; a dead one refuses it."""
    try:
        with anyio.fail_after(2):
            stream = await anyio.connect_tcp("127.0.0.1" if "::1" not in settings.url else "::1", settings.port)
        await stream.aclose()
        return True
    except (OSError, TimeoutError):
        return False


async def _guarded(op: Operation, session: ClientSession, settings: OpsSettings) -> Any:
    """Run ``op``; abort it if the server stops accepting connections (the MCP client would otherwise wait
    forever for a response that will never come). Two consecutive refusals, 1 s apart, mean the server is gone."""
    box: dict[str, Any] = {}

    async def run(scope: anyio.CancelScope) -> None:
        box["value"] = await op(session)
        scope.cancel()

    async def watch() -> None:
        refusals = 0
        while True:
            await anyio.sleep(1.0)
            refusals = 0 if await _port_alive(settings) else refusals + 1
            if refusals >= 2:
                raise ConnectionError("memory server stopped accepting connections")

    async with anyio.create_task_group() as tg:
        tg.start_soon(run, tg.cancel_scope)
        tg.start_soon(watch)
    return box["value"]


async def _open(stack: AsyncExitStack, settings: OpsSettings) -> tuple[ClientSession, types.InitializeResult]:
    client = await stack.enter_async_context(http_client(settings))
    read, write, _ = await stack.enter_async_context(streamable_http_client(settings.url, http_client=client))
    session = await stack.enter_async_context(ClientSession(read, write))
    return session, await session.initialize()


async def owner(
    settings: OpsSettings,
    requests: MemoryObjectReceiveStream,
    ready: Callable[[types.InitializeResult | BaseException], None],
) -> None:
    """Single owner of the upstream connection. ``ready`` is called once with the first initialize result (or the
    first connection error). After a failure, the next request triggers one reconnection attempt before it is
    served; if that attempt fails the request is answered "unavailable" (nothing was sent)."""
    first = True
    carry: _Request | None = None  # received, not yet sent upstream
    inflight: _Request | None = None  # sent upstream, no answer yet
    async with requests:
        while True:
            try:
                async with AsyncExitStack() as stack:  # entered AND exited in this task only
                    session, init = await _open(stack, settings)
                    if first:
                        first = False
                        ready(init)
                    while True:
                        req, carry = (carry, None) if carry is not None else (await _next_or_none(requests), None)
                        if req is None:
                            return  # channel closed: relay shutting down
                        inflight = req
                        try:
                            result = await _guarded(req.op, session, settings)
                        except Exception as exc:  # transport failure while the request was in flight
                            inflight = None
                            await _reply(req, _Failed(exc, sent=True))
                            break  # leave the stack here (same task); reconnect on the next request
                        inflight = None
                        await _reply(req, result)
            except Exception as exc:  # connection lost (transport task group failed), or could not (re)connect
                if first:
                    ready(exc)
                    return
                if inflight is not None:  # the transport died under an in-flight request: outcome uncertain
                    await _reply(inflight, _Failed(exc, sent=True))
                    inflight = None
                if carry is not None:
                    await _reply(carry, _Failed(exc, sent=False))
                    carry = None
            carry = await _next_or_none(requests)
            if carry is None:
                return


async def _next_or_none(requests: MemoryObjectReceiveStream) -> _Request | None:
    try:
        return await requests.receive()
    except (anyio.EndOfStream, anyio.ClosedResourceError):
        return None


async def _reply(req: _Request, value: Any) -> None:
    try:
        async with req.reply:
            await req.reply.send(value)
    except (anyio.BrokenResourceError, anyio.ClosedResourceError):
        pass


class Relay:
    """Handler-side facade: submit an operation to the owner and wait for its result."""

    def __init__(self, send: MemoryObjectSendStream) -> None:
        self._send = send

    async def submit(self, op: Operation, *, replay: bool) -> Any:
        for attempt in (0, 1):
            reply_send, reply_recv = anyio.create_memory_object_stream(1)
            await self._send.send(_Request(op, reply_send))
            async with reply_recv:
                value = await reply_recv.receive()
            if not isinstance(value, _Failed):
                return value
            if not value.sent:
                raise RuntimeError(UNAVAILABLE) from value.exc
            if not replay or attempt == 1:
                raise RuntimeError(UNCERTAIN) from value.exc
        raise AssertionError("unreachable")


async def serve(settings: OpsSettings) -> None:
    send, recv = anyio.create_memory_object_stream(16)
    init_box: dict[str, Any] = {}
    init_event = anyio.Event()

    def ready(value) -> None:
        init_box["value"] = value
        init_event.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(owner, settings, recv, ready)
        await init_event.wait()
        init = init_box["value"]
        if isinstance(init, BaseException):
            print(f"memory-os relay: cannot reach {settings.url}: {type(init).__name__}", file=sys.stderr)
            await send.aclose()
            raise SystemExit(1)
        relay = Relay(send)
        server = Server(init.serverInfo.name, instructions=init.instructions)

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return (await relay.submit(lambda s: s.list_tools(), replay=True)).tools

        @server.call_tool(validate_input=False)
        async def call_tool(name: str, arguments: dict):
            res = await relay.submit(lambda s: s.call_tool(name, arguments), replay=False)
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
            await send.aclose()  # the owner exits its connection in its own task


def main() -> None:
    anyio.run(serve, OpsSettings.from_env())


if __name__ == "__main__":
    main()
