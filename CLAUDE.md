# Memory OS AI — Instructions for Claude Code

Follow the shared memory protocol in [`AGENTS.md`](AGENTS.md): it is the same for Claude, Codex and any other
agent connected to the memory-os-v4 server, so work can be resumed from one AI to another.

Claude Code specifics:
- Connect through the relay (`memory-os-admin setup` prints the `.mcp.json` entry); the token file stays private.
- Optional hooks: `SessionStart` → `memory-os-admin hook session-start` (project brief injected at start, resume,
  `/clear` and after compaction); `Stop` → `memory-os-admin hook stop-nudge` (asks for a save with facts after
  `MEMORY_NUDGE_THRESHOLD` tool calls without one). See `v4/docs/OPERATIONS.md`.
- Developing this repository: v4 engine in `v4/src/memory_os_v4`, tests `pytest v4/tests`, v3 in `src/memory_os_ai`.
