# Codex CLI Bridge — Memory OS AI

## Installation

```bash
# Auto-install
python -m memory_os_ai.setup codex

# Or manually — copy to your project or global config:
mkdir -p ~/.codex
cp bridges/codex/.codex/mcp.json ~/.codex/mcp.json
```

## Usage with Codex

Codex CLI supports MCP tools, resources, and elicitation natively.
Once configured, Memory OS AI tools appear automatically in Codex sessions.

### System instructions for Codex

Add to your `codex.md` or `AGENTS.md`:
```
You have access to a persistent semantic memory system via memory_* MCP tools.
- Call memory_session_brief({}) at the start of each session to recall context.
- Call memory_chat_save({...}) regularly to persist important exchanges.
- Use memory_search({query: "..."}) for semantic search across all indexed documents.
```

## Supported Features

| MCP Feature | Codex Support | Memory OS AI |
|-------------|--------------|--------------|
| Tools | ✅ | ✅ 14 tools |
| Resources | ✅ | 🔜 Planned |
| Elicitation | ✅ | 🔜 Planned |
| stdio | ✅ | ✅ Default |
| HTTP streaming | ✅ | ✅ `--sse` flag |
