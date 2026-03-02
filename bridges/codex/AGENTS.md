# Memory OS AI — Instructions for Codex CLI

## RULE #1: You have persistent memory — USE IT

You are connected to a semantic memory system (FAISS + JSONL) via MCP tools.
You MUST use it actively to maintain continuity across sessions.

## Session Start (mandatory)

At the START of every session, call `memory_session_brief` BEFORE responding:
```
memory_session_brief({})
```
This retrieves your memory of past sessions, projects, decisions, and pending tasks.

## Context Recovery

Call `memory_session_brief` whenever:
- You cannot recall something the user references
- The session is getting long and you need a refresh
- The user says "remember...", "we said that...", "like last time..."
- You feel uncertain about the state of a project or past decision

## Regular Saves

Call `memory_chat_save` to persist the current exchange:
1. After significant tasks (commit, PR, feature, fix)
2. Every ~10 exchanges
3. When the user says "remember this" or "this is important"
4. Before ending a long session
5. After architecture decisions or technical choices

Always include a `summary` field — it's what gets retrieved first:
```
memory_chat_save({
  "messages": [...],
  "summary": "Implemented cross-project memory sharing — 4 new tools, 20 tests",
  "project_label": "my-project"
})
```

## Search Before Guessing

Use `memory_search({"query": "..."})` or `memory_get_context({"query": "..."})` 
before answering from training data alone, especially for questions about:
- Past decisions, code changes, or architecture choices
- Project-specific context you should know but might not
- Anything the user implies "you should remember"

## Available Tools (18)

| Tool | When to Use |
|------|------------|
| `memory_session_brief` | **FIRST** + on context loss |
| `memory_chat_save` | **REGULARLY** — after tasks/decisions |
| `memory_compact` | Context too long → compress memory |
| `memory_project_link` | Share memory across projects |
| `memory_project_unlink` | Remove cross-project link |
| `memory_project_list` | List linked projects |
| `memory_ingest` | Index documents (PDF, TXT, DOCX, audio) |
| `memory_search` | Semantic search in memory |
| `memory_search_occurrences` | Count keyword occurrences |
| `memory_get_context` | Targeted context retrieval |
| `memory_list_documents` | List indexed documents |
| `memory_transcribe` | Transcribe audio files |
| `memory_status` | Engine status |
| `memory_chat_sync` | Sync chat sources |
| `memory_chat_source_add` | Add a chat source |
| `memory_chat_source_remove` | Remove a chat source |
| `memory_chat_status` | Chat sync status |
| `memory_chat_auto_detect` | Detect VS Code workspaces |
