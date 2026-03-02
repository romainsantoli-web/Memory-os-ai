# Memory OS AI — Instructions for Claude Code

## RULE #1: You have persistent memory — USE IT

You have access to a semantic memory system (FAISS + JSONL) via MCP tools.
You MUST use it actively to maintain context across sessions.

## Session Start (mandatory)

At the START of every conversation, call `memory_session_brief` BEFORE responding:
```
memory_session_brief({})
```

## Context Recovery (anti-forgetting)

Call `memory_session_brief` whenever:
- You lose track of the conversation
- The user references something you can't find
- The conversation is long (>20 exchanges) and you need a refresh
- The user says "remember...", "we said that...", "like last time..."

## Regular Saves (anti-loss)

Call `memory_chat_save` to persist the current conversation:
1. After significant tasks (commit, PR, feature)
2. Every ~10 exchanges
3. When the user says "remember this"
4. Before ending a long session
5. After architecture decisions

Include a `summary` — it's what gets retrieved first by `memory_session_brief`.

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
