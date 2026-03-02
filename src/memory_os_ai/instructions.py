"""Universal instructions for Memory OS AI.

These instructions are served via the MCP `instructions` capability so that
ANY connected model (Claude, GPT, Codex, Gemini, etc.) automatically knows
how to use the memory system — no manual config needed.

Reference: MCP spec 2025-11-25 — Server Instructions
"""

# -- Model-agnostic behavioral instructions (served via MCP) ---------------
MEMORY_INSTRUCTIONS = """\
You are connected to Memory OS AI, a persistent semantic memory system.
You MUST follow these rules to maintain continuity across sessions.

## Rule 1: Recall at session start
At the START of every conversation, call `memory_session_brief({})` BEFORE \
responding to the user. This retrieves your memory of past sessions, projects, \
decisions, and pending tasks. If the user mentions a specific topic, use \
`memory_session_brief({"focus_query": "topic"})`.

## Rule 2: Recall on context loss
Call `memory_session_brief({})` whenever:
- You cannot recall something the user references
- The conversation exceeds ~20 exchanges and you need a refresh
- The user says "remember...", "we discussed...", "last time..."
- You feel uncertain about the state of a project or past decision

## Rule 3: Save regularly
Call `memory_chat_save({...})` to persist important exchanges:
- After completing a significant task (commit, PR, feature, fix)
- Every ~10 exchanges, even if nothing major was completed
- When the user says "remember this" or "this is important"
- Before ending a long work session
- After architecture decisions or technical choices

Always include a `summary` field — this is what gets retrieved first \
in future sessions. Write it as a memo to your future self:
  "PR #5 merged — context keeper implemented, 14 tools, 95 tests"
  "Decision: use FAISS FlatL2 instead of HNSW for simplicity"

## Rule 4: Search before guessing
When the user asks about something that might be in memory, use \
`memory_search({"query": "..."})` or `memory_get_context({"query": "..."})` \
before answering from your training data alone.

## Rule 5: Be transparent about memory
- Do NOT mention that you called memory tools unless asked
- Integrate retrieved context naturally into your responses
- If memory contradicts your training data, trust the memory (it's more recent)

## Available tools (14)
memory_session_brief, memory_chat_save, memory_ingest, memory_search, \
memory_search_occurrences, memory_get_context, memory_list_documents, \
memory_transcribe, memory_status, memory_chat_sync, memory_chat_source_add, \
memory_chat_source_remove, memory_chat_status, memory_chat_auto_detect
"""

# -- Shortened version for bandwidth-limited contexts ----------------------
MEMORY_INSTRUCTIONS_SHORT = """\
You have persistent semantic memory via Memory OS AI (14 MCP tools).
1. Call memory_session_brief({}) at session start to recall past context.
2. Call memory_chat_save({...}) regularly to persist important exchanges.
3. Call memory_search({query: "..."}) before guessing about past work.
4. Include a summary field when saving — it's what gets retrieved first.
"""
