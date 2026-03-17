---
title: "Give Your AI Agent Infinite Memory in 5 Minutes"
published: false
description: "How to add persistent semantic memory to Claude Code, Copilot, Codex, or ChatGPT using Memory OS AI — an open-source MCP server."
tags: ["mcp", "ai", "memory", "claude", "copilot"]
canonical_url: null
cover_image: null
---

# Give Your AI Agent Infinite Memory in 5 Minutes

Every AI coding assistant forgets everything between sessions. You explain your architecture on Monday, and by Tuesday it's gone. You paste the same context, re-explain the same decisions, re-describe the same conventions — over and over.

**Memory OS AI** fixes this. It's an open-source MCP server that gives any AI agent persistent, semantic memory — working with Claude Code, VS Code Copilot, Codex CLI, ChatGPT, and any MCP-compatible client.

## The Problem: AI Amnesia

Here's what happens without persistent memory:

```
Session 1: "Our API uses snake_case, PostgreSQL, and domain-driven design"
Session 2: "What conventions do we use?" → "I don't have context about your project"
Session 3: *pastes 500 lines of context again*
```

Your AI agent is powerful but stateless. It processes millions of tokens per session, then forgets everything. You become the memory — copying, pasting, re-explaining.

## The Solution: 18 MCP Tools, Zero Cloud

Memory OS AI runs **100% locally**. No API calls, no cloud storage, no data leaving your machine.

It provides 18 MCP tools that your AI agent uses automatically:

| Category | Tools | What they do |
|----------|-------|-------------|
| **Session** | `memory_session_brief`, `memory_chat_save` | Auto-recall at start, auto-save during work |
| **Search** | `memory_search`, `memory_get_context`, `memory_search_occurrences` | Semantic search across all indexed content |
| **Ingest** | `memory_ingest`, `memory_transcribe` | Index PDFs, DOCX, code, audio, images |
| **Chat** | `memory_chat_sync`, `memory_chat_source_add`, `memory_chat_auto_detect` | Extract and index conversations from AI clients |
| **Cross-project** | `memory_project_link`, `memory_project_unlink`, `memory_project_list` | Share memory across repositories |
| **Maintenance** | `memory_compact`, `memory_status`, `memory_list_documents` | Deduplicate, monitor, manage |

Under the hood: **FAISS** for vector search + **SentenceTransformers** for embeddings + **JSONL** for conversation persistence.

## Setup: One Command

### Install

```bash
pip install memory-os-ai
```

### Auto-configure for your AI client

```bash
# Pick your client:
memory-os-ai setup claude-code     # Claude Code
memory-os-ai setup vscode          # VS Code Copilot
memory-os-ai setup codex           # Codex CLI
memory-os-ai setup claude-desktop  # Claude Desktop
memory-os-ai setup all             # All of the above
```

That's it. The setup command writes the correct MCP config for your client and injects memory instructions so the agent knows how to use it.

### Or configure manually

**VS Code** (`.vscode/mcp.json`):
```json
{
  "servers": {
    "memory-os-ai": {
      "command": "memory-os-ai",
      "args": [],
      "env": {
        "MEMORY_CACHE_DIR": "${workspaceFolder}/.memory"
      }
    }
  }
}
```

**Claude Code** (`.claude/mcp.json`):
```json
{
  "mcpServers": {
    "memory-os-ai": {
      "command": "memory-os-ai",
      "args": []
    }
  }
}
```

**Docker** (for remote/team setups):
```bash
docker run -d -p 8765:8765 -v memory-data:/data roamino/memory-os-ai:latest
```

## How It Works

### 1. Session Start — Automatic Recall

When your AI agent starts a session, it calls `memory_session_brief`. This returns:
- The last 5 conversation summaries
- Key decisions and architecture notes
- Active tasks and their status
- Relevant context from indexed documents

The agent gets full context in ~200ms, without you doing anything.

### 2. During Work — Automatic Save

After significant actions (commits, PRs, architecture decisions), the agent calls `memory_chat_save` with a summary. This persists the conversation to JSONL and indexes key content in FAISS.

### 3. Search — Semantic, Not Keyword

When the agent needs context, `memory_search` uses FAISS vector similarity (not grep):

```
Query: "how do we handle authentication?"
→ Finds: discussion about JWT tokens from 3 weeks ago
→ Finds: auth middleware implementation notes
→ Finds: decision to use OAuth2 with PKCE
```

### 4. Cross-Project — Shared Knowledge

Link projects together so memory spans your entire codebase:

```
memory_project_link({"target_path": "/path/to/other/project"})
```

Now searches in one project surface relevant context from linked projects.

### 5. Multi-Format Ingestion

Feed it anything:

```
memory_ingest({"path": "./docs"})        # PDFs, DOCX, TXT, MD
memory_ingest({"path": "./specs.pdf"})    # Technical specs
memory_transcribe({"path": "./meeting.mp3"})  # Audio → text → memory
```

## Architecture

```
┌──────────────┐     MCP (stdio/SSE/HTTP)     ┌──────────────────┐
│  AI Agent    │ ◄──────────────────────────► │  Memory OS AI    │
│  (Claude,    │     18 tools                  │                  │
│   Copilot,   │                               │  ┌────────────┐ │
│   Codex...)  │                               │  │ FAISS Index│ │
└──────────────┘                               │  │ (vectors)  │ │
                                               │  └────────────┘ │
                                               │  ┌────────────┐ │
                                               │  │ JSONL Store│ │
                                               │  │ (messages) │ │
                                               │  └────────────┘ │
                                               │  ┌────────────┐ │
                                               │  │ Sentence   │ │
                                               │  │ Transformer│ │
                                               │  └────────────┘ │
                                               └──────────────────┘
```

- **FAISS** (Facebook AI Similarity Search): Efficient vector index, runs on CPU or GPU
- **SentenceTransformers** (`all-MiniLM-L6-v2`): 384-dim embeddings, fast inference
- **JSONL**: Append-only conversation store, human-readable, git-friendly
- **3 transports**: stdio (local), SSE (streaming), Streamable HTTP (remote)

Everything runs locally. The embedding model downloads once (~80MB) and caches permanently.

## Real-World Impact

Before Memory OS AI:
- ⏱️ 5-10 min context setup per session
- 📋 Maintaining context docs manually
- 🔄 Re-explaining decisions every session
- 😤 "We already discussed this..."

After Memory OS AI:
- ⚡ 200ms context recall
- 🧠 Agent remembers everything automatically
- 🔗 Cross-project context sharing
- 📊 348 tests, 96% coverage, production-ready

## Install & Links

```bash
# PyPI
pip install memory-os-ai

# Docker
docker pull roamino/memory-os-ai:3.0.1

# Source
git clone https://github.com/romainsantoli-web/Memory-os-ai.git
```

| Channel | Link |
|---------|------|
| **PyPI** | [pypi.org/project/memory-os-ai](https://pypi.org/project/memory-os-ai/) |
| **Docker Hub** | [hub.docker.com/r/roamino/memory-os-ai](https://hub.docker.com/r/roamino/memory-os-ai) |
| **MCP Registry** | `io.github.romainsantoli-web/memory-os-ai` |
| **GitHub** | [github.com/romainsantoli-web/Memory-os-ai](https://github.com/romainsantoli-web/Memory-os-ai) |

---

*Memory OS AI is open-source (LGPL-3.0) and part of the [firm-ecosystem](https://github.com/romainsantoli-web/setup-vs-agent-firm) — a complete stack for AI agent firms.*

⚠️ Contenu généré par IA — validation humaine requise avant utilisation.
