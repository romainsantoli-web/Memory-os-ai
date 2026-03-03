# Memory OS AI

> Adaptive memory system for AI agents — universal MCP server for **Claude Code, Codex CLI, VS Code Copilot, ChatGPT**, and any MCP-compatible client.

[![Tests](https://img.shields.io/badge/tests-348%20passed-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![MCP](https://img.shields.io/badge/MCP-2025--11--25-purple)]()
[![VS Code](https://img.shields.io/badge/VS%20Code-MCP%20Ready-007ACC?logo=visualstudiocode)]()
[![Version](https://img.shields.io/badge/version-3.0.0-orange)]()
[![License](https://img.shields.io/badge/license-LGPL--3.0-orange)]()

## Concept

Memory OS AI transforms your local documents (PDF, DOCX, images, audio) into a **semantic memory** queryable by any AI model through the **MCP** (Model Context Protocol).

```
┌──────────────────────────────────┐
│  AI Client (any MCP-compatible)  │
│  Claude Code / Codex / Copilot   │
│  ChatGPT / custom agents         │
├──────────────────────────────────┤
│         MCP Protocol             │
│   stdio / SSE / Streamable HTTP  │
├──────────────────────────────────┤
│      Memory OS AI Server         │
│  ┌────────┐  ┌───────────────┐   │
│  │ FAISS  │  │ Chat Extractor│   │
│  │ Index  │  │ (4 sources)   │   │
│  └────────┘  └───────────────┘   │
│  ┌────────────────────────────┐  │
│  │ Cross-Project Linking      │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
```

## Features

- **18 MCP tools** for memory management, search, chat persistence, and project linking
- **Semantic search** with FAISS + SentenceTransformers (all-MiniLM-L6-v2)
- **Multi-format ingestion**: PDF, DOCX, TXT, images (OCR), audio (Whisper), PPTX
- **Chat extraction**: auto-detects Claude, ChatGPT, Copilot, and terminal history
- **Cross-project linking**: share memory across multiple workspaces
- **3 transports**: stdio (default), SSE (`--sse`), Streamable HTTP (`--http`)
- **MCP Resources**: `memory://documents/*`, `memory://logs/conversation`, `memory://linked/*`
- **100% local**: all data stays on your machine — no cloud dependency

## 18 MCP Tools

| Tool | Description |
|------|-------------|
| `memory_ingest` | Index a folder of documents into FAISS |
| `memory_search` | Semantic search across all indexed content |
| `memory_search_occurrences` | Count keyword occurrences across documents |
| `memory_get_context` | Get relevant context for the current task |
| `memory_list_documents` | List all indexed documents with stats |
| `memory_transcribe` | Transcribe audio files (Whisper) |
| `memory_status` | Engine status (index size, model, device) |
| `memory_compact` | Compact/deduplicate the FAISS index |
| `memory_chat_sync` | Sync messages from configured chat sources |
| `memory_chat_source_add` | Add a chat source (Claude, ChatGPT, etc.) |
| `memory_chat_source_remove` | Remove a chat source |
| `memory_chat_status` | Status of all chat sources |
| `memory_chat_auto_detect` | Auto-detect chat workspaces on disk |
| `memory_session_brief` | Full memory briefing for session start |
| `memory_chat_save` | Persist conversation messages to memory |
| `memory_project_link` | Link another project's memory |
| `memory_project_unlink` | Unlink a project |
| `memory_project_list` | List all linked projects |

## Quick Start

### Prerequisites

- Python 3.10+
- Optional: `tesseract` (OCR), `ffmpeg` (audio), `antiword` (legacy .doc)

```bash
# macOS
brew install tesseract ffmpeg antiword

# Ubuntu/Debian
sudo apt-get install tesseract-ocr ffmpeg antiword
```

### Install

```bash
git clone https://github.com/romainsantoli-web/Memory-os-ai.git
cd Memory-os-ai
pip install -e ".[dev,audio]"
```

### Auto-Setup (recommended)

```bash
# Setup for your AI client:
memory-os-ai setup claude-code    # Claude Code
memory-os-ai setup codex          # Codex CLI
memory-os-ai setup vscode         # VS Code Copilot
memory-os-ai setup claude-desktop # Claude Desktop
memory-os-ai setup chatgpt        # ChatGPT (manual bridge)
memory-os-ai setup all            # All of the above

# Check status:
memory-os-ai setup status
```

### Manual Start

```bash
# stdio (default — Claude Code, VS Code, Codex)
memory-os-ai

# SSE transport (port 8765)
memory-os-ai --sse

# Streamable HTTP (port 8765)
memory-os-ai --http
```

## Project Structure

```
Memory-os-ai/
├── src/memory_os_ai/
│   ├── __init__.py          # Public API: MemoryEngine, ChatExtractor, TOOL_MODELS
│   ├── __main__.py          # python -m memory_os_ai entry point
│   ├── server.py            # MCP server — 18 tools, 3 transports, resources
│   ├── engine.py            # FAISS engine — indexing, search, compact, session brief
│   ├── models.py            # 18 Pydantic models + TOOL_MODELS registry
│   ├── chat_extractor.py    # 4 extractors: Claude, ChatGPT, Copilot, terminal
│   ├── instructions.py      # MEMORY_INSTRUCTIONS for AI clients
│   └── setup.py             # Auto-setup CLI for 5 AI clients
├── bridges/
│   ├── claude-code/         # CLAUDE.md with memory rules
│   ├── claude-desktop/      # config.json for Claude Desktop
│   ├── codex/               # AGENTS.md for Codex CLI
│   ├── vscode/              # mcp.json for VS Code
│   └── chatgpt/             # mcp-connection.json for ChatGPT
├── tests/                   # 348 tests — 96% coverage
│   ├── test_memory.py       # Engine + models (60 tests)
│   ├── test_chat_extractor.py  # Chat extraction (39 tests)
│   ├── test_bridges.py      # Bridge configs (22 tests)
│   ├── test_gaps.py         # Compact, cross-project, resources (34 tests)
│   ├── test_server_dispatch.py # Server dispatch + async (61 tests)
│   ├── test_setup.py        # Setup CLI targets
│   ├── test_z_coverage_boost.py # Coverage boost (35 tests)
│   └── test_zz_full_coverage.py # Full coverage (97 tests)
├── pyproject.toml           # v3.0.0 — deps, scripts, coverage config
├── Dockerfile               # Container deployment
└── README.md
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_CACHE_DIR` | `~/.memory-os-ai` | Cache / FAISS index directory |
| `MEMORY_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `MEMORY_API_KEY` | *(none)* | Optional API key for SSE/HTTP auth |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=memory_os_ai --cov-report=term-missing

# Coverage threshold: 80% (enforced in pyproject.toml)
```

## License

GNU Lesser General Public License v3.0 (LGPL-3.0). See [LICENSE](LICENSE) for details.

For commercial licensing, contact romainsantoli@gmail.com.

## Part of the OpenClaw Ecosystem

Memory OS AI is designed to work alongside the **OpenClaw** agent infrastructure:

| Repo | Description |
|------|-------------|
| [**setup-vs-agent-firm**](https://github.com/romainsantoli-web/setup-vs-agent-firm) | Factory for AI agent firms — 28 SKILL.md, 5 SOUL.md, 15 sectors |
| [**mcp-openclaw-extensions**](https://github.com/romainsantoli-web/setup-vs-agent-firm/tree/main/mcp-openclaw-extensions) | 115 MCP tools — security audit, A2A bridge, fleet management |
| **Memory OS AI** *(this repo)* | Semantic memory + chat persistence — universal MCP bridge |

Together they form a complete stack: **memory** (this repo) → **skills & souls** (setup-vs-agent-firm) → **security & orchestration** (mcp-openclaw-extensions).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

⚠️ Contenu généré par IA — validation humaine requise avant utilisation.
