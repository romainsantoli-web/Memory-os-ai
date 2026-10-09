<!-- mcp-name: io.github.romainsantoli-web/memory-os-ai -->
# Memory OS AI

> Adaptive memory system for AI agents — universal MCP server for **Claude Code, Codex CLI, VS Code Copilot, ChatGPT**, and any MCP-compatible client.

[![CI](https://github.com/romainsantoli-web/Memory-os-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/romainsantoli-web/Memory-os-ai/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![MCP](https://img.shields.io/badge/MCP-2025--11--25-purple)]()
[![VS Code](https://img.shields.io/badge/VS%20Code-MCP%20Ready-007ACC?logo=visualstudiocode)]()
[![Version](https://img.shields.io/badge/version-4.0.0-orange)]()
[![License](https://img.shields.io/badge/license-LGPL--3.0-orange)]()

## What's new in 4.0 — Memory OS v4

`pip install memory-os-ai` now ships **two engines**:

| Command | Engine | Notes |
|---|---|---|
| `memory-os-ai` / `memory-os-v4` | **v4** (`memory_os_v4`) — default | typed bi-temporal memory (episodes, facts, summaries, procedures), hybrid dense + BM25 + keyword retrieval with multi-query RRF, core blocks, consolidation, 16 MCP tools |
| `memory-os-ai-v3` | v3 (`memory_os_ai`) | the previous FAISS document server, unchanged API (21 tools) |

```bash
memory-os-ai --selfcheck                         # exercise every tool once
memory-os-ai                                     # stdio MCP server
memory-os-ai --transport http --port 8765 --warm # one shared HTTP server for all clients
```

Data lives in `MEMORY_V4_DIR` (default `~/.memory-os-v4`). Existing v3 memories can be imported with the `memory_migrate_v3` tool.
Exploratory LoCoMo evidence-retrieval Recall@10 (`v4/bench`): v3 0.175 → v4 0.538 with production defaults, 0.815 with offline LLM enrichment and reranking. This is retrieval recall on 1,536 questions after excluding adversarial/empty-evidence items; it does not measure answer accuracy or agent task success. The LongMemEval pilot uses a 150-question **oracle** sample, not the full benchmark. See the [frozen evaluation protocol](v4/bench/protocols/v1.md) before making final comparisons.
Details: [`v4/README.md`](v4/README.md), [`v4/docs/ARCHITECTURE.md`](v4/docs/ARCHITECTURE.md), [`v4/RESULTS.md`](v4/RESULTS.md).

> **Upgrading from 3.x:** the `memory-os-ai` command now starts v4. To keep v3, point your MCP config at `memory-os-ai-v3`.

## v3 document engine

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

### v3 features

- **21 MCP tools** for memory management, search, chat persistence, project linking, and cloud storage
- **Semantic search** with FAISS + SentenceTransformers (all-MiniLM-L6-v2)
- **Multi-format ingestion**: PDF, DOCX, TXT, images (OCR), audio (Whisper), PPTX
- **Chat extraction**: auto-detects Claude, ChatGPT, Copilot, and terminal history
- **Cross-project linking**: share memory across multiple workspaces
- **Cloud storage overflow**: auto-backup to Google Drive, iCloud, Dropbox, OneDrive, S3, Azure, Box, B2
- **3 transports**: stdio (default), SSE (`--sse`), Streamable HTTP (`--http`)
- **MCP Resources**: `memory://documents/*`, `memory://logs/conversation`, `memory://linked/*`
- **Local-first**: all data on your machine by default, cloud only when disk runs low

### v3 MCP tools

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
| `memory_cloud_configure` | Configure cloud storage backend for overflow |
| `memory_cloud_status` | Show local disk + cloud storage status |
| `memory_cloud_sync` | Push/pull/auto-sync between local and cloud |

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

### Configure v4 in your MCP client

Use the installed command in your client's MCP server configuration:

```json
{
  "mcpServers": {
    "memory-os-ai": {
      "command": "/absolute/path/to/venv/bin/memory-os-ai",
      "env": {"MEMORY_V4_DIR": "/absolute/path/to/memory-data"}
    }
  }
}
```

The default v4 model is `ibm-granite/granite-embedding-97m-multilingual-r2`.
The first semantic operation downloads model weights; cache them before enabling
`HF_HUB_OFFLINE=1`. `MEMORY_V4_EMBED_MODEL=hash` is a fast diagnostic option.

```bash
memory-os-ai --selfcheck
memory-os-ai --transport http --host 127.0.0.1 --port 8765 --warm
```

Keep the v4 data directory dedicated to this engine. For a shared HTTP service,
use one server process for all clients. See [v4 configuration](v4/README.md).

### Legacy v3 configuration

The setup module configures **v3** clients:

```bash
python -m memory_os_ai.setup claude-code
python -m memory_os_ai.setup codex
python -m memory_os_ai.setup vscode
python -m memory_os_ai.setup claude-desktop
python -m memory_os_ai.setup chatgpt
python -m memory_os_ai.setup status
memory-os-ai-v3
memory-os-ai-v3 --sse
memory-os-ai-v3 --http
```

## Project Structure

```
Memory-os-ai/
├── src/memory_os_ai/
│   ├── __init__.py          # Public API: MemoryEngine, ChatExtractor, TOOL_MODELS
│   ├── __main__.py          # python -m memory_os_ai entry point
│   ├── server.py            # MCP server — 21 tools, 3 transports, resources
│   ├── engine.py            # FAISS engine — indexing, search, compact, session brief
│   ├── cloud_storage.py     # 8 cloud backends (GDrive, iCloud, Dropbox, OneDrive, S3, Azure, Box, B2)
│   ├── storage_router.py    # Smart routing: local-first with cloud overflow
│   ├── models.py            # 21 Pydantic models + TOOL_MODELS registry
│   ├── chat_extractor.py    # 4 extractors: Claude, ChatGPT, Copilot, terminal
│   ├── instructions.py      # MEMORY_INSTRUCTIONS for AI clients
│   └── setup.py             # Auto-setup CLI for 5 AI clients
├── bridges/
│   ├── claude-code/         # CLAUDE.md with memory rules
│   ├── claude-desktop/      # config.json for Claude Desktop
│   ├── codex/               # AGENTS.md for Codex CLI
│   ├── vscode/              # mcp.json for VS Code
│   └── chatgpt/             # mcp-connection.json for ChatGPT
├── tests/                   # 410+ tests — 96% coverage
│   ├── test_memory.py       # Engine + models (60 tests)
│   ├── test_chat_extractor.py  # Chat extraction (39 tests)
│   ├── test_bridges.py      # Bridge configs (22 tests)
│   ├── test_gaps.py         # Compact, cross-project, resources (34 tests)
│   ├── test_server_dispatch.py # Server dispatch + async (61 tests)
│   ├── test_setup.py        # Setup CLI targets
│   ├── test_z_coverage_boost.py # Coverage boost (35 tests)
│   └── test_zz_full_coverage.py # Full coverage (97 tests)
├── pyproject.toml           # v3.1.0 — deps, scripts, coverage config + cloud optional deps
├── Dockerfile               # Container deployment
└── README.md
```

## v3 Cloud Storage (v3.1.0)

When local disk runs low (< 500 MB free by default), memory data automatically overflows to a configured cloud backend.

### Supported Providers

| Provider | Install | Credentials |
|----------|---------|-------------|
| **Google Drive** | `pip install memory-os-ai[cloud-gdrive]` | `credentials_json` or `token_json` + `folder_id` |
| **iCloud Drive** | *(macOS native, no extra deps)* | `container` name (default: `memory-os-ai`) |
| **Dropbox** | `pip install memory-os-ai[cloud-dropbox]` | `access_token` + `folder` |
| **OneDrive** | *(auto-detects mount)* or Graph API | `mount_path` or `access_token` |
| **Amazon S3** | `pip install memory-os-ai[cloud-s3]` | `bucket`, `aws_access_key_id`, `aws_secret_access_key` |
| **Azure Blob** | `pip install memory-os-ai[cloud-azure]` | `connection_string` + `container` |
| **Box** | `pip install memory-os-ai[cloud-box]` | `access_token` + `folder_id` |
| **Backblaze B2** | `pip install memory-os-ai[cloud-b2]` | `application_key_id`, `application_key`, `bucket_name` |
| **All providers** | `pip install memory-os-ai[cloud-all]` | — |

### Usage

```bash
# Configure via environment (auto-activates on server start)
export MEMORY_CLOUD_PROVIDER=icloud
export MEMORY_CLOUD_CONFIG='{"container": "memory-os-ai"}'
memory-os-ai-v3

# Or configure at runtime via MCP tool:
#   memory_cloud_configure(provider="s3", credentials={"bucket": "my-bucket", ...})
#   memory_cloud_status()       → local disk + cloud usage
#   memory_cloud_sync("push")   → backup to cloud
#   memory_cloud_sync("pull")   → restore from cloud
#   memory_cloud_sync("auto")   → offload if disk low
```

## v3 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_CACHE_DIR` | `~/.memory-os-ai` | Cache / FAISS index directory |
| `MEMORY_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `MEMORY_API_KEY` | *(none)* | Optional API key for SSE/HTTP auth |
| `MEMORY_CLOUD_PROVIDER` | *(none)* | Cloud provider name (see table above) |
| `MEMORY_CLOUD_CONFIG` | *(none)* | JSON credentials or path to JSON file |
| `MEMORY_DISK_THRESHOLD` | `524288000` | Bytes free before cloud overflow (500 MB) |

## Development and distribution qualification

Python 3.11 and 3.12 are the CI matrix; package metadata requires Python >=3.11.

```bash
python -m pip install -e ".[dev]"
ruff check src tests v4/src v4/tests v4/bench scripts --select E9,F
python -m pytest tests --cov=memory_os_ai --cov-report=xml:coverage-v3.xml
python -m pytest v4/tests -m "not slow" --cov=memory_os_v4 --cov-report=xml:coverage-v4.xml
python -m build
python -m twine check dist/*
```

CI blocks on lint, distinct v3/v4 coverage (80% minimum each), and installed
wheel/sdist MCP startup outside the source checkout. A separate CPU job downloads
and caches Granite/MiniLM, then runs real semantic retrieval and persistence
checks offline. It uses no paid API and has a 25-minute timeout. Model downloads
and CPU inference make its cold run slower than deterministic tests.
See [release qualification](docs/DISTRIBUTION.md).

## License

GNU Lesser General Public License v3 or later (`LGPL-3.0-or-later`).
See [LICENSE](LICENSE) for LGPLv3 and [LICENSE.GPL-3.0](LICENSE.GPL-3.0) for its required GPLv3 companion.
The maintainer confirmed this license choice on 2026-10-09; the former GPL-only
license file has been retained as the companion rather than discarded.

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
