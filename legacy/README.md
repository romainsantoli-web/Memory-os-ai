# Legacy — Memory OS AI v1

These files are the original Memory OS AI implementation using a local **Mistral-7B GGUF** model
via `llama-cpp-python`. They are preserved for reference only.

> **Do not use these files directly.** The active implementation is in `src/memory_os_ai/`
> and runs as an MCP server for VS Code Copilot (see `README_v2.md`).

## Files

| File | Description |
|------|-------------|
| `memory_os_ai.py` | CLI app — FAISS + SentenceTransformers + Mistral-7B (4 Go GGUF) |
| `memory_os_ai_gui.py` | PySide2 desktop GUI wrapping `memory_os_ai.py` |
| `convert_to_sentence_models.py` | Utility to convert embeddings to SentenceTransformer format |

## Dependencies (not installed in v2)

- `llama-cpp-python` — Mistral-7B GGUF inference
- `PySide2` — Desktop GUI
- `textract` — Legacy document extraction
