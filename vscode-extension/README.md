# Memory OS AI — VS Code Extension

VS Code extension that auto-configures Memory OS AI as an MCP server for Copilot.

## Features

- **Auto-setup**: Writes `.vscode/mcp.json` on workspace open
- **Status**: Check memory engine status from command palette
- **Search**: Query your memory from VS Code
- **Ingest**: Index folders into memory via UI

## Development

```bash
cd vscode-extension
npm install
npm run compile
```

Press `F5` in VS Code to launch the Extension Development Host.

## Publish

```bash
npm install -g @vscode/vsce
vsce package
vsce publish
```

## Prerequisites

Requires `memory-os-ai` Python package installed:

```bash
pip install memory-os-ai
```
