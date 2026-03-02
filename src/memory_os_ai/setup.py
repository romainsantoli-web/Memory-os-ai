"""Auto-setup script for Memory OS AI bridges.

Generates ready-to-use configuration files for any supported model/client.

Usage:
    python -m memory_os_ai.setup claude-desktop   # Configure Claude Desktop
    python -m memory_os_ai.setup claude-code       # Configure Claude Code
    python -m memory_os_ai.setup codex             # Configure Codex CLI
    python -m memory_os_ai.setup vscode            # Configure VS Code Copilot
    python -m memory_os_ai.setup chatgpt           # Show ChatGPT instructions
    python -m memory_os_ai.setup all               # Configure all local clients
    python -m memory_os_ai.setup status            # Check all configurations
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_INSTALL_DIR = str(Path(__file__).resolve().parent.parent.parent)  # repo root
_HOME = str(Path.home())
_SRC_DIR = os.path.join(_INSTALL_DIR, "src")


def _get_python() -> str:
    """Find the best python executable."""
    venv_python = os.path.join(_INSTALL_DIR, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def _ensure_cache_dir() -> str:
    """Create ~/.memory-os-ai cache directory."""
    cache = os.path.join(_HOME, ".memory-os-ai")
    os.makedirs(cache, exist_ok=True)
    return cache


def _make_config(python_path: str | None = None) -> dict:
    """Build the base MCP server config dict."""
    py = python_path or _get_python()
    return {
        "command": py,
        "args": ["-m", "memory_os_ai.server"],
        "env": {
            "PYTHONPATH": _SRC_DIR,
            "MEMORY_WORKSPACE": _HOME,
            "MEMORY_CACHE_DIR": os.path.join(_HOME, ".memory-os-ai"),
            "MEMORY_MODEL": "all-MiniLM-L6-v2",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }


# ---------------------------------------------------------------------------
# Claude Desktop
# ---------------------------------------------------------------------------
def _claude_desktop_config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif system == "Linux":
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    elif system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    raise RuntimeError(f"Unsupported platform: {system}")


def setup_claude_desktop() -> str:
    _ensure_cache_dir()
    config_path = _claude_desktop_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    existing = {}
    if config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)

    servers = existing.get("mcpServers", {})
    cfg = _make_config()
    servers["memory-os-ai"] = cfg
    existing["mcpServers"] = servers

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)

    return f"✅ Claude Desktop configured: {config_path}"


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------
def setup_claude_code(project_dir: str | None = None) -> str:
    _ensure_cache_dir()
    target = Path(project_dir or os.getcwd())
    mcp_file = target / ".mcp.json"

    existing = {}
    if mcp_file.exists():
        with open(mcp_file) as f:
            existing = json.load(f)

    servers = existing.get("mcpServers", {})
    cfg = _make_config()
    servers["memory-os-ai"] = cfg
    existing["mcpServers"] = servers

    with open(mcp_file, "w") as f:
        json.dump(existing, f, indent=2)

    # Also copy CLAUDE.md if not present
    claude_md = target / "CLAUDE.md"
    source_claude_md = Path(_INSTALL_DIR) / "bridges" / "claude-code" / "CLAUDE.md"
    if not claude_md.exists() and source_claude_md.exists():
        shutil.copy2(source_claude_md, claude_md)

    return f"✅ Claude Code configured: {mcp_file}"


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------
def setup_codex(global_config: bool = True) -> str:
    _ensure_cache_dir()

    if global_config:
        target = Path.home() / ".codex"
    else:
        target = Path(os.getcwd()) / ".codex"

    target.mkdir(parents=True, exist_ok=True)
    mcp_file = target / "mcp.json"

    existing = {}
    if mcp_file.exists():
        with open(mcp_file) as f:
            existing = json.load(f)

    servers = existing.get("mcpServers", {})
    cfg = _make_config()
    servers["memory-os-ai"] = cfg
    existing["mcpServers"] = servers

    with open(mcp_file, "w") as f:
        json.dump(existing, f, indent=2)

    return f"✅ Codex CLI configured: {mcp_file}"


# ---------------------------------------------------------------------------
# VS Code Copilot
# ---------------------------------------------------------------------------
def setup_vscode(project_dir: str | None = None) -> str:
    _ensure_cache_dir()
    target = Path(project_dir or os.getcwd())
    vscode_dir = target / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    mcp_file = vscode_dir / "mcp.json"

    existing = {}
    if mcp_file.exists():
        with open(mcp_file) as f:
            existing = json.load(f)

    servers = existing.get("servers", {})
    py = _get_python()
    servers["memory-os-ai"] = {
        "type": "stdio",
        "command": py,
        "args": ["-m", "memory_os_ai.server"],
        "cwd": _SRC_DIR,
        "env": {
            "PYTHONPATH": _SRC_DIR,
            "MEMORY_WORKSPACE": "${workspaceFolder}",
            "MEMORY_MODEL": "all-MiniLM-L6-v2",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }
    existing["servers"] = servers

    with open(mcp_file, "w") as f:
        json.dump(existing, f, indent=2)

    # Copy copilot-instructions.md if not present
    github_dir = target / ".github"
    github_dir.mkdir(parents=True, exist_ok=True)
    instructions_file = github_dir / "copilot-instructions.md"
    source_instructions = Path(_INSTALL_DIR) / ".github" / "copilot-instructions.md"
    if not instructions_file.exists() and source_instructions.exists():
        shutil.copy2(source_instructions, instructions_file)

    return f"✅ VS Code Copilot configured: {mcp_file}"


# ---------------------------------------------------------------------------
# ChatGPT (instructions only — needs remote server)
# ---------------------------------------------------------------------------
def setup_chatgpt() -> str:
    return """
✅ ChatGPT setup instructions:

1. Start the SSE server:
   MEMORY_API_KEY=your-secret python -m memory_os_ai.server --sse

2. Expose it (e.g., via ngrok):
   ngrok http 8765

3. In ChatGPT Settings → Connections → Add MCP Server:
   URL: https://your-ngrok-url/sse
   Auth: Bearer your-secret

See bridges/chatgpt/README.md for Docker and cloud deployment options.
"""


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------
def check_status() -> str:
    lines = ["Memory OS AI — Bridge Status\n"]

    # Claude Desktop
    try:
        path = _claude_desktop_config_path()
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            if "memory-os-ai" in cfg.get("mcpServers", {}):
                lines.append(f"  ✅ Claude Desktop  — {path}")
            else:
                lines.append(f"  ⚠️  Claude Desktop  — config exists but memory-os-ai not found")
        else:
            lines.append(f"  ❌ Claude Desktop  — not configured")
    except Exception:
        lines.append(f"  ❌ Claude Desktop  — not configured")

    # Codex
    codex_path = Path.home() / ".codex" / "mcp.json"
    if codex_path.exists():
        with open(codex_path) as f:
            cfg = json.load(f)
        if "memory-os-ai" in cfg.get("mcpServers", {}):
            lines.append(f"  ✅ Codex CLI       — {codex_path}")
        else:
            lines.append(f"  ⚠️  Codex CLI       — config exists but memory-os-ai not found")
    else:
        lines.append(f"  ❌ Codex CLI       — not configured")

    # VS Code (current project)
    vscode_path = Path(os.getcwd()) / ".vscode" / "mcp.json"
    if vscode_path.exists():
        lines.append(f"  ✅ VS Code Copilot — {vscode_path}")
    else:
        lines.append(f"  ❌ VS Code Copilot — not configured in current project")

    # Claude Code (current project)
    mcp_path = Path(os.getcwd()) / ".mcp.json"
    if mcp_path.exists():
        lines.append(f"  ✅ Claude Code     — {mcp_path}")
    else:
        lines.append(f"  ❌ Claude Code     — not configured in current project")

    # Cache dir
    cache = Path.home() / ".memory-os-ai"
    if cache.exists():
        files = list(cache.iterdir())
        lines.append(f"\n  📁 Cache: {cache} ({len(files)} files)")
    else:
        lines.append(f"\n  📁 Cache: not created yet")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
TARGETS = {
    "claude-desktop": setup_claude_desktop,
    "claude-code": setup_claude_code,
    "codex": setup_codex,
    "vscode": setup_vscode,
    "chatgpt": setup_chatgpt,
    "status": check_status,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m memory_os_ai.setup <target>")
        print(f"\nTargets: {', '.join(TARGETS.keys())}, all")
        print("\nExamples:")
        print("  python -m memory_os_ai.setup claude-desktop  # Configure Claude Desktop")
        print("  python -m memory_os_ai.setup all              # Configure all local clients")
        print("  python -m memory_os_ai.setup status           # Check configuration status")
        sys.exit(0)

    target = sys.argv[1]

    if target == "all":
        results = []
        for name in ["claude-desktop", "claude-code", "codex", "vscode"]:
            try:
                result = TARGETS[name]()
                results.append(result)
            except Exception as e:
                results.append(f"❌ {name}: {e}")
        results.append(setup_chatgpt())
        print("\n".join(results))
    elif target in TARGETS:
        print(TARGETS[target]())
    else:
        print(f"Unknown target: {target}")
        print(f"Available: {', '.join(TARGETS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
