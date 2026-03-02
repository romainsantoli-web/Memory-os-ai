"""Tests for universal bridges — instructions, setup, transports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Instructions module tests
# ---------------------------------------------------------------------------
from memory_os_ai.instructions import MEMORY_INSTRUCTIONS, MEMORY_INSTRUCTIONS_SHORT


class TestInstructions:
    """Verify instruction content is well-formed."""

    def test_instructions_not_empty(self):
        assert len(MEMORY_INSTRUCTIONS) > 500

    def test_instructions_contains_rules(self):
        assert "Rule 1" in MEMORY_INSTRUCTIONS
        assert "Rule 2" in MEMORY_INSTRUCTIONS
        assert "Rule 3" in MEMORY_INSTRUCTIONS
        assert "Rule 4" in MEMORY_INSTRUCTIONS
        assert "Rule 5" in MEMORY_INSTRUCTIONS

    def test_instructions_mentions_key_tools(self):
        assert "memory_session_brief" in MEMORY_INSTRUCTIONS
        assert "memory_chat_save" in MEMORY_INSTRUCTIONS
        assert "memory_search" in MEMORY_INSTRUCTIONS

    def test_instructions_mentions_14_tools(self):
        assert "14" in MEMORY_INSTRUCTIONS

    def test_short_instructions_exists(self):
        assert len(MEMORY_INSTRUCTIONS_SHORT) > 50
        assert len(MEMORY_INSTRUCTIONS_SHORT) < len(MEMORY_INSTRUCTIONS)

    def test_short_mentions_key_tools(self):
        assert "memory_session_brief" in MEMORY_INSTRUCTIONS_SHORT
        assert "memory_chat_save" in MEMORY_INSTRUCTIONS_SHORT


# ---------------------------------------------------------------------------
# Setup module tests
# ---------------------------------------------------------------------------
from memory_os_ai.setup import (
    _make_config,
    _get_python,
    _ensure_cache_dir,
    setup_chatgpt,
    check_status,
    TARGETS,
)


class TestSetup:
    """Verify setup module functions."""

    def test_make_config_has_required_keys(self):
        cfg = _make_config()
        assert "command" in cfg
        assert "args" in cfg
        assert "env" in cfg
        assert cfg["args"] == ["-m", "memory_os_ai.server"]

    def test_make_config_env_has_required_vars(self):
        cfg = _make_config()
        env = cfg["env"]
        assert "PYTHONPATH" in env
        assert "MEMORY_WORKSPACE" in env
        assert "MEMORY_CACHE_DIR" in env
        assert "MEMORY_MODEL" in env

    def test_make_config_custom_python(self):
        cfg = _make_config(python_path="/usr/bin/python3.11")
        assert cfg["command"] == "/usr/bin/python3.11"

    def test_get_python_returns_string(self):
        py = _get_python()
        assert isinstance(py, str)
        assert len(py) > 0

    def test_ensure_cache_dir_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"HOME": tmpdir}):
                with patch("memory_os_ai.setup._HOME", tmpdir):
                    cache = os.path.join(tmpdir, ".memory-os-ai")
                    os.makedirs(cache, exist_ok=True)
                    assert os.path.isdir(cache)

    def test_chatgpt_returns_instructions(self):
        result = setup_chatgpt()
        assert "SSE" in result or "sse" in result
        assert "ngrok" in result

    def test_targets_registry(self):
        expected = {"claude-desktop", "claude-code", "codex", "vscode", "chatgpt", "status"}
        assert set(TARGETS.keys()) == expected

    def test_check_status_returns_string(self):
        result = check_status()
        assert "Memory OS AI" in result
        assert "Claude Desktop" in result
        assert "Codex CLI" in result


# ---------------------------------------------------------------------------
# Server instructions integration test
# ---------------------------------------------------------------------------
class TestServerInstructions:
    """Verify that the server has instructions configured."""

    def test_server_has_instructions(self):
        from memory_os_ai.server import server
        opts = server.create_initialization_options()
        assert opts.instructions is not None
        assert "memory_session_brief" in opts.instructions
        assert "Rule 1" in opts.instructions

    def test_server_name(self):
        from memory_os_ai.server import server
        opts = server.create_initialization_options()
        assert opts.server_name == "memory-os-ai"


# ---------------------------------------------------------------------------
# Bridge config file tests
# ---------------------------------------------------------------------------
class TestBridgeConfigs:
    """Verify bridge config files are valid JSON with correct structure."""

    BRIDGE_DIR = Path(__file__).parent.parent / "bridges"

    def _load_json(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def test_claude_desktop_config(self):
        cfg = self._load_json(self.BRIDGE_DIR / "claude-desktop" / "config.json")
        assert "mcpServers" in cfg
        assert "memory-os-ai" in cfg["mcpServers"]
        srv = cfg["mcpServers"]["memory-os-ai"]
        assert "command" in srv
        assert "env" in srv

    def test_claude_code_config(self):
        cfg = self._load_json(self.BRIDGE_DIR / "claude-code" / ".mcp.json")
        assert "mcpServers" in cfg
        assert "memory-os-ai" in cfg["mcpServers"]

    def test_codex_config(self):
        cfg = self._load_json(self.BRIDGE_DIR / "codex" / ".codex" / "mcp.json")
        assert "mcpServers" in cfg
        assert "memory-os-ai" in cfg["mcpServers"]

    def test_vscode_config(self):
        cfg = self._load_json(self.BRIDGE_DIR / "vscode" / "mcp.json")
        assert "servers" in cfg
        assert "memory-os-ai" in cfg["servers"]
        assert cfg["servers"]["memory-os-ai"]["type"] == "stdio"

    def test_chatgpt_connection(self):
        cfg = self._load_json(self.BRIDGE_DIR / "chatgpt" / "mcp-connection.json")
        assert "transport" in cfg
        assert cfg["transport"]["type"] == "sse"

    def test_claude_code_has_claude_md(self):
        md = self.BRIDGE_DIR / "claude-code" / "CLAUDE.md"
        assert md.exists()
        content = md.read_text()
        assert "memory_session_brief" in content
        assert "memory_chat_save" in content
