"""Tests for setup.py — auto-setup CLI for all client bridges."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch


from memory_os_ai.setup import (
    _ensure_cache_dir,
    _get_python,
    _make_config,
    setup_claude_code,
    setup_codex,
    setup_vscode,
    setup_chatgpt,
    check_status,
    TARGETS,
    _claude_desktop_config_path,
    setup_claude_desktop,
)


class TestHelpers:
    """Test setup utility functions."""

    def test_get_python_returns_string(self):
        result = _get_python()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ensure_cache_dir_creates(self, tmp_path):
        tmp_path / ".memory-os-ai"
        with patch("memory_os_ai.setup._HOME", str(tmp_path)):
            result = _ensure_cache_dir()
        assert os.path.isdir(result)

    def test_make_config_structure(self):
        cfg = _make_config()
        assert "command" in cfg
        assert "args" in cfg
        assert "env" in cfg
        assert "-m" in cfg["args"]
        assert "memory_os_ai.server" in cfg["args"]
        assert "PYTHONPATH" in cfg["env"]
        assert "MEMORY_CACHE_DIR" in cfg["env"]

    def test_make_config_custom_python(self):
        cfg = _make_config(python_path="/usr/bin/python3")
        assert cfg["command"] == "/usr/bin/python3"


class TestClaudeDesktop:
    """Test Claude Desktop setup."""

    def test_config_path_exists(self):
        path = _claude_desktop_config_path()
        assert isinstance(path, Path)

    def test_setup_creates_config(self, tmp_path):
        config_path = tmp_path / "claude_desktop_config.json"
        with patch("memory_os_ai.setup._claude_desktop_config_path", return_value=config_path):
            with patch("memory_os_ai.setup._HOME", str(tmp_path)):
                result = setup_claude_desktop()
        assert "✅" in result
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "memory-os-ai" in data["mcpServers"]

    def test_setup_preserves_existing(self, tmp_path):
        config_path = tmp_path / "claude_desktop_config.json"
        # Pre-existing config
        config_path.write_text(json.dumps({
            "mcpServers": {"other-server": {"command": "other"}}
        }))
        with patch("memory_os_ai.setup._claude_desktop_config_path", return_value=config_path):
            with patch("memory_os_ai.setup._HOME", str(tmp_path)):
                setup_claude_desktop()
        data = json.loads(config_path.read_text())
        assert "other-server" in data["mcpServers"]
        assert "memory-os-ai" in data["mcpServers"]


class TestClaudeCode:
    """Test Claude Code setup."""

    def test_setup_creates_mcp_json(self, tmp_path):
        with patch("memory_os_ai.setup._HOME", str(tmp_path)):
            result = setup_claude_code(project_dir=str(tmp_path))
        assert "✅" in result
        mcp_file = tmp_path / ".mcp.json"
        assert mcp_file.exists()
        data = json.loads(mcp_file.read_text())
        assert "memory-os-ai" in data["mcpServers"]

    def test_setup_preserves_existing(self, tmp_path):
        mcp_file = tmp_path / ".mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {"existing": {}}}))
        with patch("memory_os_ai.setup._HOME", str(tmp_path)):
            setup_claude_code(project_dir=str(tmp_path))
        data = json.loads(mcp_file.read_text())
        assert "existing" in data["mcpServers"]
        assert "memory-os-ai" in data["mcpServers"]


class TestCodex:
    """Test Codex CLI setup."""

    def test_setup_global(self, tmp_path):
        with patch("memory_os_ai.setup._HOME", str(tmp_path)):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result = setup_codex(global_config=True)
        assert "✅" in result
        codex_dir = tmp_path / ".codex"
        assert codex_dir.exists()
        mcp_file = codex_dir / "mcp.json"
        assert mcp_file.exists()

    def test_setup_local(self, tmp_path):
        with patch("memory_os_ai.setup._HOME", str(tmp_path)):
            with patch("os.getcwd", return_value=str(tmp_path)):
                result = setup_codex(global_config=False)
        assert "✅" in result


class TestVSCode:
    """Test VS Code Copilot setup."""

    def test_setup_creates_vscode_dir(self, tmp_path):
        with patch("memory_os_ai.setup._HOME", str(tmp_path)):
            result = setup_vscode(project_dir=str(tmp_path))
        assert "✅" in result
        mcp_file = tmp_path / ".vscode" / "mcp.json"
        assert mcp_file.exists()
        data = json.loads(mcp_file.read_text())
        assert "memory-os-ai" in data["servers"]

    def test_setup_preserves_existing(self, tmp_path):
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        mcp_file = vscode_dir / "mcp.json"
        mcp_file.write_text(json.dumps({"servers": {"other": {}}}))
        with patch("memory_os_ai.setup._HOME", str(tmp_path)):
            setup_vscode(project_dir=str(tmp_path))
        data = json.loads(mcp_file.read_text())
        assert "other" in data["servers"]
        assert "memory-os-ai" in data["servers"]


class TestChatGPT:
    """Test ChatGPT setup."""

    def test_returns_instructions(self):
        result = setup_chatgpt()
        assert "ChatGPT" in result
        assert "MEMORY_API_KEY" in result
        assert "--sse" in result


class TestCheckStatus:
    """Test status check."""

    def test_status_returns_string(self):
        result = check_status()
        assert isinstance(result, str)
        assert "Bridge Status" in result

    def test_status_mentions_all_clients(self, tmp_path):
        with patch("memory_os_ai.setup._claude_desktop_config_path",
                    return_value=tmp_path / "nonexistent.json"):
            result = check_status()
        # Should mention all clients even if not configured
        assert "Claude Desktop" in result
        assert "Codex CLI" in result or "Codex" in result
        assert "VS Code" in result


class TestTargets:
    """Test TARGETS registry."""

    def test_all_targets_present(self):
        expected = {"claude-desktop", "claude-code", "codex", "vscode", "chatgpt", "status"}
        assert set(TARGETS.keys()) == expected

    def test_all_targets_callable(self):
        for name, func in TARGETS.items():
            assert callable(func), f"Target {name} is not callable"
