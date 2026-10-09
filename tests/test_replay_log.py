"""Conversation log replay + real Streamable HTTP round trip."""

import json
import os
from unittest.mock import patch

import pytest

import memory_os_ai.server as srv


def _write_log(path, records, extra_lines=()):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        for line in extra_lines:
            f.write(line + "\n")


class TestReplayConversationLog:
    def test_builds_segments_like_chat_save(self, tmp_path):
        _write_log(tmp_path / "_conversation_log.jsonl", [
            {"project": "p1", "summary": "sum one",
             "messages": [{"role": "user", "content": "hello"},
                          {"role": "assistant", "content": "world"}]},
            {"project": None, "summary": None,
             "messages": [{"role": "user", "content": "x"}, {"role": "user", "content": ""}]},
        ], extra_lines=["{broken", ""])
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}), \
             patch.object(srv._engine, "ingest_segments",
                          return_value={"added_segments": 4}) as ing:
            res = srv._replay_conversation_log()
        segs = ing.call_args[0][0]
        assert segs == ["[summary:p1] sum one", "[user] hello", "[assistant] world", "[user] x"]
        assert ing.call_args[1]["source_label"] == "replay_log"
        assert res == {"ok": True, "records": 2, "bad_lines": 1, "added_segments": 4}

    def test_missing_log(self, tmp_path):
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
            assert srv._replay_conversation_log()["records"] == 0

    def test_disabled(self, tmp_path):
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path), "MEMORY_REPLAY_LOG": "0"}):
            assert srv._replay_conversation_log()["skipped"] is True

    def test_run_survives_replay_error(self):
        with patch("memory_os_ai.server._replay_conversation_log", side_effect=RuntimeError("boom")), \
             patch("memory_os_ai.server.run_http") as mock_http, \
             patch("sys.argv", ["memory-os-ai", "--http"]):
            srv.run()
            mock_http.assert_called_once()
            assert mock_http.call_args[1]["host"] == "127.0.0.1"
