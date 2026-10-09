"""Letta-style core memory blocks: named, char-limited, always-in-context text
sections (``user_profile``, ``project_state``, ``working_rules`` by default),
append-only persisted with "last wins" semantics."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("memory_os_v4")

DEFAULT_BLOCKS = ("user_profile", "project_state", "working_rules")


@dataclass
class Block:
    name: str
    text: str
    updated_at: float


class BlockStore:
    """Core blocks backed by an append-only ``blocks.jsonl``."""

    def __init__(self, path: Path, char_limit: int = 2000) -> None:
        self.path = Path(path)
        self.char_limit = char_limit
        self._blocks: dict[str, Block] = {}
        self._history: dict[str, list[Block]] = defaultdict(list)
        self.load()

    def load(self) -> None:
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        block = Block(rec["name"], rec["text"], rec["updated_at"])
                    except (json.JSONDecodeError, KeyError):
                        continue
                    self._history[block.name].append(block)
                    self._blocks[block.name] = block
        for name in DEFAULT_BLOCKS:
            self._blocks.setdefault(name, Block(name, "", time.time()))

    def get(self, name: str) -> str:
        block = self._blocks.get(name)
        return block.text if block else ""

    def set(self, name: str, text: str) -> None:
        if len(text) > self.char_limit:
            logger.warning(
                "block %r exceeds char limit (%d > %d); truncating", name, len(text), self.char_limit
            )
            text = text[: self.char_limit]
        block = Block(name, text, time.time())
        self._blocks[name] = block
        self._history[name].append(block)
        self._append(block)

    def append(self, name: str, text: str) -> None:
        current = self.get(name)
        combined = f"{current}\n{text}".strip() if current else text
        self.set(name, combined)

    def history(self, name: str) -> list[Block]:
        return list(self._history.get(name, []))

    def all(self) -> dict[str, str]:
        return {name: block.text for name, block in self._blocks.items()}

    def _append(self, block: Block) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"name": block.name, "text": block.text, "updated_at": block.updated_at},
                    ensure_ascii=False,
                )
                + "\n"
            )
