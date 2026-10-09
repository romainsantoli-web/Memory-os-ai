"""Append-only core blocks with explicit global sharing and project isolation.

Legacy records without scope are assigned to 'default'; a legacy file cannot
prove that any particular block was intended to be shared with every project.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("memory_os_v4")
DEFAULT_BLOCKS = ("user_profile", "project_state", "working_rules")


@dataclass
class Block:
    name: str
    text: str
    updated_at: float
    project: str | None = "default"
    scope: str = "project"


class BlockStore:
    """Global blocks are visible explicitly; project blocks override them."""

    def __init__(self, path: Path, char_limit: int = 2000) -> None:
        self.path = Path(path)
        self.char_limit = char_limit
        self._blocks: dict[tuple[str | None, str], Block] = {}
        self._history: dict[tuple[str | None, str], list[Block]] = defaultdict(list)
        self.load()

    def load(self) -> None:
        self._blocks.clear()
        self._history.clear()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    scope = rec.get("scope", "project")
                    if scope not in {"project", "global"}:
                        continue
                    project = None if scope == "global" else rec.get("project", "default")
                    if scope == "project" and (not isinstance(project, str) or project == "*"):
                        continue
                    block = Block(rec["name"], rec["text"], rec["updated_at"], project, scope)
                    if not isinstance(block.name, str) or not isinstance(block.text, str):
                        continue
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                key = (project, block.name)
                self._history[key].append(block)
                self._blocks[key] = block

    @staticmethod
    def _target(project: str, scope: str) -> str | None:
        if scope not in {"project", "global"}:
            raise ValueError("block scope must be project or global")
        return None if scope == "global" else ("default" if project == "*" else project)

    def get(self, name: str, project: str = "default", scope: str = "project") -> str:
        target = self._target(project, scope)
        block = self._blocks.get((target, name))
        if block is None and scope == "project":
            block = self._blocks.get((None, name))
        return block.text if block else ""

    def set(self, name: str, text: str, project: str = "default", scope: str = "project") -> None:
        target = self._target(project, scope)
        if len(text) > self.char_limit:
            logger.warning("block %r exceeds char limit; truncating", name)
            text = text[:self.char_limit]
        block = Block(name, text, time.time(), target, scope)
        key = (target, name)
        self._blocks[key] = block
        self._history[key].append(block)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(block), ensure_ascii=False) + "\n")

    def append(self, name: str, text: str, project: str = "default", scope: str = "project") -> None:
        # An append edits the target's own contents, not inherited globals.
        current = self._blocks.get((self._target(project, scope), name))
        combined = f"{current.text}\n{text}".strip() if current and current.text else text
        self.set(name, combined, project=project, scope=scope)

    def history(self, name: str, project: str = "default", scope: str = "project") -> list[Block]:
        return list(self._history.get((self._target(project, scope), name), []))

    def all(self, project: str | None = "default", scope: str = "project") -> dict[str, str]:
        if scope not in {"project", "global"}:
            raise ValueError("block scope must be project or global")
        result = dict.fromkeys(DEFAULT_BLOCKS, "")
        result.update({name: block.text for (target, name), block in self._blocks.items() if target is None})
        if scope == "global":
            return result
        if project is None or project == "*":
            result.update({f"{target}/{name}": block.text for (target, name), block in self._blocks.items()
                           if target is not None})
        else:
            result.update({name: block.text for (target, name), block in self._blocks.items() if target == project})
        return result
