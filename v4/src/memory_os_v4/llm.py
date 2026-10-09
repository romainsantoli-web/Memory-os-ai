"""Minimal OpenAI-compatible chat client (urllib, no new dependency) for optional
LLM-assisted fact extraction. Disabled unless ``MEMORY_V4_LLM_URL`` is set and a
1-second probe of ``/models`` succeeds; result is cached. Never required for
tests or core operation.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger("memory_os_v4")

_SYSTEM_PROMPT = (
    "You extract atomic, ADD-only memory facts from a conversation snippet. "
    "Return ONLY a JSON list of objects with a required 'text' field (one "
    "self-contained, temporally-phrased statement per fact) and optional "
    "'entities', 'event_time', 'importance', 'tags'. Never emit UPDATE or "
    "DELETE operations, never restate facts already present in the provided "
    "existing snippets, and never wrap the JSON in prose or markdown."
)


class LLMClient:
    """Thin OpenAI-compatible ``/chat/completions`` client."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 1.0,
    ) -> None:
        self.base_url = base_url or os.environ.get("MEMORY_V4_LLM_URL")
        self.model = model or os.environ.get("MEMORY_V4_LLM_MODEL", "local-model")
        self.timeout = timeout
        self._available_cache: bool | None = None

    def available(self) -> bool:
        """True if a base URL is configured and ``/models`` answers within 1s."""
        if self._available_cache is not None:
            return self._available_cache
        if not self.base_url:
            self._available_cache = False
            return False
        try:
            req = urllib.request.Request(self.base_url.rstrip("/") + "/models")
            with urllib.request.urlopen(req, timeout=1.0) as resp:  # noqa: S310
                self._available_cache = 200 <= resp.status < 300
        except Exception:  # noqa: BLE001 - any failure means "unavailable"
            self._available_cache = False
        return self._available_cache

    def extract_facts(self, text: str, existing_snippets: list[str] | None = None) -> list[dict]:
        """Extract atomic facts via the configured LLM; returns [] if unavailable
        or on any parsing/network error."""
        if not self.available():
            return []
        existing = existing_snippets or []
        user_prompt = (
            "Existing snippets (do not repeat):\n"
            + "\n".join(f"- {s}" for s in existing[:20])
            + f"\n\nConversation snippet:\n{text}\n\nJSON list of new facts:"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.base_url.rstrip("/") + "/chat/completions",  # type: ignore[union-attr]
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            facts = json.loads(content)
            if not isinstance(facts, list):
                return []
            return [f for f in facts if isinstance(f, dict) and f.get("text")]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("LLM fact extraction failed: %s", exc)
            return []
