"""'*' project scope: reads span every project, writes stay on a concrete label."""

from __future__ import annotations

import pytest

import memory_os_v4.server as srv
from memory_os_v4.config import Config
from memory_os_v4.memory import MemoryV4


@pytest.fixture
def engine(tmp_path, monkeypatch):
    eng = MemoryV4(config=Config(data_dir=str(tmp_path), embed_model="hash"))
    srv.set_engine(eng)
    yield eng
    srv.set_engine(None)


def _save(project, text):
    return srv.dispatch("memory_chat_save", {
        "messages": [{"role": "user", "content": text}], "project_label": project})


def test_resolvers(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_PROJECT", "*")
    assert srv._project_of(None) is None
    assert srv._project_of("alpha") == "alpha"
    assert srv._write_project_of(None) == "default"
    assert srv._write_project_of("*") == "default"
    monkeypatch.setenv("MEMORY_V4_PROJECT", "beta")
    assert srv._project_of(None) == "beta"
    assert srv._project_of("*") is None
    assert srv._write_project_of(None) == "beta"


def test_star_reads_across_projects(engine, monkeypatch):
    monkeypatch.setenv("MEMORY_V4_PROJECT", "*")
    _save("alpha", "Le serveur VPS de Londres est chez OVH")
    _save("beta", "Le moteur Kraken Futures est a Londres chez Equinix")

    hits = srv.dispatch("memory_search", {"query": "Londres"})
    assert {h["project"] for h in hits["hits"]} == {"alpha", "beta"}

    only_alpha = srv.dispatch("memory_search", {"query": "Londres", "project": "alpha"})
    assert {h["project"] for h in only_alpha["hits"]} == {"alpha"}

    occ = srv.dispatch("memory_search_occurrences", {"keyword": "Londres"})
    assert occ["count"] == 2

    brief = srv.dispatch("memory_session_brief", {})
    assert len(brief["recent_episodes"]) == 2


def test_star_write_lands_in_default(engine, monkeypatch):
    monkeypatch.setenv("MEMORY_V4_PROJECT", "*")
    _save(None, "note sans projet")
    assert {it.project for it in engine.store.all()} == {"default"}
