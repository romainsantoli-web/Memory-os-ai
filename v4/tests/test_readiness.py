"""Authenticated readiness proves runtime data/index state, not just liveness."""
from concurrent.futures import ThreadPoolExecutor
import json

import httpx
import pytest

from memory_os_v4 import Config, MemoryV4, server
from memory_os_v4.readiness import engine_readiness

TOKEN = "synthetic-ready-token-" + "x" * 32
COMMIT = "a" * 40


@pytest.fixture
def app_engine(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text(TOKEN)
    token.chmod(0o600)
    monkeypatch.setenv("MEMORY_V4_TOKEN_FILE", str(token))
    monkeypatch.setenv("MEMORY_V4_BUILD_COMMIT", COMMIT)
    with MemoryV4(config=Config(data_dir=str(tmp_path / "store"), embed_model="hash")) as engine:
        monkeypatch.setattr(server, "_engine", engine)
        yield server._build_http_app("127.0.0.1", 8765), engine


async def response(app, path="/ready", headers=None):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
        return await client.get("http://127.0.0.1:8765" + path,
                                headers=headers if headers is not None else {"Authorization": "Bearer " + TOKEN})


@pytest.mark.asyncio
async def test_authenticated_ready_after_write_and_dirty_access_counter(app_engine):
    app, engine = app_engine
    fact = engine.add_facts([{"text": "Synthetic user uses Neptune."}], project="A")["added"][0]
    engine.store.touch(fact)
    result = await response(app)
    assert result.status_code == 200
    assert result.json()["ready"] is True
    assert result.json()["version"] == "4.0.1" and result.json()["commit"] == COMMIT
    assert result.json()["commit_source"] == "deployment_manifest"


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,status", [({}, 401), ({"Authorization": "Bearer wrong"}, 401),
                         ({"Authorization": "Bearer " + TOKEN, "Origin": "https://attacker.invalid"}, 403)])
async def test_ready_protected_like_health(app_engine, headers, status):
    app, _ = app_engine
    assert (await response(app, headers=headers)).status_code == status


@pytest.mark.asyncio
async def test_unloaded_engine_unready_without_loading(app_engine, monkeypatch):
    app, _ = app_engine
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "get_engine", lambda: pytest.fail("Readiness must not create an engine"))
    result = await response(app)
    assert result.status_code == 503 and result.json()["reason"] == "engine_not_loaded"
    assert (await response(app, "/health")).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["malformed", "truncated", "missing", "changed", "blocks", "requests",
                                    "vector", "bm25", "entities", "facets", "poisoned", "closed",
                                    "metadata", "writer_lease"])
async def test_corrupt_sources_or_incomplete_indexes_not_ready(app_engine, damage):
    app, engine = app_engine
    fact = engine.add_facts([{"text": "Synthetic evidence."}], project="A")["added"][0]
    source = engine.store.items_path
    if damage == "malformed":
        with source.open("ab") as f: f.write(b"not-json\n")
    elif damage == "truncated":
        source.write_bytes(source.read_bytes()[:-1])
    elif damage == "missing":
        source.unlink()
    elif damage == "changed":
        rows = list(map(json.loads, source.read_bytes().splitlines()))
        rows[-1]["text"] = "Externally changed"
        source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif damage == "blocks":
        engine.blocks.path.write_text("not-json\n")
    elif damage == "requests":
        engine.requests.path.write_text("not-json\n")
    elif damage == "poisoned":
        engine.requests._poisoned = True
    elif damage == "closed":
        engine.close()
    elif damage == "metadata":
        engine.store.meta_path.write_text("not-json\n")
    elif damage == "writer_lease":
        engine._writer_lock.close()
    else:
        {"vector": engine.vector_index, "bm25": engine.bm25, "entities": engine.entities,
         "facets": engine.facet_index}[damage].remove([fact] if damage == "vector" else fact)
    result = await response(app)
    assert result.status_code == 503 and result.json()["ready"] is False
    assert "Synthetic evidence" not in result.text
    assert (await response(app, "/health")).status_code == 200


def test_busy_engine_returns_bounded_non_ready(app_engine):
    _, engine = app_engine
    with engine._lock:
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(engine_readiness, engine).result(timeout=2) == (False, "engine_busy")


@pytest.mark.asyncio
async def test_malformed_source_skipped_by_loader_still_not_ready(app_engine):
    _, engine = app_engine
    engine.store.items_path.write_text("not-json\n")
    engine.store.load()
    assert engine.store.items == {}
    assert engine_readiness(engine) == (False, "store_integrity")


@pytest.mark.asyncio
async def test_missing_model_returns_non_ready(app_engine):
    app, engine = app_engine
    engine.config.embed_model = "synthetic-unloaded-model"
    assert (await response(app)).json()["reason"] == "model_not_loaded"


@pytest.mark.asyncio
async def test_unknown_build_commit_does_not_claim_ready(app_engine, monkeypatch):
    _, _engine = app_engine
    monkeypatch.setattr("memory_os_v4.readiness.build_identity", lambda: {
        "version": "4.0.1", "commit": None, "commit_source": "unknown", "source_dirty": None})
    app = server._build_http_app("127.0.0.1", 8765)
    assert (await response(app)).json()["reason"] == "build_unqualified"


@pytest.mark.asyncio
async def test_dirty_build_does_not_claim_ready(app_engine, monkeypatch):
    monkeypatch.setattr("memory_os_v4.readiness.build_identity", lambda: {
        "version": "4.0.1", "commit": COMMIT, "commit_source": "checkout", "source_dirty": True})
    app = server._build_http_app("127.0.0.1", 8765)
    assert (await response(app)).json()["reason"] == "build_unqualified"


def test_invalid_build_commit_rejected_at_startup(app_engine, monkeypatch):
    monkeypatch.setenv("MEMORY_V4_BUILD_COMMIT", "not-a-commit")
    with pytest.raises(ValueError, match="complete"):
        server._build_http_app("127.0.0.1", 8765)
