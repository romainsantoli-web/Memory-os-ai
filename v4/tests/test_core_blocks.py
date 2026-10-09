"""Core blocks: limit enforcement, append, history, persistence."""

from memory_os_v4.blocks import DEFAULT_BLOCKS, BlockStore


def test_default_blocks_present_empty(tmp_path):
    store = BlockStore(tmp_path / "blocks.jsonl", char_limit=100)
    for name in DEFAULT_BLOCKS:
        assert store.get(name) == ""


def test_set_and_get(tmp_path):
    store = BlockStore(tmp_path / "blocks.jsonl", char_limit=100)
    store.set("user_profile", "ML/HFT engineer, Apple Silicon.")
    assert store.get("user_profile") == "ML/HFT engineer, Apple Silicon."


def test_set_truncates_over_limit(tmp_path):
    store = BlockStore(tmp_path / "blocks.jsonl", char_limit=10)
    store.set("working_rules", "this text is definitely longer than ten chars")
    assert len(store.get("working_rules")) == 10


def test_append_combines_text(tmp_path):
    store = BlockStore(tmp_path / "blocks.jsonl", char_limit=200)
    store.set("project_state", "Phase 1 done.")
    store.append("project_state", "Phase 2 started.")
    assert "Phase 1 done." in store.get("project_state")
    assert "Phase 2 started." in store.get("project_state")


def test_history_tracks_versions(tmp_path):
    store = BlockStore(tmp_path / "blocks.jsonl", char_limit=200)
    store.set("user_profile", "v1")
    store.set("user_profile", "v2")
    hist = store.history("user_profile")
    assert [b.text for b in hist] == ["v1", "v2"]


def test_all_returns_all_blocks(tmp_path):
    store = BlockStore(tmp_path / "blocks.jsonl", char_limit=200)
    store.set("user_profile", "hello")
    all_blocks = store.all()
    assert all_blocks["user_profile"] == "hello"
    for name in DEFAULT_BLOCKS:
        assert name in all_blocks


def test_persistence_roundtrip_last_wins(tmp_path):
    path = tmp_path / "blocks.jsonl"
    store = BlockStore(path, char_limit=200)
    store.set("working_rules", "rule A")
    store.set("working_rules", "rule B")
    reloaded = BlockStore(path, char_limit=200)
    assert reloaded.get("working_rules") == "rule B"
    assert len(reloaded.history("working_rules")) == 2
