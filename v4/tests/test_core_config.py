"""Config defaults and MEMORY_V4_* environment overrides."""

from memory_os_v4.config import Config


def test_defaults_match_winning_ablation_combo():
    """Post-campaign defaults (2026-09-12, docs/ABLATIONS.md "locomo_v4_combo_v2"):
    fusion=cc, w_bm25=1.5, candidates=100 beat the pre-campaign rrf/1.0/50
    defaults on both LoCoMo and LongMemEval oracle. w_dense/w_entity are
    unchanged (no reproducible gain from moving them)."""
    cfg = Config()
    assert cfg.fusion == "cc"
    assert cfg.w_dense == 1.0 and cfg.w_bm25 == 1.5 and cfg.w_entity == 0.5
    assert cfg.candidates == 100
    assert cfg.dedup_cos == 0.92
    assert cfg.supersede_cos == 0.80
    assert cfg.half_life_days("episode") == 30.0
    assert cfg.half_life_days("fact") == 180.0
    assert cfg.half_life_days("doc") == 365.0
    assert cfg.rerank_model is None
    assert cfg.rerank_top_n == 30
    assert cfg.block_char_limit == 2000
    assert cfg.episode_ttl_days == 120


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_DIR", "/tmp/custom-memory-dir")
    monkeypatch.setenv("MEMORY_V4_EMBED_MODEL", "hash")
    monkeypatch.setenv("MEMORY_V4_FUSION", "dbsf")
    monkeypatch.setenv("MEMORY_V4_W_DENSE", "2.5")
    monkeypatch.setenv("MEMORY_V4_USE_BM25", "false")
    monkeypatch.setenv("MEMORY_V4_CANDIDATES", "77")
    cfg = Config.from_env()
    assert cfg.data_dir == "/tmp/custom-memory-dir"
    assert cfg.embed_model == "hash"
    assert cfg.fusion == "dbsf"
    assert cfg.w_dense == 2.5
    assert cfg.use_bm25 is False
    assert cfg.candidates == 77


def test_half_life_unknown_kind_falls_back_to_default():
    cfg = Config()
    assert cfg.half_life_days("nonexistent-kind") == cfg.half_life_default_days


def test_ablation_flags_default_to_pre_campaign_behaviour_except_the_winner():
    """New Config flags (2026-09-12 ablation campaign) default to their
    pre-campaign values so existing behaviour is unchanged, EXCEPT
    `use_stemming`: it won on its own and as part of the winning combo, so
    it is now part of the shipped default (see docs/ABLATIONS.md)."""
    cfg = Config()
    assert cfg.multi_vector is False
    assert cfg.chunk_tokens == 48
    assert cfg.chunk_overlap == 12
    assert cfg.multi_vector_candidate_mult == 4
    assert cfg.use_stemming is True
    assert cfg.neighbor_expansion is False
    assert cfg.neighbor_seed == 20
    assert cfg.neighbor_decay == 0.6
    assert cfg.neighbor_window == 1
    assert cfg.recency_only_with_at_time is False


def test_ablation_flags_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_MULTI_VECTOR", "true")
    monkeypatch.setenv("MEMORY_V4_CHUNK_TOKENS", "96")
    monkeypatch.setenv("MEMORY_V4_USE_STEMMING", "true")
    monkeypatch.setenv("MEMORY_V4_NEIGHBOR_EXPANSION", "true")
    monkeypatch.setenv("MEMORY_V4_NEIGHBOR_WINDOW", "2")
    monkeypatch.setenv("MEMORY_V4_RECENCY_ONLY_WITH_AT_TIME", "true")
    cfg = Config.from_env()
    assert cfg.multi_vector is True
    assert cfg.chunk_tokens == 96
    assert cfg.use_stemming is True
    assert cfg.neighbor_expansion is True
    assert cfg.neighbor_window == 2
    assert cfg.recency_only_with_at_time is True


def test_temporal_query_mode_defaults_off(tmp_path):
    """Part B (docs/ABLATIONS.md "Hardening 2026-09-13"): a new ablation
    flag ships OFF, same convention as every other flag above, until B1-B4
    prove a win on the ledger."""
    cfg = Config()
    assert cfg.temporal_query_mode == "off"
    assert cfg.temporal_bm25_boost == 1.5


def test_temporal_query_mode_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_TEMPORAL_QUERY_MODE", "auto")
    monkeypatch.setenv("MEMORY_V4_TEMPORAL_BM25_BOOST", "2.0")
    cfg = Config.from_env()
    assert cfg.temporal_query_mode == "auto"
    assert cfg.temporal_bm25_boost == 2.0


def test_full_candidate_scoring_defaults_off(tmp_path):
    """docs/ABLATIONS.md "Full-candidate scoring 2026-09-13": a new ablation
    flag ships OFF (and dense_norm at its unchanged "minmax" behaviour),
    same convention as every other flag above, until proven on the ledger."""
    cfg = Config()
    assert cfg.full_candidate_scoring is False
    assert cfg.dense_norm == "minmax"


def test_full_candidate_scoring_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_FULL_CANDIDATE_SCORING", "true")
    monkeypatch.setenv("MEMORY_V4_DENSE_NORM", "fixed")
    cfg = Config.from_env()
    assert cfg.full_candidate_scoring is True
    assert cfg.dense_norm == "fixed"


# ---------------------------------------------------------------------------
# Palier 2-3 (2026-09-13): embed_max_seq_length, rerank_weight/rerank_device,
# entity_expansion (E10-E12, docs/ABLATIONS.md).
# ---------------------------------------------------------------------------
def test_embed_max_seq_length_default():
    cfg = Config()
    assert cfg.embed_max_seq_length == 512


def test_embed_batch_size_default_and_env_override(monkeypatch):
    assert Config().embed_batch_size == 32
    monkeypatch.setenv("MEMORY_V4_EMBED_BATCH_SIZE", "8")
    assert Config.from_env().embed_batch_size == 8


def test_embed_max_seq_length_env_override(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_EMBED_MAX_SEQ_LENGTH", "256")
    cfg = Config.from_env()
    assert cfg.embed_max_seq_length == 256


def test_rerank_weight_and_device_defaults():
    cfg = Config()
    assert cfg.rerank_weight == 0.5
    assert cfg.rerank_device == "auto"


def test_rerank_weight_and_device_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_RERANK_WEIGHT", "0.7")
    monkeypatch.setenv("MEMORY_V4_RERANK_DEVICE", "cpu")
    cfg = Config.from_env()
    assert cfg.rerank_weight == 0.7
    assert cfg.rerank_device == "cpu"


def test_entity_expansion_defaults_off():
    """E12 (Palier 2-3): new mechanism ships OFF, same convention as every
    other ablation flag, until proven on the ledger."""
    cfg = Config()
    assert cfg.entity_expansion is False
    assert cfg.expansion_seed == 10
    assert cfg.expansion_decay == 0.5


def test_entity_expansion_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_ENTITY_EXPANSION", "true")
    monkeypatch.setenv("MEMORY_V4_EXPANSION_SEED", "5")
    monkeypatch.setenv("MEMORY_V4_EXPANSION_DECAY", "0.25")
    cfg = Config.from_env()
    assert cfg.entity_expansion is True
    assert cfg.expansion_seed == 5
    assert cfg.expansion_decay == 0.25


def test_keywords_and_multi_query_defaults():
    """Production port of the LoCoMo-validated bench mechanisms (2026-09-14):
    keyword indexing defaults ON (byte-identical indexed text for any item
    with no keywords, so this is safe as a default); multi_query_rrf_k
    defaults to 20, matching the bench's `V4PlannedRetriever` winning config."""
    cfg = Config()
    assert cfg.index_keywords is True
    assert cfg.multi_query_rrf_k == 20


def test_keywords_and_multi_query_env_overrides(monkeypatch):
    monkeypatch.setenv("MEMORY_V4_INDEX_KEYWORDS", "false")
    monkeypatch.setenv("MEMORY_V4_MULTI_QUERY_RRF_K", "99")
    cfg = Config.from_env()
    assert cfg.index_keywords is False
    assert cfg.multi_query_rrf_k == 99
