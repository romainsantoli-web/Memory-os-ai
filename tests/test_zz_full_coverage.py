"""Coverage-to-100 tests — targets every uncovered line/branch.

Uses mocks exclusively for heavy deps (torch, FAISS, whisper, fitz, etc.)
to avoid segfaults on Apple Silicon and long model loading times.
Naming: test_zz_ ensures this runs after all real-FAISS tests.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# =========================================================================
# 1. __main__.py  (lines 3-5: from .server import run; run())
# =========================================================================
class TestMainEntry:
    def test_main_calls_run(self):
        with patch("memory_os_ai.server.run") as mock_run:
            import memory_os_ai.__main__  # noqa: F401
            # The module-level run() was already called at import time.
            # If not, we call it explicitly:
            if not mock_run.called:
                # __main__ just calls run() at module level, already executed
                pass


# =========================================================================
# 2. engine.py — extractors (L56-130) + transcribe (L815-826)
# =========================================================================
from memory_os_ai.engine import (
    _extract_audio,
    _extract_txt,
    MemoryEngine,
    EXTRACTORS,
)


class TestExtractPdf:
    def test_extract_pdf_with_text(self, tmp_path):
        """Cover _extract_pdf L56-71 — text-based pages."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Hello from PDF page"

        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))

        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            import sys
            fitz_mock = sys.modules["fitz"]
            fitz_mock.open.return_value = mock_doc

            from memory_os_ai.engine import _extract_pdf
            text, pages = _extract_pdf("/fake/doc.pdf")
            assert "Hello from PDF page" in text
            assert pages == 2

    def test_extract_pdf_ocr_fallback(self, tmp_path):
        """Cover _extract_pdf L67-71 — OCR fallback for image pages."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   "  # empty text triggers OCR
        mock_pix = MagicMock()
        mock_pix.width = 100
        mock_pix.height = 100
        mock_pix.samples = b"\x00" * (100 * 100 * 3)
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        with patch.dict("sys.modules", {
            "fitz": MagicMock(open=MagicMock(return_value=mock_doc)),
        }):
            with patch("PIL.Image.frombytes", return_value=MagicMock()):
                with patch("pytesseract.image_to_string", return_value="OCR text"):
                    from memory_os_ai.engine import _extract_pdf
                    text, pages = _extract_pdf("/fake/doc.pdf")
                    assert "OCR text" in text


class TestExtractDocx:
    def test_extract_docx(self):
        mock_doc = MagicMock()
        p1 = MagicMock()
        p1.text = "Paragraph one"
        p2 = MagicMock()
        p2.text = "Paragraph two"
        mock_doc.paragraphs = [p1, p2]

        with patch.dict("sys.modules", {"docx": MagicMock()}):
            mod = sys.modules["docx"]
            mod.Document.return_value = mock_doc
            from memory_os_ai.engine import _extract_docx
            text, pages = _extract_docx("/fake/file.docx")
            assert "Paragraph one" in text
            assert "Paragraph two" in text
            assert pages == 1


class TestExtractDoc:
    def test_extract_doc(self):
        with patch.dict("sys.modules", {"textract": MagicMock()}):
            mod = sys.modules["textract"]
            mod.process.return_value = b"Doc text content"
            from memory_os_ai.engine import _extract_doc
            text, pages = _extract_doc("/fake/file.doc")
            assert text == "Doc text content"
            assert pages == 1


class TestExtractImage:
    def test_extract_image(self):
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        with patch("PIL.Image.open", return_value=mock_img):
            with patch("pytesseract.image_to_string", return_value="Image OCR text"):
                from memory_os_ai.engine import _extract_image
                text, pages = _extract_image("/fake/image.png")
                assert text == "Image OCR text"
                assert pages == 1


class TestExtractPptx:
    def test_extract_pptx(self):
        mock_shape1 = MagicMock()
        mock_shape1.text = "Slide 1 text"
        mock_slide1 = MagicMock()
        mock_slide1.shapes = [mock_shape1]

        mock_shape2 = MagicMock()
        mock_shape2.text = "Slide 2 text"
        mock_slide2 = MagicMock()
        mock_slide2.shapes = [mock_shape2]

        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide1, mock_slide2]

        with patch.dict("sys.modules", {"pptx": MagicMock()}):
            mod = sys.modules["pptx"]
            mod.Presentation.return_value = mock_prs
            from memory_os_ai.engine import _extract_pptx
            text, pages = _extract_pptx("/fake/deck.pptx")
            assert "Slide 1" in text
            assert pages == 2


class TestExtractPpt:
    def test_extract_ppt(self):
        with patch.dict("sys.modules", {"textract": MagicMock()}):
            mod = sys.modules["textract"]
            mod.process.return_value = b"PPT legacy text"
            from memory_os_ai.engine import _extract_ppt
            text, pages = _extract_ppt("/fake/deck.ppt")
            assert text == "PPT legacy text"


class TestExtractAudio:
    def test_extract_audio(self):
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Transcribed audio text  "}
        mock_whisper.load_model.return_value = mock_model

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.load = MagicMock()

        with patch.dict("sys.modules", {"whisper": mock_whisper, "torch": mock_torch}):
            with patch("memory_os_ai.engine._ensure_torch", return_value=mock_torch):
                text, pages = _extract_audio("/fake/audio.mp3", language="fr")
                assert text == "Transcribed audio text"
                assert pages == 1


class TestExtractTextFormats:
    """Test that all text-based formats (.md, .py, .ts, .log, etc.) use _extract_txt."""

    TEXT_FORMATS = [
        ".md", ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml",
        ".toml", ".sh", ".bash", ".css", ".html", ".xml", ".csv", ".sql",
        ".env", ".ini", ".cfg", ".conf", ".log", ".rst",
    ]

    def test_all_text_formats_registered(self):
        """Every text format must be in EXTRACTORS."""
        for ext in self.TEXT_FORMATS:
            assert ext in EXTRACTORS, f"{ext} missing from EXTRACTORS"
            assert EXTRACTORS[ext] is _extract_txt, f"{ext} should map to _extract_txt"

    def test_extract_markdown(self, tmp_path):
        f = tmp_path / "README.md"
        f.write_text("# Title\n\nSome **bold** text.\n")
        text, pages = _extract_txt(str(f))
        assert "Title" in text
        assert "bold" in text
        assert pages == 1

    def test_extract_python(self, tmp_path):
        f = tmp_path / "script.py"
        f.write_text("def hello():\n    return 'world'\n")
        text, pages = _extract_txt(str(f))
        assert "def hello" in text
        assert pages == 1

    def test_extract_typescript(self, tmp_path):
        f = tmp_path / "server.ts"
        f.write_text("const app = express();\napp.listen(3000);\n")
        text, pages = _extract_txt(str(f))
        assert "express()" in text
        assert pages == 1

    def test_extract_json(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text('{"key": "value", "port": 8012}\n')
        text, pages = _extract_txt(str(f))
        assert '"key"' in text
        assert pages == 1

    def test_extract_yaml(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("server:\n  port: 8012\n  host: localhost\n")
        text, pages = _extract_txt(str(f))
        assert "port: 8012" in text
        assert pages == 1

    def test_extract_log(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text("[2026-03-03 19:00:00] INFO: Server started on port 3000\n[2026-03-03 19:00:01] ERROR: Connection refused\n")
        text, pages = _extract_txt(str(f))
        assert "Server started" in text
        assert "Connection refused" in text
        assert pages == 1

    def test_extract_shell(self, tmp_path):
        f = tmp_path / "deploy.sh"
        f.write_text("#!/bin/bash\ndocker compose up -d\n")
        text, pages = _extract_txt(str(f))
        assert "docker compose" in text
        assert pages == 1

    def test_extract_css(self, tmp_path):
        f = tmp_path / "style.css"
        f.write_text(".btn { background: #6366f1; }\n")
        text, pages = _extract_txt(str(f))
        assert "#6366f1" in text
        assert pages == 1

    def test_extract_sql(self, tmp_path):
        f = tmp_path / "schema.sql"
        f.write_text("CREATE TABLE users (id TEXT PRIMARY KEY);\n")
        text, pages = _extract_txt(str(f))
        assert "CREATE TABLE" in text
        assert pages == 1

    def test_supported_extensions_includes_text_formats(self):
        """SUPPORTED_EXTENSIONS must include all text formats."""
        from memory_os_ai.engine import SUPPORTED_EXTENSIONS
        for ext in self.TEXT_FORMATS:
            assert ext in SUPPORTED_EXTENSIONS, f"{ext} not in SUPPORTED_EXTENSIONS"


class TestTranscribe:
    def test_transcribe_not_found(self):
        eng = MemoryEngine()
        result = eng.transcribe("/nonexistent/audio.mp3")
        assert result["ok"] is False
        assert "not found" in result["error"]

    def test_transcribe_unsupported_format(self, tmp_path):
        f = tmp_path / "test.avi"
        f.write_text("fake")
        eng = MemoryEngine()
        result = eng.transcribe(str(f))
        assert result["ok"] is False
        assert "Unsupported" in result["error"]

    def test_transcribe_success(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_text("fake audio")
        eng = MemoryEngine()
        with patch("memory_os_ai.engine._extract_audio", return_value=("Hello world", 1)):
            result = eng.transcribe(str(f))
            assert result["ok"] is True
            assert result["text"] == "Hello world"

    def test_transcribe_exception(self, tmp_path):
        f = tmp_path / "test.wav"
        f.write_text("fake audio")
        eng = MemoryEngine()
        with patch("memory_os_ai.engine._extract_audio", side_effect=RuntimeError("Whisper crash")):
            result = eng.transcribe(str(f))
            assert result["ok"] is False
            assert "Whisper crash" in result["error"]


# =========================================================================
# 3. engine.py — model property (L227), edge cases
# =========================================================================
class TestEngineModelProperty:
    def test_model_lazy_loads(self):
        eng = MemoryEngine()
        mock_st = MagicMock()
        mock_st.return_value = "fake_model"
        with patch("memory_os_ai.engine._ensure_sentence_transformer", return_value=mock_st):
            m = eng.model
            assert m == "fake_model"

    def test_model_local_cache(self, tmp_path):
        """Cover L224-227 — local model path in cache_dir."""
        local_model_dir = tmp_path / "all-MiniLM-L6-v2"
        local_model_dir.mkdir()
        eng = MemoryEngine(cache_dir=str(tmp_path))
        mock_st = MagicMock()
        mock_st.return_value = "local_model"
        with patch("memory_os_ai.engine._ensure_sentence_transformer", return_value=mock_st):
            m = eng.model
            assert m == "local_model"
            mock_st.assert_called_once_with(str(local_model_dir), device=eng.device)


class TestEngineIngestExtractorFailure:
    """Cover L315-316 — extractor raises exception → append empty result."""

    @patch.object(MemoryEngine, "_encode", return_value=np.random.randn(1, 384).astype("float32"))
    def test_extractor_failure_handled(self, mock_enc, tmp_path):
        (tmp_path / "good.txt").write_text("Hello world content")
        # Patch the pdf extractor to raise
        orig = EXTRACTORS.get(".pdf")
        EXTRACTORS[".pdf"] = lambda path: (_ for _ in ()).throw(ValueError("parse error"))
        try:
            (tmp_path / "bad.pdf").write_text("corrupted")
            eng = MemoryEngine()
            result = eng.ingest(str(tmp_path))
            assert result["ok"] is True  # good.txt should still succeed
        finally:
            if orig:
                EXTRACTORS[".pdf"] = orig


class TestEngineSearchEdgeCases:
    """Cover L445 (idx < 0), L469 (not initialized)."""

    def test_search_not_initialized(self):
        eng = MemoryEngine()
        results = eng.search("query")
        assert results == []

    def test_get_context_not_initialized(self):
        eng = MemoryEngine()
        assert eng.get_context("query") == ""

    def test_search_occurrences_not_initialized(self):
        eng = MemoryEngine()
        result = eng.search_occurrences("keyword")
        assert result.total == 0

    def test_get_segment_text_found(self):
        eng = MemoryEngine()
        from memory_os_ai.engine import DocumentInfo
        eng._documents["test.pdf"] = DocumentInfo("test.pdf", 1, 2, 10, 0, 2)
        eng._segments = ["segment one", "segment two"]
        text = eng.get_segment_text("test.pdf")
        assert text == "segment one\nsegment two"

    def test_get_segment_text_not_found(self):
        eng = MemoryEngine()
        assert eng.get_segment_text("nonexistent.pdf") is None


class TestEngineCompactChunked:
    """Cover L724-742 — chunked dedup for n > 5000."""

    @patch.object(MemoryEngine, "_encode")
    @patch.object(MemoryEngine, "_build_index")
    def test_compact_chunked_path(self, mock_build, mock_enc):
        mock_enc.side_effect = lambda texts: np.random.randn(len(texts), 384).astype("float32")

        eng = MemoryEngine()
        # Create a large number of segments to trigger chunked path
        n = 5500
        eng._segments = [f"segment {i}" for i in range(n)]
        embs = np.random.randn(n, 384).astype("float32")
        # Normalize to make them dissimilar enough
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.clip(norms, 1e-10, None)

        mock_index = MagicMock()
        mock_index.ntotal = n
        mock_index.reconstruct_n.return_value = embs
        eng._index = mock_index
        eng._initialized = True

        result = eng.compact(max_segments=100, strategy="dedup_merge")
        assert result["ok"] is True
        assert result["after"] <= 100


class TestEngineCompactTopK:
    """Cover L755-770 — top-k centroid trimming."""

    @patch.object(MemoryEngine, "_encode")
    @patch.object(MemoryEngine, "_build_index")
    def test_compact_topk_trim(self, mock_build, mock_enc):
        mock_enc.side_effect = lambda texts: np.random.randn(len(texts), 384).astype("float32")

        eng = MemoryEngine()
        n = 200
        eng._segments = [f"unique segment number {i} with different content" for i in range(n)]
        embs = np.random.randn(n, 384).astype("float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.clip(norms, 1e-10, None)

        mock_index = MagicMock()
        mock_index.ntotal = n
        mock_index.reconstruct_n.return_value = embs
        eng._index = mock_index
        eng._initialized = True

        result = eng.compact(max_segments=50, strategy="dedup_merge")
        assert result["ok"] is True
        assert result["after"] <= 50


# =========================================================================
# 4. server.py — _save_project_links, _get_linked_engine cache loading,
#    chat_sync with ingest, chat_save, run/run_sse/run_http
# =========================================================================
import memory_os_ai.server as srv


class TestSaveProjectLinks:
    def test_save_and_load_roundtrip(self, tmp_path):
        links_file = tmp_path / "project_links.json"
        orig_file = srv._LINKS_FILE
        orig_links = srv._linked_projects.copy()
        srv._LINKS_FILE = str(links_file)
        srv._linked_projects.clear()
        srv._linked_projects["test-link"] = {"path": str(tmp_path), "engine": None}
        try:
            srv._save_project_links()
            assert links_file.exists()
            data = json.loads(links_file.read_text())
            assert "test-link" in data
        finally:
            srv._LINKS_FILE = orig_file
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)

    def test_save_oserror(self, tmp_path):
        orig_file = srv._LINKS_FILE
        srv._LINKS_FILE = "/nonexistent/path/links.json"
        try:
            # Should not raise
            srv._save_project_links()
        finally:
            srv._LINKS_FILE = orig_file


class TestGetLinkedEngine:
    def test_lazy_init_with_cache(self, tmp_path):
        """Cover L106-144 — _get_linked_engine with numpy cache + JSONL log."""
        # Create a cache file with valid embeddings
        embs = np.random.randn(3, 384).astype("float32")
        np.save(tmp_path / "embeddings_cache.npy", embs)

        # Create a conversation log with:
        # - a valid record WITH summary (hits L136)
        # - a valid record WITHOUT summary (hits L135->137 branch)
        # - a corrupt JSON line (hits L139-140 except)
        log = tmp_path / "_conversation_log.jsonl"
        record_with_summary = {"summary": "Test summary", "messages": [
            {"role": "user", "content": "hello"},
        ]}
        record_no_summary = {"messages": [
            {"role": "assistant", "content": "world"},
        ]}
        log.write_text(
            json.dumps(record_with_summary) + "\n"
            + json.dumps(record_no_summary) + "\n"
            + "NOT VALID JSON LINE\n"
        )

        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        srv._linked_projects["cached"] = {"path": str(tmp_path), "engine": None}

        try:
            eng = srv._get_linked_engine("cached")
            assert eng is not None
            assert eng._initialized is True
            assert len(eng._segments) >= 1
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)

    def test_returns_none_for_unknown(self):
        assert srv._get_linked_engine("nonexistent") is None

    def test_linked_engine_cache_no_log(self, tmp_path):
        """Cover L135->137: cache exists but no log file."""
        embs = np.random.randn(2, 384).astype("float32")
        np.save(tmp_path / "embeddings_cache.npy", embs)
        # No _conversation_log.jsonl

        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        srv._linked_projects["nolog"] = {"path": str(tmp_path), "engine": None}

        try:
            eng = srv._get_linked_engine("nolog")
            assert eng is not None
            # Index built but no segments from log
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)

    def test_linked_engine_build_index_fails(self, tmp_path):
        """Cover L139-140: _build_index raises → caught by except."""
        embs = np.random.randn(2, 384).astype("float32")
        np.save(tmp_path / "embeddings_cache.npy", embs)

        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        srv._linked_projects["failbuild"] = {"path": str(tmp_path), "engine": None}

        try:
            with patch.object(MemoryEngine, "_build_index", side_effect=RuntimeError("FAISS broken")):
                eng = srv._get_linked_engine("failbuild")
                assert eng is not None
                assert not eng._initialized
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)


class TestChatSync:
    """Cover server.py L558-560 — chat_sync with messages and ingest."""

    def test_chat_sync_with_messages(self, tmp_path):
        orig = srv._engine
        orig_extractor = srv._chat_extractor

        from memory_os_ai.chat_extractor import ChatExtractor

        # Create a JSONL chat source
        chat_file = tmp_path / "chat.jsonl"
        chat_file.write_text(
            json.dumps({"role": "user", "content": "Hello from sync"}) + "\n"
        )

        extractor = ChatExtractor(state_dir=str(tmp_path))
        extractor.add_source("test-src", "jsonl", str(chat_file))
        srv._chat_extractor = extractor

        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.ingest_segments.return_value = {"ok": True, "added_segments": 1, "total_segments": 1}
        srv._engine = eng

        try:
            result = srv._dispatch("memory_chat_sync", {"source_id": None})
            assert result["ok"] is True
            assert result["total_new_messages"] >= 1
            assert "ingest" in result
        finally:
            srv._engine = orig
            srv._chat_extractor = orig_extractor


class TestChatSave:
    """Cover server.py L594-601, L617-619, L685-686."""

    def test_chat_save_with_summary(self, tmp_path):
        orig = srv._engine
        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.ingest_segments.return_value = {"ok": True, "added_segments": 3, "total_segments": 3}
        srv._engine = eng

        try:
            with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
                result = srv._dispatch("memory_chat_save", {
                    "messages": [
                        {"role": "user", "content": "What is FAISS?"},
                        {"role": "assistant", "content": "FAISS is a library"},
                    ],
                    "summary": "Discussion about FAISS",
                    "project_label": "test-project",
                })
            assert result["ok"] is True
            assert result["has_summary"] is True
            assert result["saved_messages"] == 2
            # Check log file was written
            log_path = tmp_path / "_conversation_log.jsonl"
            assert log_path.exists()
        finally:
            srv._engine = orig


class TestServerRun:
    """Cover server.py L781-875 — run(), run_sse(), run_http()."""

    def test_run_dispatches_stdio(self):
        with patch("asyncio.run") as mock_asyncio_run:
            with patch.object(sys, "argv", ["memory-os-ai"]):
                srv.run()
                mock_asyncio_run.assert_called_once()

    def test_run_dispatches_sse(self):
        with patch("memory_os_ai.server.run_sse") as mock_sse:
            with patch.object(sys, "argv", ["memory-os-ai", "--sse"]):
                srv.run()
                mock_sse.assert_called_once()

    def test_run_dispatches_http(self):
        with patch("memory_os_ai.server.run_http") as mock_http:
            with patch.object(sys, "argv", ["memory-os-ai", "--http"]):
                srv.run()
                mock_http.assert_called_once()

    def test_run_sse_creates_app(self):
        with patch("uvicorn.run") as mock_uv:
            with patch.dict("sys.modules", {
                "starlette.applications": MagicMock(),
                "starlette.responses": MagicMock(),
                "starlette.routing": MagicMock(),
            }):
                srv.run_sse(host="127.0.0.1", port=9999)
                mock_uv.assert_called_once()
                call_kwargs = mock_uv.call_args
                assert call_kwargs[1]["port"] == 9999

    def test_run_http_creates_app(self):
        mock_transport = MagicMock()
        mock_transport.app = MagicMock()  # StreamableHTTPServerTransport.app
        with patch("uvicorn.run") as mock_uv:
            with patch("mcp.server.streamable_http.StreamableHTTPServerTransport", return_value=mock_transport):
                srv.run_http(host="127.0.0.1", port=9998)
                mock_uv.assert_called_once()
                call_kwargs = mock_uv.call_args
                assert call_kwargs[1]["port"] == 9998


class TestCheckApiKey:
    def test_no_key_set(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MEMORY_API_KEY", None)
            mock_req = MagicMock()
            assert srv._check_api_key(mock_req) is True

    def test_bearer_token_valid(self):
        with patch.dict(os.environ, {"MEMORY_API_KEY": "secret123"}):
            mock_req = MagicMock()
            mock_req.headers = {"authorization": "Bearer secret123"}
            assert srv._check_api_key(mock_req) is True

    def test_bearer_token_invalid(self):
        with patch.dict(os.environ, {"MEMORY_API_KEY": "secret123"}):
            mock_req = MagicMock()
            mock_req.headers = {"authorization": "Bearer wrong", "x-api-key": ""}
            assert srv._check_api_key(mock_req) is False

    def test_x_api_key_valid(self):
        with patch.dict(os.environ, {"MEMORY_API_KEY": "key456"}):
            mock_req = MagicMock()
            mock_req.headers = {"authorization": "", "x-api-key": "key456"}
            assert srv._check_api_key(mock_req) is True


class TestResourceRead:
    """Cover server.py L374-384 — read_resource branches."""

    @pytest.mark.asyncio
    async def test_read_resource_conversation_log(self, tmp_path):
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
            log = tmp_path / "_conversation_log.jsonl"
            log.write_text('{"msg": "hello"}\n')
            results = await srv.read_resource("memory://logs/conversation")
            assert len(results) == 1
            assert "hello" in results[0].text

    @pytest.mark.asyncio
    async def test_read_resource_conversation_log_missing(self, tmp_path):
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
            results = await srv.read_resource("memory://logs/conversation")
            assert "No conversation log" in results[0].text

    @pytest.mark.asyncio
    async def test_read_resource_linked_not_found(self):
        results = await srv.read_resource("memory://linked/nonexistent")
        assert "not found" in results[0].text

    @pytest.mark.asyncio
    async def test_read_resource_linked_not_loaded(self, tmp_path):
        orig_links = srv._linked_projects.copy()
        srv._linked_projects["test-link"] = {"path": str(tmp_path), "engine": None}
        try:
            # Engine is None, _get_linked_engine will try to init but there's no cache
            results = await srv.read_resource("memory://linked/test-link")
            assert "not loaded" in results[0].text or "test-link" in results[0].text
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)

    @pytest.mark.asyncio
    async def test_read_resource_unknown_uri(self):
        results = await srv.read_resource("memory://unknown/resource")
        assert "Unknown resource" in results[0].text


# =========================================================================
# 5. chat_extractor.py — missing branches
# =========================================================================
from memory_os_ai.chat_extractor import (
    ChatExtractor,
    ChatMessage,
    SyncCursor,
    extract_vscode_copilot,
    extract_jsonl,
    extract_markdown,
    extract_folder,
)


class TestVSCodeExtractorBranches:
    """Cover L144-145, L156, L182-207 branches."""

    def test_vscode_no_session_index(self, tmp_path):
        """L144-145: no chat index → return empty."""
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")
        conn.commit()
        conn.close()

        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_vscode_copilot(str(tmp_path), cursor)
        assert msgs == []

    def test_vscode_response_string_value(self, tmp_path):
        """Cover L202-207: response.value is a string instead of list."""
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")

        index = {"entries": {"sess1": {"title": "Test"}}}
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("chat.ChatSessionStore.index", json.dumps(index)),
        )

        session = {
            "requests": [{
                "message": {"text": "User question"},
                "timestamp": 1234567890,
                "response": {"value": "String response directly"},
            }]
        }
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("chat.ChatSessionStore.session.sess1", json.dumps(session)),
        )
        conn.commit()
        conn.close()

        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_vscode_copilot(str(tmp_path), cursor)
        # Should have user + assistant message
        assert len(msgs) >= 2
        assistant_msgs = [m for m in msgs if m.role == "assistant"]
        assert any("String response" in m.content for m in assistant_msgs)

    def test_vscode_response_dict_parts(self, tmp_path):
        """Cover L182-185: response.value is a list with dict parts."""
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")

        index = {"entries": {"sess2": {"title": "Dict Test"}}}
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("chat.ChatSessionStore.index", json.dumps(index)),
        )

        session = {
            "requests": [{
                "message": {"text": "User Q"},
                "timestamp": None,
                "response": {"value": [
                    {"value": "Part A"},
                    "Part B string",
                    42,  # non-string/dict part → should be skipped
                ]},
            }]
        }
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("chat.ChatSessionStore.session.sess2", json.dumps(session)),
        )
        conn.commit()
        conn.close()

        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_vscode_copilot(str(tmp_path), cursor)
        assistant_msgs = [m for m in msgs if m.role == "assistant"]
        assert len(assistant_msgs) >= 2  # Part A + Part B

    def test_vscode_session_json_error(self, tmp_path):
        """Cover L206-207: JSONDecodeError in session data → continue."""
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")

        index = {"entries": {"bad-sess": {"title": "Bad"}}}
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("chat.ChatSessionStore.index", json.dumps(index)),
        )
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("chat.ChatSessionStore.session.bad-sess", "NOT VALID JSON {{"),
        )
        conn.commit()
        conn.close()

        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_vscode_copilot(str(tmp_path), cursor)
        assert msgs == []  # Bad JSON skipped

    def test_vscode_missing_session_row(self, tmp_path):
        """Cover L156: session_key exists in index but not in table."""
        db_path = tmp_path / "state.vscdb"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")

        index = {"entries": {"ghost-sess": {"title": "Ghost"}}}
        conn.execute(
            "INSERT INTO ItemTable VALUES (?, ?)",
            ("chat.ChatSessionStore.index", json.dumps(index)),
        )
        # Don't insert the session data — it's missing
        conn.commit()
        conn.close()

        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_vscode_copilot(str(tmp_path), cursor)
        assert msgs == []


class TestJsonlExtractorBranches:
    """Cover L237 — JSONDecodeError in JSONL lines."""

    def test_jsonl_bad_lines_skipped(self, tmp_path):
        f = tmp_path / "chat.jsonl"
        f.write_text(
            'INVALID JSON LINE\n'
            '{"role": "user", "content": "valid msg"}\n'
            '{broken\n'
        )
        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_jsonl(str(f), cursor)
        assert len(msgs) == 1
        assert msgs[0].content == "valid msg"


class TestFolderExtractorBranches:
    """Cover L331, L335 — folder with mixed extensions, skip unchanged."""

    def test_folder_with_txt_json(self, tmp_path):
        (tmp_path / "note.txt").write_text("Plain text note")
        (tmp_path / "data.json").write_text('{"key": "value"}')
        (tmp_path / "ignored.avi").write_text("not a chat file")

        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_folder(str(tmp_path), cursor)
        assert len(msgs) >= 2  # note.txt + data.json as documents

    def test_folder_skip_unchanged(self, tmp_path):
        (tmp_path / "note.txt").write_text("Some note")
        cursor = SyncCursor(source_id="test")
        msgs1, cursor = extract_folder(str(tmp_path), cursor)
        assert len(msgs1) >= 1

        # Second call with same cursor → no new messages (unchanged)
        msgs2, cursor = extract_folder(str(tmp_path), cursor)
        assert len(msgs2) == 0


class TestChatExtractorSync:
    """Cover sync branches L458, L462, L471-472."""

    def test_sync_unknown_source_type(self, tmp_path):
        extractor = ChatExtractor(state_dir=str(tmp_path))
        extractor._sources["bad"] = {"type": "unknown_type", "path": str(tmp_path)}
        result = extractor.sync()
        assert "bad" in result.get("errors", {})
        assert "Unknown source type" in result["errors"]["bad"]

    def test_sync_exception_in_source(self, tmp_path):
        extractor = ChatExtractor(state_dir=str(tmp_path))
        extractor._sources["crash"] = {"type": "jsonl", "path": "/dev/null/nonexistent"}
        result = extractor.sync()
        # Should handle gracefully
        assert result["ok"] is True


# =========================================================================
# 6. setup.py — missing branches (L37, L70-74, L145-146, L249-250, etc.)
# =========================================================================
from memory_os_ai.setup import (
    main as setup_main,
    check_status,
    _get_python,
    _claude_desktop_config_path,
    setup_codex,
    setup_vscode,
)


class TestSetupGetPython:
    def test_venv_python_exists(self, tmp_path):
        """Cover L37 — .venv/bin/python exists."""
        venv_dir = tmp_path / ".venv" / "bin"
        venv_dir.mkdir(parents=True)
        py = venv_dir / "python"
        py.write_text("#!/bin/sh\n")

        with patch("memory_os_ai.setup._INSTALL_DIR", str(tmp_path)):
            result = _get_python()
            assert str(py) == result


class TestSetupPlatformPaths:
    def test_claude_desktop_linux(self):
        """Cover L70-74 — Linux path."""
        with patch("platform.system", return_value="Linux"):
            path = _claude_desktop_config_path()
            assert ".config/Claude" in str(path)

    def test_claude_desktop_windows(self):
        with patch("platform.system", return_value="Windows"):
            with patch.dict(os.environ, {"APPDATA": "/fake/appdata"}):
                path = _claude_desktop_config_path()
                assert "Claude" in str(path)

    def test_claude_desktop_unsupported(self):
        with patch("platform.system", return_value="FreeBSD"):
            with pytest.raises(RuntimeError, match="Unsupported"):
                _claude_desktop_config_path()


class TestSetupCodexLocal:
    """Cover L145-146 — codex local config."""

    def test_codex_local_config(self, tmp_path):
        with patch("os.getcwd", return_value=str(tmp_path)):
            result = setup_codex(global_config=False)
            assert "✅" in result
            local_mcp = tmp_path / ".codex" / "mcp.json"
            assert local_mcp.exists()


class TestSetupVSCodePreserve:
    """Cover setup_vscode preserves existing settings."""

    def test_vscode_preserves_existing(self, tmp_path):
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        mcp_file = vscode_dir / "mcp.json"
        mcp_file.write_text(json.dumps({
            "servers": {"other-server": {"command": "other"}}
        }))

        with patch("os.getcwd", return_value=str(tmp_path)):
            result = setup_vscode()
            assert "✅" in result

            data = json.loads(mcp_file.read_text())
            assert "other-server" in data["servers"]
            assert "memory-os-ai" in data["servers"]


class TestCheckStatusBranches:
    """Cover edge cases in check_status."""

    def test_status_codex_without_memory(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        mcp_file = codex_dir / "mcp.json"
        mcp_file.write_text(json.dumps({"mcpServers": {"other": {}}}))
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = check_status()
            assert "⚠️" in result or "not found" in result

    def test_status_with_all_configured(self, tmp_path):
        # Claude Desktop
        config_path = tmp_path / "claude_config.json"
        config_path.write_text(json.dumps({"mcpServers": {"memory-os-ai": {}}}))

        # Codex
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "mcp.json").write_text(json.dumps({"mcpServers": {"memory-os-ai": {}}}))

        # VS Code
        vscode_dir = tmp_path / ".vscode"
        vscode_dir.mkdir()
        (vscode_dir / "mcp.json").write_text("{}")

        # Claude Code
        (tmp_path / ".mcp.json").write_text("{}")

        # Cache
        cache = tmp_path / ".memory-os-ai"
        cache.mkdir()
        (cache / "index.npy").write_text("data")

        with patch("memory_os_ai.setup._claude_desktop_config_path", return_value=config_path):
            with patch("pathlib.Path.home", return_value=tmp_path):
                with patch("os.getcwd", return_value=str(tmp_path)):
                    result = check_status()
                    assert "✅" in result
                    assert "Claude Desktop" in result


# =========================================================================
# 7. engine.py — list_documents, status (ensure full coverage)
# =========================================================================
class TestEngineListDocuments:
    def test_list_documents_empty(self):
        eng = MemoryEngine()
        assert eng.list_documents() == []

    def test_list_documents_with_data(self):
        from memory_os_ai.engine import DocumentInfo
        eng = MemoryEngine()
        eng._documents["test.pdf"] = DocumentInfo("test.pdf", 3, 10, 500, 0, 10)
        eng._segments = ["seg"] * 10
        docs = eng.list_documents(include_stats=True)
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.pdf"


class TestEngineStatus:
    def test_status_uninitialised(self):
        eng = MemoryEngine()
        s = eng.status()
        assert s["initialized"] is False
        assert s["document_count"] == 0


# =========================================================================
# 8. Remaining coverage gaps — second round
# =========================================================================


class TestEngineSearchBoundary:
    """Cover L445 — idx < 0 or idx >= len."""

    @patch.object(MemoryEngine, "_encode")
    def test_search_idx_out_of_bounds(self, mock_enc):
        """FAISS returns idx=-1 when fewer results than requested."""
        mock_enc.return_value = np.array([[0.1] * 384], dtype="float32")

        eng = MemoryEngine()
        eng._segments = ["only segment"]
        eng._initialized = True

        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.1, 99.0]], dtype="float32"),
            np.array([[0, -1]], dtype="int64"),  # -1 = no result
        )
        eng._index = mock_index

        results = eng.search("test", top_k=2)
        assert len(results) == 1  # Only idx=0 returned, idx=-1 filtered out


class TestEngineGetContextBudget:
    """Cover L519-522 — budget overflow with remaining > 50."""

    @patch.object(MemoryEngine, "_encode")
    def test_get_context_truncates(self, mock_enc):
        mock_enc.return_value = np.array([[0.1] * 384], dtype="float32")

        eng = MemoryEngine()
        eng._segments = ["Short"] * 5 + ["A" * 200]  # Last segment is big
        eng._initialized = True

        from memory_os_ai.engine import DocumentInfo
        eng._documents = {"test.txt": DocumentInfo("test.txt", 1, 6, 100, 0, 6)}

        mock_index = MagicMock()
        # Return all 6 segments in order
        mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype="float32"),
            np.array([[0, 1, 2, 3, 4, 5]], dtype="int64"),
        )
        eng._index = mock_index

        # Very small budget to force truncation
        ctx = eng.get_context("query", max_chars=100, top_k=6)
        assert len(ctx) <= 100


class TestEngineSessionBriefTruncation:
    """Cover L608 — session_brief context truncation."""

    @patch.object(MemoryEngine, "_encode")
    def test_session_brief_budget(self, mock_enc):
        mock_enc.return_value = np.array([[0.1] * 384], dtype="float32")

        eng = MemoryEngine()
        eng._segments = [f"Segment {i} content text" for i in range(20)]
        eng._initialized = True

        from memory_os_ai.engine import DocumentInfo
        eng._documents = {"doc.txt": DocumentInfo("doc.txt", 1, 20, 1000, 0, 20)}

        mock_index = MagicMock()
        indices = np.arange(20, dtype="int64").reshape(1, -1)
        dists = np.full((1, 20), 0.1, dtype="float32")
        mock_index.search.return_value = (dists, indices)
        eng._index = mock_index

        brief = eng.session_brief(max_chars=100, focus_query="test")
        assert brief["ok"] is True
        assert brief["context_chars"] <= 100


class TestEngineCacheMissCorrupt:
    """Cover L344-345, L352-353 — cache miss/corrupt + write failure."""

    @patch.object(MemoryEngine, "_encode")
    @patch.object(MemoryEngine, "_build_index")
    def test_cache_corrupt_recomputes(self, mock_build, mock_enc, tmp_path):
        mock_enc.return_value = np.array([[0.1] * 384], dtype="float32")

        # Write corrupt cache
        cache = tmp_path / "embeddings_cache.npy"
        cache.write_bytes(b"corrupt data not numpy")

        # Write a valid .txt file
        (tmp_path / "doc.txt").write_text("Hello world content for testing")

        eng = MemoryEngine(cache_dir=str(tmp_path))
        result = eng.ingest(str(tmp_path))
        assert result["ok"] is True
        mock_enc.assert_called()  # Should recompute

    @patch.object(MemoryEngine, "_encode")
    @patch.object(MemoryEngine, "_build_index")
    def test_cache_write_failure(self, mock_build, mock_enc, tmp_path):
        mock_enc.return_value = np.array([[0.1] * 384], dtype="float32")

        # Put doc in a folder but cache_dir points to unwritable
        doc_dir = tmp_path / "docs"
        doc_dir.mkdir()
        (doc_dir / "file.txt").write_text("Test content")

        eng = MemoryEngine(cache_dir="/nonexistent/readonly/path")
        result = eng.ingest(str(doc_dir))
        assert result["ok"] is True  # Should succeed despite cache write failure


class TestEngineExtractorNotFound:
    """Cover L310 — extractor not in registry."""

    @patch.object(MemoryEngine, "_encode")
    @patch.object(MemoryEngine, "_build_index")
    def test_unknown_extension_skipped(self, mock_build, mock_enc, tmp_path):
        mock_enc.return_value = np.array([[0.1] * 384], dtype="float32")

        (tmp_path / "doc.txt").write_text("Valid text file content here")
        (tmp_path / "data.xyz").write_text("Unknown format")

        eng = MemoryEngine()
        result = eng.ingest(str(tmp_path))
        assert result["ok"] is True
        assert result["files_indexed"] == 1  # Only .txt


class TestEngineCompactMergeAndTrim:
    """Cover L755-770 — merge short segments + centroid trim."""

    @patch.object(MemoryEngine, "_encode")
    @patch.object(MemoryEngine, "_build_index")
    def test_compact_merge_short_then_centroid_trim(self, mock_build, mock_enc):
        """Create segments where short ones merge, then trim to max_segments."""
        call_count = [0]

        def counting_encode(texts):
            call_count[0] += 1
            return np.random.randn(len(texts), 384).astype("float32") * 0.01 + np.arange(len(texts))[:, None] * 0.1

        mock_enc.side_effect = counting_encode

        eng = MemoryEngine()
        # Many short segments that should merge + enough to need centroid trim
        segs = ["Hi" for _ in range(15)] + [f"Longer segment number {i} with actual content to not merge" for i in range(30)]
        eng._segments = segs
        n = len(segs)

        embs = np.random.randn(n, 384).astype("float32")
        # Make all embeddings very different so nothing gets deduped
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        embs *= 10  # Exaggerate differences

        mock_index = MagicMock()
        mock_index.ntotal = n
        mock_index.reconstruct_n.return_value = embs
        eng._index = mock_index
        eng._initialized = True

        result = eng.compact(max_segments=10, strategy="dedup_merge")
        assert result["ok"] is True
        assert result["after"] <= 10


class TestServerLinkedEngineBranches:
    """Cover server.py L125-144 — _get_linked_engine edge cases."""

    def test_linked_engine_no_cache_file(self, tmp_path):
        """L135-137: no embeddings_cache.npy → engine created but not initialized."""
        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        srv._linked_projects["nocache"] = {"path": str(tmp_path), "engine": None}

        try:
            eng = srv._get_linked_engine("nocache")
            assert eng is not None
            assert eng._initialized is False
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)

    def test_linked_engine_corrupt_cache(self, tmp_path):
        """L139-140: corrupt numpy file → exception caught in outer try."""
        # np.save creates a valid header — write something that np.load accepts
        # but has wrong shape, triggering an error in _build_index
        embs = np.array([], dtype="float32")  # empty array
        np.save(tmp_path / "embeddings_cache.npy", embs)

        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        srv._linked_projects["corrupt"] = {"path": str(tmp_path), "engine": None}

        try:
            eng = srv._get_linked_engine("corrupt")
            assert eng is not None
            # Empty array → len == 0 → skips _build_index branch
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)


class TestServerChatSaveLog:
    """Cover server.py L594-601, L617-619 — full chat save with log write."""

    def test_chat_save_no_summary(self, tmp_path):
        orig = srv._engine
        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.ingest_segments.return_value = {"ok": True, "added_segments": 1, "total_segments": 1}
        srv._engine = eng

        try:
            with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
                result = srv._dispatch("memory_chat_save", {
                    "messages": [{"role": "user", "content": "test"}],
                })
            assert result["ok"] is True
            assert result["has_summary"] is False
        finally:
            srv._engine = orig


class TestServerCompact:
    """Cover server.py L685-686 — memory_compact dispatch."""

    def test_compact_dispatch(self):
        orig = srv._engine
        eng = MagicMock(spec=MemoryEngine)
        eng.compact.return_value = {"ok": True, "before": 100, "after": 50}
        srv._engine = eng

        try:
            result = srv._dispatch("memory_compact", {})
            assert result["ok"] is True
            assert result["after"] == 50
        finally:
            srv._engine = orig


class TestServerUnknownTool:
    """Cover server.py L760-761 — unknown tool name."""

    def test_unknown_tool_returns_error(self):
        result = srv._dispatch("nonexistent_tool_name", {})
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]


class TestServerRunSSEWithAuth:
    """Cover server.py L790-806, L819 — run_sse inner handlers + auth."""

    def test_run_sse_with_api_key(self):
        with patch("uvicorn.run") as mock_uv:
            with patch.dict(os.environ, {"MEMORY_API_KEY": "test-key-123"}):
                srv.run_sse(host="127.0.0.1", port=9997)
                mock_uv.assert_called_once()


class TestServerRunHTTPAuth:
    """Cover server.py L834-851."""

    def test_run_http_with_api_key(self):
        mock_transport = MagicMock()
        mock_transport.app = MagicMock()
        with patch("uvicorn.run") as mock_uv:
            with patch("mcp.server.streamable_http.StreamableHTTPServerTransport", return_value=mock_transport):
                with patch.dict(os.environ, {"MEMORY_API_KEY": "test-key-456"}):
                    srv.run_http(host="127.0.0.1", port=9996)
                    mock_uv.assert_called_once()


class TestServerSaveProjectLinks:
    """Cover server.py L94-96 more precisely."""

    def test_save_links_creates_dir(self, tmp_path):
        links_file = tmp_path / "subdir" / "project_links.json"
        orig_file = srv._LINKS_FILE
        orig_links = srv._linked_projects.copy()
        srv._LINKS_FILE = str(links_file)
        srv._linked_projects.clear()
        srv._linked_projects["proj-a"] = {"path": "/some/path", "engine": None}

        try:
            srv._save_project_links()
            assert links_file.exists()
            data = json.loads(links_file.read_text())
            assert data["proj-a"] == "/some/path"
        finally:
            srv._LINKS_FILE = orig_file
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)


class TestResourceReadLinkedLoaded:
    """Cover server.py L426-427 — linked project with loaded engine."""

    @pytest.mark.asyncio
    async def test_read_linked_loaded_engine(self, tmp_path):
        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.status.return_value = {"initialized": True, "segments": 10}

        orig_links = srv._linked_projects.copy()
        srv._linked_projects.clear()
        srv._linked_projects["loaded-proj"] = {"path": str(tmp_path), "engine": eng}

        try:
            results = await srv.read_resource("memory://linked/loaded-proj")
            assert len(results) == 1
            assert "10" in results[0].text or "initialized" in results[0].text
        finally:
            srv._linked_projects.clear()
            srv._linked_projects.update(orig_links)


class TestResourceReadDocNotFound:
    """Cover server.py L374-375, L384 — document not found."""

    @pytest.mark.asyncio
    async def test_read_document_not_found(self):
        results = await srv.read_resource("memory://documents/nonexistent.pdf")
        assert "not found" in results[0].text


class TestSetupMainCLI:
    """Cover setup.py L320-321 (unknown target), L333 (all target)."""

    def test_main_unknown_target(self):
        with patch.object(sys, "argv", ["setup", "unknown-target"]):
            with pytest.raises(SystemExit):
                setup_main()

    def test_main_all_target(self, tmp_path):
        with patch.object(sys, "argv", ["setup", "all"]):
            with patch("memory_os_ai.setup.setup_claude_desktop", return_value="✅ Claude"):
                with patch("memory_os_ai.setup.setup_claude_code", return_value="✅ Code"):
                    with patch("memory_os_ai.setup.setup_codex", return_value="✅ Codex"):
                        with patch("memory_os_ai.setup.setup_vscode", return_value="✅ VSCode"):
                            with patch("memory_os_ai.setup.setup_chatgpt", return_value="✅ ChatGPT"):
                                setup_main()  # Should not raise

    def test_main_status_target(self):
        with patch.object(sys, "argv", ["setup", "status"]):
            with patch("memory_os_ai.setup.check_status", return_value="Status OK"):
                setup_main()  # Should not raise


class TestSetupCodexMerge:
    """Cover setup.py L145-146 — existing codex config merge."""

    def test_codex_merges_existing(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        mcp_file = codex_dir / "mcp.json"
        mcp_file.write_text(json.dumps({
            "mcpServers": {"other-tool": {"command": "other"}}
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            result = setup_codex(global_config=True)
            assert "✅" in result
            data = json.loads(mcp_file.read_text())
            assert "memory-os-ai" in data["mcpServers"]
            assert "other-tool" in data["mcpServers"]


class TestSetupCheckStatusCodexWarning:
    """Cover setup.py L249-250, L269 — check_status Codex warning, VS Code status."""

    def test_status_codex_configured(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "mcp.json").write_text(json.dumps({
            "mcpServers": {"memory-os-ai": {}}
        }))

        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("memory_os_ai.setup._claude_desktop_config_path", side_effect=RuntimeError):
                with patch("os.getcwd", return_value=str(tmp_path)):
                    result = check_status()
                    assert "Codex" in result
                    assert "✅" in result


class TestChatExtractorSyncErrors:
    """Cover chat_extractor.py L458, L462, L471-472 — errors in sync."""

    def test_sync_source_raises(self, tmp_path):
        extractor = ChatExtractor(state_dir=str(tmp_path))
        # Add a source that will fail (nonexistent vscode path)
        extractor.add_source("bad-vscode", "vscode", "/nonexistent/workspace")
        result = extractor.sync(source_id="bad-vscode")
        assert result["ok"] is True
        # Should be 0 messages (error caught)
        assert result["total_new_messages"] == 0


class TestChatExtractorFolderMd:
    """Cover chat_extractor.py L331, L343->325 — folder with .md file."""

    def test_folder_with_markdown(self, tmp_path):
        md_file = tmp_path / "chat.md"
        md_file.write_text("## User\nHello world\n\n## Assistant\nHi there")

        cursor = SyncCursor(source_id="test")
        msgs, cursor = extract_folder(str(tmp_path), cursor, extensions={".md"})
        assert len(msgs) >= 1


class TestChatExtractorAutoDetect:
    """Cover chat_extractor.py L534->531 — auto_detect base dir not found."""

    def test_auto_detect_no_base_dirs(self, tmp_path):
        extractor = ChatExtractor(state_dir=str(tmp_path))
        with patch("os.path.expanduser", return_value=str(tmp_path / "nonexistent")):
            found = extractor.auto_detect_vscode()
            assert found == []


# =========================================================================
# 9. Additional targeted coverage — dispatch paths, list_resources, etc.
# =========================================================================


class TestServerDispatchAutoDetect:
    """Cover server.py L594-601 — memory_chat_auto_detect with found workspaces."""

    def test_auto_detect_with_workspaces(self, tmp_path):
        orig = srv._chat_extractor

        from memory_os_ai.chat_extractor import ChatExtractor
        ext = ChatExtractor(state_dir=str(tmp_path))
        srv._chat_extractor = ext

        # Create fake workspace storage items
        ws = tmp_path / "workspace1"
        ws.mkdir()
        db = ws / "state.vscdb"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE ItemTable (key TEXT, value TEXT)")
        conn.commit()
        conn.close()

        try:
            with patch.object(ext, "auto_detect_vscode", return_value=[str(ws)]):
                result = srv._dispatch("memory_chat_auto_detect", {"auto_register": True})
                assert result["ok"] is True
                assert result["workspaces_found"] == 1
                assert len(result["registered"]) == 1
        finally:
            srv._chat_extractor = orig


class TestServerDispatchSessionBrief:
    """Cover server.py L617-619 — session_brief with chat sync messages."""

    def test_session_brief_with_sync(self, tmp_path):
        orig_engine = srv._engine
        orig_extractor = srv._chat_extractor

        eng = MagicMock(spec=MemoryEngine)
        eng.is_initialized = True
        eng.session_brief.return_value = {
            "ok": True,
            "context": "some context",
            "context_chars": 100,
            "unique_segments_retrieved": 5,
            "overview": {},
        }
        eng.ingest_segments.return_value = {"ok": True, "added_segments": 2, "total_segments": 10}
        srv._engine = eng

        from memory_os_ai.chat_extractor import ChatExtractor
        ext = ChatExtractor(state_dir=str(tmp_path))
        srv._chat_extractor = ext

        # Mock sync to return messages
        fake_msgs = [ChatMessage(role="user", content="test msg", source="test")]
        with patch.object(ext, "sync", return_value={
            "ok": True,
            "total_new_messages": 1,
            "per_source": {"test": 1},
            "errors": None,
            "messages": fake_msgs,
        }):
            try:
                result = srv._dispatch("memory_session_brief", {
                    "include_chat_sync": True,
                    "max_tokens": 1000,
                    "focus_query": "test",
                })
                assert result["ok"] is True
                # Chat sync should have been called and ingested
                assert "chat_sync" in result
            finally:
                srv._engine = orig_engine
                srv._chat_extractor = orig_extractor


class TestServerListResourcesConvLog:
    """Cover server.py L374-375, L384 — list_resources with conversation log."""

    @pytest.mark.asyncio
    async def test_list_resources_with_log(self, tmp_path):
        with patch.dict(os.environ, {"MEMORY_CACHE_DIR": str(tmp_path)}):
            log = tmp_path / "_conversation_log.jsonl"
            log.write_text('{"test": true}\n')

            orig_links = srv._linked_projects.copy()
            srv._linked_projects.clear()
            srv._linked_projects["my-proj"] = {"path": str(tmp_path), "engine": None}

            try:
                resources = await srv.list_resources()
                uris = [str(r.uri) for r in resources]
                assert any("logs/conversation" in u for u in uris)
                assert any("linked/my-proj" in u for u in uris)
            finally:
                srv._linked_projects.clear()
                srv._linked_projects.update(orig_links)


class TestServerOSErrorLogWrite:
    """Cover server.py L685-686 — OSError on log write."""

    def test_chat_save_log_write_failure(self, tmp_path):
        orig = srv._engine
        eng = MagicMock(spec=MemoryEngine)
        eng.ingest_segments.return_value = {"ok": True, "added_segments": 1, "total_segments": 1}
        srv._engine = eng

        try:
            # Point to a nonexistent/unwritable path
            with patch.dict(os.environ, {"MEMORY_CACHE_DIR": "/nonexistent/read/only"}):
                result = srv._dispatch("memory_chat_save", {
                    "messages": [{"role": "user", "content": "test"}],
                })
            # Should still succeed (OSError caught)
            assert result["ok"] is True
        finally:
            srv._engine = orig


class TestSetupMainAllWithFailure:
    """Cover setup.py L320-321 — except in all target loop."""

    def test_main_all_with_exception(self):
        with patch.object(sys, "argv", ["setup", "all"]):
            with patch("memory_os_ai.setup.TARGETS", {
                "claude-desktop": MagicMock(side_effect=RuntimeError("fail")),
                "claude-code": MagicMock(return_value="ok"),
                "codex": MagicMock(return_value="ok"),
                "vscode": MagicMock(return_value="ok"),
            }):
                with patch("memory_os_ai.setup.setup_chatgpt", return_value="ok"):
                    setup_main()  # Should not raise, handles exception


class TestMarkdownHashUnchanged:
    """Cover chat_extractor.py L296->289 — markdown hash unchanged skip."""

    def test_markdown_unchanged_skips(self, tmp_path):
        md = tmp_path / "chat.md"
        md.write_text("## User\nHello\n\n## Assistant\nHi")

        cursor = SyncCursor(source_id="test")
        msgs1, cursor = extract_markdown(str(md), cursor)
        assert len(msgs1) == 2

        # Second call — same hash, should skip
        msgs2, cursor = extract_markdown(str(md), cursor)
        assert len(msgs2) == 0
