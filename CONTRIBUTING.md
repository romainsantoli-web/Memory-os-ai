# Contributing to Memory OS AI

Use Python 3.11 or 3.12. Install from the repository root:

```bash
python -m pip install -e ".[dev]"
```

Add `audio` only when working on Whisper ingestion; Tesseract, FFmpeg and Antiword
are system dependencies for the corresponding v3 document formats.

Production engines are `src/memory_os_ai` (v3) and `v4/src/memory_os_v4` (v4).
The installed commands `memory-os-ai` and `memory-os-v4` start v4;
`memory-os-ai-v3` starts v3. Keep both engines covered when changing shared
packaging or client configurations.

```bash
ruff check src tests v4/src v4/tests v4/bench scripts --select E9,F
ruff check src tests scripts
python -m pytest tests --cov=memory_os_ai --cov-report=term-missing
python -m pytest v4/tests -m "not slow" --cov=memory_os_v4 --cov-report=term-missing
python -m build
python -m twine check dist/*
```

The CI matrix, artifact installation checks and real default-model job are
described in [distribution qualification](docs/DISTRIBUTION.md). Semantic tests
require cached public model weights; deterministic tests use fake/hash embeddings.
Evaluation claims must follow the [versioned protocol](v4/bench/protocols/v1.md).

Open a pull request describing the concrete problem, resulting behavior and
validation. Preserve existing research ledgers and third-party datasets. Do not
commit secrets or user memory data.

Contributions are under `LGPL-3.0-or-later`, as confirmed by the maintainer.
[LICENSE](LICENSE) contains LGPLv3; [LICENSE.GPL-3.0](LICENSE.GPL-3.0) contains the
required GPLv3 companion. Preserve copyright and attribution notices.
