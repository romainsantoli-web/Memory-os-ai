# Distribution qualification

The CI workflow qualifies Python 3.11 and 3.12 on Ubuntu. Newer Python versions
may satisfy package metadata but are not yet part of the qualified matrix.

Required checks:

- Blocking Ruff E9/F checks across both engine trees, their tests, benchmark
  adapters and qualification scripts; full E/F checks for v3 and scripts.
- Independent v3 and v4 test runs, each with a coverage XML artifact and an
  80% minimum. Deterministic v4 coverage excludes the separately run slow test.
- Wheel and sdist build, `twine check`, a fresh environment installation and
  `pip check`, on both supported Python versions.
- Real MCP initialize/list-tools exchanges for all three installed console
  entry points outside the checkout, with no `PYTHONPATH` source override;
  both v4 aliases also execute `--help` and `--selfcheck`.
- A separate CPU semantic job downloads the configured default Granite and
  legacy MiniLM weights, caches them, then runs offline. The default smoke
  disables lexical channels, checks a paraphrase match and checks persistence
  after reload. Hash embeddings cannot pass this qualification.

The default model job uses public weights and no paid APIs. Cold downloads and
CPU inference have a 25-minute timeout. A missing cache/network failure is a
qualification failure, not a reason to substitute hash vectors silently.

CI artifacts are candidates. Release only the exact distributions whose checks
passed, record their SHA-256 values and run migration/rollback checks required by
[T06's release process](../README.md#development-and-distribution-qualification).
No release or upload is performed by ordinary CI. The PyPI workflow independently
checks the tag matches package metadata and repeats installation qualification.

## License provenance

The initial commit `a3a23e4` carried a GPLv3 `LICENSE`, while packaging and README
advertised LGPLv3. On 2026-10-09 the maintainer explicitly confirmed
`LGPL-3.0-or-later`, matching the announced package license. `LICENSE` now contains
the official GNU LGPLv3 text downloaded from
<https://www.gnu.org/licenses/lgpl-3.0.txt>. The former GPLv3 text is preserved in
`LICENSE.GPL-3.0`, the required companion. Package metadata includes both files
and the SPDX expression. Copyright/attribution remain unchanged.
