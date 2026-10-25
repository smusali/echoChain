# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each dated heading is the UTC calendar day that version was published to PyPI (`YYYY-MM-DD`). `Unreleased` has no date.

## [Unreleased]

Next release is `0.2.0`, tracked in [Upgrade to v0.2.0](https://github.com/smusali/echoChain/issues/74).

### Added

- Each run writes `manifest.json` and a complete JSONL row under `runs/<UTC>_<run id>/`.
- `--out` chooses the run root, and `--budget` stops a run with exit code 3.

### Changed

- Chat calls share one client, retry only transient failures, and stop at a wall-clock deadline.
- Completions record tokens, cost, and finish reason. Empty or truncated replies are tried once more.

### Removed

- The `backoff` dependency.

## [0.1.0] - 2026-10-25

### Added

- Offline tests, Ruff, mypy, pre-commit, and CI for Python 3.11 through 3.15.
- `.env.example`, with the API key redacted in logs, errors, and run files.
- The package is published on PyPI as `echochain`.

### Changed

- The project started as llm-telephone in June 2025 and was renamed to echoChain.
- Documentation matches the code: README claims, related work, and contributor setup.
- Rotating models are current OpenRouter snapshots, and the judge is GPT-4.1 mini.
- Each saved cycle records the requested model and the model that served the call.

### Fixed

- A fresh install no longer imports `httpx` from the chat client.
- The API key is read when a run starts, so `--help` works when it is unset.
- Configured model IDs are checked against the OpenRouter catalog before a paid call.

[Unreleased]: https://github.com/smusali/echoChain/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/smusali/echoChain/releases/tag/v0.1.0
