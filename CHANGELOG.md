# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each dated heading is the UTC calendar day that version was published to PyPI (`YYYY-MM-DD`). `Unreleased` has no date.

## [Unreleased]

Next release is `0.2.0`, tracked in [Upgrade to v0.2.0](https://github.com/smusali/echoChain/issues/74).

### Added

- An experiment TOML file, with CLI flags overriding environment variables, then the file, then the built-in defaults. `echochain config show` prints the resolved settings and redacts the API key.
- `run`, `config show`, `models`, `export`, and `analyze`. `echochain --start` is still `echochain run --start`.
- `--dry-run` prints a token ceiling and a cost ceiling from the OpenRouter model list. A priced estimate above `confirm_above_usd` (default `$1`) waits for `--yes`.
- Each run writes `manifest.json` and a complete JSONL row under `runs/<UTC>_<run id>/`.
- `--out` chooses the run root, and `--budget` stops a run with exit code 3.
- Named schedules (`alternate`, `swap`, `factorial`, `latin`, and `fixed:<model>`) and the control conditions `same-model`, `identity`, and `paraphrase`.
- `seeds/v1.toml`, a versioned corpus. Runs can select a seed by id or a split.
- `--condition`, `--repeats`, and `--dry-run`. `--dry-run` prints the call count and, when a key is set, a cost upper bound.
- Exit code `1` for a generation failure, a judge failure, or an API failure. `0` is only a finished threshold or a finished cycle cap.
- `results/pilot-2026-10/` records the runs that finished before the OpenRouter account could no longer reserve a call. The directory includes the manifests, the JSONL, the plot, and a paragraph on those limits.

### Changed

- Chat calls share one client, retry only transient failures, and stop at a wall-clock deadline.
- Completions record tokens, cost, and finish reason. Empty or truncated replies are tried once more.
- Answer steps are scored for fidelity. Question steps are scored for answerability. Stop rules and plots read fidelity only.
- A run keeps its planned length. The manifest records why it ended.

### Fixed

- A judge parse failure is stored as no score. It no longer enters the metric, and one bad reply no longer ends the run.

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
