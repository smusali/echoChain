# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-24

### Added

- Offline tests, Ruff, mypy, pre-commit, and CI for Python 3.11 through 3.15.
- `.env.example`, with the API key redacted in logs, errors, and run files.

### Changed

- The project started as llm-telephone in June 2025 and was renamed to echoChain.
- Rotating models are current OpenRouter snapshots, and the judge is GPT-4.1 mini.
- Each saved cycle records the requested model and the model that served the call.

### Fixed

- A fresh install no longer imports `httpx` from the chat client.
- The API key is read when a run starts, so `--help` works when it is unset.
- Configured model IDs are checked against the OpenRouter catalog before a paid call.
