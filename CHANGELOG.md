# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Each dated heading is the UTC calendar day that version was published to PyPI (`YYYY-MM-DD`). `Unreleased` has no date.

## [Unreleased]

Next release is `0.2.0`, tracked in [Upgrade to v0.2.0](https://github.com/smusali/echoChain/issues/74).

### Added

- `echochain analyze` prints the restricted mean of cycles to degradation up to `max_cycles`, with a seed-clustered bootstrap interval, the median or `not reached`, and the share of runs that degraded.
- A credit refusal that outlasts the waits finishes the run with `insufficient_credits` and exit code 3. `echochain run --resume BATCH_ID` continues the unfinished cells, and `--dry-run` on that resume counts only those cells.
- The final status line names the last recorded step, the planned length, and, for a cycle-cap stop, where the threshold was first met.
- Rubric version 2 asks for an anchored 1 to 5 score and the existing 1 to 10 score, plus one verdict per frozen corpus claim. Claim fidelity is the primary metric. The stop rule still reads the 1 to 10 mean.
- Later answers are judged against the previous answer as well as the seed.
- Each call sends `top_p` and a derived sampling seed. The manifest stores `sampling_seed`, and a row stores whether the provider echoed the seed.
- `echochain analyze` prints clustered bootstrap intervals, a Kaplan-Meier table, and pairwise log-rank tests, and can write those tables plus two SVG figures.
- `docs/METHODOLOGY.md` and `docs/judge-validation.md` define the measurement and the agreement gate. The labels for that gate are not collected.
- A sanitized chat cassette is replayed in the offline suite. `.github/workflows/smoke.yml` runs one live cycle nightly when an API key is configured.
- `echochain run --resume` continues an interrupted, failed, or unfinished run from its last complete step.
- Provider pins accept `data_collection` and `zdr`.
- CI enforces 80% coverage and runs `pip-audit`. Dependabot watches the lock file and GitHub Actions.
- `SECURITY.md` describes how to report a vulnerability privately.
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

- Every chat call asks OpenRouter to require the requested parameters and not to fall back to another provider.
- The manifest lists the provider that served each model. A catchable crash records `failed` and an error summary.
- Development installs use pytest 9.0.3 or newer.
- Chat calls share one client, retry only transient failures, and stop at a wall-clock deadline.
- Completions record tokens, cost, and finish reason. Empty or truncated replies are tried once more.
- Answer steps are scored for fidelity. Question steps are scored for answerability. Stop rules and plots read fidelity only.
- A run keeps its planned length. The manifest records why it ended.

### Fixed

- A judge parse failure is stored as no score. It no longer enters the metric, and one bad reply no longer ends the run.

### Removed

- Runs no longer write `chain.csv`. `echochain export --format csv` prints those columns.
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
