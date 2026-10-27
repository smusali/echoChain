# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-27

### Added

- Subcommands `run`, `config show`, `models`, `export`, and `analyze`, with `echochain --start` kept as `echochain run --start`.
- Settings from a TOML file and `ECHOCHAIN_*` variables, with CLI flags taking precedence.
- Seed files with per-seed claims, selected by id with `--seed`.
- Schedules `alternate`, `swap`, `factorial`, `latin`, and `fixed:<model>`, plus `same-model`, `identity`, and `paraphrase` controls.
- `--repeats` for batches, and `--resume` to continue an interrupted run or batch.
- A `manifest.json` and one JSONL row per step for every run, described by the JSON Schemas in `schemas/`.
- Claim fidelity, an anchored 1 to 5 score, and a comparison with the previous answer on every answer step.
- Judge-free length, token-survival, and generic-word metrics, plus cosine and entailment with the `baseline` extra.
- `echochain analyze` for bootstrap curves, Kaplan-Meier survival, restricted mean time to degradation, and log-rank tests.
- `--dry-run` for call, token, and cost upper bounds, with `--yes` required above `confirm_above_usd`.
- `--budget` to stop a run at a USD cap with exit code 3.
- Provider pins with `order`, `allow_fallbacks`, `data_collection`, and `zdr`.
- `--sampling-seed` and `top_p` are sent on every call and stored on the manifest.
- Exit codes 0, 1, 2, 3, and 130, listed in `--help`.

### Changed

- Question steps are scored for answerability and answer steps for fidelity.
- Every chat call requires the requested parameters and disables provider fallbacks unless a pin allows them.
- Chat calls share one HTTP client, retry transient failures, and stop at a per-call deadline.

### Removed

- Runs no longer write `chain.csv`, so use `echochain export --format csv` instead.
- The `backoff` dependency.

### Fixed

- An unparsable judge reply is stored as no score instead of ending the run.
- An empty or truncated reply is retried once instead of being recorded as an answer.

## [0.1.0] - 2026-10-25

### Added

- The first PyPI release, installed with `pip install echochain`.
- `OPENROUTER_API_KEY` from the environment or `.env`, redacted in logs, errors, and run files.

### Changed

- The project is renamed from llm-telephone to echoChain.
- Rotating models are current OpenRouter snapshots, and the judge is GPT-4.1 mini.
- Each saved cycle records the requested model and the model that served it.

### Fixed

- A fresh install no longer fails on a missing `httpx` import.
- `--help` and `--version` work without an API key.
- Configured model ids are checked against the OpenRouter catalog before a paid call.

[Unreleased]: https://github.com/smusali/echoChain/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/smusali/echoChain/releases/tag/v0.2.0
[0.1.0]: https://github.com/smusali/echoChain/releases/tag/v0.1.0
