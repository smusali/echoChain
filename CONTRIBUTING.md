# Contributing to echoChain

Checks in this document do not need an OpenRouter API key. The only step that spends money is [Paid smoke test](#paid-smoke-test).

## Prerequisites

- Python 3.11+
- Git

## Setup

1. Clone the repository.

   ```bash
   git clone https://github.com/smusali/echoChain.git
   cd echoChain
   ```

2. Install the package and the development tools.

   CI installs from the lock file:

   ```bash
   uv sync --frozen --extra dev
   ```

   With pip, create a virtual environment first:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e '.[dev]'
   ```

3. Install the git hooks.

   ```bash
   uv run pre-commit install
   ```

   If you installed with pip and the virtual environment is active:

   ```bash
   pre-commit install
   ```

## Tests

The default suite is offline. It does not read `OPENROUTER_API_KEY`. It fails if a test opens a network socket.

```bash
uv run pytest
uv run pre-commit run --all-files
```

With pip, after activating the virtual environment:

```bash
pytest
pre-commit run --all-files
```

`pre-commit run --all-files` runs Ruff and mypy. Ruff formats to 88 characters. mypy runs in strict mode on `echochain` and `tests`.

## Commit messages

The history uses these types:

- `feat:` for a new capability
- `fix:` for a bug fix
- `docs:` for documentation

A scope is optional: `fix(package): make a fresh install work offline`.

## Add a model

Edit `ROTATING_MODELS` in [`echochain/config.py`](echochain/config.py).

Before a paid call, `check_models` in [`echochain/client.py`](echochain/client.py) requires every configured ID, including the judge, to be in the OpenRouter catalog and to advertise `response_format` and `seed`. Leave `JUDGE_MODEL` out of `ROTATING_MODELS`.

[`tests/test_cli.py`](tests/test_cli.py) builds its catalog fixture from `configured_models()`, so the offline preflight tests follow the list.

## Add a condition

A condition or a rotation schedule is not configurable. The loop in [`echochain/engine.py`](echochain/engine.py) alternates phase and model from the same cycle index. Schedules and control arms are D1 and D2 in [#16](https://github.com/smusali/echoChain/issues/16). Named experiment config is F1 in [#39](https://github.com/smusali/echoChain/issues/39).

## Paid smoke test

This step calls OpenRouter and spends money. Nothing else in this document does.

There is no `--budget` flag yet (E7 in [#13](https://github.com/smusali/echoChain/issues/13)). A run stops when the running mean of scores is at or below 2, or after 30 cycles, whichever comes first. Plan for up to 30 cycles.

```bash
export OPENROUTER_API_KEY="your-api-key-here"
echochain --start "Redis uses single-threaded event loops with epoll for I/O multiplexing"
```

You can also put the key in `.env`. Copy [`.env.example`](.env.example) and fill in `OPENROUTER_API_KEY`. The process never accepts the key as an argument.

## Report an issue

- Use a descriptive title.
- Include the Python version and the operating system.
- Give the steps, the command, and the error text. Leave the API key out.

## Open a pull request

1. Fork the repository and create a branch.
2. Make the change. Add docstrings on new public functions.
3. Run the [tests](#tests).
4. Open a pull request that says what changed and how you tested it.

## Ideas

Open work is tracked in the [issue list](https://github.com/smusali/echoChain/issues).

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
