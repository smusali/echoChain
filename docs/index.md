# echoChain

echoChain measures how an answer changes when models rewrite it in a loop. The
repository is [smusali/echoChain](https://github.com/smusali/echoChain).

Set `OPENROUTER_API_KEY` in the environment or in `.env`.

## Install

```bash
pip install echochain
```

The optional `baseline` extra adds sentence-transformer scores:

```bash
pip install "echochain[baseline]"
```

## A first run

Dry-run from the
[example config](https://github.com/smusali/echoChain/blob/main/examples/echochain.toml):

```bash
echochain run --config examples/echochain.toml --dry-run
```

Run one live cycle:

```bash
echochain run --start "Redis uses epoll for I/O" --max-cycles 1 --yes
```

Where to go next:

- [Usage](USAGE.md) for commands, flags, configuration, output files, and exit codes.
- [Methodology](METHODOLOGY.md) for scores, conditions, analysis, and limits.
- [Contributing](CONTRIBUTING.md) for the local workflow and the release process.
