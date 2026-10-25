# echoChain 📞

[![CI](https://github.com/smusali/echoChain/actions/workflows/ci.yml/badge.svg)](https://github.com/smusali/echoChain/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/echochain.svg)](https://pypi.org/project/echochain/)
[![Python versions](https://img.shields.io/pypi/pyversions/echochain.svg)](https://pypi.org/project/echochain/)

Research prototype. Runs write raw cycles to `experiments/`. This repository has no study measurements yet.

echoChain passes a technical statement through a telephone-game chain of language models and records a judge score at each step ([`echochain/engine.py`](https://github.com/smusali/echoChain/blob/main/echochain/engine.py)). [Related work](#related-work) names the studies this follows and the three differences in what it is built to measure.

## What is echoChain?

The loop starts from a technical statement, turns that statement into a question, turns the question back into an answer, and repeats. [`echochain/engine.py`](https://github.com/smusali/echoChain/blob/main/echochain/engine.py) runs the loop. A judge model is asked for a JSON score of how much of the seed remains. The score is not a validated measurement. See [Known issues](#known-issues).

### The process

1. **Start** with a technical statement (for example, "Redis uses single-threaded event loops with epoll for I/O multiplexing").
2. **Transform** an answer into a question with the prompt in [`echochain/prompts.py`](https://github.com/smusali/echoChain/blob/main/echochain/prompts.py).
3. **Transform** that question back into an answer with the other prompt in the same file.
4. **Repeat.** The loop stops when the running mean of scores is at or below 2, or after 30 cycles ([`STOP_SCORE`](https://github.com/smusali/echoChain/blob/main/echochain/config.py) and [`MAX_CYCLES`](https://github.com/smusali/echoChain/blob/main/echochain/config.py) in [`echochain/config.py`](https://github.com/smusali/echoChain/blob/main/echochain/config.py)). See [Known issues](#known-issues).
5. **Record** each cycle. The judge call is described under [Key features](#key-features).

### Example chain

The table below is an illustration of the loop. It is not a recorded experiment.

| Cycle | Phase | Content |
|-------|-------|---------|
| 0 | Seed | `Redis uses single-threaded event loops with epoll for I/O multiplexing` |
| 1 | A→Q | `How does Redis achieve high concurrency without multiple threads?` |
| 2 | Q→A | `Redis uses asynchronous I/O and an event-driven architecture` |
| 3 | A→Q | `What techniques help Redis handle many connections efficiently?` |
| ... | ... | *[degradation continues]* |

## Quick start

### Prerequisites

- Python 3.11+. [CI](https://github.com/smusali/echoChain/blob/main/.github/workflows/ci.yml) tests 3.11 through 3.15 on Linux, macOS, and Windows.
- An OpenRouter API key for a live run ([get one here](https://openrouter.ai/)). `--help`, `--version`, and the test suite do not need a key.

### Installation

```bash
pip install echochain
```

The PyPI project name is `echochain`. That is the [PEP 503](https://peps.python.org/pep-0503/) form of `echoChain`.

[`.github/workflows/release.yml`](https://github.com/smusali/echoChain/blob/main/.github/workflows/release.yml) publishes each release with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/). The workflow uses a GitHub OIDC token and stores no PyPI API token. Sigstore signs each distribution.

### Checkout

1. **Clone the repository**
   ```bash
   git clone https://github.com/smusali/echoChain.git
   cd echoChain
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package**
   ```bash
   pip install -e ".[dev]"
   ```
   For a pinned install, use `uv sync --frozen --extra dev` from the lock file. [CI](https://github.com/smusali/echoChain/blob/main/.github/workflows/ci.yml) uses that command.

4. **Configure the API key**

   echoChain reads `OPENROUTER_API_KEY` from the environment or from `.env` ([`load_api_key`](https://github.com/smusali/echoChain/blob/main/echochain/config.py)). It never accepts the key as a command-line argument ([`test_help_without_api_key`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py)).
   ```bash
   cp .env.example .env
   ```
   You can also export it in the shell:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```
   `--help` and `--version` work without a key ([`test_help_without_api_key`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py), [`test_version_flag`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py)). A run without one exits with code 2 ([`test_missing_key_exits`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py)).

### Basic usage

```bash
# Run a single experiment
echochain --start "Redis uses single-threaded event loops with epoll for I/O multiplexing"

# The same entry point as a module
python -m echochain --start "Redis uses single-threaded event loops with epoll for I/O multiplexing"

# Check results
ls experiments/
cat experiments/redis_uses_single-threaded_*/chain.csv
```

The `echochain` command is [`echochain.cli:main`](https://github.com/smusali/echoChain/blob/main/echochain/cli.py). `python -m echochain` calls the same function from [`echochain/__main__.py`](https://github.com/smusali/echoChain/blob/main/echochain/__main__.py). A live run calls OpenRouter and spends money. The directory name is the first three words of the seed plus a UTC timestamp ([`mk_exp_dir`](https://github.com/smusali/echoChain/blob/main/echochain/persistence.py)).

## Output

Each run creates:

- A timestamped directory under `experiments/` ([`mk_exp_dir`](https://github.com/smusali/echoChain/blob/main/echochain/persistence.py)).
- `chain.csv` and `chain.jsonl`, both flushed after every cycle ([`Recorder.log`](https://github.com/smusali/echoChain/blob/main/echochain/persistence.py)). [`test_fake_cycle_records_served_model`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py) reads both files.

A crash can still leave an unfinished run. Crash-safe runs are E6 in [#15](https://github.com/smusali/echoChain/issues/15).

### Sample output

The rows below show the file shape and the current model IDs. They are not measurements from a run.

```csv
cycle,phase,model,served_model,text,score
1,A→Q,google/gemini-3.8-flash,google/gemini-3.8-flash,How does Redis handle concurrent requests?,8
2,Q→A,deepseek/deepseek-v4.1-flash-1017,deepseek/deepseek-v4.1-flash-1017,Redis uses asynchronous I/O operations,7
3,A→Q,mistralai/mistral-small-2603,mistralai/mistral-small-2603,What enables efficient I/O in databases?,5
```

`model` is the ID that was requested. `served_model` is the ID the provider returned. Those columns are written by [`Recorder.log`](https://github.com/smusali/echoChain/blob/main/echochain/persistence.py) and checked by [`test_fake_cycle_records_served_model`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py). Current IDs are [`ROTATING_MODELS`](https://github.com/smusali/echoChain/blob/main/echochain/config.py). Runs recorded with `google/gemini-2.0-flash-001` or `anthropic/claude-sonnet-4` cannot be reproduced. Those models have been retired. Current Claude Sonnet IDs are not in the rotation because the OpenRouter catalog does not advertise `seed` for them, and every configured model must advertise `response_format` and `seed` ([`check_models`](https://github.com/smusali/echoChain/blob/main/echochain/client.py), [`test_preflight_missing_seed`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py)).

## Architecture

```
echoChain/
├── echochain/           # Core package
│   ├── cli.py           # Command line
│   ├── engine.py        # Telephone game orchestration
│   ├── client.py        # OpenRouter API communication
│   ├── judge.py         # Judge scoring
│   ├── prompts.py       # Jinja2 templates
│   ├── config.py        # Models, stop score, cycle limit
│   └── persistence.py   # CSV and JSONL output
├── tests/               # Offline tests
├── experiments/         # Directories created by a run
├── CHANGELOG.md         # Release notes
└── pyproject.toml       # Package metadata
```

## Key features

- **Model rotation.** Each cycle takes the next ID in [`ROTATING_MODELS`](https://github.com/smusali/echoChain/blob/main/echochain/config.py). [`echochain/engine.py`](https://github.com/smusali/echoChain/blob/main/echochain/engine.py) advances that list once per cycle. Cycle 1 uses the first ID. [`test_fake_cycle_records_served_model`](https://github.com/smusali/echoChain/blob/main/tests/test_cli.py) records that ID and the served model. The test stops after one cycle because the fake judge score is 1, which is at or below the stop score.
- **Judge.** [`score`](https://github.com/smusali/echoChain/blob/main/echochain/judge.py) asks [`JUDGE_MODEL`](https://github.com/smusali/echoChain/blob/main/echochain/config.py) (`openai/gpt-4.1-mini`) for a JSON `score`. The same test stores the score the fake judge returns. The [prompt](https://github.com/smusali/echoChain/blob/main/echochain/prompts.py) asks for an integer from 1 to 10. The code does not check that range. See [Known issues](#known-issues).
- **Two output files.** [`Recorder.log`](https://github.com/smusali/echoChain/blob/main/echochain/persistence.py) appends the cycle to `chain.csv` and `chain.jsonl` and flushes both. The same test reads both files.

## Known issues

Version 0.1.0 still measures degradation incorrectly in four ways. All four are tracked in [#16](https://github.com/smusali/echoChain/issues/16) until 0.2.0:

- **[C1](https://github.com/smusali/echoChain/issues/16).** A judge parse failure is stored as `1` (`_FALLBACK_SCORE` in [`echochain/judge.py`](https://github.com/smusali/echoChain/blob/main/echochain/judge.py)) and enters the running mean in [`echochain/engine.py`](https://github.com/smusali/echoChain/blob/main/echochain/engine.py).
- **[C3](https://github.com/smusali/echoChain/issues/16).** Every step, including a question, is scored against the seed with the answer prompt ([`echochain/engine.py`](https://github.com/smusali/echoChain/blob/main/echochain/engine.py), [`echochain/prompts.py`](https://github.com/smusali/echoChain/blob/main/echochain/prompts.py)).
- **[D1](https://github.com/smusali/echoChain/issues/16).** Phase and model are both chosen from the cycle index, so those effects are not isolated ([`echochain/engine.py`](https://github.com/smusali/echoChain/blob/main/echochain/engine.py)).
- **[D5](https://github.com/smusali/echoChain/issues/16).** The run stops on the running mean, and [`echochain/cli.py`](https://github.com/smusali/echoChain/blob/main/echochain/cli.py) always prints `Degradation complete`.

## Related work

Iterative distortion through language models has already been studied. echoChain is built to differ in three ways, and two of those differences are not in this version:

1. **Technical claims.** A model can keep the topic and lose the mechanism. The seed examples and the judge prompt ask about technical meaning. This version still collapses that question into one score.
2. **Model rotation as a variable.** Rotation is meant to be isolated from the question and answer phase. This version does not isolate it (D1 and D2 in [#16](https://github.com/smusali/echoChain/issues/16)).
3. **Claim-by-claim fidelity.** Fidelity is meant to be scored claim by claim, following FActScore, rather than as one 1-10 score (C7 in [#19](https://github.com/smusali/echoChain/issues/19)). This version still asks for one score.

Cited work:

- Amr Mohamed, Mingmeng Geng, Michalis Vazirgiannis, and Guokan Shang. "LLM as a Broken Telephone: Iterative Generation Distorts Information." ACL 2025. DOI [10.18653/v1/2025.acl-long.371](https://doi.org/10.18653/v1/2025.acl-long.371). [Anthology](https://aclanthology.org/2025.acl-long.371/), [arXiv:2502.20258](https://arxiv.org/abs/2502.20258). Translation and paraphrase chains accumulate distortion. Restricted prompting and lower temperature reduce it and do not remove it.
- Jérémy Perez, Grgur Kovač, Corentin Léger, Cédric Colas, Gaia Molinaro, Maxime Derex, Pierre-Yves Oudeyer, and Clément Moulin-Frier. "When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings." ICLR 2025. [arXiv:2407.04503](https://arxiv.org/abs/2407.04503). Introduces the attractor effect.
- Sabbir Mollah, Rohit Gupta, Sirnam Swetha, Qingyang Liu, Ahnaf Munir, and Mubarak Shah. "The Telephone Game: Evaluating Semantic Drift in Unified Models." 2025. [arXiv:2509.04438](https://arxiv.org/abs/2509.04438). Measures drift between images and text.
- Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." EMNLP 2023. [Anthology](https://aclanthology.org/2023.emnlp-main.741/).
- A public tool that runs the same kind of chain: [yymss/llm-telephone](https://github.com/yymss/llm-telephone).

## Documentation

- [CONTRIBUTING.md](https://github.com/smusali/echoChain/blob/main/CONTRIBUTING.md): setup, offline tests, and the one paid smoke test
- [CHANGELOG.md](https://github.com/smusali/echoChain/blob/main/CHANGELOG.md): release notes
- [LICENSE](https://github.com/smusali/echoChain/blob/main/LICENSE): MIT License

## Contributing

See [CONTRIBUTING.md](https://github.com/smusali/echoChain/blob/main/CONTRIBUTING.md) for the development setup, the offline tests, and the pull request process.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/smusali/echoChain/blob/main/LICENSE) file for details.

## Acknowledgments

- Built for research into how language models preserve technical information.
- Uses [OpenRouter](https://openrouter.ai/) for access to the model APIs.
- Inspired by the telephone game, and by the studies in [Related work](#related-work).
