# echoChain 📞

[![CI](https://github.com/smusali/echoChain/actions/workflows/ci.yml/badge.svg)](https://github.com/smusali/echoChain/actions/workflows/ci.yml)

A research tool that measures how technical information degrades when passed through a "telephone game" chain of different Large Language Models.

## 🎯 What is echoChain?

echoChain implements a digital version of the classic "telephone game" using multiple Large Language Models. It measures how technical information erodes when repeatedly transformed between **Answer → Question → Answer** cycles across different LLM providers.

### The Process

1. **Start** with a technical statement (e.g., "Redis uses single-threaded event loops with epoll for I/O multiplexing")
2. **Transform** to question: "What question does this answer?"
3. **Transform** back to answer: "What's the answer to this question?"
4. **Repeat** until information degrades below threshold
5. **Measure** semantic similarity at each step using LLM-as-a-Judge

### Example Chain

| Cycle | Phase | Content |
|-------|-------|---------|
| 0 | Seed | `Redis uses single-threaded event loops with epoll for I/O multiplexing` |
| 1 | A→Q | `How does Redis achieve high concurrency without multiple threads?` |
| 2 | Q→A | `Redis uses asynchronous I/O and an event-driven architecture` |
| 3 | A→Q | `What techniques help Redis handle many connections efficiently?` |
| ... | ... | *[degradation continues]* |

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/smusali/echoChain.git
   cd echoChain
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install the package**
   ```bash
   pip install -e ".[dev]"
   ```
   For a pinned install, use `uv sync --frozen --extra dev` from the lock file.

4. **Configure API key**
   Copy the example env file and set your key. echoChain reads `OPENROUTER_API_KEY` from the environment or from `.env`. It never accepts the key as a command-line argument.
   ```bash
   cp .env.example .env
   ```
   You can also export it in the shell:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```
   `--help` and `--version` work without a key. A run without one exits with code 2.

### Basic Usage

```bash
# Run a single experiment
echochain --start "Redis uses single-threaded event loops with epoll for I/O multiplexing"

# The same entry point as a module
python -m echochain --start "Redis uses single-threaded event loops with epoll for I/O multiplexing"

# Check results
ls experiments/
cat experiments/redis_uses_single-threaded_*/chain.csv
```

## 📊 Output & Results

Each experiment generates:

- **Timestamped directory** in `experiments/`
- **CSV file** (`chain.csv`) with structured data
- **JSONL file** (`chain.jsonl`) with complete experiment logs

### Sample Output

```csv
cycle,phase,model,served_model,text,score
1,A→Q,google/gemini-3.8-flash,google/gemini-3.8-flash,How does Redis handle concurrent requests?,8
2,Q→A,deepseek/deepseek-v4.1-flash-1017,deepseek/deepseek-v4.1-flash-1017,Redis uses asynchronous I/O operations,7
3,A→Q,mistralai/mistral-small-2603,mistralai/mistral-small-2603,What enables efficient I/O in databases?,5
```

`model` is the ID that was requested. `served_model` is the ID the provider returned. Runs recorded with `google/gemini-2.0-flash-001` or `anthropic/claude-sonnet-4` cannot be reproduced; those models have been retired. Current Claude Sonnet IDs are not in the rotation because the OpenRouter catalog does not advertise `seed` for them, and every configured model must advertise `response_format` and `seed`.

## 🏗️ Architecture

```
echoChain/
├── echochain/           # 📦 Core package
│   ├── cli.py           # 🎯 Command line
│   ├── engine.py        # 🔄 Telephone game orchestration
│   ├── client.py        # 🌐 OpenRouter API communication
│   ├── judge.py         # ⚖️  LLM-as-a-Judge scoring
│   ├── prompts.py       # 📝 Jinja2 template management
│   ├── config.py        # ⚙️  System configuration
│   └── persistence.py   # 💾 Data storage & experiments
├── tests/               # 🧪 Offline tests
├── experiments/         # 📈 Generated experiment data
├── CHANGELOG.md         # 📒 Release notes
└── pyproject.toml       # 📋 Package metadata
```

## ✨ Key Features

- **🔄 Model Rotation**: Cycles through different LLM providers automatically
- **⚖️ LLM-as-a-Judge**: GPT-4.1 mini evaluates semantic similarity (1-10 scale)
- **📊 Dual Storage**: Results saved in both CSV and JSONL formats
- **🛡️ Error Resilience**: Exponential backoff, timeout handling, graceful degradation
- **⏱️ Real-time Logging**: Immediate data persistence prevents loss
- **🎯 Smart Termination**: Stops when degradation threshold reached or max cycles exceeded

## Known issues

Version 0.1.0 still measures degradation incorrectly in four ways:

- A judge failure is stored as a score.
- Questions are scored with the answer rubric.
- The rotating model and the phase change together, so those two effects are mixed.
- A run stops on a running mean.

## 📚 Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Release notes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development setup and contribution guidelines
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Future enhancement roadmap
- **[LICENSE](LICENSE)** - MIT License details

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development environment setup
- Code style guidelines
- Testing procedures
- Pull request process


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for research into LLM behavior and information preservation
- Uses [OpenRouter](https://openrouter.ai/) for unified LLM API access
- Inspired by the classic "telephone game" concept

---
