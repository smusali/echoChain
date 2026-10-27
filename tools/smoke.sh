#!/bin/sh
set -eu
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY is unset; skipping"
  exit 0
fi
uv run echochain models
uv run echochain run --start "Redis uses single-threaded event loops with epoll for I/O multiplexing" --max-cycles 1 --repeats 1 --budget 0.10 --yes
