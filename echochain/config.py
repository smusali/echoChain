from __future__ import annotations

import logging
import os
from pathlib import Path

ROTATING_MODELS = [
    "google/gemini-3.8-flash",
    "deepseek/deepseek-v4.1-flash-1017",
    "mistralai/mistral-small-2603",
]
JUDGE_MODEL = "openai/gpt-4.1-mini"
MAX_CYCLES = 30
STOP_SCORE = 2
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/smusali/echoChain",
    "X-Title": "echoChain",
}
REQUIRED_PARAMETERS = ("response_format", "seed")
MISSING_API_KEY = "OPENROUTER_API_KEY is not set; export it or add it to `.env`"
REDACTED = "[REDACTED]"
_API_KEY_ENV = "OPENROUTER_API_KEY"
_EXPORT_PREFIX = "export "
_MIN_QUOTED_LENGTH = 2


class MissingAPIKeyError(RuntimeError):
    pass


def configured_models() -> tuple[str, ...]:
    """Return the rotating models and the judge.

    Returns:
        Model IDs that must pass preflight.
    """
    return (*ROTATING_MODELS, JUDGE_MODEL)


def load_api_key(env_path: Path | None = None) -> str:
    """Return the OpenRouter key from the environment or `.env`.

    Args:
        env_path: `.env` path. Defaults to `.env` in the working directory.

    Returns:
        The non-empty API key.

    Raises:
        MissingAPIKeyError: The key is missing from the environment and `.env`.
    """
    env_value = os.environ.get(_API_KEY_ENV, "").strip()
    if env_value:
        _install_redaction()
        return env_value
    path = Path(".env") if env_path is None else env_path
    file_value = _dotenv_value(path, _API_KEY_ENV).strip()
    if file_value:
        os.environ[_API_KEY_ENV] = file_value
        _install_redaction()
        return file_value
    raise MissingAPIKeyError(MISSING_API_KEY)


def redact(text: str) -> str:
    """Replace the API key in text.

    Args:
        text: Text that may contain the key.

    Returns:
        Text with the key replaced by `[REDACTED]`.
    """
    secret = os.environ.get(_API_KEY_ENV, "").strip()
    if not secret:
        return text
    return text.replace(secret, REDACTED)


def _dotenv_value(path: Path, name: str) -> str:
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix(_EXPORT_PREFIX)
        key, separator, value = line.partition("=")
        if separator == "" or key.strip() != name:
            continue
        return _unquote(value.strip())
    return ""


def _unquote(value: str) -> str:
    if len(value) < _MIN_QUOTED_LENGTH:
        return value
    if value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _install_redaction() -> None:
    logging.setLoggerClass(_RedactingLogger)
    _attach_filter(logging.getLogger())
    for existing in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(existing, logging.Logger):
            _attach_filter(existing)


def _attach_filter(logger: logging.Logger) -> None:
    if any(isinstance(item, _RedactFilter) for item in logger.filters):
        return
    logger.addFilter(_RedactFilter())


class _RedactingLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        super().__init__(name, level)
        self.addFilter(_RedactFilter())


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        secret = os.environ.get(_API_KEY_ENV, "").strip()
        if secret:
            _redact_record(record, secret)
        return True


def _redact_record(record: logging.LogRecord, secret: str) -> None:
    if isinstance(record.msg, str):
        record.msg = record.msg.replace(secret, REDACTED)
    args = record.args
    if isinstance(args, dict):
        record.args = {
            key: _redact_object(value, secret) for key, value in args.items()
        }
    elif isinstance(args, tuple):
        record.args = tuple(_redact_object(value, secret) for value in args)
    if record.exc_text:
        record.exc_text = record.exc_text.replace(secret, REDACTED)


def _redact_object(value: object, secret: str) -> object:
    if isinstance(value, str):
        return value.replace(secret, REDACTED)
    return value
