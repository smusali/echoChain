import os
from pathlib import Path

def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no external dependency)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

_load_dotenv()

ROTATING_MODELS = [
    "google/gemini-2.0-flash-001",  # fast & cheap
    "anthropic/claude-sonnet-4",  # good at paraphrasing
]

JUDGE_MODEL = "openai/gpt-4o-mini"  # reserved for scoring only
MAX_CYCLES = 30
STOP_SCORE = 2  # ≤ 2 ⇒ terminate
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Copy .env.example to .env and set the key, or export it."
    )
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/smusali/llm-telephone",
    "X-Title": "LLM Telephone (dev)",
}

def redact(text: str) -> str:
    """Redact the API key from any string (logs, errors, manifests)."""
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY in text:
        return text.replace(OPENROUTER_API_KEY, "[REDACTED]")
    return text
