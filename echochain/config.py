import os

ROTATING_MODELS = [
    "google/gemini-2.0-flash-001",  # fast & cheap
    "anthropic/claude-sonnet-4",  # good at paraphrasing
]

JUDGE_MODEL = "openai/gpt-4o-mini"  # reserved for scoring only
MAX_CYCLES = 30
STOP_SCORE = 2  # ≤ 2 ⇒ terminate
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/smusali/echoChain",
    "X-Title": "echoChain",
}
