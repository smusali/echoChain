"""Basic tests for secret redaction."""
import os
import unittest
from unittest.mock import patch

class TestRedaction(unittest.TestCase):
    def test_redact_replaces_key(self):
        sentinel = "sk-or-v1-SENTINEL-TEST-KEY-98765"
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": sentinel}):
            # Re-import to pick up the env (module-level load)
            import importlib
            import llm_tel.config as cfg
            importlib.reload(cfg)
            text = f"API error: invalid key {sentinel} in response"
            redacted = cfg.redact(text)
            self.assertNotIn(sentinel, redacted)
            self.assertIn("[REDACTED]", redacted)

if __name__ == "__main__":
    unittest.main()
