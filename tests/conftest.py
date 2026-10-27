from __future__ import annotations

import os

from hypothesis import settings

settings.register_profile("ci", derandomize=True)
if os.environ.get("CI"):
    settings.load_profile("ci")
