from __future__ import annotations

from pathlib import Path

import pytest

from echochain.coercion import ConfigError
from echochain.settings import resolve_settings


def test_precedence_int_float_list_path_pins_and_start(tmp_path: Path) -> None:
    config = tmp_path / "echochain.toml"
    config.write_text(
        "\n".join(
            (
                "max_cycles = 4",
                "temperature = 0.2",
                'models = ["file/model"]',
                'out = "from-file"',
                'seed = "from-file"',
                "",
                '[providers."file/model"]',
                'order = ["file-provider"]',
                "allow_fallbacks = false",
            )
        ),
        encoding="utf-8",
    )
    env = {
        "ECHOCHAIN_MAX_CYCLES": "5",
        "ECHOCHAIN_TEMPERATURE": "0.4",
        "ECHOCHAIN_MODELS": "env/model",
        "ECHOCHAIN_OUT": "from-env",
        "ECHOCHAIN_PROVIDERS": "env/model=env-provider|true",
    }
    flags = {
        "max_cycles": 6,
        "temperature": 0.6,
        "models": ["flag/model"],
        "out": tmp_path / "from-flag",
        "providers": "flag/model=flag-provider|false",
        "start": "inline seed",
    }
    chosen = resolve_settings(config=config, environ=env, overrides=flags)
    assert chosen.max_cycles == 6
    assert chosen.temperature == 0.6
    assert chosen.models == ("flag/model",)
    assert chosen.out == tmp_path / "from-flag"
    assert chosen.providers[0][0] == "flag/model"
    assert chosen.providers[0][1].order == ("flag-provider",)
    assert chosen.start == "inline seed"
    assert chosen.seed is None


def test_start_rejects_a_seed_selector(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="provide one of start or seed"):
        resolve_settings(
            overrides={"start": "text", "seed": "redis-epoll"},
            environ={},
        )


def test_shared_family_is_rejected() -> None:
    with pytest.raises(ConfigError, match="shares a family"):
        resolve_settings(
            overrides={"models": ["openai/a"], "judge": ["openai/b"]},
            environ={},
        )
