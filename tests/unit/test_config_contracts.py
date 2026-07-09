from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path("configs")
ENV_PLACEHOLDER_PREFIX = "${"
ENV_PLACEHOLDER_SUFFIX = "}"
SENSITIVE_KEYS = {
    "api_key",
    "mineru_api_key",
    "connection_string",
    "oauth_token",
    "password",
    "private_key_path",
    "sas_token",
}
TEST_PROVIDER_NAMES = {"fake", "hash"}
TEST_CONFIG_STEM_SUFFIXES = ("smoke", "test")


def test_example_configs_do_not_inline_secret_values() -> None:
    violations: list[str] = []
    for config_path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = _load_config(config_path)
        for dotted_path, value in _flatten(config):
            key = dotted_path.rsplit(".", 1)[-1]
            if key not in SENSITIVE_KEYS:
                continue
            if not isinstance(value, str) or not (
                value.startswith(ENV_PLACEHOLDER_PREFIX)
                and value.endswith(ENV_PLACEHOLDER_SUFFIX)
            ):
                violations.append(f"{config_path}:{dotted_path}={value!r}")

    assert violations == []


def test_fake_and_hash_providers_are_confined_to_smoke_or_test_configs() -> None:
    violations: list[str] = []
    for config_path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = _load_config(config_path)
        test_provider_paths = [
            dotted_path
            for dotted_path, value in _flatten(config)
            if dotted_path.endswith(".provider") and value in TEST_PROVIDER_NAMES
        ]
        if test_provider_paths and not config_path.stem.endswith(TEST_CONFIG_STEM_SUFFIXES):
            violations.extend(f"{config_path}:{path}" for path in test_provider_paths)

    assert violations == []


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(item, child_prefix))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_flatten(item, f"{prefix}[{index}]"))
        return rows
    return [(prefix, value)]
