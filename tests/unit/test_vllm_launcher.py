"""Protect the concise quick-start launcher and its validated serving profile."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_qwen36_launcher_passes_pinned_defaults_and_environment_overrides(
    tmp_path: Path,
) -> None:
    """Exercise the shell boundary without loading vLLM or an actual model."""

    arguments_path = tmp_path / "arguments.txt"
    environment_path = tmp_path / "environment.txt"
    fake_vllm = tmp_path / "vllm"
    fake_vllm.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$ARGUMENTS_PATH"\n'
        'printf \'%s\\n\' "$VLLM_MARLIN_USE_ATOMIC_ADD" > "$ENVIRONMENT_PATH"\n',
        encoding="utf-8",
    )
    fake_vllm.chmod(0o755)

    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "ARGUMENTS_PATH": str(arguments_path),
        "ENVIRONMENT_PATH": str(environment_path),
        "VLLM_PORT": "9000",
        "VLLM_MAX_NUM_SEQS": "2",
    }
    subprocess.run(
        ["bash", "deploy/vllm/serve-qwen36.sh"],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    arguments = arguments_path.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["serve", "nvidia/Qwen3.6-35B-A3B-NVFP4"]
    assert _option(arguments, "--revision") == "491c2f1ea524c639598bf8fa787a93fed5a6fbce"
    assert _option(arguments, "--port") == "9000"
    assert _option(arguments, "--max-num-seqs") == "2"
    assert _option(arguments, "--max-num-batched-tokens") == "32768"
    assert _option(arguments, "--gpu-memory-utilization") == "0.50"
    assert "--enable-prefix-caching" in arguments
    assert environment_path.read_text(encoding="utf-8").strip() == "1"


def _option(arguments: list[str], name: str) -> str:
    """Return the value immediately following a required launcher option."""

    index = arguments.index(name)
    return arguments[index + 1]
