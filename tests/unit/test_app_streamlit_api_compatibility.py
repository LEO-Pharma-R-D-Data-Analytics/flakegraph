"""Guard the app against Streamlit APIs the deployment target does not have.

Streamlit-in-Snowflake runs the version pinned in ``app/environment.yml``, which
trails the latest release. Developing against a newer Streamlit silently accepts
parameters that do not exist there, and the failure surfaces only as a traceback
in a deployed app — ``st.popover(key=...)`` reached production exactly this way.

Pinning the local Streamlit to the deployed version makes the import-time API the
same one Snowflake offers, and this module fails the build if the pins drift or a
call uses a parameter the installed version does not accept.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
from typing import Any

import streamlit as st
import yaml

_APP_ROOT = pathlib.Path(__file__).resolve().parents[2] / "app"
_REPOSITORY_ROOT = _APP_ROOT.parent


def _environment_streamlit_pin() -> str:
    environment = yaml.safe_load((_APP_ROOT / "environment.yml").read_text(encoding="utf-8"))
    for dependency in environment["dependencies"]:
        text = str(dependency)
        if text.startswith("streamlit="):
            return text.split("=", 1)[1]
    raise AssertionError("app/environment.yml must pin streamlit")


def test_local_streamlit_matches_the_deployed_version() -> None:
    """What the tests exercise must be what Snowflake actually runs."""

    pyproject = (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pinned = set(re.findall(r'"streamlit==([0-9][^"]*)"', pyproject))

    assert pinned, "pyproject must pin streamlit for the app and dev extras"
    assert pinned == {_environment_streamlit_pin()}
    assert st.__version__ == _environment_streamlit_pin()


def _unsupported_calls() -> list[str]:
    findings: list[str] = []
    for path in sorted(_APP_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute):
                continue
            value = function.value
            if not isinstance(value, ast.Name) or value.id != "st":
                continue
            target: Any = getattr(st, function.attr, None)
            if target is None:
                findings.append(f"{path.name}:{node.lineno}: st.{function.attr} does not exist")
                continue
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
                continue
            for keyword in node.keywords:
                if keyword.arg and keyword.arg not in signature.parameters:
                    findings.append(
                        f"{path.name}:{node.lineno}: "
                        f"st.{function.attr}(..., {keyword.arg}=) is not supported"
                    )
    return findings


def test_every_streamlit_call_is_supported_by_the_deployed_version() -> None:
    """A parameter the deployed Streamlit lacks is a runtime crash, not a warning."""

    assert _unsupported_calls() == []
