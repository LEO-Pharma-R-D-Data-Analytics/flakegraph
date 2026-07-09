from __future__ import annotations

import json
from collections.abc import Sequence

from kg_processor.application.azure_openai_access import (
    AzureCliResult,
    run_azure_openai_access_check,
)


def test_azure_openai_access_check_reports_account_deployments_and_key_without_secret() -> None:
    report = run_azure_openai_access_check(
        subscription="sub",
        resource_group="rg",
        account="ai-account",
        llm_deployment="gpt-4.1-mini-2025-04-14",
        embedding_deployment="text-embedding-3-small",
        command_runner=_successful_runner,
    )

    assert report.ok
    assert report.endpoint == "https://example.cognitiveservices.azure.com/"
    assert [deployment.name for deployment in report.deployments] == [
        "gpt-4.1-mini-2025-04-14",
        "text-embedding-3-small",
    ]
    assert {check.name for check in report.checks} == {
        "account",
        "deployments",
        "llm_deployment",
        "embedding_deployment",
        "api_key",
    }
    assert "super-secret-key" not in report.model_dump_json()


def test_azure_openai_access_check_reports_missing_deployment_and_key_failure() -> None:
    report = run_azure_openai_access_check(
        subscription="sub",
        resource_group="rg",
        account="ai-account",
        llm_deployment="missing-chat",
        embedding_deployment="text-embedding-3-small",
        command_runner=_key_failure_runner,
    )

    assert not report.ok
    missing = next(check for check in report.checks if check.name == "llm_deployment")
    key = next(check for check in report.checks if check.name == "api_key")
    assert missing.details == {"deployment": "missing-chat"}
    assert key.details["returncode"] == 1
    assert "not authorized" in key.details["stderr"]


def test_azure_openai_access_check_reports_invalid_account_json() -> None:
    report = run_azure_openai_access_check(
        subscription="sub",
        resource_group="rg",
        account="ai-account",
        command_runner=_invalid_account_runner,
    )

    assert not report.ok
    assert report.endpoint is None
    account = next(check for check in report.checks if check.name == "account")
    assert account.message == "Azure CLI returned invalid JSON for account"


def _successful_runner(command: Sequence[str], _timeout_seconds: int) -> AzureCliResult:
    command_text = " ".join(command)
    if " account show " in f" {command_text} ":
        return AzureCliResult(
            returncode=0,
            stdout=json.dumps(
                {
                    "kind": "AIServices",
                    "location": "swedencentral",
                    "properties": {
                        "endpoint": "https://example.cognitiveservices.azure.com/",
                        "customSubDomainName": "example",
                    },
                    "sku": {"name": "S0"},
                }
            ),
            stderr="",
        )
    if " deployment list " in f" {command_text} ":
        return AzureCliResult(returncode=0, stdout=json.dumps(_deployments()), stderr="")
    if " keys list " in f" {command_text} ":
        return AzureCliResult(returncode=0, stdout="super-secret-key\n", stderr="")
    raise AssertionError(f"Unexpected command: {command_text}")


def _key_failure_runner(command: Sequence[str], timeout_seconds: int) -> AzureCliResult:
    if " keys list " in f" {' '.join(command)} ":
        return AzureCliResult(returncode=1, stdout="", stderr="not authorized")
    return _successful_runner(command, timeout_seconds)


def _invalid_account_runner(command: Sequence[str], timeout_seconds: int) -> AzureCliResult:
    if " account show " in f" {' '.join(command)} ":
        return AzureCliResult(returncode=0, stdout="not-json", stderr="")
    return _successful_runner(command, timeout_seconds)


def _deployments() -> list[dict[str, object]]:
    return [
        {
            "name": "gpt-4.1-mini-2025-04-14",
            "properties": {
                "model": {
                    "name": "gpt-4.1-mini",
                    "version": "2025-04-14",
                    "format": "OpenAI",
                }
            },
            "sku": {"name": "Standard", "capacity": 2000},
        },
        {
            "name": "text-embedding-3-small",
            "properties": {
                "model": {
                    "name": "text-embedding-3-small",
                    "version": "1",
                    "format": "OpenAI",
                }
            },
            "sku": {"name": "GlobalStandard", "capacity": 10000},
        },
    ]
