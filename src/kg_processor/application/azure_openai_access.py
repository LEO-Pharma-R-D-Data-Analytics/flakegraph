"""Non-mutating Azure OpenAI/Foundry access checks.

This module intentionally uses the Azure CLI instead of an SDK dependency:
operators already use `az login` during local validation, and this keeps the
container runtime focused on FlakeGraph provider dependencies. The retrieved
API key is checked only for presence and is never included in the report.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class AzureOpenAIAccessCheck(BaseModel):
    """One non-mutating Azure OpenAI readiness check."""

    name: str
    ok: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class AzureOpenAIDeployment(BaseModel):
    """Deployment metadata returned by the Azure CLI access check."""

    name: str
    model: str | None = None
    version: str | None = None
    format: str | None = None
    sku: str | None = None
    capacity: int | None = None


class AzureOpenAIAccessReport(BaseModel):
    """Aggregate Azure OpenAI access report consumed by the CLI."""

    ok: bool
    subscription: str
    resource_group: str
    account: str
    endpoint: str | None = None
    deployments: list[AzureOpenAIDeployment] = Field(default_factory=list)
    checks: list[AzureOpenAIAccessCheck]


@dataclass(frozen=True)
class AzureCliResult:
    """Captured Azure CLI command result for production or test runners."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], int], AzureCliResult]


def run_azure_openai_access_check(
    *,
    subscription: str,
    resource_group: str,
    account: str,
    llm_deployment: str | None = None,
    embedding_deployment: str | None = None,
    command_runner: CommandRunner | None = None,
    timeout_seconds: int = 30,
) -> AzureOpenAIAccessReport:
    """Check Azure OpenAI account, deployments, and key visibility via `az`."""

    runner = command_runner or _subprocess_runner
    checks: list[AzureOpenAIAccessCheck] = []

    account_payload, account_check = _account_details(
        runner,
        timeout_seconds,
        subscription,
        resource_group,
        account,
    )
    checks.append(account_check)
    endpoint = _account_endpoint(account_payload)

    deployment_payload, deployment_check = _deployment_details(
        runner,
        timeout_seconds,
        subscription,
        resource_group,
        account,
    )
    checks.append(deployment_check)
    deployments = _normalize_deployments(deployment_payload)
    deployment_names = {deployment.name for deployment in deployments}

    if llm_deployment:
        checks.append(
            _deployment_presence_check("llm_deployment", llm_deployment, deployment_names)
        )
    if embedding_deployment:
        checks.append(
            _deployment_presence_check(
                "embedding_deployment",
                embedding_deployment,
                deployment_names,
            )
        )

    checks.append(
        _api_key_check(
            runner,
            timeout_seconds,
            subscription,
            resource_group,
            account,
        )
    )

    return AzureOpenAIAccessReport(
        ok=all(check.ok for check in checks),
        subscription=subscription,
        resource_group=resource_group,
        account=account,
        endpoint=endpoint,
        deployments=deployments,
        checks=checks,
    )


def _account_details(
    runner: CommandRunner,
    timeout_seconds: int,
    subscription: str,
    resource_group: str,
    account: str,
) -> tuple[dict[str, Any] | None, AzureOpenAIAccessCheck]:
    result = runner(
        [
            "az",
            "cognitiveservices",
            "account",
            "show",
            "--subscription",
            subscription,
            "--resource-group",
            resource_group,
            "--name",
            account,
            "-o",
            "json",
        ],
        timeout_seconds,
    )
    if result.returncode != 0:
        return None, _failed_cli_check(
            "account",
            "Azure AI Services account is not visible",
            result,
        )
    payload = _parse_json(result.stdout)
    if not isinstance(payload, dict):
        return None, _invalid_json_check("account")
    return payload, AzureOpenAIAccessCheck(
        name="account",
        ok=True,
        message="Azure AI Services account is visible",
        details={
            "kind": payload.get("kind"),
            "location": payload.get("location"),
            "sku": _nested(payload, "sku", "name"),
            "custom_subdomain": _nested(payload, "properties", "customSubDomainName"),
        },
    )


def _deployment_details(
    runner: CommandRunner,
    timeout_seconds: int,
    subscription: str,
    resource_group: str,
    account: str,
) -> tuple[list[dict[str, Any]] | None, AzureOpenAIAccessCheck]:
    result = runner(
        [
            "az",
            "cognitiveservices",
            "account",
            "deployment",
            "list",
            "--subscription",
            subscription,
            "--resource-group",
            resource_group,
            "--name",
            account,
            "-o",
            "json",
        ],
        timeout_seconds,
    )
    if result.returncode != 0:
        return None, _failed_cli_check(
            "deployments",
            "Azure OpenAI deployments are not visible",
            result,
        )
    payload = _parse_json(result.stdout)
    if not isinstance(payload, list):
        return None, _invalid_json_check("deployments")
    deployments = [item for item in payload if isinstance(item, dict)]
    return deployments, AzureOpenAIAccessCheck(
        name="deployments",
        ok=bool(deployments),
        message=(
            "Azure OpenAI deployments are visible"
            if deployments
            else "No Azure OpenAI deployments are visible"
        ),
        details={"count": len(deployments)},
    )


def _api_key_check(
    runner: CommandRunner,
    timeout_seconds: int,
    subscription: str,
    resource_group: str,
    account: str,
) -> AzureOpenAIAccessCheck:
    result = runner(
        [
            "az",
            "cognitiveservices",
            "account",
            "keys",
            "list",
            "--subscription",
            subscription,
            "--resource-group",
            resource_group,
            "--name",
            account,
            "--query",
            "key1",
            "-o",
            "tsv",
        ],
        timeout_seconds,
    )
    if result.returncode != 0:
        return _failed_cli_check("api_key", "Azure OpenAI API key is not readable", result)
    return AzureOpenAIAccessCheck(
        name="api_key",
        ok=bool(result.stdout.strip()),
        message=(
            "Azure OpenAI API key is readable"
            if result.stdout.strip()
            else "Azure OpenAI API key lookup returned an empty value"
        ),
        details={"key_name": "key1", "present": bool(result.stdout.strip())},
    )


def _deployment_presence_check(
    name: str,
    deployment: str,
    deployment_names: set[str],
) -> AzureOpenAIAccessCheck:
    visible = deployment in deployment_names
    return AzureOpenAIAccessCheck(
        name=name,
        ok=visible,
        message=(
            f"Deployment is visible: {deployment}"
            if visible
            else f"Deployment is not visible: {deployment}"
        ),
        details={"deployment": deployment},
    )


def _normalize_deployments(
    payload: list[dict[str, Any]] | None,
) -> list[AzureOpenAIDeployment]:
    deployments: list[AzureOpenAIDeployment] = []
    for item in payload or []:
        capacity = _nested(item, "sku", "capacity")
        deployments.append(
            AzureOpenAIDeployment(
                name=str(item.get("name", "")),
                model=_optional_string(_nested(item, "properties", "model", "name")),
                version=_optional_string(_nested(item, "properties", "model", "version")),
                format=_optional_string(_nested(item, "properties", "model", "format")),
                sku=_optional_string(_nested(item, "sku", "name")),
                capacity=capacity if isinstance(capacity, int) else None,
            )
        )
    return [deployment for deployment in deployments if deployment.name]


def _account_endpoint(payload: dict[str, Any] | None) -> str | None:
    endpoint = _nested(payload or {}, "properties", "endpoint")
    return endpoint if isinstance(endpoint, str) and endpoint else None


def _subprocess_runner(command: Sequence[str], timeout_seconds: int) -> AzureCliResult:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return AzureCliResult(returncode=127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        return AzureCliResult(
            returncode=124,
            stdout=_exception_text(exc.stdout),
            stderr=str(exc),
        )
    return AzureCliResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _failed_cli_check(
    name: str,
    message: str,
    result: AzureCliResult,
) -> AzureOpenAIAccessCheck:
    return AzureOpenAIAccessCheck(
        name=name,
        ok=False,
        message=message,
        details={
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
        },
    )


def _invalid_json_check(name: str) -> AzureOpenAIAccessCheck:
    return AzureOpenAIAccessCheck(
        name=name,
        ok=False,
        message=f"Azure CLI returned invalid JSON for {name}",
    )


def _parse_json(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _exception_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _nested(value: dict[str, Any], *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
