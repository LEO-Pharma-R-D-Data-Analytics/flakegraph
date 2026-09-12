"""Pure Snowpark Container Services job-spec composition for the Streamlit app."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

import yaml
from flakegraph_app.configuration import sensitive_config_key

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_IMAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9._/-]*(?::[A-Za-z0-9_.-]+)?$")
_IMAGE_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_QUALIFIED_OBJECT_PARTS = 3


@dataclass(frozen=True)
class SpcsLaunch:
    """Contain one validated staged specification and its execution coordinates."""

    spec_yaml: str
    spec_stage: str
    spec_file: str
    compute_pool: str
    service_identifier: str
    runtime_config: dict[str, Any]

    @property
    def execute_sql(self) -> str:
        """Render the fixed-shape SQL that starts this asynchronous job service."""

        return (
            "EXECUTE JOB SERVICE\n"
            f"  IN COMPUTE POOL {self.compute_pool}\n"
            f"  NAME = {self.service_identifier}\n"
            "  ASYNC = TRUE\n"
            f"  FROM {self.spec_stage}\n"
            f"  SPEC = '{self.spec_file}';"
        )


def service_name_for_job(job_id: str) -> str:
    """Return the SPCS job service a run is executed by.

    Derived from the run alone so anything holding a run id can find the service
    that carries its logs and its terminal state, without having to rebuild the
    run's configuration first. Defined once because two callers now depend on it
    agreeing exactly: submission creates the service, and status reconciles a run
    against it.
    """

    return f"FLAKEGRAPH_APP_{hashlib.sha256(job_id.encode()).hexdigest()[:16].upper()}"


def build_spcs_launch(config: dict[str, Any]) -> SpcsLaunch:
    """Build a credential-free SPCS job from the app's effective run config."""

    job = _mapping(config, "job")
    snowflake = _mapping(config, "snowflake")
    job_id = _required_text(job, "job_id", "job")
    service_name = service_name_for_job(job_id)
    database = _identifier(_required_text(snowflake, "database", "snowflake"), "database")
    schema = _identifier(_required_text(snowflake, "schema", "snowflake"), "schema")
    compute_pool = _identifier(
        _required_text(snowflake, "compute_pool", "snowflake"),
        "compute pool",
    )
    spec_stage = _stage(_required_text(snowflake, "service_spec_stage", "snowflake"))
    image = _image_reference(snowflake)

    runtime_config = _without_secrets(config)
    runtime_job = _mapping(runtime_config, "job")
    runtime_job["lease_owner"] = service_name.lower()
    runtime_snowflake = _mapping(runtime_config, "snowflake")
    # SPCS supplies connection coordinates paired with its short-lived OAuth
    # token. Client-side values can point at another account host or principal,
    # so they must not override the service identity inside the container.
    for key in (
        "account",
        "host",
        "user",
        "password",
        "authenticator",
        "private_key_path",
        "oauth_token",
        "oauth_token_path",
    ):
        runtime_snowflake.pop(key, None)
    runtime_snowflake["service_name"] = service_name
    runtime_snowflake["authenticator"] = "oauth"
    runtime_snowflake["oauth_token_path"] = "/snowflake/session/token"

    env = {
        "KG_CONFIG_FILE": "configs/snowflake-cortex.yaml",
        "KG_CONFIG_JSON": json.dumps(
            runtime_config,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "KG_RUNTIME": "spcs",
        "KG_SNOWFLAKE_AUTHENTICATOR": "oauth",
        "KG_SNOWFLAKE_OAUTH_TOKEN_PATH": "/snowflake/session/token",
    }
    resources = {
        "requests": {
            "cpu": str(snowflake.get("service_cpu_request") or "500m"),
            "memory": str(snowflake.get("service_memory_request") or "3Gi"),
        },
        "limits": {
            "cpu": str(snowflake.get("service_cpu_limit") or "1"),
            "memory": str(snowflake.get("service_memory_limit") or "5Gi"),
        },
    }
    spec = {
        "spec": {
            "containers": [
                {
                    "name": "flakegraph",
                    "image": image,
                    "args": ["worker", "--config", "configs/snowflake-cortex.yaml"],
                    "env": env,
                    "resources": resources,
                }
            ]
        }
    }
    spec_file = f"flakegraph-app-{hashlib.sha256(job_id.encode()).hexdigest()[:16]}.yaml"
    return SpcsLaunch(
        spec_yaml=yaml.safe_dump(spec, sort_keys=False),
        spec_stage=spec_stage,
        spec_file=spec_file,
        compute_pool=compute_pool,
        service_identifier=f"{database}.{schema}.{service_name}",
        runtime_config=runtime_config,
    )


def _without_secrets(value: Any) -> Any:
    """Recursively remove credential-bearing keys from a runtime payload."""

    if isinstance(value, dict):
        return {
            str(key): _without_secrets(item)
            for key, item in value.items()
            if not sensitive_config_key(str(key))
        }
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one mutable configuration section or fail with a precise label."""

    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Snowflake app configuration requires a {key} mapping")
    return value


def _required_text(parent: dict[str, Any], key: str, section: str) -> str:
    """Return one required non-empty scalar without accepting structured values."""

    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Snowflake app configuration requires {section}.{key}")
    return value.strip()


def _identifier(value: str, label: str) -> str:
    """Validate an unquoted Snowflake identifier used in generated SQL."""

    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Snowflake {label} must be an unquoted identifier")
    return value


def _stage(value: str) -> str:
    """Validate a fully qualified internal stage with no path component."""

    normalized = value.strip().lstrip("@")
    parts = normalized.split(".")
    if len(parts) != _QUALIFIED_OBJECT_PARTS or any(
        not _IDENTIFIER.fullmatch(part) for part in parts
    ):
        raise ValueError("Snowflake service_spec_stage must be @DATABASE.SCHEMA.STAGE")
    return "@" + ".".join(parts)


def _image_reference(snowflake: dict[str, Any]) -> str:
    """Build the SPCS repository path and optional immutable digest."""

    repository = _required_text(snowflake, "image_repository", "snowflake").lstrip("/")
    parts = repository.replace("/", ".").split(".")
    if len(parts) != _QUALIFIED_OBJECT_PARTS or any(
        not _IDENTIFIER.fullmatch(part) for part in parts
    ):
        raise ValueError("Snowflake image_repository must be DATABASE.SCHEMA.REPOSITORY")
    image_name = _required_text(snowflake, "image_name", "snowflake")
    if not _IMAGE_NAME.fullmatch(image_name):
        raise ValueError("Snowflake image_name is not a safe OCI image and tag")
    digest = str(snowflake.get("image_digest") or "").strip().lower()
    if digest and not _IMAGE_DIGEST.fullmatch(digest):
        raise ValueError("Snowflake image_digest must be sha256 followed by 64 hex digits")
    suffix = f"@{digest}" if digest else ""
    return f"/{'/'.join(parts)}/{image_name}{suffix}"
