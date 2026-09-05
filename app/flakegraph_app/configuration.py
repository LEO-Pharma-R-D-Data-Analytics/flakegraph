"""Configuration composition for application-submitted FlakeGraph runs."""

from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml
from flakegraph_app.models import IngestionRequest, SourceKind, StorageKind
from flakegraph_app.providers import embedding_dimension

# Every stage that calls a model is bounded by the same budget, and the stages
# run one after another, so each may use all of it.
_PARALLELISM_SETTINGS = (
    "extraction_parallelism",
    "resolution_parallelism",
    "community_report_parallelism",
    "description_merge_parallelism",
)
# Leave the node a little room for the runtime rather than requesting all of it.
_NODE_CPU_HEADROOM = 1.0
_NODE_MEMORY_HEADROOM_GIB = 2.0

_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ENVIRONMENT_REFERENCE = re.compile(r"^\$\{[A-Z_][A-Z0-9_]*\}$")
_SENSITIVE_MARKERS = (
    "password",
    "token",
    "secret",
    "api_key",
    "private_key",
    "connection_string",
    "database_url",
)
_SENSITIVE_EXACT_KEYS = frozenset({"dsn"})
_NON_SECRET_REFERENCE_SUFFIXES = (
    "_environment_variable",
    "_path",
    "_name",
    "_header",
    "_prefix",
)
# A credential is as often carried inside a value as named by its key: a blob
# account URL ends in a SAS signature, a database URL carries userinfo, an OCR
# endpoint can end in ``?api_key=``. Those keys are ``account_url``, ``endpoint``
# and ``url``, none of which reads as sensitive, so scalar URLs are examined as
# well. Deliberately reimplemented here rather than imported from
# ``kg_processor``: the deployed Streamlit bundle ships only this package, and
# importing the product breaks the whole application inside Snowflake with "No
# module named 'kg_processor'".
_ABSOLUTE_URL = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
# The short Azure parameters below are ordinary words elsewhere — ``sp``, ``sr``
# and ``sv`` appear in perfectly innocent query strings — so they are only
# treated as a shared access signature when the signature itself is present.
_SAS_SIGNATURE = re.compile(r"[?&]sig=", flags=re.IGNORECASE)
_SAS_QUERY_PARAMETER = re.compile(
    r"([?&](?:sig|se|sp|spr|sr|st|sv|skoid|sktid|skt|ske|sks|skv)=)[^&#\s]+",
    flags=re.IGNORECASE,
)
# The parameter name must be a credential in full. A suffix match would treat an
# ordinary ``next_token`` cursor as a secret and, through rejection, refuse to
# start a run over a pagination parameter.
# Named exhaustively rather than by substring: a parameter called ``next_token``
# is a pagination cursor, and rejecting a run over one would block ordinary work.
# The vocabulary covers the shapes real providers use to carry a credential in a
# query string — Google (``key``), Azure (``subscription-key``), GitLab
# (``private_token``) — and the AWS SigV4 set, all of which a name-suffix match
# would miss.
_SECRET_QUERY_PARAMETER = re.compile(
    r"([?&](?:"
    r"access[_-]?key(?:[_-]?id)?|access[_-]?token|api[_-]?key|apikey|auth|auth[_-]?token|"
    r"authorization|aws[_-]?access[_-]?key[_-]?id|client[_-]?secret|credential|key|"
    r"passwd|password|private[_-]?token|secret|secret[_-]?access[_-]?key|"
    r"session[_-]?token|sig|signature|subscription[_-]?key|token|"
    r"x-amz-credential|x-amz-security-token|x-amz-signature"
    r")=)[^&#\s]+",
    flags=re.IGNORECASE,
)
_URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]*:)[^\s/@]+(@)", re.IGNORECASE)
_REDACTED_VALUE = "${REDACTED}"


def load_base_config(path: Path | None) -> dict[str, Any]:
    """Load an optional YAML profile without reparsing unchanged files on reruns."""

    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return copy.deepcopy(_load_base_config_cached(str(resolved), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=32)
def _load_base_config_cached(
    resolved_path: str,
    modified_ns: int,
    size_bytes: int,
) -> dict[str, Any]:
    """Parse one immutable file revision identified by path, mtime, and size."""

    del modified_ns, size_bytes
    loaded = yaml.safe_load(Path(resolved_path).read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("FlakeGraph configuration root must be a mapping")
    return {str(key): value for key, value in loaded.items()}


def build_run_config(request: IngestionRequest) -> dict[str, Any]:
    """Overlay form selections on a reusable base profile.

    Provider credentials are represented as environment placeholders. The app
    never serializes secret values into generated YAML, logs, or session state.
    """

    _reject_literal_secrets(request.source, path="source")
    config = copy.deepcopy(load_base_config(request.base_config_path))
    # A base profile may reference an ontology by path. The container receives
    # only this configuration and the image, so the path dangles there and the
    # worker aborts before claiming a file. Resolved here against the profile it
    # came from, while that file is still readable.
    _inline_ontology_profile(config, _ontology_profile_path(config, request.base_config_path))
    _reject_literal_secrets(config, path="base configuration")
    embedding_config = _provider_config(request.embedding)
    embedding_config["dimension"] = embedding_dimension(
        request.embedding.provider,
        request.embedding.model,
        request.embedding.dimension,
    )
    overrides: dict[str, Any] = {
        "runtime": {"runtime": _runtime_value(request)},
        "job": {
            "job_id": request.job_id,
            "graph_id": request.graph_id,
            "use_file_queue": request.runtime.value == "Snowflake",
        },
        "files": {
            "include_globs": list(request.include_globs),
        },
        "ocr": _ocr_provider_config(request.ocr),
        "llm": _provider_config(request.llm),
        "embedding": embedding_config,
        "writer": {
            "provider": request.output.writer_provider,
            "output_path": str(request.output.workspace_path),
        },
        "cache": {
            "provider": request.cache_provider,
            "path": str(request.output.workspace_path.parent / "cache"),
        },
    }
    if request.output.kind == StorageKind.SNOWFLAKE:
        _deep_merge(overrides, _snowflake_output_config(request))
    # Concurrency is a property of the pipeline, not of the runtime hosting it: a
    # stage waiting on a completion holds a socket, not a core. Every runtime
    # therefore carries the operator's budget into the stages that spend it.
    _deep_merge(
        overrides,
        {"graph": provider_parallelism_settings(request.provider_parallelism)},
    )
    if request.runtime.value == "Snowflake" and request.runtime_options:
        runtime_options = {
            key: value
            for key, value in request.runtime_options.items()
            if not isinstance(value, str) or value.strip()
        }
        if runtime_options:
            _deep_merge(overrides, {"snowflake": runtime_options})
    if request.ocr.provider == "generic_http":
        overrides["generic_http_ocr"] = _external_ocr_config(request.ocr)
    _deep_merge(overrides, _source_config(request))
    _deep_merge(config, overrides)
    _sanitize_provider_sections(config, request)
    _reject_literal_secrets(config)
    return config


def run_ontology_profile(request: IngestionRequest) -> dict[str, Any] | None:
    """Return the ontology a run will carry, resolved the way submission resolves it.

    Preflight needs the same answer the submitted config will contain, and the
    ontology reaches that config by being read from the base profile's path and
    inlined. Re-deriving it here rather than re-reading the file keeps the two
    from disagreeing about which profile a run actually uses.
    """

    config = build_run_config(request)
    ontology = config.get("ontology")
    if not isinstance(ontology, Mapping):
        return None
    profile = ontology.get("profile")
    return dict(profile) if isinstance(profile, Mapping) else None


def provider_parallelism_settings(parallelism: int) -> dict[str, int]:
    """Apply one concurrency budget to every stage that calls a model."""

    bounded = max(1, min(int(parallelism), 64))
    return dict.fromkeys(_PARALLELISM_SETTINGS, bounded)


def container_resources(capacity: Mapping[str, float]) -> dict[str, str]:
    """Size the worker container from the node its compute pool provides.

    A job service occupies a node on its own and SPCS bills that node whether the
    container asks for it or not, so requesting a fraction buys nothing. An
    unreadable or implausible capacity returns nothing, leaving the caller's
    defaults in place rather than proposing a container the pool cannot schedule.
    """

    vcpu = float(capacity.get("vcpu", 0) or 0)
    memory_gib = float(capacity.get("memory_gib", 0) or 0)
    if vcpu <= 0 or memory_gib <= 0:
        return {}
    usable_cpu = max(1.0, vcpu - _NODE_CPU_HEADROOM)
    usable_memory = max(1.0, memory_gib - _NODE_MEMORY_HEADROOM_GIB)
    return {
        "service_cpu_request": f"{usable_cpu:g}",
        "service_cpu_limit": f"{vcpu:g}",
        "service_memory_request": f"{usable_memory:g}Gi",
        "service_memory_limit": f"{usable_memory:g}Gi",
    }


def write_run_config(request: IngestionRequest, destination: Path) -> Path:
    """Persist one generated non-secret run profile and return its path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(build_run_config(request), sort_keys=False),
        encoding="utf-8",
    )
    return destination


def redacted_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursively redacted copy suitable for configuration previews.

    Connection strings and database URLs are credentials even when their key
    names do not include words such as ``password``. A credential inside a URL
    value is redacted from the value itself, because the key that holds it is
    ordinarily named for the service rather than for the secret. Reference
    metadata such as environment-variable names and token file paths remains
    visible for useful preflight review.
    """

    def redact(value: Any, key: str = "") -> Any:
        if isinstance(value, Mapping):
            return {str(item_key): redact(item, str(item_key)) for item_key, item in value.items()}
        if isinstance(value, list):
            return [redact(item, key) for item in value]
        if sensitive_config_key(key):
            return _REDACTED_VALUE if value else value
        if isinstance(value, str):
            return redacted_url(value)
        return value

    return cast(dict[str, Any], redact(dict(config)))


def redacted_url(value: str) -> str:
    """Strip credentials from a scalar URL without rewriting ordinary prose.

    Only absolute URLs are examined. Descriptions, quotes, prompts, and file
    names legitimately contain words that resemble credentials, so a value is
    rewritten only where its shape says it addresses a service.
    """

    if not _ABSOLUTE_URL.match(value):
        return value
    redacted = (
        _SAS_QUERY_PARAMETER.sub(r"\1***", value)
        if _SAS_SIGNATURE.search(value)
        else value
    )
    redacted = _SECRET_QUERY_PARAMETER.sub(r"\1***", redacted)
    return _URL_PASSWORD.sub(r"\1***\2", redacted)


def effective_request_fingerprint(
    request: IngestionRequest,
    effective_config: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable approval identity for the exact effective request.

    The digest covers the composed profile plus request fields that a future
    configuration translator might intentionally omit. It includes credential
    reference names but never environment values, allowing approval comparison
    without retaining secrets in Streamlit session state.
    """

    config = dict(effective_config) if effective_config is not None else build_run_config(request)
    identity = {
        "effective_config": config,
        "request": {
            "runtime": request.runtime.value,
            "job_id": request.job_id,
            "graph_id": request.graph_id,
            "source_kind": request.source_kind.value,
            "source": dict(request.source),
            "base_config_path": str(request.base_config_path) if request.base_config_path else None,
            "include_globs": list(request.include_globs),
            "cache_provider": request.cache_provider,
            "runtime_options": dict(request.runtime_options),
            "ocr": _selection_identity(request.ocr),
            "llm": _selection_identity(request.llm),
            "embedding": _selection_identity(request.embedding),
            "output_kind": request.output.kind.value,
            "output_location": request.output.location,
        },
    }
    canonical = yaml.safe_dump(identity, sort_keys=True, allow_unicode=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sensitive_config_key(key: str) -> bool:
    """Identify keys whose values can contain literal authentication material."""

    normalized = key.strip().lower()
    if normalized.endswith(_NON_SECRET_REFERENCE_SUFFIXES):
        return False
    return normalized in _SENSITIVE_EXACT_KEYS or any(
        marker in normalized for marker in _SENSITIVE_MARKERS
    )


def environment_for_request(request: IngestionRequest) -> dict[str, str]:
    """Return inherited environment values required by selected credential references."""

    environment = dict(os.environ)
    for selection in (request.ocr, request.llm, request.embedding):
        name = selection.api_key_environment_variable
        if name and name not in environment:
            raise ValueError(f"Credential environment variable is not set: {name}")
    for name in request.output.credential_environment_variables:
        if name not in environment:
            raise ValueError(f"Credential environment variable is not set: {name}")
    return environment


def _source_config(request: IngestionRequest) -> dict[str, Any]:
    """Translate one source selection into the processing-core settings shape."""

    source = dict(request.source)
    if request.source_kind in {SourceKind.UPLOAD, SourceKind.LOCAL}:
        return {"files": {"source": "local", "input_path": str(source["path"])}}
    if request.source_kind == SourceKind.AZURE_BLOB:
        return {
            "files": {"source": "azure_blob"},
            "azure_blob": {
                "account_url": source.get("account_url"),
                "container": source.get("container"),
                "prefix": source.get("prefix"),
                "download_path": str(request.output.workspace_path.parent / "source-cache"),
            },
        }
    if request.source_kind == SourceKind.S3:
        return {
            "files": {"source": "s3"},
            "s3": {
                "bucket": source.get("bucket"),
                "prefix": source.get("prefix"),
                "endpoint_url": source.get("endpoint_url"),
                "region": source.get("region"),
                "download_path": str(request.output.workspace_path.parent / "source-cache"),
            },
        }
    if request.source_kind == SourceKind.SNOWFLAKE_STAGE:
        return {
            "files": {
                "source": "snowflake_stage",
                "stage_prefix": source.get("prefix"),
            },
            "snowflake": {"stage": source.get("stage")},
        }
    raise ValueError(f"Unsupported source kind: {request.source_kind.value}")


def _provider_config(selection: Any) -> dict[str, Any]:
    """Translate one provider selection and validated credential reference."""

    result: dict[str, Any] = {"provider": selection.provider, **dict(selection.options)}
    if selection.model:
        result["model"] = selection.model
    if selection.endpoint:
        result["endpoint"] = selection.endpoint
    if selection.api_key_environment_variable:
        name = selection.api_key_environment_variable.strip()
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"Invalid credential environment variable name: {name}")
        result["api_key"] = f"${{{name}}}"
    return result


def _sanitize_provider_sections(
    config: dict[str, Any],
    request: IngestionRequest,
) -> None:
    """Retain neutral tuning while replacing stale provider-owned connection fields."""

    config["llm"] = _provider_section(
        config.get("llm"),
        _provider_config(request.llm),
        allowed={"timeout_seconds"}
        | ({"api_version"} if request.llm.provider == "azure_openai" else set()),
    )
    embedding = _provider_config(request.embedding)
    embedding["dimension"] = embedding_dimension(
        request.embedding.provider,
        request.embedding.model,
        request.embedding.dimension,
    )
    embedding_allowed = {"batch_size"}
    if request.embedding.provider == "sentence_transformers":
        embedding_allowed.add("device")
    if request.embedding.provider == "azure_openai":
        embedding_allowed.add("api_version")
    config["embedding"] = _provider_section(
        config.get("embedding"),
        embedding,
        allowed=embedding_allowed,
    )

    ocr_allowed = {"language", "page_range", "timeout_seconds"}
    ocr_prefixes: tuple[str, ...] = ()
    if request.ocr.provider == "fallback":
        ocr_prefixes = ("fallback_", "mineru_", "tesseract_")
        ocr_allowed.add("model_cache_dir")
    elif request.ocr.provider == "mineru_internal":
        ocr_prefixes = ("mineru_",)
        ocr_allowed.add("model_cache_dir")
    elif request.ocr.provider == "tesseract_internal":
        ocr_prefixes = ("tesseract_",)
    elif request.ocr.provider == "snowflake_cortex":
        ocr_prefixes = ("snowflake_",)
    config["ocr"] = _provider_section(
        config.get("ocr"),
        _ocr_provider_config(request.ocr),
        allowed=ocr_allowed,
        allowed_prefixes=ocr_prefixes,
        excluded={"mineru_api_url", "mineru_api_key"},
    )

    if request.ocr.provider == "generic_http":
        existing = config.get("generic_http_ocr")
        neutral = (
            {
                str(key): value
                for key, value in existing.items()
                if isinstance(existing, Mapping) and key not in {"endpoint", "api_key"}
            }
            if isinstance(existing, Mapping)
            else {}
        )
        neutral.update(_external_ocr_config(request.ocr))
        config["generic_http_ocr"] = neutral
    else:
        config.pop("generic_http_ocr", None)


def _provider_section(
    existing: Any,
    selected: Mapping[str, Any],
    *,
    allowed: set[str],
    allowed_prefixes: tuple[str, ...] = (),
    excluded: set[str] | None = None,
) -> dict[str, Any]:
    """Merge selected fields over a narrow allowlist of provider-neutral tuning."""

    blocked = excluded or set()
    retained: dict[str, Any] = {}
    if isinstance(existing, Mapping):
        retained = {
            str(key): value
            for key, value in existing.items()
            if str(key) not in blocked
            and (
                str(key) in allowed
                or any(str(key).startswith(prefix) for prefix in allowed_prefixes)
            )
            and not sensitive_config_key(str(key))
        }
    retained.update(selected)
    return retained


def _reject_literal_secrets(value: Any, *, path: str = "configuration") -> None:
    """Reject literal credentials while allowing validated environment placeholders.

    Both halves of a credential are rejected: a sensitive key holding anything
    other than an environment reference, and a URL value carrying a shared-access
    signature, userinfo password, or secret query parameter whatever its key is
    called.
    """

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            item_path = f"{path}.{key}"
            if (
                sensitive_config_key(key)
                and item not in (None, "")
                and (not isinstance(item, str) or not _ENVIRONMENT_REFERENCE.fullmatch(item))
            ):
                raise ValueError(
                    f"Literal secret is not allowed in app configuration: {item_path}; "
                    "use an environment variable reference"
                )
            _reject_literal_secrets(item, path=item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_literal_secrets(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and redacted_url(value) != value:
        raise ValueError(
            f"Literal secret is not allowed in app configuration: {path}; "
            "the URL carries a credential — use an environment variable reference"
        )


def _selection_identity(selection: Any) -> dict[str, Any]:
    """Serialize one non-secret provider selection for approval fingerprinting."""

    return {
        "provider": selection.provider,
        "model": selection.model,
        "endpoint": selection.endpoint,
        "api_key_environment_variable": selection.api_key_environment_variable,
        "dimension": selection.dimension,
        "options": dict(selection.options),
    }


def _snowflake_output_config(request: IngestionRequest) -> dict[str, Any]:
    """Translate a Snowflake destination without serializing credential values."""

    target = request.output.snowflake
    if target is None:
        raise ValueError("Snowflake output requires Snowflake connection settings")
    snowflake: dict[str, Any] = {
        "account": target.account,
        "host": target.host,
        "user": target.user,
        "authenticator": target.authenticator,
        "database": target.database,
        "schema": target.schema,
        "role": target.role,
        "warehouse": target.warehouse,
        "bulk_stage": target.bulk_stage,
    }
    if target.credential_field and target.credential_environment_variable:
        snowflake[target.credential_field] = _environment_reference(
            target.credential_environment_variable
        )
        snowflake[f"{target.credential_field}_environment_variable"] = (
            target.credential_environment_variable
        )
    return {"snowflake": {key: value for key, value in snowflake.items() if value}}


def _ocr_provider_config(selection: Any) -> dict[str, Any]:
    """Map OCR transport fields to their provider-specific settings names."""

    result = {"provider": selection.provider, **dict(selection.options)}
    if selection.provider == "mineru_api":
        if selection.endpoint:
            result["mineru_api_url"] = selection.endpoint
        if selection.api_key_environment_variable:
            result["mineru_api_key"] = _environment_reference(
                selection.api_key_environment_variable
            )
    return result


def _external_ocr_config(selection: Any) -> dict[str, Any]:
    """Build the separate generic HTTP OCR response-mapping configuration."""

    result: dict[str, Any] = {}
    if selection.endpoint:
        result["endpoint"] = selection.endpoint
    if selection.api_key_environment_variable:
        result["api_key"] = _environment_reference(selection.api_key_environment_variable)
    return result


def _environment_reference(name: str) -> str:
    """Validate and render one secret-bearing environment placeholder."""

    normalized = name.strip()
    if not _ENVIRONMENT_NAME.fullmatch(normalized):
        raise ValueError(f"Invalid credential environment variable name: {normalized}")
    return f"${{{normalized}}}"


def _runtime_value(request: IngestionRequest) -> str:
    """Map UI labels to the runtime literals understood by the processing core."""

    return {
        "Local": "local",
        "Kubernetes": "kubernetes",
        "Snowflake": "spcs",
    }[request.runtime.value]


def _deep_merge(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    """Recursively merge mappings while replacing scalar and sequence leaves."""

    for key, value in source.items():
        current = target.get(key)
        if isinstance(current, MutableMapping) and isinstance(value, Mapping):
            _deep_merge(current, value)
        else:
            target[key] = copy.deepcopy(value)


def _ontology_profile_path(config: Mapping[str, Any], base_config_path: Path | None) -> Path | None:
    """Resolve a profile's ontology path relative to the repository, if it has one."""

    ontology = config.get("ontology")
    if not isinstance(ontology, Mapping):
        return None
    raw = ontology.get("profile_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    # The referenced file is read and inlined into a configuration rendered back
    # into the browser, so it must be one of the application's own profiles. An
    # absolute or parent-escaping reference addresses somewhere else entirely,
    # and a search that walks to the filesystem root reaches the same places by
    # another route, so both the reference and the result are contained.
    if base_config_path is None or candidate.is_absolute() or ".." in candidate.parts:
        return None
    roots = _profile_roots(base_config_path)
    if candidate.is_file() and _within(candidate, roots):
        return candidate
    for parent in Path(base_config_path).resolve().parents:
        if not _within(parent, roots):
            break
        resolved = parent / candidate
        if resolved.is_file():
            return resolved
    return None


def _profile_roots(base_config_path: Path) -> list[Path]:
    """Return the directories an ontology reference may resolve inside.

    The application package and whichever checkout the profile came from are the
    only trees that hold reviewed profiles.
    """

    resolved = Path(base_config_path).resolve()
    # The profile's own directory, so an ontology stored beside it resolves, and
    # the checkout it belongs to, so a profile under ``configs/`` can name a path
    # relative to the repository root as those profiles do.
    roots = [Path(__file__).resolve().parents[1], resolved.parent]
    for parent in resolved.parents:
        if (parent / "configs").is_dir() or (parent / "pyproject.toml").is_file():
            roots.append(parent)
            break
    return roots


def _within(path: Path, roots: Sequence[Path]) -> bool:
    """Return whether a resolved path lies inside one of the permitted roots."""

    resolved = path.resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in roots)


def resolve_profile_path(raw: str, *roots: Path) -> Path:
    """Resolve an operator-supplied configuration profile inside a permitted root.

    The field names one of the application's own reviewed profiles, not an
    arbitrary readable file. Its contents are parsed and rendered back into the
    browser, so without a boundary the control makes every file the app's
    identity can read — credential stores, key material, container registry
    logins — into a disclosure surface. Symlinks are followed before the check so
    a link inside a root cannot point out of it.
    """

    candidate = Path(raw.strip()).expanduser()
    resolved = candidate.resolve()
    permitted = [root.expanduser().resolve() for root in roots]
    if not any(resolved == root or resolved.is_relative_to(root) for root in permitted):
        locations = ", ".join(str(root) for root in permitted)
        raise ValueError(
            f"Base configuration must be a file under {locations}: {candidate}"
        )
    if not resolved.is_file():
        raise ValueError(f"Base configuration does not exist: {candidate}")
    return resolved


def _inline_ontology_profile(config: dict[str, Any], profile_path: Path | None) -> None:
    """Replace an ontology file reference with the profile itself.

    Deliberately implemented here rather than imported from ``kg_processor``: the
    deployed Streamlit bundle ships only this package, so importing the product
    breaks the entire application inside Snowflake with "No module named
    'kg_processor'". ``test_app_and_product_inline_ontologies_identically`` pins
    this to the product's behaviour so the two cannot drift apart unnoticed.
    """

    ontology = config.get("ontology")
    if not isinstance(ontology, dict):
        return
    if ontology.get("profile") is not None:
        ontology["profile_path"] = None
        return
    if profile_path is None:
        return
    with profile_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Ontology file must contain a mapping: {profile_path}")
    ontology["profile"] = raw
    # The path would not resolve inside the container and must not be retried.
    ontology["profile_path"] = None
