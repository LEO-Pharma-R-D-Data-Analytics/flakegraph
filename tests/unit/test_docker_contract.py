from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

_COMPOSE_PATH = Path(".github/docker-compose.smoke.yaml")


def test_dockerfile_defaults_to_production_local_open_source_image() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.13-slim" in dockerfile
    assert _docker_arg_default(dockerfile, "KG_INSTALL_MINERU") == "true"
    assert _docker_arg_default(dockerfile, "KG_INSTALL_LOCAL_EMBEDDINGS") == "true"


def test_dockerfile_installs_from_locked_uv_environment() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "COPY pyproject.toml uv.lock README.md /app/" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in dockerfile
    assert "uv sync --locked --no-dev --no-editable" in dockerfile
    assert "pip install .;" not in dockerfile
    assert "uv.lock" not in dockerignore
    assert 'ENTRYPOINT ["flakegraph"]' in dockerfile
    assert pyproject["project"]["name"] == "flakegraph"
    assert pyproject["project"]["scripts"]["flakegraph"] == "kg_processor.cli:app"
    assert pyproject["project"]["scripts"]["kg-processor"] == "kg_processor.cli:app"


def test_dockerfile_uses_noninteractive_apt_installs() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "DEBIAN_FRONTEND=noninteractive apt-get install" in dockerfile
    assert dockerfile.count("DEBIAN_FRONTEND=noninteractive apt-get install") == 2


def test_dockerfile_runs_as_non_root_with_external_cache_and_output_paths() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "useradd --create-home --shell /usr/sbin/nologin kgprocessor" in dockerfile
    assert "XDG_CACHE_HOME=/home/kgprocessor/.cache" in dockerfile
    assert "/home/kgprocessor/.cache/huggingface" in dockerfile
    assert "/home/kgprocessor/.cache/mineru" in dockerfile
    assert "/home/kgprocessor/.cache/sentence_transformers" in dockerfile
    assert "/home/kgprocessor/.cache/torch" in dockerfile
    assert "/app/out" in dockerfile
    assert "chown -R kgprocessor:kgprocessor /home/kgprocessor /app/out" in dockerfile
    assert "USER kgprocessor" in dockerfile
    assert "USER root" not in dockerfile


def test_docker_build_context_keeps_runtime_mounts_and_review_docs_out_of_image() -> None:
    dockerignore = set(Path(".dockerignore").read_text(encoding="utf-8").splitlines())
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert {"data", "docs", "tests", "out", ".venv"} <= dockerignore
    assert "COPY src /app/src" in dockerfile
    assert "COPY configs /app/configs" in dockerfile
    assert "COPY data" not in dockerfile
    assert "COPY docs" not in dockerfile
    assert "COPY tests" not in dockerfile


def test_locked_container_profiles_use_cpu_torch_on_linux() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lockfile = Path("uv.lock").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")

    sources = pyproject["tool"]["uv"]["sources"]
    indexes = {index["name"]: index for index in pyproject["tool"]["uv"]["index"]}
    assert sources["torch"] == [{"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}]
    assert sources["torchvision"] == [
        {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}
    ]
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cpu"]["explicit"] is True

    assert 'version = "2.12.1+cpu"' in lockfile
    assert 'version = "0.27.1+cpu"' in lockfile
    assert 'source = { registry = "https://download.pytorch.org/whl/cpu" }' in lockfile
    assert "nvidia-" not in lockfile
    assert "cuda-toolkit" not in lockfile
    assert 'name = "triton"' not in lockfile
    assert "KG_TORCH_INDEX_URL" not in dockerfile
    assert "KG_TORCH_INDEX_URL" not in compose


def test_compose_smoke_profiles_opt_out_of_heavy_production_dependencies() -> None:
    compose = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose["services"]

    smoke_args = services["flakegraph"]["build"]["args"]
    assert smoke_args["KG_INSTALL_MINERU"] == "false"
    assert smoke_args["KG_INSTALL_LOCAL_EMBEDDINGS"] == "false"

    tesseract_args = services["flakegraph-tesseract"]["build"]["args"]
    assert tesseract_args["KG_INSTALL_TESSERACT"] == "true"
    assert tesseract_args["KG_INSTALL_MINERU"] == "false"
    assert tesseract_args["KG_INSTALL_LOCAL_EMBEDDINGS"] == "false"

    mineru_api_args = services["flakegraph-mineru-api"]["build"]["args"]
    assert mineru_api_args["KG_INSTALL_MINERU"] == "false"
    assert mineru_api_args["KG_INSTALL_LOCAL_EMBEDDINGS"] == "false"

    vllm_args = services["flakegraph-vllm"]["build"]["args"]
    assert vllm_args["KG_INSTALL_MINERU"] == "true"
    assert vllm_args["KG_INSTALL_LOCAL_EMBEDDINGS"] == "true"


def test_compose_includes_external_mineru_api_profile() -> None:
    compose = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    service = compose["services"]["flakegraph-mineru-api"]

    assert service["profiles"] == ["mineru-api"]
    assert service["image"] == "flakegraph:local"
    assert service["command"] == ["worker", "--config", "configs/local-mineru-api.yaml"]
    assert service["environment"]["KG_MINERU_API_URL"] == "${KG_MINERU_API_URL:-}"
    assert service["environment"]["KG_MINERU_API_KEY"] == "${KG_MINERU_API_KEY:-}"


def test_compose_profiles_mount_inputs_outputs_and_do_not_inline_secrets() -> None:
    compose = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose["services"]

    for service_name, service in services.items():
        volumes = set(service["volumes"])
        assert "../data:/app/data:ro" in volumes, service_name
        assert "../out:/app/out" in volumes, service_name

    assert "mineru-cache:/home/kgprocessor/.cache/mineru" in set(
        services["flakegraph-mineru"]["volumes"]
    )
    assert "mineru-cache:/home/kgprocessor/.cache/mineru" in set(
        services["flakegraph-vllm"]["volumes"]
    )

    for service_name, service in services.items():
        environment = service.get("environment", {})
        assert isinstance(environment, dict), service_name
        for key, value in environment.items():
            if _is_sensitive_env_name(key):
                assert isinstance(value, str), (service_name, key)
                assert value.startswith("${"), (service_name, key)
                assert value.endswith(":-}"), (service_name, key)


def _docker_arg_default(dockerfile: str, name: str) -> str:
    match = re.search(rf"^ARG {name}=(.+)$", dockerfile, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"Dockerfile is missing ARG {name}")
    return match.group(1)


def _is_sensitive_env_name(name: str) -> bool:
    sensitive_fragments = ("API_KEY", "PASSWORD", "TOKEN", "SAS", "PRIVATE_KEY", "SECRET")
    return any(fragment in name.upper() for fragment in sensitive_fragments)
