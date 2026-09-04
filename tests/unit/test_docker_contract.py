from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

_COMPOSE_PATH = Path(".github/docker-compose.smoke.yaml")
_GITHUB_ACTION_SHA_RE = re.compile(r"uses: actions/[A-Za-z0-9_-]+@[a-f0-9]{40}")


def test_dockerfile_defaults_to_production_local_open_source_image() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.14.6-slim-trixie@sha256:" in dockerfile
    assert _docker_arg_default(dockerfile, "KG_INSTALL_MINERU") == "true"
    assert (
        'uv tool install --python 3.13 --with "six==1.17.0" "mineru[pipeline]==3.4.4"'
    ) in dockerfile
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv-python" in dockerfile
    assert "--extra ocr-mineru" not in dockerfile
    assert _docker_arg_default(dockerfile, "KG_INSTALL_LOCAL_EMBEDDINGS") == "true"
    assert _docker_arg_default(dockerfile, "KG_INSTALL_GLINER") == "false"
    assert _docker_arg_default(dockerfile, "KG_PRELOAD_LOCAL_EMBEDDING") == "true"
    assert "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" in dockerfile


def test_spark_dockerfile_preserves_upstream_executor_lifecycle() -> None:
    """Keep Spark, Java, entrypoint, and signal handling from one pinned runtime."""

    dockerfile = Path("Dockerfile.spark").read_text(encoding="utf-8")

    assert "FROM apache/spark:4.1.2-python3@sha256:" in dockerfile
    assert "FROM python:3.14.6-slim-trixie@sha256:" in dockerfile
    assert "COPY --from=spark /opt/java /opt/java" in dockerfile
    assert "COPY --from=spark /opt/spark /opt/spark" in dockerfile
    assert "COPY --from=spark /opt/entrypoint.sh /opt/entrypoint.sh" in dockerfile
    assert "COPY --from=spark /usr/bin/tini /usr/bin/tini" in dockerfile
    assert "graphframes-spark4_2.13:0.12.1" in dockerfile
    assert "hadoop-aws:3.4.2" in dockerfile
    assert "--extra local-embeddings" in dockerfile
    assert "SentenceTransformer('$KG_LOCAL_EMBEDDING_MODEL'" in dockerfile
    assert "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" in dockerfile
    assert "HF_HOME=/home/spark/.cache/huggingface" in dockerfile
    assert "SENTENCE_TRANSFORMERS_HOME=/home/spark/.cache/sentence_transformers" in dockerfile
    assert "USER spark" in dockerfile


def test_dockerfile_installs_from_locked_uv_environment() -> None:
    """Install locked dependencies without coupling their layer to documentation."""

    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    readme_copy = "COPY --chown=kgprocessor:kgprocessor README.md /app/README.md"
    assert "COPY pyproject.toml uv.lock /app/" in dockerfile
    assert readme_copy in dockerfile
    assert dockerfile.index("--no-install-project") < dockerfile.index(readme_copy)
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in dockerfile
    assert "uv sync --locked --no-dev --no-editable" in dockerfile
    assert dockerfile.count("&& uv cache clean") == 2
    assert "--extra extract-gliner" in dockerfile
    assert "--no-install-project" in dockerfile
    assert "pip install --upgrade pip" not in dockerfile
    assert "pip install .;" not in dockerfile
    assert "uv.lock" not in dockerignore
    assert 'ENTRYPOINT ["flakegraph"]' in dockerfile
    assert pyproject["project"]["name"] == "flakegraph"
    assert pyproject["project"]["scripts"]["flakegraph"] == "kg_processor.cli:app"


def test_dockerfile_uses_noninteractive_apt_installs() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "DEBIAN_FRONTEND=noninteractive apt-get install" in dockerfile
    assert dockerfile.count("DEBIAN_FRONTEND=noninteractive apt-get install") == 2


def test_dockerfile_runs_as_non_root_with_external_cache_and_output_paths() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "useradd --create-home --shell /usr/sbin/nologin kgprocessor" in dockerfile
    assert "XDG_CACHE_HOME=/home/kgprocessor/.cache" in dockerfile
    assert "HF_HOME=/home/kgprocessor/.cache/huggingface" in dockerfile
    assert "SENTENCE_TRANSFORMERS_HOME=/home/kgprocessor/.cache/sentence_transformers" in dockerfile
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
    assert "COPY --chown=kgprocessor:kgprocessor src /app/src" in dockerfile
    assert "COPY --chown=kgprocessor:kgprocessor configs /app/configs" in dockerfile
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
    assert "torchvision" not in sources
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cpu"]["explicit"] is True

    assert 'version = "2.13.0+cpu"' in lockfile
    assert 'name = "torchvision"' not in lockfile
    assert 'source = { registry = "https://download.pytorch.org/whl/cpu" }' in lockfile
    assert "nvidia-" not in lockfile
    assert "cuda-toolkit" not in lockfile
    assert 'name = "triton"' not in lockfile
    assert "KG_TORCH_INDEX_URL" not in dockerfile
    assert "KG_TORCH_INDEX_URL" not in compose


def test_compose_smoke_service_is_small_and_self_contained() -> None:
    compose = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"flakegraph"}
    smoke_args = services["flakegraph"]["build"]["args"]
    assert smoke_args["KG_INSTALL_MINERU"] == "false"
    assert smoke_args["KG_INSTALL_LOCAL_EMBEDDINGS"] == "false"
    service = services["flakegraph"]
    assert service["command"] == ["worker", "--config", "/app/smoke.yaml"]
    assert set(service["volumes"]) == {
        "../.github/configs/smoke.yaml:/app/smoke.yaml:ro",
        "../data:/app/data:ro",
        "flakegraph-out:/app/out",
    }
    assert "environment" not in service
    assert "flakegraph-out" in compose["volumes"]


def test_ci_workflow_pins_first_party_actions_to_commit_shas() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    action_uses = [
        line.split("#", 1)[0].strip()
        for line in workflow.splitlines()
        if line.strip().startswith("uses: actions/")
    ]

    assert action_uses
    assert all(_GITHUB_ACTION_SHA_RE.fullmatch(line) for line in action_uses)
    assert "@v4" not in workflow
    assert "@v5" not in workflow


def _docker_arg_default(dockerfile: str, name: str) -> str:
    match = re.search(rf"^ARG {name}=(.+)$", dockerfile, flags=re.MULTILINE)
    if match is None:
        raise AssertionError(f"Dockerfile is missing ARG {name}")
    return match.group(1)


def test_image_contents_do_not_inherit_the_build_host_file_modes() -> None:
    """Own and normalise what is copied in, rather than trusting the builder.

    COPY preserves the source tree's ownership and mode. A checkout synced onto
    a build machine under a restrictive umask once produced a 0640 root-owned
    entry point, which the unprivileged runtime user could not read - and the
    image failed on first request rather than at build, on one machine and not
    another. Whether the image works should not depend on the umask of whoever
    last copied the tree onto the builder.
    """

    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    copied_sources = [
        line
        for line in dockerfile.splitlines()
        if line.startswith("COPY ") and "--from=" not in line and "/app/" in line
    ]
    assert copied_sources, "expected the image to copy application sources"
    unowned = [line for line in copied_sources if "--chown=" not in line]
    # The dependency layer copies only manifests, which the install step reads
    # as root before the runtime user exists.
    assert unowned == ["COPY pyproject.toml uv.lock /app/"], unowned

    assert "chmod -R a+rX" in dockerfile
