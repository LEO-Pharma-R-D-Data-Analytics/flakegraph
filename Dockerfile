# Keep the container base aligned with pyproject/mypy/ruff and the local validation
# runtime. The digest pins the otherwise mutable Python image tag.
# Pinned deliberately: a kubectl more than one minor ahead of the API server is
# outside the supported skew.
ARG KG_KUBECTL_VERSION=v1.32.5
FROM registry.k8s.io/kubectl:${KG_KUBECTL_VERSION} AS kubectl

FROM python:3.14.6-slim-trixie@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1

ARG KG_INSTALL_MINERU=true
ARG KG_INSTALL_TESSERACT=false
ARG KG_INSTALL_LOCAL_EMBEDDINGS=true
ARG KG_INSTALL_GLINER=false
# The control plane is a component of the deployment, not a script an operator
# runs on a node by hand, so the image can serve it like anything else.
ARG KG_INSTALL_CONTROL_PLANE=true
ARG KG_PRELOAD_LOCAL_EMBEDDING=true
ARG KG_LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG KG_LOCAL_EMBEDDING_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
ARG UV_VERSION=0.11.28

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    XDG_CACHE_HOME=/home/kgprocessor/.cache \
    HF_HOME=/home/kgprocessor/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/kgprocessor/.cache/sentence_transformers
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# The worker should behave the same in Docker and SPCS: application code is
# immutable, while output and provider model caches live in writable paths owned
# by the unprivileged runtime user.
RUN useradd --create-home --shell /usr/sbin/nologin kgprocessor \
    && mkdir -p \
        /home/kgprocessor/.cache/huggingface \
        /home/kgprocessor/.cache/mineru \
        /home/kgprocessor/.cache/sentence_transformers \
        /home/kgprocessor/.cache/torch \
        /app/out \
    && chown -R kgprocessor:kgprocessor /home/kgprocessor /app/out

# Dependency resolution depends only on package metadata and the lockfile. Keep
# frequently edited documentation out of this layer so README changes do not
# reinstall OCR, embedding, and provider dependencies.
COPY pyproject.toml uv.lock /app/

RUN if [ "$KG_INSTALL_TESSERACT" = "true" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            poppler-utils \
            tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# MinerU's pipeline backend imports six without declaring it in the published
# extra. Keep it in the isolated tool environment so scanned-PDF OCR works
# without leaking MinerU dependencies into the FlakeGraph environment.
RUN if [ "$KG_INSTALL_MINERU" = "true" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential \
            libgl1 \
            libglib2.0-0 \
            libgomp1; \
    fi \
    && python -m pip install "uv==$UV_VERSION" \
    && if [ "$KG_INSTALL_MINERU" = "true" ]; then \
        uv python install 3.13 \
        && uv tool install --python 3.13 --with "six==1.17.0" "mineru[pipeline]==3.4.4"; \
    fi \
    && set -- \
    && if [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
        set -- "$@" --extra local-embeddings; \
    fi \
    && if [ "$KG_INSTALL_GLINER" = "true" ]; then set -- "$@" --extra extract-gliner; fi \
    && if [ "$KG_INSTALL_CONTROL_PLANE" = "true" ]; then set -- "$@" --extra app; fi \
    && uv sync --locked --no-dev --no-editable --no-install-project "$@" \
    && if [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ] \
        && [ "$KG_PRELOAD_LOCAL_EMBEDDING" = "true" ]; then \
        /opt/venv/bin/python -c \
          "from sentence_transformers import SentenceTransformer; SentenceTransformer('$KG_LOCAL_EMBEDDING_MODEL', revision='$KG_LOCAL_EMBEDDING_REVISION')"; \
    fi \
    && uv cache clean \
    && chown -R kgprocessor:kgprocessor /home/kgprocessor/.cache \
    && if [ "$KG_INSTALL_MINERU" = "true" ]; then \
        apt-get purge -y --auto-remove build-essential \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# The control plane queries the fleet through kubectl rather than a client
# library. Taken from the upstream release image rather than fetched: this base
# carries no curl, and that image is multi-arch, so the copy is right on arm64
# without asking what architecture we are on. COPY cannot be made conditional,
# so this is present even when the control plane is not built in - about 50 MB.
COPY --from=kubectl /bin/kubectl /usr/local/bin/kubectl

COPY README.md /app/README.md
COPY src /app/src
COPY app /app/app

# Installed environments are self-contained. Removing uv's wheel and source
# cache avoids shipping several gigabytes of duplicate build inputs to every
# worker without changing either the application or MinerU environments.
RUN set -- \
    && if [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
        set -- "$@" --extra local-embeddings; \
    fi \
    && if [ "$KG_INSTALL_GLINER" = "true" ]; then set -- "$@" --extra extract-gliner; fi \
    && if [ "$KG_INSTALL_CONTROL_PLANE" = "true" ]; then set -- "$@" --extra app; fi \
    && uv sync --locked --no-dev --no-editable "$@" \
    && uv cache clean

# Config examples are runtime inputs for local, on-prem, and SPCS execution.
# Test data and docs stay outside the image and are mounted or staged instead.
COPY configs /app/configs

USER kgprocessor

ENTRYPOINT ["flakegraph"]
CMD ["--help"]
