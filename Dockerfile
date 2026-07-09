# Keep the container base aligned with pyproject/mypy/ruff and the local validation
# runtime. A single Python minor version avoids dependency drift between tests and SPCS.
FROM python:3.13-slim

ARG KG_INSTALL_MINERU=true
ARG KG_INSTALL_TESSERACT=false
ARG KG_INSTALL_LOCAL_EMBEDDINGS=true
ARG UV_VERSION=0.11.19

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    XDG_CACHE_HOME=/home/kgprocessor/.cache
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

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src

RUN if [ "$KG_INSTALL_MINERU" = "true" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            libgl1 \
            libglib2.0-0 \
            libgomp1 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN if [ "$KG_INSTALL_TESSERACT" = "true" ]; then \
        apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            poppler-utils \
            tesseract-ocr \
        && rm -rf /var/lib/apt/lists/*; \
    fi

RUN pip install --upgrade pip "uv==$UV_VERSION" \
    && if [ "$KG_INSTALL_MINERU" = "true" ] \
        && [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
            uv sync --locked --no-dev --no-editable \
                --extra ocr-mineru \
                --extra local-embeddings; \
    elif [ "$KG_INSTALL_MINERU" = "true" ]; then \
            uv sync --locked --no-dev --no-editable \
                --extra ocr-mineru; \
    elif [ "$KG_INSTALL_LOCAL_EMBEDDINGS" = "true" ]; then \
            uv sync --locked --no-dev --no-editable \
                --extra local-embeddings; \
    else \
            uv sync --locked --no-dev --no-editable; \
    fi

# Config examples are runtime inputs for local, on-prem, and SPCS execution.
# Test data and docs stay outside the image and are mounted or staged instead.
COPY configs /app/configs

USER kgprocessor

ENTRYPOINT ["flakegraph"]
CMD ["--help"]
