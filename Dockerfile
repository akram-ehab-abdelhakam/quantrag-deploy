# QuantRAG container: one process serving REST (/health, /ask, /search) and
# the MCP Streamable HTTP endpoint (/mcp).
#
# The image contains a prebuilt index (data/index/ and/or data/index.json) but
# no secrets.  Operators inject credentials through the platform's environment
# or secret manager at run time.

# -------------------------------------------------------------------
# Stage 1: dependency installation (cached on uv.lock + pyproject.toml)
# -------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /bin/uv

# Copy only what dependency resolution needs; this layer is cached until the
# lockfile or project metadata changes.
COPY pyproject.toml uv.lock README.md ./

# Production dependencies only (the REST/MCP server needs the api extra).
RUN uv sync --frozen --no-dev --extra api --no-install-project

# -------------------------------------------------------------------
# Stage 2: runtime image (non-root, no build tools, no secrets)
# -------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN groupadd -r quantrag \
    && useradd -r -g quantrag -d /app -s /sbin/nologin quantrag

WORKDIR /app

# Reuse the exact virtual environment created in the builder.
COPY --from=builder /app/.venv /app/.venv

# Runtime code and the prebuilt knowledge base.  .dockerignore keeps tests,
# docs, zips, .env files, raw downloads and CSV manifests out of the image.
COPY src/ ./src/
COPY data/ ./data/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MCP_TRANSPORT="streamable-http" \
    PORT="8080" \
    QUANTRAG_REQUIRE_AUTH="1" \
    QUANTRAG_INDEX_DIR="data/index" \
    QUANTRAG_INDEX_FILE="data/index.json" \
    QUANTRAG_INDEX_META="data/index.meta.json" \
    QUANTRAG_CORPUS_PATH="data/chunks.jsonl"

# Credentials are supplied at run time, never baked into the image.
ENV OPENROUTER_API_KEY="" \
    OPENROUTER_MODEL="" \
    OPENROUTER_EMBEDDING_MODEL="" \
    QUANTRAG_API_KEY="" \
    QUANTRAG_API_KEYS=""

USER quantrag

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8080'))" || exit 1

# MCP_TRANSPORT=streamable-http starts the combined FastAPI app on 0.0.0.0:$PORT.
CMD ["python", "-m", "quantrag"]
