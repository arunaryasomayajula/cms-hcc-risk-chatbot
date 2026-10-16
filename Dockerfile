# CMS HCC Risk Scoring Chatbot — self-contained app image (< 1 GB).
# The LLM is Claude via the Anthropic API; no model weights live in the image.
# See docs/docker.md for the size budget and how it was reached.
#
#   docker build -t cms-hcc-chatbot .
#   docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... cms-hcc-chatbot
#   docker run --rm -p 9000:9000 -e PORT=9000 -e ANTHROPIC_API_KEY=sk-ant-... cms-hcc-chatbot

ARG PYTHON_VERSION=3.12

# ── builder: resolve + install pinned deps into an isolated venv ─────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

# binutils provides `strip`; it stays in this throwaway stage.
RUN apt-get update && apt-get install -y --no-install-recommends binutils \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock \
    # Test suites and C sources shipped inside wheels are never used at runtime.
 && find /venv -depth -type d \( -name tests -o -name test \) -path '*/site-packages/*' -exec rm -rf {} + \
 && find /venv -type f \( -name '*.pyx' -o -name '*.pxd' -o -name '*.c' -o -name '*.h' \) -delete \
    # Many manylinux wheels ship unstripped native libs; debug symbols are
    # dead weight in production (~40% of the venv's .so bytes).
 && find /venv -type f -name '*.so*' -exec strip --strip-unneeded {} + \
 && pip uninstall -y pip setuptools wheel

# Bake the TF-IDF ICD-10 index into the image so startup does not refit it and
# the runtime user never needs write access to /app/backend.
COPY backend /app/backend
COPY data /app/data
WORKDIR /app/backend
RUN python -c "from rag_icd10 import ICD10RAG; ICD10RAG().initialize()"

# ── runtime ──────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim

# Listen port. Build-time default (`--build-arg PORT=...`), overridable at run
# time with `-e PORT=...`; app.py, EXPOSE and the healthcheck all follow it.
ARG PORT=8000

ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=${PORT} \
    # Every pipeline step runs on Claude; override per step if you add a sidecar.
    LLM_PROVIDER_EXTRACT=claude \
    LLM_PROVIDER_SELECT=claude \
    LLM_PROVIDER_EXPLAIN=claude \
    # Traces contain clinical text — keep them on a volume, never in the image.
    LANGSMITH_TRACE_DIR=/app/traces

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/traces && chown app:app /app/traces

COPY --from=builder /venv /venv
COPY --from=builder /app/backend /app/backend
COPY data /app/data
COPY frontend /app/frontend

USER app
WORKDIR /app/backend
VOLUME ["/app/traces"]
EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/status', timeout=4).status == 200 else 1)"

# app.py reads HOST / PORT and validates them (see listen_address()).
CMD ["python", "app.py"]
