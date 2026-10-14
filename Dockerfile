# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
 && python -m spacy download en_core_web_sm \
 && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} + \
 && find /opt/venv -type d \( -name tests -o -name test \) -path "*/site-packages/*" -prune -exec rm -rf {} +
COPY backend /app/backend
COPY data /app/data
RUN cd /app/backend && python -c "import rag_icd10; rag_icd10.ICD10RAG().initialize()"

FROM python:3.12-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app frontend /app/frontend
USER app
WORKDIR /app/backend
EXPOSE 8000
CMD ["python", "app.py"]
