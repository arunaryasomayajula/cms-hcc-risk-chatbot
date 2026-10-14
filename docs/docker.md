# Docker image (< 1 GB)

One self-contained image: FastAPI backend, static frontend, bundled CMS data, the
pre-built TF-IDF ICD-10 index and the PHI guardrail. Every LLM step runs on Claude
via the Anthropic API, so no model weights live in the image.

```bash
docker compose up --build                     # reads ANTHROPIC_API_KEY from shell / .env
# or
docker build -t cms-hcc-chatbot .
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY cms-hcc-chatbot
```

Open http://localhost:8000. `GET /api/status` should report `initialized`,
`rag_ready` and `guardrails.available` as `true`.

## Size

Measured on Docker 29.8 (linux/amd64):

| | Before | After |
|---|---|---|
| Image on disk (uncompressed layers) | not built; would include torch + CUDA libs via `sentence-transformers` / `llm-guard` (multiple GB) | **~610 MB** |
| Download (compressed) | — | **~185 MB** |
| `/venv` | — | 463 MB |

Docker Desktop's "DISK USAGE" column (~807 MB) adds the compressed content store to the
unpacked layers; the uncompressed layer total is the number that counts against the limit.

Check it yourself:

```bash
docker image inspect cms-hcc-chatbot --format '{{.Size}}'
docker history cms-hcc-chatbot
```

## How it got there

| Change | Why |
|---|---|
| Removed `chromadb`, `sentence-transformers` | Not imported anywhere since v1.1.0 (`rag_icd10.py` uses sklearn TF-IDF). `sentence-transformers` alone pulls torch + CUDA (GBs). |
| Replaced `llm-guard` with `presidio-analyzer` + `en_core_web_sm` | LLM Guard's `Anonymize` is a wrapper over Presidio but hard-depends on torch + transformers. Presidio's default would also download `en_core_web_lg` (~560 MB); the small model is ~15 MB. |
| Declared `scikit-learn` | It was only installed transitively via `sentence-transformers`. |
| Removed `openpyxl`, `python-multipart` | No imports / no form or upload endpoints. |
| Pinned `backend/requirements.lock` | Reproducible builds; a new transitive dependency can't silently bring torch back. |
| Multi-stage build, `python:3.12-slim` | Build tools (`binutils`, pip) stay in the builder. Slim (glibc) rather than Alpine: Alpine's musl has no manylinux wheels, so numpy/scipy would compile from source. |
| `strip --strip-unneeded` on `*.so` | Wheels ship debug symbols; native libs went from 387 MB to 202 MB. |
| Drop wheel `tests/` dirs and C sources | Never used at runtime. |
| TF-IDF index built at image build | No refit on start, and the non-root user needs no write access to `/app/backend`. |
| `.dockerignore` | Keeps `.git`, evaluation, docs, tests, the AirLLM sidecar and any local traces/indexes out of the build context. |

## Runtime notes

- Runs as non-root user `app` (uid 10001). Healthcheck polls `/api/status`.
- Traces (`LANGSMITH_TRACING=1`) contain clinical text and are written to the
  `/app/traces` volume, never into the image.
- `LLM_PROVIDER_EXTRACT|SELECT|EXPLAIN` default to `claude`; set `CLAUDE_MODEL` to pick
  the model.
- Local-model providers (Ollama / vLLM / AirLLM) are out of scope for this image. The
  AirLLM sidecar needs torch and is excluded via `.dockerignore`.

## Updating dependencies

Edit `backend/requirements.txt`, then regenerate the lock and rebuild:

```bash
uv pip compile backend/requirements.txt -o backend/requirements.lock --python-version 3.12 --python-platform x86_64-unknown-linux-gnu
```

If `torch`, `transformers` or `nvidia-*` ever shows up in the lock, something reintroduced
the heavy stack. Find what pulled it in from the `# via` comments.

## References

- Docker multi-stage builds — https://docs.docker.com/build/building/multi-stage/
- Docker build best practices — https://docs.docker.com/build/building/best-practices/
- Why not Alpine for Python — https://pythonspeed.com/articles/alpine-docker-python/
- Presidio NLP engine configuration — https://microsoft.github.io/presidio/analyzer/customizing_nlp_models/
- LLM Guard Anonymize (previous implementation) — https://protectai.github.io/llm-guard/input_scanners/anonymize/
- spaCy English models — https://spacy.io/models/en
