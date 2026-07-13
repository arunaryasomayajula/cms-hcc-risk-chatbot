# Serving MedGemma with vLLM

The chatbot can use **MedGemma** — Google's medical-domain Gemma variant — for any
of its three LLM pipeline steps, served locally by **vLLM** through its
OpenAI-compatible API. The app talks to it with the `openai` SDK, so any
OpenAI-compatible endpoint works.

## 1. Prerequisites

- An NVIDIA GPU. Rough guidance:
  - `google/medgemma-4b-it` — multimodal, ~1× 16–24 GB GPU (or quantized on less).
  - `google/medgemma-27b-it` — text, ~1–2× 40–80 GB GPU (A100/H100), or quantized.
- A Hugging Face account with access to the MedGemma weights (accept the license on
  the model page) and `huggingface-cli login`.
- `pip install vllm`.

## 2. Start the vLLM server

The app runs on port **8000**, so serve vLLM on a **different port** (8001):

```bash
# Text model (recommended for this text-only pipeline)
vllm serve google/medgemma-27b-it \
  --port 8001 \
  --api-key EMPTY \
  --served-model-name google/medgemma-27b-it

# Lower-VRAM option
vllm serve google/medgemma-4b-it --port 8001 --api-key EMPTY
```

Quantization / memory knobs if needed:

```bash
vllm serve google/medgemma-27b-it --port 8001 --api-key EMPTY \
  --quantization bitsandbytes --max-model-len 8192 --gpu-memory-utilization 0.90
```

Verify it is up:

```bash
curl http://localhost:8001/v1/models -H "Authorization: Bearer EMPTY"
```

## 3. Point the app at it

Set these environment variables before launching the backend (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM OpenAI endpoint |
| `VLLM_MODEL` | `google/medgemma-27b-it` | Model name passed to vLLM |
| `VLLM_API_KEY` | `EMPTY` | Matches vLLM's `--api-key` |

Choose which step(s) use MedGemma, either per request from the sidebar dropdowns, or
as process-wide defaults via:

| Variable | Default | Values |
|---|---|---|
| `LLM_PROVIDER_EXTRACT` | `claude` | `claude` \| `medgemma` |
| `LLM_PROVIDER_SELECT` | `claude` | `claude` \| `medgemma` |
| `LLM_PROVIDER_EXPLAIN` | `claude` | `claude` \| `medgemma` |

Example (PowerShell) — MedGemma for extraction + selection, Claude for the explanation:

```powershell
$env:VLLM_BASE_URL = "http://localhost:8001/v1"
$env:VLLM_MODEL    = "google/medgemma-27b-it"
$env:LLM_PROVIDER_EXTRACT = "medgemma"
$env:LLM_PROVIDER_SELECT  = "medgemma"
python backend/app.py
```

`GET /api/config` reports whether the vLLM endpoint is reachable, and the sidebar
dropdowns are populated from it. If MedGemma is unreachable, keep the selectors on
Claude (or leave the defaults) and the app runs unchanged.

## 3a. Alternative: Ollama (CPU / no GPU)

vLLM requires Linux + an NVIDIA GPU. On a laptop, Windows, or any machine without a
GPU you can serve MedGemma with **[Ollama](https://ollama.com)** instead — it exposes
the *same* OpenAI-compatible API, so the app's provider code is identical; only the
endpoint URL and model name change.

```bash
ollama pull alibayram/medgemma:4b      # ~2.5 GB, runs on CPU (27b tag also exists)
ollama serve                           # serves the OpenAI API at http://localhost:11434/v1
```

Point the app at it:

```powershell
$env:VLLM_BASE_URL = "http://localhost:11434/v1"
$env:VLLM_MODEL    = "alibayram/medgemma:4b"
$env:VLLM_API_KEY  = "ollama"          # any non-empty string
python backend/app.py
```

Then pick **MedGemma** in the sidebar dropdowns. Note: CPU inference is much slower
than a GPU vLLM deployment (several seconds per step) — fine for testing, not for
production throughput.

## 4. Notes

- The provider abstraction lives in [`backend/llm_providers.py`](../backend/llm_providers.py).
  Any OpenAI-compatible server (vLLM, **Ollama**, TGI with the OpenAI shim, etc.) works by
  pointing `VLLM_BASE_URL`/`VLLM_MODEL` at it.
- MedGemma, like Claude, may wrap JSON in markdown fences; the existing
  `_parse_json()` helper handles both.
- Compare MedGemma vs Claude on your own cases with
  [`evaluation/run_provider_comparison.py`](../evaluation/run_provider_comparison.py).
