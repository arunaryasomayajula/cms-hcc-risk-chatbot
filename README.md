# CMS HCC Risk Scoring Chatbot

A web-based, AI-powered chatbot that accepts patient clinical notes, maps described conditions to **ICD-10-CM codes** using retrieval-augmented generation (RAG), and automatically calculates **CMS HCC v22 Medicare risk scores** for all nine payment models (payment year 2027).

> **Intended users:** Health plan analysts, risk adjustment coders, actuaries, and clinical informatics teams who need to quickly understand the Medicare Advantage risk impact of a patient's documented diagnoses.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [How the Pipeline Works](#how-the-pipeline-works)
4. [CMS HCC Model Primer](#cms-hcc-model-primer)
5. [Model Providers (Claude / vLLM / AirLLM / Ollama)](#model-providers-claude--vllm--airllm--ollama)
6. [PHI Guardrails](#phi-guardrails)
7. [Tracing](#tracing)
8. [Evaluation](#evaluation)
9. [Prerequisites](#prerequisites)
10. [Installation](#installation)
11. [Configuration](#configuration)
12. [Running the App](#running-the-app)
13. [Using the Chatbot](#using-the-chatbot)
14. [API Reference](#api-reference)
15. [Project Structure](#project-structure)
16. [CMS Data Files Referenced](#cms-data-files-referenced)
17. [Caveats and Limitations](#caveats-and-limitations)
18. [Extending the App](#extending-the-app)
19. [Changelog](#changelog)
20. [References](#references)

---

## What It Does

| Capability | Detail |
|---|---|
| **Clinical NLP** | Extracts discrete diagnoses and conditions from free-text clinical notes using an LLM (Claude, or a self-hosted open-weight model) |
| **PHI guardrail** | De-identifies names, dates, MRNs, SSNs, etc. (Microsoft Presidio + spaCy) before any text reaches an LLM |
| **Pluggable models** | Each pipeline step independently picks a **provider and a model** at request time: Claude, vLLM, AirLLM (local, layer-streamed), or Ollama |
| **Runs on your hardware** | Local inference auto-detects CPU / GTX 10-series / RTX 20-50 / Blackwell / Apple Silicon and adapts dtype, context, and token caps |
| **ICD-10 RAG search** | Searches ~10 k HCC-relevant ICD-10-CM codes using a local TF-IDF index (sklearn) — no internet or model download required |
| **Code validation** | A second LLM pass applies medical coder judgment to select the most specific, documentation-supported codes |
| **HCC calculation** | Pure-Python implementation of the CMS HCC v22 model: ICD-10 → CC → HCC (with hierarchies) → diagnosis categories → interaction terms → risk scores |
| **9 model scores** | Returns scores for all seven Continued Enrollee (CE) models and both New Enrollee (NE) models simultaneously |
| **Conversational UI** | Multi-turn chat with persistent demographics sidebar, inline result cards, and confidence bars |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (vanilla JS)                      │
│  ┌──────────────┐   ┌──────────────────────────────────────┐    │
│  │ Demographics  │   │  Chat window                         │    │
│  │  sidebar      │   │  • ICD-10 result card (table)        │    │
│  │  (DOB, sex,   │   │  • HCC flags + interactions card     │    │
│  │   OREC, dual) │   │  • 9-model risk score card           │    │
│  └──────────────┘   └──────────────────────────────────────┘    │
└──────────────────────────────┬──────────────────────────────────┘
                                │ POST /api/chat
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI backend (Python)                     │
│                                                                   │
│  Guard   Presidio  ──► de-identify PHI in the note (+ spaCy NER) │
│                                                                   │
│  Step 1  LLM (any provider) ──► Extract conditions (JSON)        │
│                                                                   │
│  Step 2  TF-IDF index (sklearn, local) ──► Candidate ICD-10s    │
│                                                                   │
│  Step 3  LLM (any provider) ──► Select best codes (JSON)         │
│                                                                   │
│  Step 4  HCCCalculator ──► 9 risk scores                         │
│                                                                   │
│  Step 5  LLM (any provider) ──► Natural language summary         │
│          (provider AND model configurable per step)              │
└──────────────────────────────┬──────────────────────────────────┘
                                │ reads reference CSVs (read-only)
                                ▼
                  data/ folder (bundled CMS reference CSVs)
```

### Component Responsibilities

| File | Role |
|---|---|
| `backend/app.py` | FastAPI app, request routing, pipeline orchestration |
| `backend/llm_providers.py` | Provider + model catalog; one OpenAI-compatible client for vLLM/AirLLM/Ollama |
| `backend/hardware.py` | Device/dtype/context policy for local inference (CPU → Blackwell) |
| `backend/airllm_server.py` | OpenAI-compatible sidecar wrapping AirLLM (separate process, port 8002) |
| `backend/json_utils.py` | Tolerant JSON extraction shared by the app and the eval harness |
| `backend/prompts.py` | The three step prompts and token budgets, shared with `evaluation/` |
| `backend/rag_icd10.py` | Loads ICD-10 descriptions, builds/caches a TF-IDF index, exposes `search()` |
| `backend/hcc_calculator.py` | Pure-Python CMS HCC v22 scoring engine (no SAS dependency) |
| `frontend/index.html` | Single-page chat UI with demographics sidebar |
| `frontend/styles.css` | Styling — layout, cards, score table color coding |
| `frontend/app.js` | API calls, dynamic card rendering, status polling |

---

## How the Pipeline Works

Every time a user submits a message that looks like clinical notes, the backend executes six sequential steps.

> **Note:** The note is first **de-identified** (see [PHI Guardrails](#phi-guardrails)), and the three LLM steps (1, 3, 5) each run on the **configured provider and model** (Claude, vLLM, AirLLM, or Ollama) (see [Model Providers](#model-providers-claude--vllm--airllm--ollama)). "The LLM" below means whichever provider is configured for that step.

### Step 1 — Condition extraction (LLM: any configured provider)

The LLM is prompted to parse the free text and return a structured JSON list of distinct medical conditions with a targeted search query for each one. Models rarely return bare JSON — [`backend/json_utils.py`](backend/json_utils.py) tolerates markdown fences (` ```json ... ``` `), surrounding prose, trailing commas, and a bare array instead of the requested object. If it still cannot recover an object, the step records a `parse_failures` entry rather than silently returning nothing.

```json
{
  "conditions": [
    {"condition": "Type 2 diabetes mellitus",
     "search_query": "type 2 diabetes mellitus without complication"},
    {"condition": "Congestive heart failure",
     "search_query": "congestive heart failure unspecified"}
  ]
}
```

### Step 2 — TF-IDF RAG search (local, no download)

For each condition's search query, a TF-IDF cosine similarity search runs against all HCC-relevant ICD-10-CM descriptions. The top 6 candidates per condition are collected (duplicates removed).

The index is built from two sources joined at startup:
- **Descriptions** from `2027 Initial ICD-10-CM Mappings.csv`
- **CC numbers** from `ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv`

Only codes that appear in the V22 mapping file are indexed (~10,248 codes), keeping results clinically relevant. The index is serialised to `backend/tfidf_index.pkl` after the first build so subsequent restarts load it in under a second.

**Why TF-IDF instead of a neural embedding model?**
ICD-10 descriptions are short, domain-specific, and highly keyword-dependent ("Type 2 diabetes mellitus with diabetic chronic kidney disease, stage 4"). TF-IDF with bigrams performs well in this vocabulary and requires no internet connection, no model download, and no GPU. Build time is ~3 seconds; search latency is <1 ms.

### Step 3 — Code selection (LLM: any configured provider)

A second LLM call acts as a medical coder review. It receives:
- The original clinical notes
- The extracted conditions
- Up to 25 TF-IDF candidates with descriptions, CC numbers, and similarity scores

Claude selects the most specific, documentation-supported codes, removes duplicates at lower specificity, and returns each code with a rationale and confidence score (0–1). The response is fence-stripped and parsed the same way as Step 1.

### Step 4 — HCC calculation (Python)

`HCCCalculator.calculate()` implements the full CMS HCC v22 scoring pipeline:

1. **Age/sex demographic variables** — 24 CE buckets (`F65_69`, `M75_79`, …) and 32 NE buckets, computed against **February 1, 2027** (the CMS payment-year age cutoff).

2. **ICD-10 → CC mapping** — Each code is looked up in `ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv`. MCE (Medicare Code Editor) age/sex edits are applied: maternity codes are rejected for males, pediatric codes for adults, etc.

3. **HCC hierarchies** — Within each clinical family, only the most severe condition counts. For example, if a patient has metastatic cancer (HCC8), the less severe neoplasm HCCs (HCC9–HCC12) are zeroed out.

4. **Diagnosis category flags** — Ten disease group flags (`CANCER`, `DIABETES`, `CHF`, `RENAL`, etc.) are set to 1 if any HCC in that group is active.

5. **Interaction terms** — 26 comorbidity interaction variables (e.g., `HCC85_gCopdCF` = CHF × COPD) capture the extra cost when specific conditions co-occur.

6. **Score = Σ(flag × coefficient)** — Every active flag is multiplied by its model-specific regression coefficient from `V22_CE_Relative_Factors.csv` or `V22_NE_Relative_Factors.csv`. The dot product is the risk score for that model.

### Step 5 — Natural language explanation (LLM: any configured provider)

Claude receives the complete analysis results (conditions, ICD-10 codes, active HCCs, interaction flags, and all nine scores) and generates a plain-English explanation of what fired, what the score means, and which payment model applies to the patient.

### Step 6 — Response assembly

The API returns:
- `response` — Claude's explanation (markdown)
- `icd10_codes` — Selected codes with description, CC, condition, rationale, confidence
- `hcc_result` — Active HCCs, triggered interactions, all 9 scores, applicable model
- `conditions_found` — Intermediate condition list from Step 1
- `conversation_history` — Full history for multi-turn follow-up questions

---

## CMS HCC Model Primer

### What is an HCC risk score?

CMS uses risk scores to adjust per-member-per-month capitation payments to Medicare Advantage plans. A score of **1.0** represents the average Medicare beneficiary; **2.0** means CMS pays twice the base rate for that member.

Scores are additive: each demographic variable, HCC, and interaction term contributes a regression coefficient to the total.

### The 9 payment models

The chatbot scores each patient on all nine models simultaneously. The **applicable model** is highlighted in the UI.

#### Continued Enrollee (CE) — 7 models

| Model column | When it applies |
|---|---|
| `COMMUNITY_NA` | Community, non-dual, aged (65+) |
| `COMMUNITY_PBA` | Community, partial-benefit dual, aged |
| `COMMUNITY_FBA` | Community, full-benefit dual, aged |
| `COMMUNITY_ND` | Community, non-dual, disabled (under 65) |
| `COMMUNITY_PBD` | Community, partial-benefit dual, disabled |
| `COMMUNITY_FBD` | Community, full-benefit dual, disabled |
| `INSTITUTIONAL` | Long-term institutional (nursing facility) |

#### New Enrollee (NE) — 2 models

| Model column | When it applies |
|---|---|
| `NEW_ENROLLEE` | First year of Medicare enrollment |
| `SNP_NEW_ENROLLEE` | New enrollee in a Special Needs Plan |

New enrollee models use only demographic variables (age/sex/Medicaid interactions) because no prior-year diagnosis data is available.

### Applicable model selection logic

```
ltimcaid == 1           → INSTITUTIONAL
age >= 65:
  dual_status == 2      → COMMUNITY_FBA
  dual_status == 1      → COMMUNITY_PBA
  else                  → COMMUNITY_NA
age < 65:
  dual_status == 2      → COMMUNITY_FBD
  dual_status == 1      → COMMUNITY_PBD
  else                  → COMMUNITY_ND
```

### HCC hierarchies (why some codes disappear)

Within each clinical category, CMS only counts the **highest-severity condition** to prevent double-counting. Examples:

| If patient has → | These are suppressed |
|---|---|
| HCC8 (Metastatic cancer) | HCC9, HCC10, HCC11, HCC12 |
| HCC17 (Diabetes with acute complications) | HCC18, HCC19 |
| HCC85 (CHF) | Lower heart failure codes |
| HCC134 (Dialysis status) | HCC135, HCC136, HCC137 |

### Comorbidity interactions (why score > sum of parts)

Twenty-six interaction variables capture pairs of conditions that together cost more than either alone. Examples:

| Interaction | Clinical meaning |
|---|---|
| `HCC85_gCopdCF` | CHF + COPD |
| `DIABETES_CHF` | Diabetes + CHF |
| `HCC85_gRenal` | CHF + renal disease |
| `SEPSIS_PRESSURE_ULCER` | Sepsis + pressure ulcer |
| `DISABLED_HCC85` | Disabled patient with CHF |

---

## Model Providers (Claude / vLLM / AirLLM / Ollama)

The three LLM steps — condition extraction, ICD-10 code selection, and the
natural-language explanation — can each be served by a **different provider and a
different model**, chosen at request time:

| Provider | What it is | Use when |
|---|---|---|
| `claude` (default) | Anthropic API | You want the best accuracy and have a key |
| `medgemma` | Any model served by **vLLM** | The model fits entirely in your VRAM |
| `airllm` | **[AirLLM](https://github.com/lyogavin/airllm)** streaming layers from disk via a local sidecar | The model is *larger* than your VRAM (a 70B in ~4 GB) |
| `ollama` | Quantized GGUF models served by **Ollama** | You want a local answer in seconds — the practical choice on CPU |

Selection is **per pipeline step**, from the sidebar's provider + model dropdowns
(populated by `GET /api/config`) or via environment defaults:

| Variable | Default | Values |
|---|---|---|
| `LLM_PROVIDER_EXTRACT` | `claude` | `claude` \| `medgemma` \| `airllm` \| `ollama` |
| `LLM_PROVIDER_SELECT`  | `claude` | ” |
| `LLM_PROVIDER_EXPLAIN` | `claude` | ” |
| `LLM_MODEL_EXTRACT` / `_SELECT` / `_EXPLAIN` | provider default | Model id for that step |
| `CLAUDE_MODEL` | `claude-sonnet-5-5` | Anthropic model id |
| `VLLM_BASE_URL` / `VLLM_MODEL` / `VLLM_API_KEY` | `http://127.0.0.1:8001/v1` · `google/medgemma-27b-it` · `EMPTY` | vLLM endpoint |
| `AIRLLM_BASE_URL` / `AIRLLM_MODEL` | `http://127.0.0.1:8002/v1` · `Qwen/Qwen2.5-0.5B-Instruct` | AirLLM sidecar |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://127.0.0.1:11434/v1` · `qwen2.5:7b-instruct` | Ollama endpoint |
| `LLM_ALLOW_UNLISTED_MODELS` | unset | Accept a model id the catalog doesn't list |
| `LOCAL_INFERENCE_CONCURRENCY` | `1` | Local calls are serialized; raise only for a backend that parallelises |

> Use `127.0.0.1` rather than `localhost` in these URLs on Windows. `localhost`
> resolves to IPv6 `::1` first and local servers usually bind IPv4 only, so every
> request pays a full timeout first — measured here at 2.02 s vs 0.01 s.

The abstraction lives in [`backend/llm_providers.py`](backend/llm_providers.py). The
last three providers are one `OpenAICompatProvider` with different endpoints, so any
OpenAI-compatible server (TGI, LM Studio, llama.cpp's server…) works by pointing one
of the `*_BASE_URL` variables at it. Each `/api/chat` response reports
`providers_used`, `models_used`, and `parse_failures`.

Guides: [`docs/airllm.md`](docs/airllm.md) (local open-weight models, per-GPU setup)
and [`docs/vllm-medgemma.md`](docs/vllm-medgemma.md) (vLLM and Ollama).

### Hardware support

Local inference adapts to the machine. [`backend/hardware.py`](backend/hardware.py)
detects the device once and derives dtype, sequence length, output caps, and client
timeouts from it — nothing is hardcoded.

| Tier | Detected as | dtype chosen | Why |
|---|---|---|---|
| CPU | `cpu` | `float32` (`bfloat16` with AVX512-BF16) | fp16 matmul is unimplemented on CPU |
| GTX 10-series (Pascal, sm_61) | `cuda` | `float32` | consumer Pascal runs fp16 at 1/64 rate — fp16 is *slower* |
| RTX 20 / GTX 16 (Turing, sm_75) | `cuda` | `float16` | real fp16, no bf16 |
| RTX 30/40, A100, H100 (sm_80+) | `cuda` | `bfloat16` | |
| RTX 50 (Blackwell, sm_120) | `cuda` | `bfloat16` | needs a **CUDA 12.8+ torch wheel** |
| Apple Silicon | `mps` | `float16` | AirLLM auto-selects its MLX path |

The most common GPU failure is a torch wheel without kernels for your card — it is
detected fine and then dies at the first kernel launch with *"no kernel image is
available for execution on the device"*. This hits **new** cards (`sm_120` missing
from older wheels) and **old** ones (`sm_61` dropped from recent wheels) alike. The
startup check compares your compute capability against `torch.cuda.get_arch_list()`
and prints the exact `pip install` line to fix it; the UI shows the same warning.

---

## PHI Guardrails

Before any clinical note is sent to an LLM (Claude, vLLM, AirLLM, or Ollama), it is scanned
by **Microsoft Presidio**'s `AnalyzerEngine` (regex/checksum recognizers + spaCy
`en_core_web_sm` NER). Detected PHI/PII — names, dates, phone numbers, emails, SSNs,
medical-record numbers, locations — is **de-identified** (replaced with placeholders
like `[REDACTED_PERSON_1]`) and the pipeline proceeds on the sanitized text. See
[`backend/guardrails.py`](backend/guardrails.py).

- **De-identify, don't block** — clinical content (diagnoses, labs, meds) is kept.
- **No re-identification** — the placeholder→value map exists only inside one
  `deidentify()` call, so PHI never enters the model output or stored conversation
  history.
- **Labelled identifiers always win** — a value after an `MRN` / `SSN` label is
  redacted with top confidence, even if spaCy mislabels it (e.g. an MRN as a date).
- **Only types are surfaced** — the `/api/chat` response returns a `guardrail` block
  with entity *types* and counts (never raw values); the UI shows a "🔒 PHI
  de-identified" banner.
- **Graceful fallback** — if `presidio-analyzer` or its spaCy model isn't installed,
  the app still boots and `/api/status` reports the guardrail as off.

| Variable | Default | Purpose |
|---|---|---|
| `GUARDRAILS_ENABLED` | `true` | set `false` to disable de-identification |
| `GUARDRAILS_SPACY_MODEL` (or `SPACY_MODEL`) | `en_core_web_sm` | spaCy NER model (install it first; `en_core_web_md` is a middle ground, `_lg` adds ~560 MB) |
| `GUARDRAILS_SCORE_THRESHOLD` | `0.35` | minimum Presidio confidence to redact |

> Earlier versions used LLM Guard's `Anonymize` scanner, which wraps the same Presidio
> analyzer but hard-depends on torch + transformers. Dropping it is what lets the
> Docker image fit in under 1 GB — see [`docs/docker.md`](docs/docker.md).

---

## Tracing

Every `/api/chat` request can be captured as a **LangSmith run tree** and written as
**one JSON file per trace** into a folder you specify.

```bash
export LANGSMITH_TRACING=1
export LANGSMITH_TRACE_DIR=/var/log/cms-hcc-traces
```

Files are named `<UTC timestamp>__<trace uuid>.json` — the UUID makes them unique, the
timestamp makes the folder sort chronologically. Each trace holds a root `chat` span
with children for the guardrail, each LLM call, the TF-IDF search, and the HCC
calculation, every one carrying its own `latency_ms`, inputs, outputs, and the
provider/model that served it.

**The note recorded in a trace is the de-identified one.** A trace is the
highest-fidelity record of what went through the pipeline, so capturing raw text would
quietly undo the [PHI guardrail](#phi-guardrails). Raw capture requires an explicit
`LANGSMITH_TRACE_RAW_INPUT=1` and logs a warning at startup. Nothing is sent off the
machine unless you also set `LANGSMITH_TRACE_UPLOAD=1` with a `LANGSMITH_API_KEY`.

| Variable | Default | Purpose |
|---|---|---|
| `LANGSMITH_TRACING` | `0` | Master on/off switch |
| `LANGSMITH_TRACE_DIR` | `backend/traces` | Output folder |
| `LANGSMITH_PROJECT` | `cms-hcc-chatbot` | Recorded in each trace |
| `LANGSMITH_TRACE_RAW_INPUT` | `0` | Capture the pre-guardrail note (PHI) |
| `LANGSMITH_TRACE_MAX_FIELD` | `20000` | Truncate any single string field |
| `LANGSMITH_TRACE_UPLOAD` | `0` | Also send to the LangSmith cloud |

`GET /api/status` reports tracing state. Full guide, file schema, and `jq` recipes:
[`docs/tracing.md`](docs/tracing.md).

---

## Evaluation

A standalone harness in [`evaluation/`](evaluation/) (separate `requirements.txt`)
covers four strategies:

| Script | Measures |
|---|---|
| `run_rag_eval.py` | ICD-10 retrieval `recall@k` / `MRR` vs a gold set (no API key) |
| `run_provider_comparison.py` | Any providers/models: code F1, HCC set, score, latency |
| `run_guardrail_eval.py` | PHI detection precision/recall + clinical-term survival |
| `run_h2o_sonar_eval.py` | H2O sonar PII-leakage / toxicity / fairness evaluators |

See [`evaluation/README.md`](evaluation/README.md) for usage and interpretation.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required for f-string syntax used in age-variable naming |
| pip | Any recent | Included with Python |
| Anthropic API key | — | Only if you use the Claude provider (default). Get one at [console.anthropic.com](https://console.anthropic.com) |
| CMS reference data | — | **Bundled in this repo** under `data/` (~1.5 MB) — nothing to download |
| ~10 MB disk | — | For the pickled TF-IDF index (`tfidf_index.pkl`, ~2.4 MB) |
| Internet | Startup only | Required for Claude API calls, and for the first download of a local model |
| GPU | Optional | Only for local inference — see below. The default Claude path needs none |

### Hardware for local models (optional)

Only relevant if you use the `airllm`, `medgemma`, or `ollama` providers.

| Your hardware | Recommended provider | Install |
|---|---|---|
| No GPU | **Ollama** | `ollama pull qwen2.5:7b-instruct` |
| Any NVIDIA GPU, model fits in VRAM | **vLLM** | `pip install vllm` |
| Any NVIDIA GPU, model exceeds VRAM | **AirLLM** | torch for your card + `backend/requirements-local.txt` |

AirLLM disk cost is roughly **2x the model size** (it keeps the original snapshot
and the layer shards). GPU users must install a torch wheel matching their card —
RTX 50-series (Blackwell) needs CUDA 12.8+; GTX 10-series needs an older wheel.
[`docs/airllm.md`](docs/airllm.md) has the matrix, and the app detects and reports a
mismatch at startup.

> **Throughput reality check.** AirLLM re-reads the whole model from disk per token.
> Measured on a CPU-only laptop (Core Ultra 9 185H): **~10.4 s/token even for a 0.5B
> model** — about 90 minutes for a single 512-token answer, and dtype makes no
> difference because it is I/O-bound. On CPU, use Ollama. AirLLM is worth it when you
> have a GPU and the model does not fit in its VRAM.

### CMS reference data (bundled)

The seven CMS reference files the app needs are **committed to this repository** under
`data/`, so the project is self-contained and runs out of the box:

```
cms-hcc-chatbot/
└── data/                                      ← CMS reference files (public domain)
    ├── icd10_mappings/
    │   └── 2027 Initial ICD-10-CM Mappings.csv     ← ICD-10 descriptions
    └── v22_internal/
        ├── ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv
        ├── V22_HCC_Hierarchies.csv
        ├── V22_Diagnosis_Categories.csv
        ├── V22_Interactions.csv
        ├── V22_CE_Relative_Factors.csv
        └── V22_NE_Relative_Factors.csv
```

To use an external copy instead (same `icd10_mappings/` + `v22_internal/` layout), set
the `CMS_DATA_PATH` environment variable; both `backend/hcc_calculator.py` and
`backend/rag_icd10.py` honor it and fall back to the bundled `data/` folder otherwise.

> These files are a small subset of the full CMS release (the V22 payment-year-2027
> initial package). The complete CMS Data collection — V28, ESRD, RxHCC, FFS
> normalization files — is not needed by this V22 chatbot and is not committed.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/arunaryasomayajula/cms-hcc-risk-chatbot.git
cd cms-hcc-risk-chatbot

# 2. (Optional but recommended) Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt
```

Dependencies installed:

| Package | Purpose |
|---|---|
| `fastapi` | Web framework for the REST API |
| `uvicorn` | ASGI server |
| `anthropic` | Official Anthropic Python SDK (Claude API) |
| `openai` | Client for every OpenAI-compatible provider (vLLM, AirLLM sidecar, Ollama) |
| `langsmith` | Run-tree tracing, written to local JSON (opt-in) |
| `presidio-analyzer` + `en_core_web_sm` | PHI/PII de-identification (Presidio recognizers + spaCy NER) |
| `scikit-learn` | TF-IDF vectorizer and cosine similarity for ICD-10 search |
| `pandas` | Loading and processing CMS reference CSVs |
| `numpy` | Array operations for similarity ranking |

> **Note:** the runtime is deliberately torch-free. `chromadb`, `sentence-transformers`,
> `llm-guard`, `openpyxl` and `python-multipart` were removed (unused, or replaced) to
> keep the Docker image under 1 GB. `backend/requirements.lock` pins the exact Linux
> versions the image installs.

#### Docker (recommended)

```bash
docker compose up --build        # needs ANTHROPIC_API_KEY in the shell or .env
```

One self-contained image (~610 MB on disk, ~185 MB download) running every
pipeline step on Claude. See [`docs/docker.md`](docs/docker.md).

#### Optional — local open-weight models (AirLLM)

Only needed for the `airllm` provider. Install a torch wheel matching your GPU
**first**, then:

```bash
pip install -r backend/requirements-local.txt
```

The file documents the per-hardware torch install (CPU / cu124 / cu128 for
Blackwell) and why torch is deliberately not pinned there. Full guide:
[`docs/airllm.md`](docs/airllm.md). The `ollama` provider needs nothing installed
here — just a running Ollama.

---

## Configuration

### Anthropic API key (required only for the `claude` provider)

Set your key as an environment variable before starting:

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-api03-...
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

**macOS / Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

> The key is read by `backend/llm_providers.py` via `os.environ.get('ANTHROPIC_API_KEY')`. It is never logged or committed. Not required if every pipeline step uses a local provider (`medgemma` / `airllm` / `ollama`) — `start.bat` and `start.sh` warn but continue.

### Optional — CMS data path override

The CMS reference files are bundled under `data/` and used automatically. To point at
an external copy (same `icd10_mappings/` + `v22_internal/` layout), set one env var:

**Windows (PowerShell):**
```powershell
$env:CMS_DATA_PATH = "C:\path\to\your\data"
```

**macOS / Linux:**
```bash
export CMS_DATA_PATH="/path/to/your/data"
```

---

## Running the App

### Windows — one-click

```cmd
start.bat
```

`start.bat` verifies the API key is set, installs requirements, then starts the server.

### Manual (any OS)

```bash
cd backend
python app.py
```

Then open **http://localhost:8000** in your browser.

### Choosing the port

Every launcher listens on `PORT` (default `8000`) and binds `HOST` (default `0.0.0.0`).
`app.py` validates the value, so a bad port fails fast with a clear message.

| Launcher | Custom port |
|---|---|
| `start.sh` | `./start.sh --port 9000` or `PORT=9000 ./start.sh` |
| `start.bat` | `start.bat --port 9000` or `set PORT=9000` first |
| `python app.py` | `PORT=9000 python app.py` |
| Docker | `docker run -p 9000:9000 -e PORT=9000 ... cms-hcc-chatbot` or `... cms-hcc-chatbot --port 9000` (the image's entrypoint is `start.sh`) |
| Compose | `PORT=9000 docker compose up` (or `PORT=9000` in `.env`) |

### Startup sequence

```
[startup]  HCC Calculator loads 6 reference CSVs         ~1 second
[startup]  ICD-10 RAG loads descriptions + CC mappings   ~1 second
[startup]  TF-IDF index built (first run) OR loaded       ~3 seconds (first run) / <1 second (cached)
[ready]    Status badge turns green ✓
```

Total cold-start time is typically **under 5 seconds**. The TF-IDF index is serialised to `backend/tfidf_index.pkl` after the first build and loaded from disk on all subsequent runs.

> If the index file becomes stale (e.g. after updating the CMS data files), delete `backend/tfidf_index.pkl` and restart — it will rebuild automatically.

---

## Using the Chatbot

### 1. Set patient demographics (left sidebar)

Before submitting clinical notes, configure:

| Field | Values | Effect on scoring |
|---|---|---|
| **Date of Birth** | Any date | Age calculated as of Feb 1, 2027 (CMS cutoff) |
| **Sex** | Male / Female | Selects age/sex coefficient cell; applied to sex-specific ICD edits |
| **Original Entitlement (OREC)** | 0 Aged, 1 Disabled, 2 ESRD, 3 Both | Drives disability flags and NE age-slot logic |
| **Dual Status** | Non-dual / Partial / Full | Selects among the three aged/disabled community model variants |
| **LTI Medicaid** | Checkbox | Flags Institutional model as applicable |
| **NE Medicaid** | Checkbox | Affects new enrollee interaction variables |

### 2. Submit clinical notes

Paste any clinical note format into the text box and press **Enter** or the send button:

- H&P notes
- SOAP notes
- Discharge summaries
- Problem lists
- Physician letter excerpts

**Example input:**
```
75-year-old female with a history of Type 2 diabetes mellitus with CKD stage 4,
congestive heart failure (EF 35%), COPD on home oxygen, and a prior MI in 2021.
Current medications include metformin, furosemide, carvedilol, and tiotropium.
Most recent HbA1c 8.4%, eGFR 22. Seen for routine follow-up.
```

**Example output (verified against live pipeline):**

| ICD-10 | Description | CC | HCC |
|---|---|---|---|
| E1122 | T2DM with diabetic chronic kidney disease | 18 | HCC18 |
| N184 | Chronic kidney disease, stage 4 | 137 | HCC137 |
| I5022 | Chronic systolic (congestive) heart failure | 85 | HCC85 |
| J449 | COPD, unspecified | 111 | HCC111 |

Active interactions: `HCC85_gDiabetesMellit`, `HCC85_gCopdCF`, `HCC85_gRenal`, `CHF_gCopdCF`, `DIABETES_CHF`

`COMMUNITY_NA` risk score: **2.268** (2.27× average Medicare cost)

### 3. Read the results

The bot returns three result cards alongside its plain-English explanation:

#### ICD-10 Codes card

| Column | Meaning |
|---|---|
| Code | ICD-10-CM code (no dots, e.g. `E1122`) |
| Description | Official CMS code description |
| CC | Condition Category number this code maps to |
| Condition | The clinical condition from the notes it satisfies |
| Confidence | Claude's confidence that this code is supported by the documentation (0–100%) |

> Codes that don't appear in the V22 mapping file (e.g. history codes like `Z8719`) receive `CC N/A` and contribute 0 to the HCC score. This is expected and correct.

#### HCC Flags card

- **Blue pills** — Active HCC numbers (after hierarchy suppression), hover for description
- **Amber pills** — Diagnosis category groups triggered (CANCER, DIABETES, RENAL, …)
- **Purple pills ⚡** — Comorbidity interaction terms that fired

#### Risk Score card

All nine model scores displayed in a table. The applicable model for this patient is highlighted with a **▶ Applies** badge.

| Score color | Meaning |
|---|---|
| Green | < 0.700 — below-average cost |
| Amber | 0.700 – 1.799 — near-average |
| Red | ≥ 1.800 — significantly above average |

### 4. Ask follow-up questions

The chat is multi-turn. After receiving scores you can ask:

- *"Why was HCC85 suppressed?"*
- *"What would the score be if we added dialysis?"*
- *"Which of these HCCs has the biggest impact on the COMMUNITY_NA score?"*
- *"Explain what the CHF-COPD interaction means for this patient."*

---

## API Reference

### `GET /api/status`

Returns initialization state. The frontend polls this every 5 seconds until `initialized: true`.

```json
{
  "initialized": true,
  "error": null,
  "rag_ready": true,
  "calculator_ready": true
}
```

### `GET /api/config`

Provider and model catalog for the sidebar selectors, plus detected hardware.
Local endpoints are probed concurrently and the result cached for 30 s
(`LLM_CATALOG_TTL`), so this returns in ~1 s even when a backend is down.

```json
{
  "providers": [
    {"name": "airllm", "label": "AirLLM (local)", "local": true,
     "reachable": true, "base_url": "http://127.0.0.1:8002/v1",
     "default_model": "Qwen/Qwen2.5-0.5B-Instruct",
     "models": [{"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5 7B",
                 "note": "~15 GB - minutes per response"}]}
  ],
  "step_defaults": {"extract": "claude", "select": "claude", "explain": "claude"},
  "model_defaults": {},
  "hardware": {"backend": "cuda", "family": "Blackwell", "dtype": "bfloat16",
               "tier": "cuda_large", "summary": "CUDA - RTX 5090 - 32.0 GB - bfloat16",
               "warnings": []},
  "guardrails": {"available": true}
}
```

Model `note` text is resolved for the detected hardware tier, so the same catalog
reads honestly on a laptop and on a workstation. `hardware.warnings` carries
actionable setup problems (e.g. a torch wheel with no kernels for your GPU).

### `GET /api/local-status`

Live AirLLM sidecar state, polled by the UI only while a local request is running.
Kept separate from `/api/status` (which is polled every 5 s) because it makes a
network call.

```json
{"airllm": {"state": "generating", "model": "Qwen/Qwen2.5-7B-Instruct",
            "layers_split": 26, "layers_total": 26,
            "tokens_generated": 41, "tokens_requested": 512, "elapsed_s": 18.3}}
```

### `POST /api/chat`

Main endpoint. Accepts clinical notes and demographics; returns ICD-10 codes, HCC results, and a conversational response.

**Request body:**
```json
{
  "message": "75F with T2DM, CHF, COPD...",
  "demographics": {
    "dob":         "1950-01-21",
    "sex":         2,
    "orec":        0,
    "dual_status": 0,
    "ltimcaid":    0,
    "nemcaid":     0
  },
  "conversation_history": [],
  "providers": {"extract": "airllm", "select": "ollama", "explain": "claude"},
  "models":    {"extract": "Qwen/Qwen2.5-7B-Instruct",
                "select":  "qwen2.5:7b-instruct",
                "explain": "claude-sonnet-5-5"}
}
```

`providers` and `models` are optional and per-step; omitted steps fall back to the
`LLM_PROVIDER_*` / `LLM_MODEL_*` environment defaults. An unknown provider or a model
that is not in that provider's catalog falls back to the default with a logged
warning rather than failing the request.

**Demographics fields:**

| Field | Type | Values |
|---|---|---|
| `dob` | string | `YYYY-MM-DD` |
| `sex` | int | `1` = Male, `2` = Female |
| `orec` | int | `0` = Aged/OASI, `1` = Disabled/DIB, `2` = ESRD, `3` = DIB+ESRD |
| `dual_status` | int | `0` = Non-dual, `1` = Partial, `2` = Full |
| `ltimcaid` | int | `0` or `1` |
| `nemcaid` | int | `0` or `1` |

**Response body:**
```json
{
  "response": "Based on this note, I identified four HCC conditions...",
  "status": "success",
  "providers_used": {"extract": "airllm", "select": "ollama", "explain": "claude"},
  "models_used":    {"extract": "Qwen/Qwen2.5-7B-Instruct",
                     "select": "qwen2.5:7b-instruct", "explain": "claude-sonnet-5-5"},
  "parse_failures": [],
  "conditions_found": [
    {"condition": "Type 2 diabetes with CKD stage 4",
     "search_query": "type 2 diabetes chronic kidney disease stage 4"}
  ],
  "icd10_codes": [
    {
      "icd10": "E1122",
      "description": "Type 2 diabetes mellitus with diabetic chronic kidney disease",
      "cc": "18",
      "condition": "Type 2 diabetes with CKD stage 4",
      "rationale": "Combination code documents both T2DM and CKD cause",
      "confidence": 0.95
    }
  ],
  "hcc_result": {
    "demographics_derived": {
      "age": 76, "sex_label": "Female",
      "orec_label": "Aged/OASI", "disabl": 0, "origdis": 0
    },
    "icd10_to_cc": {"E1122": [18], "N184": [137], "I5022": [85], "J449": [111]},
    "active_hccs": [
      {"hcc": "HCC18",  "description": "Diabetes with Chronic Complications"},
      {"hcc": "HCC85",  "description": "Congestive Heart Failure"},
      {"hcc": "HCC111", "description": "Chronic Obstructive Pulmonary Disease"},
      {"hcc": "HCC137", "description": "Chronic Kidney Disease/Severe (Stage 4)"}
    ],
    "diag_categories_triggered": ["DIABETES", "CHF", "gCopdCF", "RENAL"],
    "interactions_triggered": [
      "HCC85_gDiabetesMellit", "HCC85_gCopdCF", "HCC85_gRenal",
      "CHF_gCopdCF", "DIABETES_CHF"
    ],
    "ce_scores": {
      "COMMUNITY_NA": 2.268,
      "COMMUNITY_PBA": 2.363,
      "COMMUNITY_FBA": 2.694,
      "COMMUNITY_ND": 1.949,
      "COMMUNITY_PBD": 2.047,
      "COMMUNITY_FBD": 2.367,
      "INSTITUTIONAL": 2.452
    },
    "ne_scores": {
      "NEW_ENROLLEE": 0.892,
      "SNP_NEW_ENROLLEE": 1.480
    },
    "applicable_model": "COMMUNITY_NA",
    "applicable_score": 2.268
  },
  "conversation_history": [
    {"role": "user",      "content": "75F with T2DM..."},
    {"role": "assistant", "content": "Based on this note..."}
  ]
}
```

> The `ce_scores` and `ne_scores` values in the example above are real outputs from the live pipeline, not illustrative placeholders.

---

## Project Structure

```
cms-hcc-chatbot/
│
├── backend/
│   ├── app.py              FastAPI server, pipeline orchestration, per-step LLM calls
│   ├── llm_providers.py    Provider + model catalog (Claude/vLLM/AirLLM/Ollama)
│   ├── hardware.py         Device/dtype/context policy (CPU → Blackwell)
│   ├── airllm_server.py    OpenAI-compatible AirLLM sidecar (port 8002)
│   ├── json_utils.py       Tolerant JSON extraction (shared with evaluation/)
│   ├── prompts.py          Step prompts + token budgets (shared with evaluation/)
│   ├── tracing.py          LangSmith run trees -> one JSON file per trace
│   ├── requirements-local.txt  Optional AirLLM deps
│   ├── guardrails.py       Presidio PHI de-identification
│   ├── rag_icd10.py        TF-IDF index over 10k HCC-relevant ICD-10 descriptions
│   ├── hcc_calculator.py   CMS HCC v22 scoring engine (pure Python)
│   └── requirements.txt    Python dependencies
│
├── frontend/
│   ├── index.html          Single-page chat UI (with per-step model selectors)
│   ├── styles.css          Layout, cards, color coding
│   └── app.js              API calls, card rendering, status polling
│
├── data/                   Bundled CMS reference files (public domain, ~1.5 MB)
│   ├── icd10_mappings/     2027 Initial ICD-10-CM Mappings.csv
│   └── v22_internal/       6 CMS-HCC V22 CSVs (mappings, hierarchies, factors, …)
│
├── evaluation/            Offline eval harness (RAG, provider comparison, guardrail, H2O sonar)
│   ├── datasets/           Gold clinical + PHI cases
│   └── run_*.py            Eval scripts
│
├── docs/
│   ├── vllm-medgemma.md    MedGemma-on-vLLM (and Ollama) serving guide
│   ├── airllm.md           Local open-weight models, per-GPU setup
│   └── tracing.md          LangSmith trace capture and the PHI defaults
│
├── tests/                  Runnable standalone or under pytest
│   ├── test_hardware.py    GPU-tier policy (Pascal→Blackwell) via a faked torch
│   └── test_json_utils.py  Tolerant JSON extraction cases
│
├── .gitignore              Excludes tfidf_index.pkl, .airllm_shards/, results/, venv/, .env
├── start.bat               Windows launcher (--airllm also starts the sidecar)
├── start.sh                Linux/macOS launcher
└── README.md               This file
```

**Not committed (auto-generated at runtime):**
- `backend/tfidf_index.pkl` — serialised TF-IDF matrix and vectorizer (~2.4 MB, rebuilt in ~3s if deleted)
- `backend/.airllm_shards/` — AirLLM per-layer weights (~1x model size, re-split if deleted)
- `evaluation/results/` — eval output artifacts
- `backend/traces/` — LangSmith trace JSON (sensitive; gitignored)

---

## CMS Data Files Referenced

All files are read-only at runtime and **bundled under `data/`** (CMS reference data
is public domain). The full CMS release is not committed — only the seven files below.

| File | Location under `data/` | Purpose |
|---|---|---|
| `2027 Initial ICD-10-CM Mappings.csv` | `icd10_mappings/` | ICD-10-CM code descriptions used to build the TF-IDF index |
| `ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv` | `v22_internal/` | ICD-10 → CC mapping with MCE age/sex edit columns |
| `V22_HCC_Hierarchies.csv` | `v22_internal/` | HCC hierarchy suppression rules |
| `V22_Diagnosis_Categories.csv` | `v22_internal/` | HCC → disease group (CANCER, DIABETES, CHF, …) |
| `V22_Interactions.csv` | `v22_internal/` | 26 comorbidity interaction variable definitions |
| `V22_CE_Relative_Factors.csv` | `v22_internal/` | Regression coefficients — 7 CE models |
| `V22_NE_Relative_Factors.csv` | `v22_internal/` | Regression coefficients — 2 NE models |

Additional CMS model packages (CMS-HCC V28, ESRD V21/V24, RxHCC R08) are not used by
this V22 chatbot and are not bundled. See [Extending the App](#extending-the-app) for
guidance on incorporating those models (you'll need to add their files to `data/`).

---

## Caveats and Limitations

### Clinical and compliance

> **This tool is for analytical and educational purposes only. It is not a certified medical coding tool and must not be used as the basis for CMS risk adjustment submissions.**

1. **Not a substitute for certified coders.** ICD-10 code selection requires review by a Certified Professional Coder (CPC) or Certified Risk Adjustment Coder (CRC). Claude's selections are probabilistic and may miss specificity requirements or local coverage rules.

2. **Diagnosis source validation not performed.** CMS requires that risk-adjustable diagnoses come from acceptable encounter types (face-to-face visits, inpatient stays, etc.). This tool does not validate the source of the diagnosis.

3. **Initial payment model only.** The CMS data used (`O1 initial`) reflects preliminary 2027 payment rates. CMS publishes revised rates (final, reconciliation) throughout the year; scores will differ from final settlement figures.

4. **ESRD and RxHCC not supported.** Patients with OREC = 2 (ESRD) or 3 (DIB+ESRD) should be scored under the ESRD HCC model (E21/E24), not the standard CMS-HCC V22. This chatbot will still compute V22 scores for such patients, which may understate or misrepresent their true risk.

5. **Dual status input is manual.** Full vs. partial dual eligibility is determined from CMS enrollment files. Users must enter this manually based on their own data sources.

6. **No RAF normalization.** CMS applies a plan-average normalization factor to raw risk scores before calculating payments. The scores returned here are raw risk scores, not final payment-adjusted RAF values.

### Technical

7. **TF-IDF search is keyword-based, not semantic.** Unlike neural embeddings, TF-IDF cannot infer that "heart failure" and "cardiac decompensation" are synonymous unless both terms appear in the ICD-10 description. Claude's condition extraction step (Step 1) partially mitigates this by generating specific search queries, but very unusual clinical phrasing may yield weaker candidate retrieval. In practice, ICD-10 descriptions are terse and keyword-dense, so TF-IDF performs well for this domain.

8. **TF-IDF index is not automatically updated.** If the underlying CMS ICD-10 mapping or description files change, delete `backend/tfidf_index.pkl` to force a rebuild on the next startup.

9. **No authentication.** The API has no built-in auth layer. Do not expose this server to the public internet with real patient data. Run it on `localhost` or behind an authenticated reverse proxy.

10. **Local models are much slower, and AirLLM on CPU is impractical.** AirLLM re-reads the entire model from disk for every token — measured at ~10.4 s/token even for a 0.5B model on a CPU-only laptop, i.e. ~90 minutes for one 512-token answer. Output tokens are therefore capped per hardware tier (512 on CPU, 2048 on CUDA). Use Ollama on CPU; reserve AirLLM for GPUs running models larger than VRAM.

11. **Small open-weight models often fail the JSON steps.** Extraction and code selection require structured JSON. The parser tolerates markdown fences, surrounding prose, bare arrays, and trailing commas, but a small or truncated model still produces unusable output. When that happens the response carries `status: "partial"` and a `parse_failures` entry (surfaced in the UI) instead of silently returning an empty result — code selection falls back to the top 5 TF-IDF candidates.

12. **AirLLM tracks fast-moving libraries.** It declares `transformers>=4.49,<5.13` and reaches into model internals. Verified here against transformers 5.9.0, but if a future upgrade breaks the import, run the sidecar in its own venv (see `backend/requirements-local.txt`) — the app talks to it over HTTP, so nothing else changes.

10. **Clinical notes may contain PHI.** The built-in [PHI guardrail](#phi-guardrails) de-identifies detected identifiers before any text is sent to an LLM, but detection is not guaranteed to be exhaustive. When using the **Claude** provider, sanitized text is still sent to the Anthropic API — ensure you have appropriate data use agreements. For a fully local path, set all three steps to a self-hosted provider (`medgemma`, `airllm`, or `ollama`) so no note data leaves your environment. Always confirm the guardrail reports `available: true` at `/api/status` before submitting real notes.

11. **Python 3.10+ required.** The f-string syntax used in age-variable naming requires Python 3.10 as a minimum. Python 3.13 is verified to work.

---

## Extending the App

### Add CMS-HCC V28 (115 HCCs)

The V28 reference files are not bundled (this repo ships only the V22 subset). Obtain
them from the CMS 2027 model software release and add them under `data/` (e.g.
`data/v28_internal/`). To add a V28 calculator:

1. Copy `backend/hcc_calculator.py` to `backend/hcc_calculator_v28.py`
2. Update `INTERNAL_DATA_PATH` to point to the V28 internal data folder under `data/`
3. Change `CE_MODEL_COLS` if V28 uses different model names
4. Add a new endpoint `/api/chat-v28` in `app.py` that instantiates `HCCCalculatorV28`

### Add ESRD scoring (E21/E24)

The ESRD model has a different set of HCCs and five payment models (Dialysis CE, Dialysis NE, Graft Community, Graft Institutional, Graft NE). The reference files are present under `ESRD software E2126.87.P2/` and `ESRD_v21_2027_P1_initial_package_v1/`.

Automatically route patients with `orec == 2` or `orec == 3` to the ESRD calculator.

### Add RxHCC scoring (Part D)

The RxHCC R08 model scores members for Part D drug plan payments using NDC → RxHCC mappings instead of ICD-10 → CC. Reference files are under `RxHCC software R0826.84.*`.

### Upgrade to neural embeddings

If your environment has outbound HTTPS access to HuggingFace, you can restore the original sentence-transformer search:

1. In `backend/rag_icd10.py`, replace the `TfidfVectorizer` block with:
   ```python
   from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
   ef = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
   ```
2. Reinstall `chromadb` and `sentence-transformers` if they were removed from your environment.
3. On first run the model (~90 MB) downloads and a ChromaDB collection builds (~2–5 minutes).

### Add another model provider

Providers are declared as data in [`backend/llm_providers.py`](backend/llm_providers.py).
Any OpenAI-compatible server needs no new class — add a `ProviderSpec`:

```python
PROVIDERS["lmstudio"] = ProviderSpec(
    name="lmstudio", label="LM Studio (local)", kind="openai_compat", local=True,
    base_url="http://127.0.0.1:1234/v1", api_key="EMPTY",
    default_model="my-model",
    models=(ModelSpec("my-model", "My Model", {"*": "note shown in the dropdown"}),),
    timeout=900.0, max_retries=1,
)
```

It then appears in the sidebar dropdowns, the model catalog, `/api/config`, and the
eval harness automatically. Only a non-OpenAI wire format needs a new class
implementing `complete(system, messages, max_tokens, model=None) -> str`.

### Stream Claude responses

Replace the blocking `claude.messages.create()` calls in `app.py` with `claude.messages.stream()` and return a `StreamingResponse` from FastAPI to show the explanation as it's generated.

### Deploy to the cloud

```bash
# Example: deploy to a Linux VM
pip install gunicorn
gunicorn backend.app:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

The bundled `data/` folder ships with the repo, so no external data mount is needed
(or set `CMS_DATA_PATH` to relocate it). The `tfidf_index.pkl` file will be created in `backend/` on first startup and can be committed or mounted as a volume to avoid the rebuild on each deploy.

---

## Changelog

### v1.4.0 — 2026-10-14 (Docker image under 1 GB, Claude-only runtime)

**1. Self-contained Docker image** — new `Dockerfile`, `.dockerignore`, `docker-compose.yml`.
Multi-stage build on `python:3.12-slim`, pinned `backend/requirements.lock`, native libs
stripped, TF-IDF index baked in at build time, non-root user, healthcheck, traces on a
volume. ~610 MB on disk / ~185 MB download. Every pipeline step defaults to Claude.

**2. Torch-free runtime dependencies**
- Removed unused `chromadb`, `sentence-transformers`, `openpyxl`, `python-multipart`.
- Declared `scikit-learn`, which the TF-IDF index uses but was only installed
  transitively through `sentence-transformers`.

**3. PHI guardrail moved from LLM Guard to Presidio directly**
- Same API, placeholders and "types and counts only" reporting; spaCy `en_core_web_sm`.
- MRN / SSN labelled-identifier rules now run inside the analyzer at top confidence, so
  an MRN spaCy reads as a date, or a sample SSN Presidio would discard, is still redacted.
- New `tests/test_guardrails.py`; `run_guardrail_eval.py` scores precision/recall 1.0.

**4. Default Claude model is now `claude-sonnet-5-5`** (Claude Sonnet 5.5). Sonnet 4.6
stays in the catalog; override with `CLAUDE_MODEL`.

### v1.3.0 — 2026-08-08 (local open-weight models, per-step model picker, hardware awareness)

Adds [AirLLM](https://github.com/lyogavin/airllm) and Ollama as providers, and turns
"which model?" into a runtime choice rather than an env var set at boot.

- **AirLLM sidecar** (`backend/airllm_server.py`) — an OpenAI-compatible server
  wrapping AirLLM's layer-streaming inference, so a model larger than VRAM can run
  locally. Runs as a separate process, keeping torch out of the web app and making a
  CUDA OOM or a library break survivable. Verified against the `openai` SDK.
- **Ollama as a first-class provider** — the practical local option, especially on CPU.
- **Per-step model selection** — provider *and* model dropdowns per pipeline step,
  driven by a catalog from `GET /api/config` that merges curated entries with live
  endpoint discovery. `/api/chat` accepts `models` and returns `models_used`.
- **Hardware awareness** (`backend/hardware.py`) — detects CPU / Pascal / Turing /
  Ampere / Ada / Blackwell / Apple Silicon and derives dtype, context length, token
  caps, and client timeouts. Catches the "torch wheel has no kernels for your GPU"
  failure (`sm_120` on new cards, `sm_61` on old ones) at startup with the exact fix,
  instead of at the first kernel launch. Covered by tests that fake each GPU tier.
- **Non-blocking LLM calls** — the three steps and the spaCy guardrail now run via
  `asyncio.to_thread`, so `/api/status` keeps answering during a multi-minute local
  generation. Local calls are serialized by a semaphore. New `GET /api/local-status`
  exposes live model-load and token progress.
- **Failures stop being silent** — a tolerant JSON parser (`backend/json_utils.py`,
  handles prose, fences, bare arrays, trailing commas) plus `parse_failures` in the
  response and a UI warning, replacing a fallback that made a broken model look like
  a clean empty result. The frontend now checks `resp.ok`, so a 502 shows its reason
  rather than "(no response)".
- **De-duplication** — prompts and JSON parsing moved to `backend/prompts.py` and
  `backend/json_utils.py`, shared with `evaluation/` instead of hand-mirrored.
- **Evaluation** — `--providers` now takes `provider[=model]` specs, and code diffs
  work for three or more (previously silently skipped for anything but exactly two).
- **Fixes** — `/api/config` went from 4.5 s to 1.27 s cold / 6 ms warm by probing
  endpoints concurrently with a TTL cache and defaulting to `127.0.0.1` instead of
  `localhost` (which costs a 2 s IPv6 timeout per request on Windows); conversation
  history is now bounded; the provider cache is lock-guarded now that it is reached
  from worker threads.
- **LangSmith tracing** (`backend/tracing.py`) — each request becomes a run tree
  written as one JSON file per trace into a configurable folder, with per-span
  latency. Records the de-identified note by default and stays on-disk unless
  upload is explicitly enabled.
- `start.bat --airllm` / new `start.sh` no longer hard-fail without an Anthropic key,
  so an all-local setup can start.

### v1.2.0 — 2026-07-13 (multi-provider, guardrails, evaluation, bundled data)

Major feature release. Verified end-to-end (Claude pipeline, MedGemma via a local
OpenAI-compatible endpoint, and PHI redaction).

**1. Pluggable LLM providers, per pipeline step**
- New `backend/llm_providers.py`: `ClaudeProvider` (Anthropic SDK) and `VLLMProvider`
  (OpenAI SDK → vLLM / any OpenAI-compatible server, e.g. MedGemma).
- Each step (extract / select / explain) is independently configurable via the sidebar
  dropdowns or `LLM_PROVIDER_*` env vars. New `GET /api/config`; `/api/chat` returns
  `providers_used`.

**2. PHI/PII guardrail (LLM Guard)**
- New `backend/guardrails.py`: de-identifies detected PHI to placeholders **before** any
  text reaches an LLM; no re-identification; only entity types/counts are surfaced.
  Compatible with both modern (0.3.x) and legacy (0.0.x) `llm-guard` scan APIs.

**3. Evaluation harness (`evaluation/`)**
- RAG retrieval metrics, Claude-vs-MedGemma provider comparison, guardrail PHI eval, and
  H2O sonar evaluators, with gold datasets and an isolated `requirements.txt`.

**4. Bundled CMS reference data**
- The 7 required CMS V22 CSVs are now committed under `data/` (~1.5 MB); the app is
  self-contained. `backend/hcc_calculator.py` and `backend/rag_icd10.py` default to
  `data/` and honor a `CMS_DATA_PATH` override.

**5. Frontend & docs**
- Per-step model selectors and a "🔒 PHI de-identified" banner; new
  [`docs/vllm-medgemma.md`](docs/vllm-medgemma.md) serving guide (vLLM + Ollama).

### v1.1.0 — 2026-06-02 (post-verification fixes)

Two bugs discovered during end-to-end runtime verification were fixed and committed in [679de67](https://github.com/arunaryasomayajula/cms-hcc-risk-chatbot/commit/679de67):

**1. Replaced ChromaDB + sentence-transformers with local TF-IDF (sklearn)**

- **Root cause:** The `sentence-transformers` library attempted to download the `all-MiniLM-L6-v2` model (~90 MB) from HuggingFace at startup. On networks where HuggingFace is blocked, this caused a `[WinError 10054]` connection reset. Additionally, the download ran inside FastAPI's `ThreadPoolExecutor`, which conflicted with Anaconda's system `httpx` client on the PATH, raising `RuntimeError: Cannot send a request, as the client has been closed`.
- **Fix:** `backend/rag_icd10.py` now uses `sklearn.feature_extraction.text.TfidfVectorizer` with bigrams. The index builds from local CSV files in ~3 seconds and is cached to `backend/tfidf_index.pkl`. No internet connection is needed at startup.
- **Impact:** Startup time reduced from a potential 2–5 minutes to under 5 seconds total. Search latency is <1 ms.

**2. Added markdown code-fence stripping before JSON parsing**

- **Root cause:** Both Claude calls that return JSON (condition extraction and code selection) returned their output wrapped in ` ```json ... ``` ` markdown fences. The bare `json.loads()` calls raised `JSONDecodeError` on the backtick character, which was silently caught, leaving `conditions_found = []`. As a result, the entire RAG → HCC scoring pipeline was skipped and Claude produced a hallucinated analysis from its training knowledge instead of the actual calculated values.
- **Fix:** `backend/app.py` gained a `_parse_json(text)` helper that strips leading/trailing ` ``` ` fences (with optional `json` language tag) before parsing. Both JSON parse sites now call `_parse_json()` instead of `json.loads()`.
- **Impact:** The full 6-step pipeline now runs correctly: conditions are extracted, ICD-10 codes are retrieved and validated, HCC flags are set, and all 9 risk scores are calculated from the actual CMS reference coefficients.

### v1.0.0 — 2026-06-02

Initial release.

---

## References

### CMS Official Documentation

| Resource | URL |
|---|---|
| 2027 Advance Notice / Rate Announcement | https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics |
| CMS-HCC Risk Adjustment Model Overview | https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/risk-adjustors |
| ICD-10-CM Official Guidelines | https://www.cdc.gov/nchs/icd/icd-10-cm/index.htm |
| Medicare Managed Care Manual, Chapter 7 (Risk Adjustment) | https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Internet-Only-Manuals-IOMs |
| ESRD Prospective Payment System Model | https://www.cms.gov/medicare/end-stage-renal-disease |

### CMS Software Packages (source of reference files)

The CMS Data folder contains the following packages published by CMS for payment year 2027:

| Package | HCCs | Description |
|---|---|---|
| CMS-HCC V22 — `V2226.79.O2` | 79 | Standard MA risk model (initial, blended) — **used by this chatbot** |
| CMS-HCC V28 — `V2826.115.T2` | 115 | Standard MA risk model (transition, new) |
| ESRD-HCC V21 — `E2126.87.P2` | 87 | ESRD dialysis/graft (initial) |
| ESRD-HCC V24 — `E2426.86.T2` | 86 | ESRD (transition) |
| RxHCC R08 — `R0826.84.T2/Y1/Y2` | 84 | Part D drug plan |

### Key concepts

| Term | Definition |
|---|---|
| HCC | Hierarchical Condition Category — a clinically meaningful grouping of related diagnoses |
| CC | Condition Category — the raw grouping before hierarchy suppression |
| OREC | Original Reason for Entitlement Code — how a member first qualified for Medicare |
| RAF | Risk Adjustment Factor — the final plan-level normalized risk score used in payment |
| MCE | Medicare Code Editor — CMS validity rules for ICD-10 code/patient age-sex combinations |
| Dual | A Medicare beneficiary who also receives Medicaid benefits |
| CE | Continued Enrollee — member enrolled in the plan in both the diagnosis year and payment year |
| NE | New Enrollee — member's first year in Medicare; no prior diagnosis data used |
| TF-IDF | Term Frequency–Inverse Document Frequency — a statistical text similarity measure |

### Libraries and models

| Library | License | Link |
|---|---|---|
| FastAPI | MIT | https://fastapi.tiangolo.com |
| scikit-learn | BSD 3-Clause | https://scikit-learn.org |
| Anthropic Python SDK | MIT | https://github.com/anthropics/anthropic-sdk-python |
| pandas | BSD 3-Clause | https://pandas.pydata.org |
| numpy | BSD 3-Clause | https://numpy.org |

---

## License

This repository contains only application code. The CMS reference data files (ICD-10 mappings, HCC coefficients, model software) are published by the Centers for Medicare & Medicaid Services and are in the public domain. See individual CMS publications for terms of use.

---

*Built using CMS HCC Model V22, payment year 2027 initial rates. For questions about the underlying CMS methodology, refer to the annual Advance Notice and Rate Announcement published by CMS.*
