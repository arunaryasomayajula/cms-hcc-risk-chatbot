# CMS HCC Risk Scoring Chatbot

A web-based, AI-powered chatbot that accepts patient clinical notes, maps described conditions to **ICD-10-CM codes** using retrieval-augmented generation (RAG), and automatically calculates **CMS HCC v22 Medicare risk scores** for all nine payment models (payment year 2027).

> **Intended users:** Health plan analysts, risk adjustment coders, actuaries, and clinical informatics teams who need to quickly understand the Medicare Advantage risk impact of a patient's documented diagnoses.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [How the Pipeline Works](#how-the-pipeline-works)
4. [CMS HCC Model Primer](#cms-hcc-model-primer)
5. [Model Providers (Claude / MedGemma via vLLM)](#model-providers-claude--medgemma-via-vllm)
6. [PHI Guardrails](#phi-guardrails)
7. [Evaluation](#evaluation)
8. [Prerequisites](#prerequisites)
9. [Installation](#installation)
10. [Configuration](#configuration)
11. [Running the App](#running-the-app)
12. [Using the Chatbot](#using-the-chatbot)
13. [API Reference](#api-reference)
14. [Project Structure](#project-structure)
15. [CMS Data Files Referenced](#cms-data-files-referenced)
16. [Caveats and Limitations](#caveats-and-limitations)
17. [Extending the App](#extending-the-app)
18. [Changelog](#changelog)
19. [References](#references)

---

## What It Does

| Capability | Detail |
|---|---|
| **Clinical NLP** | Extracts discrete diagnoses and conditions from free-text clinical notes using an LLM (Claude or self-hosted MedGemma) |
| **PHI guardrail** | De-identifies names, dates, MRNs, SSNs, etc. (LLM Guard) before any text reaches an LLM |
| **Pluggable models** | Each pipeline step is independently served by Claude or MedGemma (via vLLM / any OpenAI-compatible endpoint) |
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
│  Guard   LLM Guard ──► de-identify PHI in the note (Anonymize)   │
│                                                                   │
│  Step 1  LLM (Claude | MedGemma) ──► Extract conditions (JSON)   │
│                                                                   │
│  Step 2  TF-IDF index (sklearn, local) ──► Candidate ICD-10s    │
│                                                                   │
│  Step 3  LLM (Claude | MedGemma) ──► Select best codes (JSON)    │
│                                                                   │
│  Step 4  HCCCalculator ──► 9 risk scores                         │
│                                                                   │
│  Step 5  LLM (Claude | MedGemma) ──► Natural language summary    │
│          (provider is configurable per step)                     │
└──────────────────────────────┬──────────────────────────────────┘
                                │ reads reference CSVs (read-only)
                                ▼
                  data/ folder (bundled CMS reference CSVs)
```

### Component Responsibilities

| File | Role |
|---|---|
| `backend/app.py` | FastAPI app, request routing, Claude API calls, pipeline orchestration |
| `backend/rag_icd10.py` | Loads ICD-10 descriptions, builds/caches a TF-IDF index, exposes `search()` |
| `backend/hcc_calculator.py` | Pure-Python CMS HCC v22 scoring engine (no SAS dependency) |
| `frontend/index.html` | Single-page chat UI with demographics sidebar |
| `frontend/styles.css` | Styling — layout, cards, score table color coding |
| `frontend/app.js` | API calls, dynamic card rendering, status polling |

---

## How the Pipeline Works

Every time a user submits a message that looks like clinical notes, the backend executes six sequential steps.

> **Note:** The note is first **de-identified** (see [PHI Guardrails](#phi-guardrails)), and the three LLM steps (1, 3, 5) each run on the **configured provider — Claude or MedGemma** (see [Model Providers](#model-providers-claude--medgemma-via-vllm)). "Claude" below refers to the default provider; the flow is identical for MedGemma.

### Step 1 — Condition extraction (LLM: Claude or MedGemma)

The LLM is prompted to parse the free text and return a structured JSON list of distinct medical conditions with a targeted search query for each one. Claude may return JSON wrapped in markdown code fences (` ```json ... ``` `); the backend strips these automatically before parsing.

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

### Step 3 — Code selection (LLM: Claude or MedGemma)

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

### Step 5 — Natural language explanation (LLM: Claude or MedGemma)

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

## Model Providers (Claude / MedGemma via vLLM)

The three LLM steps — condition extraction, ICD-10 code selection, and the
natural-language explanation — can each be served by a **different** model:

- **Claude** (default) via the Anthropic API.
- **MedGemma** (`google/medgemma-27b-it` / `-4b-it`) served locally by **vLLM**
  through its OpenAI-compatible API.

Selection is **per pipeline step**, set either from the sidebar dropdowns
(populated from `GET /api/config`) or via environment defaults:

| Variable | Default | Values |
|---|---|---|
| `LLM_PROVIDER_EXTRACT` | `claude` | `claude` \| `medgemma` |
| `LLM_PROVIDER_SELECT`  | `claude` | `claude` \| `medgemma` |
| `LLM_PROVIDER_EXPLAIN` | `claude` | `claude` \| `medgemma` |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Anthropic model id |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM OpenAI endpoint |
| `VLLM_MODEL` | `google/medgemma-27b-it` | model served by vLLM |
| `VLLM_API_KEY` | `EMPTY` | matches vLLM's `--api-key` |

The abstraction lives in [`backend/llm_providers.py`](backend/llm_providers.py); any
OpenAI-compatible server works by pointing `VLLM_BASE_URL`/`VLLM_MODEL` at it. See
[`docs/vllm-medgemma.md`](docs/vllm-medgemma.md) for the full serving guide. Each
`/api/chat` response includes `providers_used` so you can confirm which model served
each step.

---

## PHI Guardrails

Before any clinical note is sent to an LLM (Claude **or** MedGemma), it is scanned
by **LLM Guard**'s `Anonymize` scanner. Detected PHI/PII — names, dates, phone
numbers, emails, SSNs, medical-record numbers, locations — is **de-identified**
(replaced with placeholders like `[REDACTED_PERSON_1]`) and the pipeline proceeds on
the sanitized text. See [`backend/guardrails.py`](backend/guardrails.py).

- **De-identify, don't block** — clinical content (diagnoses, labs, meds) is kept.
- **No re-identification** — the placeholder→value map (Vault) never leaves the
  process and is discarded after each request, so PHI never enters the model output
  or stored conversation history.
- **Only types are surfaced** — the `/api/chat` response returns a `guardrail` block
  with entity *types* and counts (never raw values); the UI shows a "🔒 PHI
  de-identified" banner.
- **Graceful fallback** — if `llm-guard` or its spaCy model isn't installed, the app
  still boots and `/api/status` reports the guardrail as off.

| Variable | Default | Purpose |
|---|---|---|
| `GUARDRAILS_ENABLED` | `true` | set `false` to disable de-identification |

> After installing `llm-guard`, download the spaCy model it relies on:
> `python -m spacy download en_core_web_lg`
>
> **Platform note:** `llm-guard` pulls native packages (`spacy`, `thinc`, `blis`,
> `sentencepiece`). On Linux/macOS these install from wheels cleanly. On **Windows +
> Python 3.13** some versions have no prebuilt wheel and need a C++ toolchain; if pip
> tries to compile, force wheels with
> `pip install llm-guard "spacy>=3.8" --only-binary=:all:`, or use a Python 3.11/3.12
> environment. The guardrail code in `backend/guardrails.py` is compatible with both
> the modern (0.3.x) and legacy (0.0.x) `llm-guard` scan APIs.

---

## Evaluation

A standalone harness in [`evaluation/`](evaluation/) (separate `requirements.txt`)
covers four strategies:

| Script | Measures |
|---|---|
| `run_rag_eval.py` | ICD-10 retrieval `recall@k` / `MRR` vs a gold set (no API key) |
| `run_provider_comparison.py` | Claude vs MedGemma: code F1, HCC set, score, latency |
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
| ~100 MB disk | — | For the pickled TF-IDF index (`tfidf_index.pkl`, ~80 MB) |
| Internet | Startup only | Required only for Claude API calls — no model downloads needed |

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
| `uvicorn[standard]` | ASGI server |
| `anthropic` | Official Anthropic Python SDK (Claude API) |
| `openai` | Client for MedGemma served via vLLM's OpenAI-compatible API |
| `llm-guard` | PHI/PII de-identification (Anonymize scanner + Vault) |
| `scikit-learn` | TF-IDF vectorizer and cosine similarity for ICD-10 search |
| `pandas` | Loading and processing CMS reference CSVs |
| `numpy` | Array operations for similarity ranking |
| `python-multipart` | FastAPI form support |
| `openpyxl` | Reading `.xlsx` files if needed |

> **Note:** `chromadb` and `sentence-transformers` are listed in `requirements.txt` for completeness but are not used at runtime. The active search engine is sklearn's `TfidfVectorizer` — fully local, no model download, no internet required at startup.

---

## Configuration

### Required — Anthropic API key

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

> The key is read by `backend/llm_providers.py` via `os.environ.get('ANTHROPIC_API_KEY')`. It is never logged or committed. Not required if every pipeline step is set to MedGemma.

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
  "conversation_history": []
}
```

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
│   ├── llm_providers.py    Claude + MedGemma/vLLM provider abstraction
│   ├── guardrails.py       LLM Guard PHI de-identification
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
│   └── vllm-medgemma.md    MedGemma-on-vLLM serving guide
│
├── .gitignore              Excludes tfidf_index.pkl, evaluation/results/, venv/, .env
├── start.bat               Windows one-click launcher
└── README.md               This file
```

**Not committed (auto-generated at runtime):**
- `backend/tfidf_index.pkl` — serialised TF-IDF matrix and vectorizer (~80 MB, rebuilt in ~3s if deleted)
- `evaluation/results/` — eval output artifacts

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

10. **Clinical notes may contain PHI.** The built-in [PHI guardrail](#phi-guardrails) de-identifies detected identifiers before any text is sent to an LLM, but detection is not guaranteed to be exhaustive. When using the **Claude** provider, sanitized text is still sent to the Anthropic API — ensure you have appropriate data use agreements. For a fully local path, run **MedGemma** (self-hosted) for all steps so no note data leaves your environment. Always confirm the guardrail reports `available: true` at `/api/status` before submitting real notes.

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
