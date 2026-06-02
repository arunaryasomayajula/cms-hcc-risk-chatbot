# CMS HCC Risk Scoring Chatbot

A web-based, AI-powered chatbot that accepts patient clinical notes, maps described conditions to **ICD-10-CM codes** using retrieval-augmented generation (RAG), and automatically calculates **CMS HCC v22 Medicare risk scores** for all nine payment models (payment year 2027).

> **Intended users:** Health plan analysts, risk adjustment coders, actuaries, and clinical informatics teams who need to quickly understand the Medicare Advantage risk impact of a patient's documented diagnoses.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [How the Pipeline Works](#how-the-pipeline-works)
4. [CMS HCC Model Primer](#cms-hcc-model-primer)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the App](#running-the-app)
9. [Using the Chatbot](#using-the-chatbot)
10. [API Reference](#api-reference)
11. [Project Structure](#project-structure)
12. [CMS Data Files Referenced](#cms-data-files-referenced)
13. [Caveats and Limitations](#caveats-and-limitations)
14. [Extending the App](#extending-the-app)
15. [References](#references)

---

## What It Does

| Capability | Detail |
|---|---|
| **Clinical NLP** | Extracts discrete diagnoses and conditions from free-text clinical notes using Claude AI |
| **ICD-10 RAG search** | Semantically searches ~10 k HCC-relevant ICD-10-CM codes using a local ChromaDB vector index |
| **Code validation** | A second Claude pass applies medical coder judgment to select the most specific, documentation-supported codes |
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
│  Step 1  Claude claude-sonnet-4-6 ──► Extract conditions (JSON)  │
│                                                                   │
│  Step 2  ChromaDB (sentence-transformers) ──► Candidate ICD-10s  │
│                                                                   │
│  Step 3  Claude claude-sonnet-4-6 ──► Select best codes (JSON)   │
│                                                                   │
│  Step 4  HCCCalculator ──► 9 risk scores                         │
│                                                                   │
│  Step 5  Claude claude-sonnet-4-6 ──► Natural language summary   │
└──────────────────────────────┬──────────────────────────────────┘
                                │ reads reference CSVs (read-only)
                                ▼
                  CMS Data folder (local, not modified)
```

### Component Responsibilities

| File | Role |
|---|---|
| `backend/app.py` | FastAPI app, request routing, Claude API calls, pipeline orchestration |
| `backend/rag_icd10.py` | Loads ICD-10 descriptions, builds/loads ChromaDB, exposes `search()` |
| `backend/hcc_calculator.py` | Pure-Python CMS HCC v22 scoring engine (no SAS dependency) |
| `frontend/index.html` | Single-page chat UI with demographics sidebar |
| `frontend/styles.css` | Styling — layout, cards, score table color coding |
| `frontend/app.js` | API calls, dynamic card rendering, status polling |

---

## How the Pipeline Works

Every time a user submits a message that looks like clinical notes, the backend executes six sequential steps:

### Step 1 — Condition extraction (Claude)

Claude is prompted to parse the free text and return a structured JSON list of distinct medical conditions with a targeted search query for each one.

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

### Step 2 — Semantic RAG search (ChromaDB)

For each condition's search query, ChromaDB performs a nearest-neighbor search over the sentence-transformer embeddings of all HCC-relevant ICD-10-CM descriptions. The top 6 candidates per condition are collected (duplicates removed).

The vector index contains only ICD-10 codes that appear in the CMS HCC v22 mapping file — codes with no HCC relevance are excluded, keeping the index lean and the results clinically meaningful.

### Step 3 — Code selection (Claude)

A second Claude call acts as a medical coder review. It receives:
- The original clinical notes
- The extracted conditions
- Up to 25 RAG candidates with descriptions, CC numbers, and similarity scores

Claude selects the most specific, documentation-supported codes, removes duplicates at lower specificity, and returns each code with a rationale and confidence score (0–1).

### Step 4 — HCC calculation (Python)

`HCCCalculator.calculate()` implements the full CMS HCC v22 scoring pipeline:

1. **Age/sex demographic variables** — 24 CE buckets (`F65_69`, `M75_79`, …) and 32 NE buckets, computed against **February 1, 2027** (the CMS payment-year age cutoff).

2. **ICD-10 → CC mapping** — Each code is looked up in `ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv`. MCE (Medicare Code Editor) age/sex edits are applied: maternity codes are rejected for males, pediatric codes for adults, etc.

3. **HCC hierarchies** — Within each clinical family, only the most severe condition counts. For example, if a patient has metastatic cancer (HCC8), the less severe neoplasm HCCs (HCC9–HCC12) are zeroed out.

4. **Diagnosis category flags** — Ten disease group flags (`CANCER`, `DIABETES`, `CHF`, `RENAL`, etc.) are set to 1 if any HCC in that group is active.

5. **Interaction terms** — 26 comorbidity interaction variables (e.g., `HCC85_gCopdCF` = CHF × COPD) capture the extra cost when specific conditions co-occur.

6. **Score = Σ(flag × coefficient)** — Every active flag is multiplied by its model-specific regression coefficient from `V22_CE_Relative_Factors.csv` or `V22_NE_Relative_Factors.csv`. The dot product is the risk score for that model.

### Step 5 — Natural language explanation (Claude)

Claude receives the complete analysis results and generates a plain-English explanation of which HCCs fired, what interactions are present, what the score means, and which payment model applies to the patient.

### Step 6 — Response assembly

The API returns:
- `response` — Claude's explanation
- `icd10_codes` — Selected codes with description, CC, condition, rationale, confidence
- `hcc_result` — Active HCCs, triggered interactions, all 9 scores, applicable model
- `conversation_history` — For multi-turn follow-up questions

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

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required for `f'{x if y else "GT"}'` syntax used throughout |
| pip | Any recent | Included with Python |
| Anthropic API key | — | Get one at [console.anthropic.com](https://console.anthropic.com) |
| CMS Data folder | — | Must be at `../CMS Data/` relative to this repo (see below) |
| ~2 GB disk | — | For sentence-transformer model download and ChromaDB index |
| Internet (first run) | — | To download `all-MiniLM-L6-v2` model (~90 MB) |

### CMS Data folder location

The backend expects the CMS reference files at the path **one level above this repository**:

```
Downloads/
├── CMS Data/                         ← CMS reference files (not in this repo)
│   ├── 2027-initial-icd-10-cm-mappings/
│   │   └── 2027 Initial ICD-10-CM Mappings.csv
│   └── python-2027-initial-model-software/
│       └── CMS_HCC_v22_2027_O1_initial_package_v1/
│           └── software/CMS_HCC_v22/data/input/internal/
│               ├── ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv
│               ├── V22_HCC_Hierarchies.csv
│               ├── V22_Diagnosis_Categories.csv
│               ├── V22_Interactions.csv
│               ├── V22_CE_Relative_Factors.csv
│               └── V22_NE_Relative_Factors.csv
└── cms-hcc-risk-chatbot/             ← this repository
    ├── backend/
    └── frontend/
```

If your `CMS Data` folder is in a different location, edit the `CMS_DATA_PATH` constant at the top of both `backend/hcc_calculator.py` and `backend/rag_icd10.py`.

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
| `chromadb` | Local vector database for the ICD-10 index |
| `sentence-transformers` | `all-MiniLM-L6-v2` embeddings for semantic search |
| `pandas` | Loading and processing CMS reference CSVs |
| `numpy` | Numerical operations |
| `python-multipart` | FastAPI form support |
| `openpyxl` | Reading `.xlsx` files if needed |

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

> The key is read by `backend/app.py` via `os.environ.get('ANTHROPIC_API_KEY')`. It is never logged or committed.

### Optional — CMS Data path override

If your CMS data is not at `../CMS Data/` relative to the repo, edit the two path constants:

**`backend/hcc_calculator.py` line 11:**
```python
CMS_DATA_PATH = os.path.abspath('/your/path/to/CMS Data')
```

**`backend/rag_icd10.py` line 9:**
```python
CMS_DATA_PATH = os.path.abspath('/your/path/to/CMS Data')
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

### First-run startup sequence

```
[startup]  HCC Calculator loads 6 reference CSVs          ~1 second
[startup]  ICD-10 RAG downloads sentence-transformer model  ~30 seconds (first run only, cached after)
[startup]  ChromaDB vector index is built                   ~2-5 minutes (first run only, cached after)
[ready]    Status badge turns green ✓
```

On subsequent runs, the ChromaDB index at `backend/chroma_db/` is loaded from disk in seconds.

> **Do not interrupt the first run** during index building. If you do, delete `backend/chroma_db/` and restart.

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

### 3. Read the results

The bot returns three result cards alongside its plain-English explanation:

#### ICD-10 Codes card

| Column | Meaning |
|---|---|
| Code | ICD-10-CM code (no dots, e.g. `E1165`) |
| Description | Official CMS code description |
| CC | Condition Category number this code maps to |
| Condition | The clinical condition from the notes it satisfies |
| Confidence | Claude's confidence that this code is supported by the documentation (0–100%) |

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

Returns initialization state. The frontend polls this every 5 seconds.

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
      "icd10": "E1165",
      "description": "Type 2 diabetes mellitus with hyperglycemia",
      "cc": "18",
      "condition": "Type 2 diabetes with CKD stage 4",
      "rationale": "Documentation supports T2DM with CKD complication",
      "confidence": 0.91
    }
  ],
  "hcc_result": {
    "demographics_derived": {
      "age": 75, "sex_label": "Female",
      "orec_label": "Aged/OASI", "disabl": 0, "origdis": 0
    },
    "icd10_to_cc": {"E1165": [18]},
    "active_hccs": [
      {"hcc": "HCC18", "description": "Diabetes with Chronic Complications"},
      {"hcc": "HCC85", "description": "Congestive Heart Failure"},
      {"hcc": "HCC111", "description": "Chronic Obstructive Pulmonary Disease"},
      {"hcc": "HCC137", "description": "Chronic Kidney Disease/Severe (Stage 4)"}
    ],
    "diag_categories_triggered": ["DIABETES", "CHF", "gCopdCF", "RENAL"],
    "interactions_triggered": ["HCC85_gDiabetesMellit", "HCC85_gCopdCF", "HCC85_gRenal"],
    "ce_scores": {
      "COMMUNITY_NA": 2.341,
      "COMMUNITY_PBA": 2.287,
      "COMMUNITY_FBA": 2.563,
      "COMMUNITY_ND": 2.108,
      "COMMUNITY_PBD": 2.034,
      "COMMUNITY_FBD": 2.312,
      "INSTITUTIONAL": 1.654
    },
    "ne_scores": {
      "NEW_ENROLLEE": 1.324,
      "SNP_NEW_ENROLLEE": 1.987
    },
    "applicable_model": "COMMUNITY_NA",
    "applicable_score": 2.341
  },
  "conversation_history": [
    {"role": "user", "content": "75F with T2DM..."},
    {"role": "assistant", "content": "Based on this note..."}
  ]
}
```

---

## Project Structure

```
cms-hcc-risk-chatbot/
│
├── backend/
│   ├── app.py              FastAPI server, pipeline orchestration, Claude calls
│   ├── rag_icd10.py        ChromaDB vector index — ICD-10 semantic search
│   ├── hcc_calculator.py   CMS HCC v22 scoring engine (pure Python)
│   └── requirements.txt    Python dependencies
│
├── frontend/
│   ├── index.html          Single-page chat UI
│   ├── styles.css          Layout, cards, color coding
│   └── app.js              API calls, card rendering, status polling
│
├── .gitignore              Excludes chroma_db/, venv/, .env, __pycache__
├── start.bat               Windows one-click launcher
└── README.md               This file
```

**Not committed (auto-generated at runtime):**
- `backend/chroma_db/` — ChromaDB vector index (rebuilt on first run if absent)

---

## CMS Data Files Referenced

All files are read-only at runtime. No CMS data is committed to this repository.

| File | Location under `CMS Data/` | Purpose |
|---|---|---|
| `2027 Initial ICD-10-CM Mappings.csv` | `2027-initial-icd-10-cm-mappings/` | ICD-10-CM code descriptions used to build the RAG index |
| `ICD10_CC_mappings_CMS_HCC_2027_v22_initial.csv` | `python-2027-initial-model-software/.../internal/` | ICD-10 → CC mapping with MCE age/sex edit columns |
| `V22_HCC_Hierarchies.csv` | same `internal/` folder | HCC hierarchy suppression rules |
| `V22_Diagnosis_Categories.csv` | same `internal/` folder | HCC → disease group (CANCER, DIABETES, CHF, …) |
| `V22_Interactions.csv` | same `internal/` folder | 26 comorbidity interaction variable definitions |
| `V22_CE_Relative_Factors.csv` | same `internal/` folder | Regression coefficients — 7 CE models |
| `V22_NE_Relative_Factors.csv` | same `internal/` folder | Regression coefficients — 2 NE models |

The CMS Data folder also contains additional model packages (CMS-HCC V28, ESRD V21/V24, RxHCC R08) which are not used by this chatbot but are present in the reference files. See [Extending the App](#extending-the-app) for guidance on incorporating those models.

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

7. **First-run index build takes 2–5 minutes.** During this time, the API returns an "initializing" message. Do not restart the server.

8. **ChromaDB index is not automatically updated.** If the underlying CMS ICD-10 mapping files change, delete `backend/chroma_db/` to force a rebuild on the next startup.

9. **No authentication.** The API has no built-in auth layer. Do not expose this server to the public internet with real patient data. Run it on `localhost` or behind an authenticated reverse proxy.

10. **Clinical notes may contain PHI.** Text submitted via the chatbot is sent to the Anthropic API. Ensure you have appropriate data use agreements and de-identify notes where required before using with real patient data.

11. **Sentence-transformer model requires internet on first download.** After the first run, the model is cached locally by Hugging Face at `~/.cache/huggingface/`.

12. **Python 3.10+ required.** The f-string syntax `f'{x if y else "GT"}'` used in age-variable naming requires Python 3.12+ on some platforms if the string contains quotes. Python 3.10 is the tested minimum.

---

## Extending the App

### Add CMS-HCC V28 (115 HCCs)

The CMS Data folder already contains the V28 reference files. To add a V28 calculator:

1. Copy `backend/hcc_calculator.py` to `backend/hcc_calculator_v28.py`
2. Update `INTERNAL_DATA_PATH` to point to the V28 internal data folder
3. Change `CE_MODEL_COLS` if V28 uses different model names
4. Add a new endpoint `/api/chat-v28` in `app.py` that instantiates `HCCCalculatorV28`

### Add ESRD scoring (E21/E24)

The ESRD model has a different set of HCCs and five payment models (Dialysis CE, Dialysis NE, Graft Community, Graft Institutional, Graft NE). The reference files are present under `ESRD software E2126.87.P2/` and `ESRD_v21_2027_P1_initial_package_v1/`.

Automatically route patients with `orec == 2` or `orec == 3` to the ESRD calculator.

### Add RxHCC scoring (Part D)

The RxHCC R08 model scores members for Part D drug plan payments using NDC → RxHCC mappings instead of ICD-10 → CC. Reference files are under `RxHCC software R0826.84.*`.

### Stream Claude responses

Replace the blocking `claude.messages.create()` calls in `app.py` with `claude.messages.stream()` and return a `StreamingResponse` from FastAPI to show the explanation as it's generated.

### Deploy to the cloud

```bash
# Example: deploy to a Linux VM
pip install gunicorn
gunicorn backend.app:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Ensure the CMS Data path is absolute and accessible on the server. The `chroma_db/` directory should be on persistent storage.

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
| CMS-HCC V22 — `V2226.79.O2` | 79 | Standard MA risk model (initial, blended) |
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

### Libraries and models

| Library | License | Link |
|---|---|---|
| FastAPI | MIT | https://fastapi.tiangolo.com |
| ChromaDB | Apache 2.0 | https://www.trychroma.com |
| sentence-transformers (`all-MiniLM-L6-v2`) | Apache 2.0 | https://www.sbert.net |
| Anthropic Python SDK | MIT | https://github.com/anthropics/anthropic-sdk-python |
| pandas | BSD 3-Clause | https://pandas.pydata.org |

---

## License

This repository contains only application code. The CMS reference data files (ICD-10 mappings, HCC coefficients, model software) are published by the Centers for Medicare & Medicaid Services and are in the public domain. See individual CMS publications for terms of use.

---

*Built using CMS HCC Model V22, payment year 2027 initial rates. For questions about the underlying CMS methodology, refer to the annual Advance Notice and Rate Announcement published by CMS.*
