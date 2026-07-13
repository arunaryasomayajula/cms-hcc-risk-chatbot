# Evaluation Harness

Offline evaluation for the CMS HCC chatbot. Each script runs directly against the
real backend modules (`rag_icd10`, `hcc_calculator`, `llm_providers`, `guardrails`)
— no running server required. Results are written to `evaluation/results/`.

## Setup

```bash
# From the repo root, with the app deps already installed:
pip install -r evaluation/requirements.txt
# Guardrail eval also needs the spaCy model used by LLM Guard:
python -m spacy download en_core_web_lg
```

The CMS reference CSVs are bundled in the repo under `data/`, so the RAG index and
HCC calculator load with no extra setup (override with `CMS_DATA_PATH` if needed).

## Scripts

| Script | What it measures | Needs LLM API? |
|---|---|---|
| `run_rag_eval.py` | ICD-10 retrieval quality: `recall@k`, `MRR`, pool precision vs the gold codes | No |
| `run_provider_comparison.py` | Claude vs MedGemma: ICD-10 code F1, HCC set, risk score, per-step latency | Yes (both providers) |
| `run_guardrail_eval.py` | PHI detection precision/recall + that clinical terms survive de-identification | No (local LLM Guard) |
| `run_h2o_sonar_eval.py` | PII leakage, toxicity, fairness/bias over model outputs (H2O sonar, with fallback) | Only when generating fresh outputs |

## Running

```bash
cd evaluation

# 1. RAG retrieval metrics (fast, no API key)
python run_rag_eval.py --k 6 10

# 2. Guardrail / PHI de-identification metrics (no API key)
python run_guardrail_eval.py

# 3. Provider comparison (needs ANTHROPIC_API_KEY and a running vLLM/MedGemma)
python run_provider_comparison.py --providers claude medgemma

# 4. H2O sonar LLM evaluators (PII leakage / toxicity / bias)
python run_h2o_sonar_eval.py --provider claude
python run_h2o_sonar_eval.py --from-cache   # reuse cached outputs, no API call
```

## Datasets

- `datasets/clinical_cases.jsonl` — gold clinical notes with `expected_icd10`,
  `expected_hccs`, `search_queries`, and demographics. Extend this to grow coverage.
- `datasets/phi_cases.jsonl` — notes with labeled `expected_entities` (PHI types)
  and `must_survive` clinical terms.

## Interpreting results

- **RAG recall@k** near 1.0 means the retriever surfaces the correct codes for the
  LLM to select. Low recall with `missing` codes points to a TF-IDF gap (try a
  better `search_query` in the dataset, or a semantic retriever).
- **Provider comparison** `code_f1` shows selection accuracy vs gold; `Δ codes`
  highlights where the two models disagree. Latency is per-step wall time.
- **Guardrail** wants high `entity_recall` (few missed identifiers) and
  `clinical_survival_rate == 1.0` (no clinical terms redacted).
- **H2O sonar** `pii_leakage.leak_rate` **must be 0** — any leak means a raw
  identifier reached the model output despite de-identification.

## Notes

- `run_h2o_sonar_eval.py` targets h2o-sonar's `PiiLeakageEvaluator`,
  `ToxicityEvaluator`, and `FairnessBiasEvaluator`. The h2o-sonar programmatic API
  varies by version; if your installed version exposes a different entry point,
  adapt `run_h2o_sonar()` — the built-in fallback keeps the script usable meanwhile.
- The pipeline in `common.py` mirrors `backend/app.py`. If you change the prompts
  in `app.py`, update `common.py` to keep eval representative.
