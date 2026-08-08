"""
Prompt text and token budgets for the three LLM pipeline steps.

Single source of truth for `backend/app.py` and `evaluation/common.py`. Those two
files previously carried byte-identical copies of these prompts with a docstring
warning to hand-mirror any edit — a hazard that gets worse once local open-weight
models need prompt tuning that Claude does not.
"""
import json
from typing import Dict, List

# ── Step 2: condition extraction ─────────────────────────────────────────────

EXTRACT_SYSTEM = (
    "Extract distinct medical diagnoses and conditions from clinical notes. "
    "Return ONLY a JSON object like: "
    '{"conditions": [{"condition": "Type 2 diabetes mellitus", '
    '"search_query": "type 2 diabetes mellitus without complication"}, ...]}'
    " Include chronic and acute conditions. Be specific."
)

# ── Step 4: ICD-10 code selection ────────────────────────────────────────────

SELECT_SYSTEM = (
    "You are a certified medical coder. Given clinical notes and ICD-10 "
    "candidate codes from semantic search, select the most appropriate codes. "
    "Return ONLY JSON: "
    '{"selected_codes": [{"icd10": "E119", "description": "...", '
    '"cc": "19", "condition": "...", "rationale": "...", "confidence": 0.95}]}'
    " Use the MOST SPECIFIC code supported by the documentation. "
    "Remove codes that are not supported or are duplicates at a lower specificity."
)

# ── Step 6: conversational explanation ───────────────────────────────────────

EXPLAIN_SYSTEM = """You are a clinical coding and Medicare risk adjustment specialist.
You help users analyze patient clinical notes, identify ICD-10 diagnoses, and understand
CMS HCC (Hierarchical Condition Category) risk scores for Medicare payment year 2027.

Key knowledge:
- HCC risk scores are relative: 1.0 = average Medicare beneficiary cost
- Higher scores = higher predicted cost → higher CMS capitation payment
- Hierarchies prevent double-counting (e.g., metastatic cancer suppresses lower cancer codes)
- Interactions capture extra cost of comorbidities together (e.g., CHF + COPD)
- CE = Continued Enrollee (7 models), NE = New Enrollee (2 models)
- Model selection: Community vs Institutional, Aged (65+) vs Disabled (<65), Dual vs Non-dual

Be concise, clinically accurate, and explain risk implications clearly."""


# ── Budgets ──────────────────────────────────────────────────────────────────

#: Requested output tokens per step. Local providers may cap these further based
#: on detected hardware — see `hardware.max_new_tokens_cap`.
MAX_TOKENS: Dict[str, int] = {
    "extract": 800,
    "select": 1500,
    "explain": 2000,
}

#: Number of prior user/assistant exchanges kept in the explain step's context.
#: Unbounded history overflows a local model's finite context window; even for
#: Claude it is unnecessary cost.
MAX_HISTORY_TURNS = 6

#: TF-IDF candidates handed to the select step. Pretty-printed at indent=2 this is
#: already ~1.3-1.7k tokens; raising it risks blowing small local context windows.
MAX_SELECT_CANDIDATES = 25


def build_select_user(user_text: str, conditions_found: List[Dict], candidates: List[Dict]) -> str:
    """User-turn content for the code-selection step."""
    return (
        f"Clinical notes:\n{user_text}\n\n"
        f"Extracted conditions: {json.dumps(conditions_found)}\n\n"
        f"ICD-10 candidates:\n{json.dumps(candidates[:MAX_SELECT_CANDIDATES], indent=2)}"
    )
