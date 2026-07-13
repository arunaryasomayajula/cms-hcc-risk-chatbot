"""
Shared helpers for the evaluation harness.

Adds the backend package to sys.path and exposes a compact re-implementation of
the app.py pipeline so eval scripts can run cases directly against the real RAG
index, HCC calculator, guardrails, and LLM providers — without a live server.

The prompt text here is kept identical to backend/app.py; if you change the
prompts there, mirror them here.
"""
import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("evaluation")

# Windows consoles default to cp1252 and choke on the box-drawing chars used in
# the eval report output; force UTF-8 so scripts print cleanly everywhere.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "backend"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

DATASETS_DIR = os.path.join(_HERE, "datasets")
RESULTS_DIR = os.path.join(_HERE, "results")


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


# ── Lazy singletons for the heavy backend components ─────────────────────────

_rag = None
_calc = None


def get_rag():
    global _rag
    if _rag is None:
        from rag_icd10 import ICD10RAG
        _rag = ICD10RAG()
        _rag.initialize()
    return _rag


def get_calculator():
    global _calc
    if _calc is None:
        from hcc_calculator import HCCCalculator
        _calc = HCCCalculator()
        _calc.load_data()
    return _calc


# ── Pipeline (mirrors backend/app.py) ────────────────────────────────────────

def _parse_json(text: str) -> dict:
    import re
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def run_pipeline(
    note: str,
    demographics: Dict,
    providers: Optional[Dict[str, str]] = None,
    deidentify: bool = False,
) -> Dict:
    """Run the full extract → search → select → calculate → explain pipeline.

    Returns a dict with the intermediate artifacts plus per-step latency so eval
    scripts can compare providers. `providers` maps step -> provider name.
    """
    import llm_providers
    import guardrails as guardrails_mod

    step_providers = llm_providers.resolve_step_providers(providers)
    timings: Dict[str, float] = {}

    def run_step(step, system, messages, max_tokens):
        provider = llm_providers.get_provider(step_providers[step])
        t0 = time.perf_counter()
        out = provider.complete(system=system, messages=messages, max_tokens=max_tokens)
        timings[step] = round(time.perf_counter() - t0, 3)
        return out

    text = note
    guardrail_info = None
    if deidentify:
        deid = guardrails_mod.deidentify(note)
        text = deid.sanitized_text
        guardrail_info = deid.to_dict()

    # Step 2: extract conditions
    extract_text = run_step(
        "extract",
        system=(
            "Extract distinct medical diagnoses and conditions from clinical notes. "
            "Return ONLY a JSON object like: "
            '{"conditions": [{"condition": "Type 2 diabetes mellitus", '
            '"search_query": "type 2 diabetes mellitus without complication"}, ...]}'
            " Include chronic and acute conditions. Be specific."
        ),
        messages=[{"role": "user", "content": text}],
        max_tokens=800,
    )
    try:
        conditions = _parse_json(extract_text).get("conditions", [])
    except Exception:
        conditions = []

    # Step 3: RAG search
    rag = get_rag()
    candidates, seen = [], set()
    for cond in conditions:
        q = cond.get("search_query") or cond.get("condition", "")
        for hit in rag.search(q, n_results=6):
            if hit["icd10"] not in seen:
                seen.add(hit["icd10"])
                hit["for_condition"] = cond.get("condition", "")
                candidates.append(hit)

    # Step 4: select codes
    icd10_results = []
    if candidates:
        select_text = run_step(
            "select",
            system=(
                "You are a certified medical coder. Given clinical notes and ICD-10 "
                "candidate codes from semantic search, select the most appropriate codes. "
                "Return ONLY JSON: "
                '{"selected_codes": [{"icd10": "E119", "description": "...", '
                '"cc": "19", "condition": "...", "rationale": "...", "confidence": 0.95}]}'
                " Use the MOST SPECIFIC code supported by the documentation. "
                "Remove codes that are not supported or are duplicates at a lower specificity."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Clinical notes:\n{text}\n\n"
                    f"Extracted conditions: {json.dumps(conditions)}\n\n"
                    f"ICD-10 candidates:\n{json.dumps(candidates[:25], indent=2)}"
                ),
            }],
            max_tokens=1500,
        )
        try:
            icd10_results = _parse_json(select_text).get("selected_codes", [])
        except Exception:
            icd10_results = candidates[:5]

    # Step 5: HCC calculation
    hcc_result = None
    codes = [r.get("icd10", "") for r in icd10_results if r.get("icd10")]
    if codes and demographics:
        try:
            hcc_result = get_calculator().calculate(demographics, codes)
        except Exception as exc:
            logger.warning("HCC calc failed: %s", exc)

    return {
        "conditions": conditions,
        "candidates": candidates,
        "icd10_codes": codes,
        "icd10_results": icd10_results,
        "hcc_result": hcc_result,
        "providers_used": step_providers,
        "timings": timings,
        "guardrail": guardrail_info,
    }
