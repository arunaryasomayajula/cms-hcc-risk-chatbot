import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic

from rag_icd10 import ICD10RAG
from hcc_calculator import HCCCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

rag = ICD10RAG()
calculator = HCCCalculator()
claude = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))
executor = ThreadPoolExecutor(max_workers=2)

_initialized = False
_init_error: Optional[str] = None


def _run_init():
    global _initialized, _init_error
    try:
        calculator.load_data()
        rag.initialize()
        _initialized = True
        logger.info("All components initialized")
    except Exception as exc:
        _init_error = str(exc)
        logger.error(f"Initialization failed: {exc}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_init)
    yield
    executor.shutdown(wait=False)


app = FastAPI(title="CMS HCC Risk Scoring Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


# ── Pydantic models ────────────────────────────────────────────────────────────

class Demographics(BaseModel):
    dob: str = "1950-01-01"
    sex: int = 2          # 1=Male 2=Female
    orec: int = 0         # 0=Aged 1=Disabled 2=ESRD 3=Both
    ltimcaid: int = 0
    nemcaid: int = 0
    dual_status: int = 0  # 0=Non-dual 1=Partial 2=Full


class ChatRequest(BaseModel):
    message: str
    demographics: Optional[Demographics] = None
    conversation_history: Optional[List[dict]] = []


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    idx = os.path.join(FRONTEND, 'index.html')
    if os.path.exists(idx):
        return HTMLResponse(open(idx, encoding='utf-8').read())
    return {"message": "CMS HCC Chatbot API — open /static/index.html"}


@app.get("/api/status")
async def status():
    return {
        "initialized": _initialized,
        "error": _init_error,
        "rag_ready": rag.is_ready,
        "calculator_ready": calculator.is_ready,
    }


SYSTEM_PROMPT = """You are a clinical coding and Medicare risk adjustment specialist.
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


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not _initialized:
        return {
            "response": "⏳ System is still initializing (building ICD-10 index). This takes ~2-5 minutes on first run. Please wait and try again.",
            "status": "initializing",
            "icd10_codes": [],
            "hcc_result": None,
            "conversation_history": req.conversation_history or [],
        }

    demo_dict = req.demographics.model_dump() if req.demographics else {}
    history = list(req.conversation_history or [])
    user_text = req.message.strip()

    if not user_text:
        raise HTTPException(400, "Empty message")

    # ── Step 1: detect if this looks like clinical notes ──────────────────────
    clinical_keywords = [
        'patient', 'diagnosis', 'history', 'hx', 'presents', 'prescribed',
        'medications', 'assessment', 'plan', 'chief complaint', 'pmh', 'icd',
        'hospital', 'chronic', 'acute', 'disorder', 'disease', 'syndrome',
        'bilateral', 'unilateral', 'follow-up', 'years old', 'yo ', 'y/o',
        'blood pressure', 'glucose', 'hemoglobin', 'creatinine',
    ]
    is_clinical = (
        len(user_text) > 80 or
        sum(1 for kw in clinical_keywords if kw in user_text.lower()) >= 2
    )

    icd10_results: list = []
    hcc_result: Optional[dict] = None
    conditions_found: list = []

    if is_clinical:
        # ── Step 2: extract conditions ────────────────────────────────────────
        extract_resp = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system=(
                "Extract distinct medical diagnoses and conditions from clinical notes. "
                "Return ONLY a JSON object like: "
                '{"conditions": [{"condition": "Type 2 diabetes mellitus", '
                '"search_query": "type 2 diabetes mellitus without complication"}, ...]}'
                " Include chronic and acute conditions. Be specific."
            ),
            messages=[{"role": "user", "content": user_text}],
        )
        try:
            conditions_found = json.loads(extract_resp.content[0].text).get("conditions", [])
        except (json.JSONDecodeError, IndexError, AttributeError):
            conditions_found = []

        # ── Step 3: RAG search per condition ─────────────────────────────────
        candidates: list = []
        seen_codes: set = set()
        for cond in conditions_found:
            q = cond.get("search_query") or cond.get("condition", "")
            for hit in rag.search(q, n_results=6):
                if hit["icd10"] not in seen_codes:
                    seen_codes.add(hit["icd10"])
                    hit["for_condition"] = cond.get("condition", "")
                    candidates.append(hit)

        # ── Step 4: Claude selects best codes ────────────────────────────────
        if candidates:
            select_resp = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
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
                        f"Clinical notes:\n{user_text}\n\n"
                        f"Extracted conditions: {json.dumps(conditions_found)}\n\n"
                        f"ICD-10 candidates:\n{json.dumps(candidates[:25], indent=2)}"
                    )
                }],
            )
            try:
                icd10_results = json.loads(select_resp.content[0].text).get("selected_codes", [])
            except (json.JSONDecodeError, IndexError, AttributeError):
                icd10_results = candidates[:5]

        # ── Step 5: HCC calculation ───────────────────────────────────────────
        if icd10_results and demo_dict:
            codes = [r.get("icd10", "") for r in icd10_results if r.get("icd10")]
            try:
                hcc_result = calculator.calculate(demo_dict, codes)
            except Exception as exc:
                logger.error(f"HCC calculation error: {exc}", exc_info=True)

    # ── Step 6: Generate conversational explanation ───────────────────────────
    context_parts = []
    if conditions_found:
        context_parts.append(f"Conditions extracted: {[c['condition'] for c in conditions_found]}")
    if icd10_results:
        code_summary = [f"{r['icd10']} ({r['description']}, CC{r['cc']})" for r in icd10_results]
        context_parts.append(f"ICD-10 codes selected: {code_summary}")
    if hcc_result:
        active = [h['hcc'] for h in hcc_result.get('active_hccs', [])]
        context_parts.append(
            f"Active HCCs: {active}\n"
            f"Applicable model: {hcc_result.get('applicable_model')}\n"
            f"Risk score: {hcc_result.get('applicable_score')}\n"
            f"All CE scores: {hcc_result.get('ce_scores')}\n"
            f"Interactions triggered: {hcc_result.get('interactions_triggered')}"
        )

    augmented = user_text
    if context_parts:
        augmented = (
            user_text
            + "\n\n--- Analysis results (use these to craft your response) ---\n"
            + "\n".join(context_parts)
        )

    messages = history + [{"role": "user", "content": augmented}]
    final_resp = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    assistant_text = final_resp.content[0].text

    # Return clean history (don't expose the augmented system context)
    clean_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    return {
        "response": assistant_text,
        "icd10_codes": icd10_results,
        "hcc_result": hcc_result,
        "conditions_found": conditions_found,
        "conversation_history": clean_history,
        "status": "success",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
