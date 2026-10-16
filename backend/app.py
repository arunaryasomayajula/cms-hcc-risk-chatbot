import os
import sys
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

from rag_icd10 import ICD10RAG
from hcc_calculator import HCCCalculator
from json_utils import try_parse_json_object
import llm_providers
import prompts
import guardrails
import tracing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

rag = ICD10RAG()
calculator = HCCCalculator()
executor = ThreadPoolExecutor(max_workers=2)

# Local backends hold one model in memory and are not concurrency-safe, so calls
# to them are serialized process-wide. Raise this only if you point the app at
# something that genuinely parallelises (multi-GPU vLLM, a remote endpoint).
_LOCAL_SEM = asyncio.Semaphore(int(os.environ.get("LOCAL_INFERENCE_CONCURRENCY", "1")))

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
    # Per-step provider overrides, e.g. {"extract": "airllm", "explain": "claude"}
    providers: Optional[dict] = None
    # Per-step model overrides, e.g. {"extract": "Qwen/Qwen2.5-7B-Instruct"}
    models: Optional[dict] = None


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
        "guardrails": guardrails.status(),
        "tracing": tracing.status(),
    }


@app.get("/api/config")
async def config():
    """Provider + model catalog so the frontend can populate the per-step selectors."""
    # config_summary() probes local endpoints over the network; keep it off the loop.
    summary = await asyncio.to_thread(llm_providers.config_summary)
    return {**summary, "guardrails": guardrails.status()}


@app.get("/api/local-status")
async def local_status():
    """Live AirLLM sidecar progress (model load %, tokens generated).

    Deliberately separate from /api/status, which the UI polls every 5 s and which
    must stay a pure in-memory read — this one makes a network call.
    """
    return await asyncio.to_thread(llm_providers.local_health)


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
    # Cap history: unbounded context overflows a local model's window, and costs
    # money on Claude. Each turn is a user+assistant pair.
    history = list(req.conversation_history or [])[-2 * prompts.MAX_HISTORY_TURNS:]
    raw_text = req.message.strip()

    if not raw_text:
        raise HTTPException(400, "Empty message")

    # ── Guardrail: de-identify PHI before anything is sent to an LLM ──────
    # spaCy NER is synchronous and slow enough to stall the event loop.
    deid = await asyncio.to_thread(guardrails.deidentify, raw_text)
    user_text = deid.sanitized_text  # everything downstream uses the sanitized text
    guard_info = deid.to_dict()

    # ── Resolve which provider + model serves each pipeline step ──────────
    step_providers = llm_providers.resolve_step_providers(req.providers)
    step_models = llm_providers.resolve_step_models(step_providers, req.models)
    parse_failures: list = []

    # One JSON trace file per request. `tr` is a no-op when tracing is disabled, so
    # nothing below needs to branch. The note recorded here is the SANITIZED one
    # unless LANGSMITH_TRACE_RAW_INPUT is set — see backend/tracing.py.
    with tracing.trace(
        "chat",
        message=(raw_text if tracing.RAW_INPUT else user_text),
        demographics=demo_dict,
        history_turns=len(history) // 2,
        metadata={
            "providers": step_providers,
            "models": step_models,
            "guardrail_redacted": guard_info.get("redacted"),
        },
    ) as tr:
        # Record what the guardrail removed, never the identifiers themselves.
        with tr.span("guardrail.deidentify", run_type="tool", chars_in=len(raw_text)) as sp:
            sp.output(redacted=guard_info.get("redacted"), chars_out=len(user_text),
                      findings=guard_info.get("findings", []))

        async def run_step(step: str, system: str, messages: list, max_tokens: int) -> str:
            name, model = step_providers[step], step_models[step]

            def _call() -> str:
                return llm_providers.get_provider(name).complete(
                    system=system, messages=messages, max_tokens=max_tokens, model=model,
                )

            with tr.span(f"llm.{step}", run_type="llm", system=system, messages=messages) as sp:
                sp.metadata(provider=name, model=model, max_tokens=max_tokens,
                            local=llm_providers.is_local(name))
                # to_thread is what keeps /api/status answering during a multi-minute
                # local generation; without it the whole server appears hung.
                if llm_providers.is_local(name):
                    async with _LOCAL_SEM:
                        out = await asyncio.to_thread(_call)
                else:
                    out = await asyncio.to_thread(_call)
                sp.output(text=out)
                return out

        # ── Step 1: detect if this looks like clinical notes ──────────────
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
        tr.metadata(is_clinical=is_clinical)

        icd10_results: list = []
        hcc_result: Optional[dict] = None
        conditions_found: list = []

        if is_clinical:
            # ── Step 2: extract conditions ──────────────────────
            try:
                extract_text = await run_step(
                    "extract",
                    system=prompts.EXTRACT_SYSTEM,
                    messages=[{"role": "user", "content": user_text}],
                    max_tokens=prompts.MAX_TOKENS["extract"],
                )
            except Exception as exc:
                logger.error(f"Condition extraction call failed: {exc}", exc_info=True)
                raise HTTPException(502, f"LLM provider error (extract): {exc}")

            parsed, err = try_parse_json_object(extract_text, list_key="conditions")
            if err:
                # Record rather than swallow: an empty list here is indistinguishable
                # from "the note had no conditions", which hides a broken local model.
                logger.warning(f"Condition extraction parse failed: {err}")
                parse_failures.append({
                    "step": "extract", "error": err, "raw_preview": (extract_text or "")[:300],
                })
                conditions_found = []
            else:
                conditions_found = parsed.get("conditions", [])

            # ── Step 3: RAG search per condition ─────────────────
            with tr.span("rag.search", run_type="retriever", conditions=conditions_found) as sp:
                candidates: list = []
                seen_codes: set = set()
                for cond in conditions_found:
                    q = cond.get("search_query") or cond.get("condition", "")
                    for hit in rag.search(q, n_results=6):
                        if hit["icd10"] not in seen_codes:
                            seen_codes.add(hit["icd10"])
                            hit["for_condition"] = cond.get("condition", "")
                            candidates.append(hit)
                sp.output(candidate_count=len(candidates),
                          candidates=candidates[:prompts.MAX_SELECT_CANDIDATES])

            # ── Step 4: select best codes ──────────────────────
            if candidates:
                try:
                    select_text = await run_step(
                        "select",
                        system=prompts.SELECT_SYSTEM,
                        messages=[{
                            "role": "user",
                            "content": prompts.build_select_user(user_text, conditions_found, candidates),
                        }],
                        max_tokens=prompts.MAX_TOKENS["select"],
                    )
                except Exception as exc:
                    logger.error(f"Code selection call failed: {exc}", exc_info=True)
                    raise HTTPException(502, f"LLM provider error (select): {exc}")

                parsed, err = try_parse_json_object(select_text, list_key="selected_codes")
                if err:
                    logger.warning(f"Code selection parse failed: {err}")
                    parse_failures.append({
                        "step": "select", "error": err, "raw_preview": (select_text or "")[:300],
                        "fallback": "top_5_candidates",
                    })
                    icd10_results = candidates[:5]
                else:
                    icd10_results = parsed.get("selected_codes", [])

            # ── Step 5: HCC calculation ───────────────────────
            if icd10_results and demo_dict:
                codes = [r.get("icd10", "") for r in icd10_results if r.get("icd10")]
                with tr.span("hcc.calculate", run_type="tool",
                             codes=codes, demographics=demo_dict) as sp:
                    try:
                        hcc_result = calculator.calculate(demo_dict, codes)
                        sp.output(
                            active_hccs=[h["hcc"] for h in (hcc_result or {}).get("active_hccs", [])],
                            applicable_model=(hcc_result or {}).get("applicable_model"),
                            applicable_score=(hcc_result or {}).get("applicable_score"),
                            interactions=(hcc_result or {}).get("interactions_triggered"),
                        )
                    except Exception as exc:
                        # Non-fatal by design: the explanation step still runs.
                        logger.error(f"HCC calculation error: {exc}", exc_info=True)
                        sp.error(exc)

        # ── Step 6: Generate conversational explanation ───────────────
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
        try:
            assistant_text = await run_step(
                "explain",
                system=prompts.EXPLAIN_SYSTEM,
                messages=messages,
                max_tokens=prompts.MAX_TOKENS["explain"],
            )
        except Exception as exc:
            logger.error(f"Explanation call failed: {exc}", exc_info=True)
            raise HTTPException(502, f"LLM provider error (explain): {exc}")

        # Return clean history (sanitized text only — PHI is never stored)
        clean_history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

        tr.output(response=assistant_text,
                  icd10_codes=[r.get("icd10") for r in icd10_results],
                  parse_failures=parse_failures,
                  status="partial" if parse_failures else "success")

        return {
            "response": assistant_text,
            "icd10_codes": icd10_results,
            "hcc_result": hcc_result,
            "conditions_found": conditions_found,
            "conversation_history": clean_history,
            "guardrail": guard_info,
            "providers_used": step_providers,
            "models_used": step_models,
            "parse_failures": parse_failures,
            "status": "partial" if parse_failures else "success",
        }


DEFAULT_PORT = 8000


def listen_address() -> tuple:
    """(host, port) from the HOST / PORT env vars, shared by every launcher.

    Exits with a readable message on a bad PORT rather than a uvicorn traceback.
    """
    host = os.environ.get("HOST", "").strip() or "0.0.0.0"
    raw = os.environ.get("PORT", "").strip() or str(DEFAULT_PORT)
    try:
        port = int(raw)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        sys.exit(f"Invalid PORT={raw!r}: expected an integer between 1 and 65535.")
    return host, port


if __name__ == "__main__":
    import uvicorn
    host, port = listen_address()
    # Pass the app object, not "app:app": the import string would load this
    # module a second time (as `app` alongside `__main__`).
    uvicorn.run(app, host=host, port=port)
