"""
OpenAI-compatible HTTP server for AirLLM.

AirLLM ships no server, so this sidecar wraps it in the `/v1/chat/completions`
shape the rest of the app already speaks (the same interface used for vLLM and
Ollama). Run it as a SEPARATE process from the web app:

    python backend/airllm_server.py          # serves on 127.0.0.1:8002

Why a sidecar rather than importing AirLLM into app.py:
  * Model loading decomposes and re-saves the weights layer-by-layer — minutes,
    and gigabytes of disk. In-process that would block the first request and the
    whole event loop with it.
  * A CUDA OOM or a torch/transformers incompatibility kills only this process.
  * The web app keeps zero torch dependency, so GPU users can install a CUDA
    wheel here (or in a dedicated venv) without touching the app's environment.

Device, dtype, sequence length and token caps all come from `hardware.profile()`,
which spans CPU through GTX 10-series, RTX 20/30/40/50 and datacenter cards.
AirLLM itself has no fallback logic: its `device` default is the literal string
"cuda:0" and its `dtype` default is float16, both wrong on several of those.
"""
import os

# Must precede `import torch`: Anaconda ships two OpenMP runtimes and torch aborts
# with "OMP: Error #15 ... libiomp5md.dll already initialized" on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc  # noqa: E402
import glob  # noqa: E402
import logging  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from typing import Dict, List, Literal, Optional, Union  # noqa: E402

import torch  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

import hardware  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("airllm_server")

HW = hardware.profile()

# AirLLM's `clean_memory()` calls `torch.cuda.empty_cache()` unguarded from its
# per-layer post-hook. It is a no-op on today's CPU builds but is not contractually
# so; neutralise it off-CUDA rather than risk a crash mid-generation.
if HW.backend != "cuda":
    torch.cuda.empty_cache = lambda *a, **k: None  # type: ignore[assignment]

if HW.backend == "cpu" and HW.cpu_threads:
    torch.set_num_threads(HW.cpu_threads)

for _w in HW.warnings:
    logger.warning(_w)
logger.info("Hardware: %s", hardware.summary(HW))


# ── Configuration ────────────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


DEFAULT_MODEL = _env("AIRLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CATALOG = [m.strip() for m in _env("AIRLLM_MODELS").split(",") if m.strip()]
COMPRESSION = _env("AIRLLM_COMPRESSION") or None
SHARDS_DIR = _env("AIRLLM_SHARDS_DIR") or os.path.join(os.path.dirname(__file__), ".airllm_shards")
PREFETCH = _env_bool("AIRLLM_PREFETCH", True)
PRELOAD = _env_bool("AIRLLM_PRELOAD", False)
ALLOW_SWITCH = _env_bool("AIRLLM_ALLOW_SWITCH", True)
DELETE_ORIGINAL = _env_bool("AIRLLM_DELETE_ORIGINAL", False)
HF_TOKEN = _env("HF_TOKEN") or _env("HUGGING_FACE_HUB_TOKEN") or None
HOST = _env("AIRLLM_HOST", "127.0.0.1")
PORT = int(_env("AIRLLM_PORT", "8002"))

if COMPRESSION and not HW.supports_bnb:
    logger.warning(
        "AIRLLM_COMPRESSION=%s ignored: bitsandbytes is unavailable or unsupported on this "
        "device (%s). Loading in %s instead.", COMPRESSION, hardware.summary(HW), HW.dtype,
    )
    COMPRESSION = None


# ── OpenAI wire format ───────────────────────────────────────────────────────
# Field sets below are exactly what the `openai` SDK's pydantic models require.
# `finish_reason` in particular is a strict Literal — emitting null or "eos"
# raises inside the client rather than in our code.

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = ""


class ChatCompletionRequest(BaseModel):
    # The SDK sends stream/n/presence_penalty/etc. regardless; ignore the rest.
    model_config = ConfigDict(extra="ignore")

    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None   # newer SDK spelling
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False


class RespMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class RespChoice(BaseModel):
    index: int = 0
    message: RespMessage
    finish_reason: Literal["stop", "length"] = "stop"
    logprobs: None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[RespChoice]
    usage: Usage


def _error(status: int, message: str, err_type: str = "invalid_request_error") -> JSONResponse:
    """OpenAI error envelope, so the client raises a readable APIStatusError.

    FastAPI's default {"detail": ...} shape is swallowed by the SDK.
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": None}},
    )


# ── Prompt rendering ─────────────────────────────────────────────────────────

def _render_prompt(tok, messages: List[ChatMessage]) -> str:
    """Apply the model's chat template, degrading gracefully."""
    dicts = [{"role": m.role, "content": m.content} for m in messages]

    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(dicts, add_generation_prompt=True, tokenize=False)
        except Exception as exc:
            logger.warning("chat template failed (%s); retrying with system folded in", exc)
            # Gemma and Mistral templates reject a standalone `system` role.
            folded, system_text = [], ""
            for m in dicts:
                if m["role"] == "system":
                    system_text += m["content"] + "\n\n"
                else:
                    content = m["content"]
                    if system_text and m["role"] == "user":
                        content, system_text = system_text + content, ""
                    folded.append({"role": m["role"], "content": content})
            try:
                return tok.apply_chat_template(folded, add_generation_prompt=True, tokenize=False)
            except Exception as exc2:
                logger.warning("folded chat template also failed (%s); using plain format", exc2)

    parts = []
    for m in dicts:
        parts.append(f"{m['role'].capitalize()}: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _apply_stop(text: str, stop: Optional[Union[str, List[str]]]):
    """Truncate at the first stop string. Returns (text, hit)."""
    if not stop:
        return text, False
    needles = [stop] if isinstance(stop, str) else list(stop)
    cut = None
    for needle in needles:
        if not needle:
            continue
        idx = text.find(needle)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    return (text[:cut], True) if cut is not None else (text, False)


# ── Model manager ────────────────────────────────────────────────────────────

class _ModelManager:
    """Holds at most one AirLLM model.

    Two locks on purpose. `_lock` serializes load AND generate because AirLLM is
    not concurrency-safe, and it can be held for many minutes. `_status_lock`
    guards only the small status dict so /health stays answerable throughout —
    without that split, the progress endpoint would block behind the work it is
    supposed to be reporting on.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._model = None
        self._model_id: Optional[str] = None
        self._status: Dict = {
            "state": "idle",
            "model": None,
            "message": "",
            "started_at": None,
            "elapsed_s": 0.0,
            "layers_total": 0,
            "layers_split": 0,
            "tokens_generated": 0,
            "tokens_requested": 0,
            "error": None,
        }

    # -- status ------------------------------------------------------------

    #: States where elapsed_s should keep ticking. Anywhere else the work is over
    #: and a growing timer would just look like a hang.
    _ACTIVE = ("loading", "splitting", "generating")

    def _set(self, **kw):
        with self._status_lock:
            self._status.update(kw)
            started = self._status.get("started_at")
            if started:
                self._status["elapsed_s"] = round(time.time() - started, 1)
            if self._status.get("state") not in self._ACTIVE:
                self._status["started_at"] = None

    def status(self) -> Dict:
        with self._status_lock:
            snap = dict(self._status)
        if snap.get("started_at") and snap.get("state") in self._ACTIVE:
            snap["elapsed_s"] = round(time.time() - snap["started_at"], 1)
        snap["loaded_model"] = self._model_id
        return snap

    @property
    def loaded_model(self) -> Optional[str]:
        return self._model_id

    # -- loading -----------------------------------------------------------

    def _watch_shards(self, model_id: str, stop: threading.Event):
        """Poll the shard directory so the slow first split gets a real progress bar.

        Reads no AirLLM internals — it just counts the per-layer safetensors files
        being written, which is exactly the work being waited on.
        """
        safe = model_id.replace("/", "_")
        patterns = [
            os.path.join(SHARDS_DIR, "**", "*.safetensors"),
            os.path.join(SHARDS_DIR, safe, "**", "*.safetensors"),
        ]
        while not stop.wait(2.0):
            count = 0
            for pattern in patterns:
                try:
                    count = max(count, len(glob.glob(pattern, recursive=True)))
                except Exception:
                    pass
            self._set(layers_split=count, state="splitting" if count else "loading")

    def _load_locked(self, model_id: str):
        if self._model is not None and self._model_id == model_id:
            return

        if self._model is not None and not ALLOW_SWITCH:
            raise RuntimeError(
                f"Model {self._model_id!r} is already loaded and AIRLLM_ALLOW_SWITCH is off; "
                f"cannot switch to {model_id!r}."
            )

        if self._model is not None:
            logger.info("Evicting %s to load %s", self._model_id, model_id)
            self._model = None
            self._model_id = None
            gc.collect()
            if HW.backend == "cuda":
                # VRAM is not returned to the allocator without this.
                torch.cuda.empty_cache()

        self._set(state="loading", model=model_id, started_at=time.time(), error=None,
                  layers_split=0, layers_total=0, tokens_generated=0, tokens_requested=0,
                  message=f"Loading {model_id} on {HW.device} ({HW.dtype})")
        logger.info("Loading %s on %s (%s, max_seq_len=%d)",
                    model_id, HW.device, HW.dtype, HW.max_seq_len)

        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
            hidden = int(getattr(cfg, "num_hidden_layers", 0) or 0)
            # AirLLM writes one shard per hidden layer PLUS the input embeddings and
            # the final norm/lm_head, so the shard count runs 2 above the layer count.
            self._set(layers_total=hidden + 2 if hidden else 0)
        except Exception as exc:
            logger.warning("Could not read layer count for %s: %s", model_id, exc)

        stop = threading.Event()
        watcher = threading.Thread(target=self._watch_shards, args=(model_id, stop), daemon=True)
        watcher.start()

        try:
            os.makedirs(SHARDS_DIR, exist_ok=True)
            from airllm import AutoModel
            self._model = AutoModel.from_pretrained(
                model_id,
                device=HW.device,
                dtype=hardware.torch_dtype(),
                max_seq_len=HW.max_seq_len,
                layer_shards_saving_path=SHARDS_DIR,
                compression=COMPRESSION,
                prefetching=PREFETCH,
                delete_original=DELETE_ORIGINAL,
                hf_token=HF_TOKEN,
            )
            self._model_id = model_id
            self._set(state="ready", message=f"{model_id} ready on {HW.device}")
            logger.info("Loaded %s", model_id)
        except Exception as exc:
            self._model = None
            self._model_id = None
            self._set(state="error", error=f"{type(exc).__name__}: {exc}",
                      message=f"Failed to load {model_id}")
            logger.error("Failed to load %s: %s", model_id, exc, exc_info=True)
            raise
        finally:
            stop.set()

    def ensure_loaded(self, model_id: str):
        with self._lock:
            self._load_locked(model_id)

    # -- generation --------------------------------------------------------

    def generate(self, model_id: str, messages: List[ChatMessage], max_new_tokens: int,
                 temperature: Optional[float], top_p: Optional[float],
                 stop: Optional[Union[str, List[str]]]) -> Dict:
        with self._lock:
            self._load_locked(model_id)
            model = self._model
            tok = model.tokenizer

            # Left truncation matters: the instruction that says "Return ONLY JSON"
            # sits at the END of the select-step prompt. Truncating from the right
            # would silently drop exactly the part that shapes the output.
            try:
                tok.truncation_side = "left"
            except Exception:
                pass

            prompt = _render_prompt(tok, messages)
            max_input = max(16, HW.max_seq_len - max_new_tokens)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input)
            input_ids = enc["input_ids"].to(HW.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(HW.device)
            input_len = int(input_ids.shape[1])
            if input_len >= max_input:
                logger.warning("Prompt truncated to %d tokens (max_seq_len=%d, max_new_tokens=%d)",
                               max_input, HW.max_seq_len, max_new_tokens)

            self._set(state="generating", tokens_generated=0, tokens_requested=max_new_tokens,
                      started_at=time.time(), message=f"Generating with {model_id}")

            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "pad_token_id": getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
            }
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p or 1.0)
            else:
                gen_kwargs.update(do_sample=False)  # OpenAI temperature=0 means greedy

            processors = self._token_counter()
            if processors is not None:
                gen_kwargs["logits_processor"] = processors

            t0 = time.perf_counter()
            try:
                try:
                    out = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
                except TypeError:
                    # A few AirLLM architecture classes reject attention_mask.
                    out = model.generate(input_ids, **gen_kwargs)
            except Exception as exc:
                self._set(state="error", error=f"{type(exc).__name__}: {exc}")
                raise
            elapsed = time.perf_counter() - t0

            text, completion_tokens, finish = self._decode(out, tok, prompt, input_len, max_new_tokens)
            text, hit_stop = _apply_stop(text, stop)
            if hit_stop:
                finish = "stop"

            self._set(state="ready", message=f"{model_id} ready on {HW.device}")
            logger.info("Generated %d tokens in %.1fs (%.2f tok/s) with %s",
                        completion_tokens, elapsed,
                        completion_tokens / elapsed if elapsed else 0.0, model_id)

            return {
                "text": text.strip(),
                "prompt_tokens": input_len,
                "completion_tokens": completion_tokens,
                "finish_reason": finish,
            }

    def _token_counter(self):
        """LogitsProcessor that ticks the progress counter — cheaper than streaming."""
        try:
            from transformers import LogitsProcessor, LogitsProcessorList
        except Exception:
            return None

        manager = self

        class _Counter(LogitsProcessor):
            def __call__(self, input_ids, scores):
                with manager._status_lock:
                    manager._status["tokens_generated"] += 1
                return scores

        return LogitsProcessorList([_Counter()])

    @staticmethod
    def _decode(out, tok, prompt: str, input_len: int, max_new_tokens: int):
        """Slice new tokens by index. Never string-match the prompt off the front."""
        if isinstance(out, str):
            # A couple of AirLLM architecture classes return decoded text directly.
            text = out[len(prompt):] if out.startswith(prompt) else out
            approx = len(tok(text)["input_ids"]) if text else 0
            return text, approx, ("length" if approx >= max_new_tokens else "stop")

        sequences = getattr(out, "sequences", out)
        seq = sequences[0]
        new_ids = seq[input_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        if text.startswith(prompt):  # belt and braces
            text = text[len(prompt):]
        count = int(len(new_ids))
        return text, count, ("length" if count >= max_new_tokens else "stop")


MANAGER = _ModelManager()


# ── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_app: FastAPI):
    if PRELOAD:
        # Background thread, not await: loading can take minutes and must not
        # delay the port opening — /health is how progress is reported.
        threading.Thread(target=_safe_load, args=(DEFAULT_MODEL,), daemon=True).start()
    yield


app = FastAPI(title="AirLLM OpenAI-compatible server", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        **MANAGER.status(),
        "hardware": hardware.describe(),
        "config": {
            "default_model": DEFAULT_MODEL,
            "max_seq_len": HW.max_seq_len,
            "max_new_tokens_cap": HW.max_new_tokens_cap,
            "compression": COMPRESSION,
            "shards_dir": SHARDS_DIR,
            "allow_switch": ALLOW_SWITCH,
        },
    }


@app.get("/v1/models")
def list_models():
    ids = list(dict.fromkeys([DEFAULT_MODEL, *CATALOG, *( [MANAGER.loaded_model] if MANAGER.loaded_model else [] )]))
    return {
        "object": "list",
        # All four fields are required by the SDK's Model type.
        "data": [{"id": i, "object": "model", "created": 0, "owned_by": "airllm"} for i in ids],
    }


@app.post("/v1/models/load")
def load_model(body: Dict):
    """Warm a model without spending a generation. Returns immediately."""
    model_id = (body or {}).get("model") or DEFAULT_MODEL
    threading.Thread(target=lambda: _safe_load(model_id), daemon=True).start()
    return JSONResponse(status_code=202, content={"status": "loading", "model": model_id})


def _safe_load(model_id: str):
    try:
        MANAGER.ensure_loaded(model_id)
    except Exception:
        logger.error("Background load of %s failed", model_id, exc_info=True)


# NOTE: `def`, not `async def`. Starlette runs sync routes in a threadpool, so a
# multi-minute generation holding the manager lock cannot block /health.
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return _error(400, "This server does not support stream=true.")
    if not req.messages:
        return _error(400, "`messages` must not be empty.")

    model_id = req.model or DEFAULT_MODEL
    requested = req.max_tokens or req.max_completion_tokens or 512
    max_new_tokens = max(1, min(int(requested), HW.max_new_tokens_cap))
    if max_new_tokens < requested:
        logger.info("Capping max_tokens %d -> %d for the %s tier", requested, max_new_tokens, HW.tier)

    state = MANAGER.status()
    if state.get("state") in ("loading", "splitting") and MANAGER.loaded_model != model_id:
        total, done = state.get("layers_total") or 0, state.get("layers_split") or 0
        progress = f" (layer {done}/{total})" if total else ""
        return _error(
            503,
            f"AirLLM is still loading {state.get('model')}{progress}. Retry once /health reports ready.",
            err_type="server_error",
        )

    try:
        result = MANAGER.generate(
            model_id=model_id,
            messages=req.messages,
            max_new_tokens=max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )
    except torch.cuda.OutOfMemoryError as exc:
        # The dominant GPU-tier failure. A bare 500 here is unreadable.
        return _error(
            507,
            f"CUDA out of memory loading/running {model_id} on {HW.name} ({HW.vram_gb} GB). "
            f"Try a lower AIRLLM_MAX_SEQ_LEN (currently {HW.max_seq_len}) or a smaller model. "
            f"Details: {exc}",
            err_type="server_error",
        )
    except Exception as exc:
        logger.error("Generation failed for %s: %s", model_id, exc, exc_info=True)
        return _error(500, f"{type(exc).__name__}: {exc}", err_type="server_error")

    return ChatCompletionResponse(
        id="chatcmpl-" + uuid.uuid4().hex[:24],
        created=int(time.time()),
        model=model_id,
        choices=[RespChoice(
            index=0,
            message=RespMessage(content=result["text"]),
            finish_reason=result["finish_reason"],
        )],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        ),
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("AirLLM server on http://%s:%d (default model: %s)", HOST, PORT, DEFAULT_MODEL)
    # One worker only: the model is process-local state guarded by a threading lock.
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
