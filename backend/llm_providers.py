"""
LLM provider abstraction with a per-provider model catalog.

Each pipeline step (condition extraction, code selection, explanation) picks a
provider AND a model at request time. Four providers ship:

- "claude"   — Anthropic Claude via the official `anthropic` SDK.
- "medgemma" — a model served by vLLM through its OpenAI-compatible API.
- "airllm"   — open-weight models run locally by the AirLLM sidecar
               (`backend/airllm_server.py`), for models too large for VRAM.
- "ollama"   — small GGUF models served locally by Ollama, the practical local
               option when you want an answer in seconds rather than minutes.

The last three are all OpenAI-compatible, so they are three configurations of one
`OpenAICompatProvider` rather than three classes.

Providers expose:

    provider.complete(system, messages, max_tokens, model=None) -> str

`model` is a per-request field for every backend here, which is why it is an
argument to `complete()` rather than construction state — caching a provider
instance per (name, model) would build N identical HTTP clients for no gain.
"""
import logging
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Configuration (env-driven) ───────────────────────────────────────────────

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

# Default to 127.0.0.1, not "localhost". On Windows "localhost" resolves to ::1
# first; local servers usually bind IPv4 only, so every probe eats a full timeout
# before falling back — measured at 2.02 s vs 0.01 s here. urllib applies the
# timeout PER address family, so an unreachable host costs double.
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/medgemma-27b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

AIRLLM_BASE_URL = os.environ.get("AIRLLM_BASE_URL", "http://127.0.0.1:8002/v1")
AIRLLM_MODEL = os.environ.get("AIRLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
AIRLLM_API_KEY = os.environ.get("AIRLLM_API_KEY", "EMPTY")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")

# Per-step defaults. Any of these may be overridden per request.
STEP_DEFAULTS = {
    "extract": os.environ.get("LLM_PROVIDER_EXTRACT", "claude"),
    "select": os.environ.get("LLM_PROVIDER_SELECT", "claude"),
    "explain": os.environ.get("LLM_PROVIDER_EXPLAIN", "claude"),
}

MODEL_DEFAULTS = {
    "extract": os.environ.get("LLM_MODEL_EXTRACT", ""),
    "select": os.environ.get("LLM_MODEL_SELECT", ""),
    "explain": os.environ.get("LLM_MODEL_EXPLAIN", ""),
}

#: Accept a model id that discovery did not report (e.g. a freshly pulled Ollama tag).
ALLOW_UNLISTED_MODELS = os.environ.get("LLM_ALLOW_UNLISTED_MODELS", "").lower() in ("1", "true", "yes")

_CATALOG_TTL = float(os.environ.get("LLM_CATALOG_TTL", "30"))
_PROBE_TIMEOUT = float(os.environ.get("LLM_PROBE_TIMEOUT", "1.2"))


# ── Catalog types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    """A selectable model.

    `notes` is keyed by hardware tier ("cpu", "cuda_tiny", "cuda_small",
    "cuda_large") so the same catalog reads honestly on a laptop and on a
    workstation: a 70B is "hours per response, demo only" on CPU and genuinely
    usable on a 24 GB card. "*" is the fallback key.
    """
    id: str
    label: str
    notes: Dict[str, str] = field(default_factory=dict)

    def note_for(self, tier: Optional[str]) -> str:
        return self.notes.get(tier or "*", self.notes.get("*", ""))


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    label: str
    kind: str                       # "anthropic" | "openai_compat"
    local: bool = False
    base_url: str = ""
    api_key: str = ""
    default_model: str = ""
    models: Tuple[ModelSpec, ...] = ()
    note: str = ""
    timeout: float = 600.0
    max_retries: int = 2
    max_tokens_cap: Optional[int] = None


def _airllm_models() -> Tuple[ModelSpec, ...]:
    """Open-weight models AirLLM can stream layer-by-layer.

    AirLLM's premise is that VRAM constrains a single layer, not the whole model,
    so a 70B runs on a small card — slowly. Disk is the real constraint: it keeps
    the original snapshot AND the layer shards, roughly 2x the model size.
    """
    return (
        ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5 0.5B (smoke test)", {
            "cpu": "~1 GB · seconds per token · use to verify the setup, not for real coding",
            "*": "~1 GB · fast · smoke-test model",
        }),
        ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5 1.5B", {
            "cpu": "~3 GB · slow but usable for the extract step",
            "*": "~3 GB · quick",
        }),
        ModelSpec("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B", {
            "cpu": "~15 GB · minutes per response",
            "cuda_tiny": "~15 GB · workable",
            "*": "~15 GB · good general default",
        }),
        ModelSpec("mistralai/Mistral-7B-Instruct-v0.3", "Mistral 7B", {
            "cpu": "~15 GB · minutes per response",
            "*": "~15 GB",
        }),
        ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B", {
            "cpu": "~16 GB · gated repo (needs HF_TOKEN) · minutes per response",
            "*": "~16 GB · gated repo (needs HF_TOKEN)",
        }),
        ModelSpec("Qwen/Qwen2.5-32B-Instruct", "Qwen2.5 32B", {
            "cpu": "~65 GB on disk · tens of minutes per response",
            "cuda_tiny": "~65 GB on disk · slow but runs",
            "*": "~65 GB on disk · strong quality",
        }),
        ModelSpec("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B", {
            "cpu": "~140 GB on disk · HOURS per response · demo only",
            "cuda_tiny": "~140 GB on disk · slow; this is AirLLM's design point",
            "*": "~140 GB on disk · gated repo (needs HF_TOKEN)",
        }),
        ModelSpec("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5 72B", {
            "cpu": "~145 GB on disk · HOURS per response · demo only",
            "*": "~145 GB on disk",
        }),
    )


def _ollama_models() -> Tuple[ModelSpec, ...]:
    return (
        ModelSpec("qwen2.5:3b", "Qwen2.5 3B", {"*": "~2 GB · fastest"}),
        ModelSpec("gemma3:4b", "Gemma 3 4B", {"*": "~3.3 GB"}),
        ModelSpec("alibayram/medgemma:4b", "MedGemma 4B", {"*": "~2.5 GB · medical domain"}),
        ModelSpec("qwen2.5:7b-instruct", "Qwen2.5 7B", {"*": "~4.7 GB · best general default"}),
        ModelSpec("llama3.1:8b", "Llama 3.1 8B", {"*": "~4.9 GB"}),
        ModelSpec("phi4:14b", "Phi-4 14B", {"*": "~9 GB · strong quality"}),
        ModelSpec("qwen2.5:32b", "Qwen2.5 32B", {
            "cpu": "~20 GB · very slow on CPU",
            "*": "~20 GB · needs a large GPU",
        }),
    )


def _claude_models() -> Tuple[ModelSpec, ...]:
    return (
        ModelSpec("claude-sonnet-5-5", "Claude Sonnet 5.5", {"*": "balanced default"}),
        ModelSpec("claude-sonnet-4-6", "Claude Sonnet 4.6", {"*": "previous default"}),
        ModelSpec("claude-opus-4-1", "Claude Opus 4.1", {"*": "highest quality"}),
        ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5", {"*": "fastest / cheapest"}),
    )


def _local_tuning() -> Tuple[float, Optional[int]]:
    """(timeout, max_tokens_cap) for the AirLLM provider, from detected hardware.

    The `openai` SDK defaults to a 600 s read timeout with 2 retries — on a slow
    CPU generation that means timing out and then doing the same work twice more.
    Retries are disabled for AirLLM at every tier; a multi-minute generation is
    never worth repeating.
    """
    try:
        import hardware
        p = hardware.profile()
        return p.client_timeout, p.max_new_tokens_cap
    except Exception:
        return 3600.0, 512  # torch absent: assume the slow case


_AIRLLM_TIMEOUT, _AIRLLM_CAP = _local_tuning()

PROVIDERS: Dict[str, ProviderSpec] = {
    "claude": ProviderSpec(
        name="claude", label="Claude", kind="anthropic",
        default_model=CLAUDE_MODEL, models=_claude_models(),
        note="Anthropic API — requires ANTHROPIC_API_KEY.",
    ),
    "medgemma": ProviderSpec(
        name="medgemma", label="MedGemma (vLLM)", kind="openai_compat", local=True,
        base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, default_model=VLLM_MODEL,
        models=(ModelSpec(VLLM_MODEL, VLLM_MODEL, {"*": "served by vLLM"}),),
        note="vLLM OpenAI endpoint. Best when the model fits entirely in VRAM.",
        timeout=600.0, max_retries=2,
    ),
    "airllm": ProviderSpec(
        name="airllm", label="AirLLM (local)", kind="openai_compat", local=True,
        base_url=AIRLLM_BASE_URL, api_key=AIRLLM_API_KEY, default_model=AIRLLM_MODEL,
        models=_airllm_models(),
        note="Local layer-streamed inference. Use for models too large for VRAM; "
             "expect minutes per response without a GPU.",
        timeout=_AIRLLM_TIMEOUT, max_retries=0, max_tokens_cap=_AIRLLM_CAP,
    ),
    "ollama": ProviderSpec(
        name="ollama", label="Ollama (local)", kind="openai_compat", local=True,
        base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, default_model=OLLAMA_MODEL,
        models=_ollama_models(),
        note="Local GGUF models. The practical local option — seconds per response.",
        timeout=900.0, max_retries=1,
    ),
}

AVAILABLE_PROVIDERS = tuple(PROVIDERS)


def is_local(name: str) -> bool:
    spec = PROVIDERS.get((name or "").lower().strip())
    return bool(spec and spec.local)


# ── Provider implementations ─────────────────────────────────────────────────

class LLMProvider:
    """Common interface. `complete` returns the raw assistant text."""

    name: str = "base"

    def complete(self, system: str, messages: List[Dict], max_tokens: int,
                 model: Optional[str] = None) -> str:
        raise NotImplementedError


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, spec: ProviderSpec):
        import anthropic  # imported lazily so the app boots without the SDK
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — the 'claude' provider is unavailable."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._default_model = spec.default_model

    def complete(self, system: str, messages: List[Dict], max_tokens: int,
                 model: Optional[str] = None) -> str:
        resp = self._client.messages.create(
            model=model or self._default_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text


class OpenAICompatProvider(LLMProvider):
    """Any OpenAI-compatible endpoint: vLLM, the AirLLM sidecar, Ollama, TGI…"""

    def __init__(self, spec: ProviderSpec):
        from openai import OpenAI  # lazy import
        self.name = spec.name
        self._default_model = spec.default_model
        self._cap = spec.max_tokens_cap
        self._client = OpenAI(
            base_url=spec.base_url,
            api_key=spec.api_key or "EMPTY",
            timeout=spec.timeout,
            max_retries=spec.max_retries,
        )

    def complete(self, system: str, messages: List[Dict], max_tokens: int,
                 model: Optional[str] = None) -> str:
        if self._cap:
            max_tokens = min(max_tokens, self._cap)
        # Anthropic keeps `system` separate; OpenAI wants it as the first message.
        oai_messages: List[Dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        resp = self._client.chat.completions.create(
            model=model or self._default_model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return resp.choices[0].message.content or ""


# ── Factory / registry ───────────────────────────────────────────────────────

_PROVIDER_CACHE: Dict[str, LLMProvider] = {}
_CACHE_LOCK = threading.Lock()


def get_provider(name: str) -> LLMProvider:
    """Return a memoized provider instance. Raises with a clear message on failure."""
    name = (name or "").lower().strip()
    spec = PROVIDERS.get(name)
    if spec is None:
        raise ValueError(
            f"Unknown LLM provider '{name}'. Choose one of: {', '.join(AVAILABLE_PROVIDERS)}."
        )
    # app.py now calls providers from worker threads, so the check-and-set races.
    with _CACHE_LOCK:
        if name not in _PROVIDER_CACHE:
            _PROVIDER_CACHE[name] = (
                ClaudeProvider(spec) if spec.kind == "anthropic" else OpenAICompatProvider(spec)
            )
        return _PROVIDER_CACHE[name]


def resolve_step_providers(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Merge per-request overrides onto the env defaults, validating names.

    Returns a dict like {"extract": "claude", "select": "airllm", "explain": "claude"}.
    Unknown provider names silently fall back to the step default so a bad UI value
    can never break a request.
    """
    overrides = overrides or {}
    resolved: Dict[str, str] = {}
    for step, default in STEP_DEFAULTS.items():
        choice = str(overrides.get(step, default) or default).lower().strip()
        if choice not in PROVIDERS:
            logger.warning("Ignoring unknown provider %r for step %r", choice, step)
            choice = default
        resolved[step] = choice
    return resolved


def resolve_step_models(step_providers: Dict[str, str],
                        overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Pick a model id per step, relative to that step's already-resolved provider.

    A model is only meaningful alongside its provider, which is why this takes the
    resolved providers rather than reading STEP_DEFAULTS itself. Unknown ids fall
    back to the provider default with a warning — same contract as
    `resolve_step_providers`, so a stale dropdown value cannot break a request.
    """
    overrides = overrides or {}
    resolved: Dict[str, str] = {}

    for step, provider_name in step_providers.items():
        spec = PROVIDERS.get(provider_name)
        if spec is None:
            resolved[step] = ""
            continue

        allowed = {m.id for m in _models_for(spec)}
        candidate = str(overrides.get(step) or MODEL_DEFAULTS.get(step) or "").strip()

        if candidate and candidate not in allowed and not ALLOW_UNLISTED_MODELS:
            logger.warning(
                "Ignoring unknown model %r for provider %r (step %r); using %r",
                candidate, provider_name, step, spec.default_model,
            )
            candidate = ""

        resolved[step] = candidate or spec.default_model

    return resolved


# ── Endpoint discovery ───────────────────────────────────────────────────────

_DISCOVERY: Dict[str, object] = {"at": 0.0, "data": {}}
_DISCOVERY_LOCK = threading.Lock()


def _probe(spec: ProviderSpec) -> Tuple[bool, List[str]]:
    """GET {base_url}/models. Returns (reachable, model ids)."""
    if spec.kind != "openai_compat":
        return True, []
    url = spec.base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {spec.api_key or 'EMPTY'}"})
    try:
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT) as resp:
            if resp.status != 200:
                return False, []
            import json
            payload = json.loads(resp.read().decode("utf-8"))
        ids = [m.get("id") for m in payload.get("data", []) if isinstance(m, dict) and m.get("id")]
        return True, ids
    except Exception:
        return False, []


def discover(force: bool = False) -> Dict[str, Tuple[bool, List[str]]]:
    """Probe every OpenAI-compatible endpoint concurrently, memoized briefly.

    Serially this would be one timeout per endpoint (previously a single 1.5 s
    stall in /api/config; with three endpoints it would be 4.5 s). Concurrent
    probing plus a short TTL keeps it at roughly one timeout, once per TTL.
    """
    now = time.time()
    with _DISCOVERY_LOCK:
        if not force and now - float(_DISCOVERY["at"]) < _CATALOG_TTL and _DISCOVERY["data"]:
            return dict(_DISCOVERY["data"])  # type: ignore[arg-type]

    targets = [s for s in PROVIDERS.values() if s.kind == "openai_compat"]
    results: Dict[str, Tuple[bool, List[str]]] = {}
    if targets:
        with ThreadPoolExecutor(max_workers=len(targets)) as pool:
            for spec, outcome in zip(targets, pool.map(_probe, targets)):
                results[spec.name] = outcome

    with _DISCOVERY_LOCK:
        _DISCOVERY["at"] = now
        _DISCOVERY["data"] = results
    return results


def _models_for(spec: ProviderSpec, discovered: Optional[List[str]] = None) -> List[ModelSpec]:
    """Curated entries first, then anything the live endpoint reports that we lack."""
    if discovered is None:
        discovered = discover().get(spec.name, (False, []))[1]

    models = list(spec.models)
    known = {m.id for m in models}
    for model_id in discovered or []:
        if model_id not in known:
            models.append(ModelSpec(model_id, model_id, {"*": "discovered on the endpoint"}))
            known.add(model_id)

    if spec.default_model and spec.default_model not in known:
        models.insert(0, ModelSpec(spec.default_model, spec.default_model, {}))
    return models


def local_health() -> Dict:
    """Live status of the AirLLM sidecar, for /api/local-status."""
    spec = PROVIDERS["airllm"]
    url = spec.base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    try:
        with urllib.request.urlopen(url + "/health", timeout=2.0) as resp:
            import json
            return {"airllm": json.loads(resp.read().decode("utf-8"))}
    except Exception as exc:
        return {"airllm": None, "error": str(exc)}


def config_summary() -> Dict:
    """Serializable config for the frontend to populate provider + model selectors."""
    probes = discover()

    # Hardware comes from the sidecar when it is up (it is the process that owns
    # torch); fall back to local detection so a GPU box still self-describes.
    hw = None
    tier = None
    airllm_probe = probes.get("airllm", (False, []))
    if airllm_probe[0]:
        health = local_health().get("airllm") or {}
        hw = health.get("hardware")
    if hw is None:
        try:
            import hardware as hw_mod
            hw = hw_mod.describe()
        except Exception:
            hw = None
    if hw:
        tier = hw.get("tier")

    providers = []
    for spec in PROVIDERS.values():
        reachable, discovered = probes.get(spec.name, (True, []))
        models = _models_for(spec, discovered)
        providers.append({
            "name": spec.name,
            "label": spec.label,
            "local": spec.local,
            "kind": spec.kind,
            "base_url": spec.base_url,
            "reachable": reachable,
            "default_model": spec.default_model,
            "note": spec.note,
            "models": [
                {"id": m.id, "label": m.label, "note": m.note_for(tier)} for m in models
            ],
        })

    return {
        "providers": providers,
        "step_defaults": dict(STEP_DEFAULTS),
        "model_defaults": {k: v for k, v in MODEL_DEFAULTS.items() if v},
        "hardware": hw,
    }
