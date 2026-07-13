"""
LLM provider abstraction.

Lets each pipeline step (condition extraction, code selection, explanation) be
served by a different backend model. Two providers are supported:

- "claude"   — Anthropic Claude via the official `anthropic` SDK.
- "medgemma" — Google MedGemma (or any model) served locally by vLLM through its
               OpenAI-compatible API, accessed with the `openai` SDK.

Both expose the same tiny interface:

    provider.complete(system, messages, max_tokens) -> str

so `app.py` can swap them per step without caring which one it is.
"""
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ── Configuration (env-driven) ───────────────────────────────────────────────

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/medgemma-27b-it")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# Per-step defaults. Any of these may be overridden per request.
STEP_DEFAULTS = {
    "extract": os.environ.get("LLM_PROVIDER_EXTRACT", "claude"),
    "select": os.environ.get("LLM_PROVIDER_SELECT", "claude"),
    "explain": os.environ.get("LLM_PROVIDER_EXPLAIN", "claude"),
}

AVAILABLE_PROVIDERS = ("claude", "medgemma")


# ── Provider implementations ─────────────────────────────────────────────────

class LLMProvider:
    """Common interface. `complete` returns the raw assistant text."""

    name: str = "base"

    def complete(self, system: str, messages: List[Dict], max_tokens: int) -> str:
        raise NotImplementedError


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self):
        import anthropic  # imported lazily so the app boots without the SDK
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — the 'claude' provider is unavailable."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = CLAUDE_MODEL

    def complete(self, system: str, messages: List[Dict], max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text


class VLLMProvider(LLMProvider):
    """MedGemma (or any model) served by vLLM via its OpenAI-compatible API."""

    name = "medgemma"

    def __init__(self):
        from openai import OpenAI  # lazy import
        self._client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
        self._model = VLLM_MODEL

    def complete(self, system: str, messages: List[Dict], max_tokens: int) -> str:
        # Anthropic keeps `system` separate; OpenAI wants it as the first message.
        oai_messages: List[Dict] = []
        if system:
            oai_messages.append({"role": "system", "content": system})
        oai_messages.extend(messages)

        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return resp.choices[0].message.content or ""


# ── Factory / registry ───────────────────────────────────────────────────────

_PROVIDER_CACHE: Dict[str, LLMProvider] = {}
_PROVIDER_CLASSES = {
    "claude": ClaudeProvider,
    "medgemma": VLLMProvider,
}


def get_provider(name: str) -> LLMProvider:
    """Return a memoized provider instance. Raises with a clear message on failure."""
    name = (name or "").lower().strip()
    if name not in _PROVIDER_CLASSES:
        raise ValueError(
            f"Unknown LLM provider '{name}'. Choose one of: {', '.join(AVAILABLE_PROVIDERS)}."
        )
    if name not in _PROVIDER_CACHE:
        _PROVIDER_CACHE[name] = _PROVIDER_CLASSES[name]()
    return _PROVIDER_CACHE[name]


def resolve_step_providers(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Merge per-request overrides onto the env defaults, validating names.

    Returns a dict like {"extract": "claude", "select": "medgemma", "explain": "claude"}.
    Unknown provider names silently fall back to the step default so a bad UI value
    can never break a request.
    """
    overrides = overrides or {}
    resolved: Dict[str, str] = {}
    for step, default in STEP_DEFAULTS.items():
        choice = str(overrides.get(step, default) or default).lower().strip()
        if choice not in AVAILABLE_PROVIDERS:
            logger.warning("Ignoring unknown provider %r for step %r", choice, step)
            choice = default
        resolved[step] = choice
    return resolved


def vllm_reachable(timeout: float = 1.5) -> bool:
    """Best-effort check that the vLLM OpenAI endpoint is up (for /api/config)."""
    try:
        import urllib.request

        url = VLLM_BASE_URL.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {VLLM_API_KEY}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def config_summary() -> Dict:
    """Serializable config for the frontend to populate model selectors."""
    return {
        "available_providers": list(AVAILABLE_PROVIDERS),
        "step_defaults": dict(STEP_DEFAULTS),
        "claude_model": CLAUDE_MODEL,
        "vllm_model": VLLM_MODEL,
        "vllm_base_url": VLLM_BASE_URL,
        "vllm_reachable": vllm_reachable(),
    }
