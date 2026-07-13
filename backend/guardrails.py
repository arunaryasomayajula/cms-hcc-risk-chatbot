"""
PHI / PII guardrails using LLM Guard (de-identify, then proceed).

Before any clinical note is sent to an LLM — Claude or MedGemma — it is scanned
with LLM Guard's `Anonymize` input scanner. Detected PHI (names, dates, phone
numbers, SSNs, medical-record numbers, ...) is replaced with placeholders such as
`[REDACTED_PERSON_1]`, and the sanitized text is what the pipeline actually uses.

Design choices:
- **De-identify, don't block.** The clinical content the HCC pipeline needs
  (diagnoses, labs, meds) is preserved; only identifiers are removed.
- **No de-anonymization.** The Vault (placeholder → real value map) never leaves
  this process and is discarded after each request, so PHI never re-enters the
  model output or the stored conversation.
- **Never surface raw values.** `findings` reports only entity *types* and counts.
- **Graceful fallback.** If `llm-guard` (or its spaCy model) is not installed, the
  app still boots; `deidentify()` returns the text unchanged and flags itself as
  disabled so the operator knows the guardrail is off.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

GUARDRAILS_ENABLED = os.environ.get("GUARDRAILS_ENABLED", "true").lower() in ("1", "true", "yes")

# Entity types tuned for clinical notes (Presidio recognizer names, via LLM Guard).
DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "DATE_TIME",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_LICENSE",
    "LOCATION",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "URL",
]

# Supplementary regex for medical-record-number-like identifiers that generic
# recognizers often miss (e.g. "MRN: 00123456", "MRN 4432219").
_MRN_RE = re.compile(r"\b(?:MRN|Medical Record(?: Number)?)\s*[:#]?\s*([A-Za-z0-9\-]{4,})", re.IGNORECASE)
_PLACEHOLDER_TYPE_RE = re.compile(r"\[REDACTED_([A-Z_]+?)_\d+\]")


@dataclass
class DeidResult:
    sanitized_text: str
    redacted: bool = False
    findings: List[Dict] = field(default_factory=list)  # [{"entity_type": ..., "count": N}]
    enabled: bool = True
    note: Optional[str] = None  # populated when the guardrail is unavailable

    def to_dict(self) -> Dict:
        return {
            "redacted": self.redacted,
            "findings": self.findings,
            "enabled": self.enabled,
            "note": self.note,
        }


class Guardrails:
    """Lazily-initialized LLM Guard wrapper. One analyzer, reused across requests."""

    def __init__(self):
        self._scanner = None
        self._vault = None
        self._available = False
        self._init_error: Optional[str] = None
        self._initialized = False

    @property
    def available(self) -> bool:
        self._ensure_init()
        return self._available

    def status(self) -> Dict:
        self._ensure_init()
        return {
            "enabled": GUARDRAILS_ENABLED,
            "available": self._available,
            "error": self._init_error,
            "entity_types": DEFAULT_ENTITY_TYPES,
        }

    def _ensure_init(self):
        if self._initialized:
            return
        self._initialized = True
        if not GUARDRAILS_ENABLED:
            self._init_error = "disabled via GUARDRAILS_ENABLED"
            return
        try:
            from llm_guard.input_scanners import Anonymize
            from llm_guard.vault import Vault

            self._vault = Vault()
            self._scanner = Anonymize(
                self._vault,
                entity_types=DEFAULT_ENTITY_TYPES,
            )
            self._available = True
            logger.info("Guardrails ready — LLM Guard Anonymize scanner initialized")
        except Exception as exc:  # missing package, missing spaCy model, etc.
            self._init_error = str(exc)
            logger.warning(
                "Guardrails unavailable (%s). Clinical notes will NOT be de-identified.",
                exc,
            )

    def deidentify(self, text: str) -> DeidResult:
        """Redact PHI from `text`. Always returns usable text (original if disabled)."""
        self._ensure_init()

        if not GUARDRAILS_ENABLED:
            return DeidResult(text, enabled=False, note="Guardrails disabled")
        if not self._available:
            return DeidResult(
                text, enabled=False,
                note=f"Guardrails unavailable: {self._init_error}",
            )

        try:
            before = len(self._vault.get())
            # llm-guard's scan() return shape varies by version: older releases
            # return (sanitized, is_valid); newer ones (sanitized, is_valid, risk).
            result = self._scanner.scan(text)
            sanitized = result[0] if isinstance(result, (list, tuple)) else result
            new_entries = self._vault.get()[before:]
        except Exception as exc:
            logger.error("De-identification failed, passing text through: %s", exc)
            return DeidResult(text, enabled=True, note=f"scan error: {exc}")

        # Supplementary MRN pass on the already-sanitized text.
        sanitized, mrn_count = self._redact_mrn(sanitized)

        findings = self._summarize(new_entries)
        if mrn_count:
            findings.append({"entity_type": "MRN", "count": mrn_count})

        redacted = bool(findings)
        # Do not retain PHI: clear this request's entries from the vault.
        self._clear_vault()

        return DeidResult(sanitized_text=sanitized, redacted=redacted, findings=findings)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _redact_mrn(text: str) -> tuple:
        count = 0

        def _sub(m):
            nonlocal count
            count += 1
            return f"MRN [REDACTED_MRN_{count}]"

        return _MRN_RE.sub(_sub, text), count

    @staticmethod
    def _summarize(vault_entries: List) -> List[Dict]:
        """Count findings by entity type from vault (placeholder, value) tuples.

        Only the placeholder is inspected — raw values are never read or returned.
        """
        counts: Dict[str, int] = {}
        for entry in vault_entries:
            placeholder = entry[0] if isinstance(entry, (list, tuple)) else str(entry)
            m = _PLACEHOLDER_TYPE_RE.search(placeholder)
            etype = m.group(1) if m else "UNKNOWN"
            counts[etype] = counts.get(etype, 0) + 1
        return [{"entity_type": k, "count": v} for k, v in sorted(counts.items())]

    def _clear_vault(self):
        """Best-effort reset so PHI is not retained between requests."""
        try:
            self._vault._tuples.clear()  # LLM Guard stores entries in a private list
        except Exception:
            # Fall back to a fresh vault + rebind (rare; keeps memory clean).
            try:
                from llm_guard.vault import Vault
                from llm_guard.input_scanners import Anonymize

                self._vault = Vault()
                self._scanner = Anonymize(self._vault, entity_types=DEFAULT_ENTITY_TYPES)
            except Exception:
                pass


# Module-level singleton
_guard = Guardrails()


def deidentify(text: str) -> DeidResult:
    return _guard.deidentify(text)


def status() -> Dict:
    return _guard.status()
