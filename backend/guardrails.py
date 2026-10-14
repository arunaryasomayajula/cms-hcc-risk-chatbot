"""
PHI / PII guardrails using Microsoft Presidio + a small spaCy model
(de-identify, then proceed).

Before any clinical note is sent to an LLM — Claude or MedGemma — it is scanned
with Presidio's analyzer. Detected PHI (names, dates, phone numbers, SSNs,
medical-record numbers, ...) is replaced with placeholders such as
`[REDACTED_PERSON_1]`, and the sanitized text is what the pipeline actually uses.

Design choices:
- **De-identify, don't block.** The clinical content the HCC pipeline needs
  (diagnoses, labs, meds) is preserved; only identifiers are removed.
- **No de-anonymization.** No placeholder -> real value map is kept, so PHI never
  re-enters the model output or the stored conversation.
- **Never surface raw values.** `findings` reports only entity *types* and counts.
- **Lightweight.** Presidio + `en_core_web_sm` has no torch dependency, which keeps
  the Docker image small. Override the model with SPACY_MODEL (e.g. en_core_web_lg)
  if you can afford the size for better name recall.
- **Graceful fallback.** If `presidio-analyzer` (or its spaCy model) is not installed,
  the app still boots; `deidentify()` returns the text unchanged and flags itself as
  disabled so the operator knows the guardrail is off.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
SCORE_THRESHOLD = float(os.environ.get("GUARDRAILS_SCORE_THRESHOLD", "0.4"))
GUARDRAILS_ENABLED = os.environ.get("GUARDRAILS_ENABLED", "true").lower() in ("1", "true", "yes")

# Entity types tuned for clinical notes (Presidio recognizer names).
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
    """Lazily-initialized Presidio wrapper. One analyzer, reused across requests."""

    def __init__(self):
        self._analyzer = None
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
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            nlp = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
            }).create_engine()
            self._analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
            self._available = True
            logger.info("Guardrails ready — Presidio analyzer initialized (%s)", SPACY_MODEL)
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
            results = self._analyzer.analyze(
                text=text, language="en", entities=DEFAULT_ENTITY_TYPES,
                score_threshold=SCORE_THRESHOLD,
            )
            sanitized, counts = self._replace(text, results)
        except Exception as exc:
            logger.error("De-identification failed, passing text through: %s", exc)
            return DeidResult(text, enabled=True, note=f"scan error: {exc}")

        # Supplementary MRN pass on the already-sanitized text.
        sanitized, mrn_count = self._redact_mrn(sanitized)

        findings = [{"entity_type": k, "count": v} for k, v in sorted(counts.items())]
        if mrn_count:
            findings.append({"entity_type": "MRN", "count": mrn_count})

        return DeidResult(sanitized_text=sanitized, redacted=bool(findings), findings=findings)

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
    def _replace(text: str, results) -> tuple:
        """Swap detected spans for numbered placeholders.

        Overlapping spans keep the earliest/longest one. Only entity types and
        counts are returned; raw values are never retained.
        """
        ordered = sorted(results, key=lambda r: (r.start, -(r.end - r.start), -r.score))
        kept, last_end = [], -1
        for r in ordered:
            if r.start >= last_end:
                kept.append(r)
                last_end = r.end
        counts: Dict[str, int] = {}
        out, pos = [], 0
        for r in kept:
            counts[r.entity_type] = counts.get(r.entity_type, 0) + 1
            out.append(text[pos:r.start])
            out.append(f"[REDACTED_{r.entity_type}_{counts[r.entity_type]}]")
            pos = r.end
        out.append(text[pos:])
        return "".join(out), counts


# Module-level singleton
_guard = Guardrails()


def deidentify(text: str) -> DeidResult:
    return _guard.deidentify(text)


def status() -> Dict:
    return _guard.status()
