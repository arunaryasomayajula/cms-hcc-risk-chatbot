"""
PHI / PII guardrails using Microsoft Presidio (de-identify, then proceed).

Before any clinical note is sent to an LLM — Claude or MedGemma — it is scanned
with Presidio's `AnalyzerEngine` (regex/checksum recognizers + spaCy NER).
Detected PHI (names, dates, phone numbers, SSNs, medical-record numbers, ...) is
replaced with placeholders such as `[REDACTED_PERSON_1]`, and the sanitized text
is what the pipeline actually uses.

Presidio is used directly rather than through LLM Guard: LLM Guard's `Anonymize`
wraps the same analyzer but hard-depends on torch + transformers, which alone
put the Docker image several GB over budget (see docs/docker.md).

Design choices:
- **De-identify, don't block.** The clinical content the HCC pipeline needs
  (diagnoses, labs, meds) is preserved; only identifiers are removed.
- **No de-anonymization.** The placeholder → real value map exists only inside
  one `deidentify()` call, so PHI never re-enters the model output or the stored
  conversation.
- **Never surface raw values.** `findings` reports only entity *types* and counts.
- **Graceful fallback.** If `presidio-analyzer` (or its spaCy model) is not
  installed, the app still boots; `deidentify()` returns the text unchanged and
  flags itself as disabled so the operator knows the guardrail is off.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

GUARDRAILS_ENABLED = os.environ.get("GUARDRAILS_ENABLED", "true").lower() in ("1", "true", "yes")
SPACY_MODEL = os.environ.get("GUARDRAILS_SPACY_MODEL") or os.environ.get("SPACY_MODEL", "en_core_web_sm")
SCORE_THRESHOLD = float(os.environ.get("GUARDRAILS_SCORE_THRESHOLD", "0.35"))

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

# Labelled identifiers. A value written after an "MRN" / "SSN" label is PHI no
# matter what it looks like, so these score 1.0 and win any overlap — otherwise
# spaCy tags "MRN: 00123456" as a DATE, and Presidio's generic SSN recognizer
# discards well-known sample numbers such as 123-45-6789. The label itself is
# kept (lookbehind; Presidio matches with the `regex` module, which allows
# variable-width lookbehind).
_LABELLED_PATTERNS = {
    "MRN": r"(?<=\b(?:MRN|Medical Record(?: Number)?)\s*[:#]?\s*)[A-Za-z0-9\-]{4,}",
    "US_SSN": r"(?<=\b(?:SSN|Social Security(?: Number| No\.?)?)\s*[:#]?\s*)\d{3}[- ]?\d{2}[- ]?\d{4}\b",
}
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
            "spacy_model": SPACY_MODEL,
            "score_threshold": SCORE_THRESHOLD,
        }

    def _ensure_init(self):
        if self._initialized:
            return
        self._initialized = True
        if not GUARDRAILS_ENABLED:
            self._init_error = "disabled via GUARDRAILS_ENABLED"
            return
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Name the model explicitly: Presidio's default config asks for
            # en_core_web_lg (~560 MB) and tries to download it at runtime.
            nlp_engine = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
            }).create_engine()
            # Presidio warns once per non-English recognizer it skips; silence
            # just that construction step.
            presidio_log = logging.getLogger("presidio-analyzer")
            prev_level = presidio_log.level
            presidio_log.setLevel(logging.ERROR)
            try:
                self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            finally:
                presidio_log.setLevel(prev_level)
            for entity, regex in _LABELLED_PATTERNS.items():
                self._analyzer.registry.add_recognizer(PatternRecognizer(
                    supported_entity=entity,
                    name=f"Labelled{entity}Recognizer",
                    patterns=[Pattern(f"labelled {entity}", regex, 1.0)],
                ))
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
                text=text,
                entities=DEFAULT_ENTITY_TYPES + ["MRN"],
                language="en",
                score_threshold=SCORE_THRESHOLD,
            )
            sanitized, placeholders = self._replace(text, results)
        except Exception as exc:
            logger.error("De-identification failed, passing text through: %s", exc)
            return DeidResult(text, enabled=True, note=f"scan error: {exc}")

        findings = self._summarize(placeholders)
        return DeidResult(sanitized_text=sanitized, redacted=bool(findings), findings=findings)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _replace(text: str, results: List) -> tuple:
        """Swap detected spans for `[REDACTED_<TYPE>_<n>]` placeholders.

        Overlapping detections are resolved by keeping the highest-scoring (then
        longest) span. Repeats of the same value share one placeholder. Returns
        the sanitized text and the distinct placeholders used — never raw values.
        """
        kept = []
        for r in sorted(results, key=lambda r: (-r.score, -(r.end - r.start))):
            if all(r.end <= k.start or r.start >= k.end for k in kept):
                kept.append(r)
        kept.sort(key=lambda r: r.start)

        by_value: Dict[tuple, str] = {}
        per_type: Dict[str, int] = {}
        out, pos = [], 0
        for r in kept:
            key = (r.entity_type, text[r.start:r.end].strip().lower())
            if key not in by_value:
                per_type[r.entity_type] = per_type.get(r.entity_type, 0) + 1
                by_value[key] = f"[REDACTED_{r.entity_type}_{per_type[r.entity_type]}]"
            out.append(text[pos:r.start])
            out.append(by_value[key])
            pos = r.end
        out.append(text[pos:])
        return "".join(out), list(by_value.values())

    @staticmethod
    def _summarize(placeholders: List[str]) -> List[Dict]:
        """Count findings by entity type from the placeholders that were issued."""
        counts: Dict[str, int] = {}
        for placeholder in placeholders:
            m = _PLACEHOLDER_TYPE_RE.search(placeholder)
            etype = m.group(1) if m else "UNKNOWN"
            counts[etype] = counts.get(etype, 0) + 1
        return [{"entity_type": k, "count": v} for k, v in sorted(counts.items())]


# Module-level singleton
_guard = Guardrails()


def deidentify(text: str) -> DeidResult:
    return _guard.deidentify(text)


def status() -> Dict:
    return _guard.status()
