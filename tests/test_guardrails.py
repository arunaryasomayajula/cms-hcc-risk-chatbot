"""
PHI guardrail tests (Presidio + spaCy).

Skipped when presidio-analyzer / en_core_web_sm are not installed — the app
itself degrades gracefully in that case (see guardrails.py).

Run with pytest:  pytest tests/test_guardrails.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

import guardrails  # noqa: E402

pytestmark = pytest.mark.skipif(
    not guardrails.status()["available"], reason="presidio / spaCy model not installed"
)


def _types(res):
    return {f["entity_type"]: f["count"] for f in res.findings}


def test_labelled_mrn_keeps_label_and_beats_spacy_date():
    res = guardrails.deidentify("MRN: 00456789. Dx: CKD stage 4.")
    assert "00456789" not in res.sanitized_text
    assert "MRN: [REDACTED_MRN_1]" in res.sanitized_text
    assert _types(res).get("MRN") == 1
    assert "DATE_TIME" not in _types(res)


def test_labelled_sample_ssn_is_redacted():
    # Presidio's generic recognizer rejects 123-45-6789 as a sample number.
    res = guardrails.deidentify("SSN 123-45-6789. History of COPD.")
    assert "123-45-6789" not in res.sanitized_text
    assert _types(res).get("US_SSN") == 1


def test_repeated_value_shares_one_placeholder():
    res = guardrails.deidentify(
        "John Smith reports chest pain. John Smith has diabetes. Email a@example.com."
    )
    assert "John Smith" not in res.sanitized_text
    assert res.sanitized_text.count("[REDACTED_PERSON_1]") == 2
    assert _types(res).get("PERSON") == 1


def test_findings_never_contain_raw_values():
    note = "Maria Gonzalez, phone 555-123-4567, email maria.g@example.com"
    res = guardrails.deidentify(note)
    blob = repr(res.to_dict())
    for raw in ("Maria", "Gonzalez", "555-123-4567", "maria.g@example.com"):
        assert raw not in blob


def test_clinical_text_without_phi_is_untouched():
    note = "Follow-up for morbid obesity, major depressive disorder and CHF."
    res = guardrails.deidentify(note)
    assert res.sanitized_text == note
    assert res.redacted is False
