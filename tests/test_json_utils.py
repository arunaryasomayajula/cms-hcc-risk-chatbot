"""
Tolerant JSON extraction tests.

Each case here is a real failure mode of small open-weight models against the
"Return ONLY JSON" prompts. The previous `_parse_json` (anchored fence regex +
json.loads) failed every one of them except the first two.

Run standalone:  python tests/test_json_utils.py
Or with pytest:  pytest tests/test_json_utils.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from json_utils import parse_json_object, try_parse_json_object, extract_json_str  # noqa: E402


def test_bare_object():
    assert parse_json_object('{"conditions": []}') == {"conditions": []}


def test_plain_fence():
    text = '```json\n{"conditions": [{"condition": "CHF"}]}\n```'
    assert parse_json_object(text)["conditions"][0]["condition"] == "CHF"


def test_fence_without_language_tag():
    assert parse_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_preamble_before_fence():
    """The single most common small-model failure; the old regex was anchored."""
    text = 'Sure! Here is the JSON:\n```json\n{"conditions": [{"condition": "CHF"}]}\n```'
    assert parse_json_object(text)["conditions"][0]["condition"] == "CHF"


def test_preamble_and_postamble():
    text = ('Here you go:\n```json\n{"conditions": [{"condition": "COPD"}]}\n```\n'
            'Let me know if you need anything else!')
    assert parse_json_object(text)["conditions"][0]["condition"] == "COPD"


def test_trailing_comma():
    assert parse_json_object('{"conditions": [{"condition": "CHF"},]}')["conditions"]


def test_preamble_fence_trailing_comma_postamble():
    """Everything wrong at once — the composite case from the plan."""
    text = ("Sure! Here's the JSON:\n```json\n"
            '{"conditions": [{"condition": "CHF"},]}\n'
            "```\nHope that helps.")
    assert parse_json_object(text) == {"conditions": [{"condition": "CHF"}]}


def test_bare_object_with_prose_no_fence():
    text = 'The extracted conditions are {"conditions": [{"condition": "CKD"}]} based on the note.'
    assert parse_json_object(text)["conditions"][0]["condition"] == "CKD"


def test_bare_array_promoted_via_list_key():
    """Small models often emit the array instead of the requested wrapper object."""
    out = parse_json_object('[{"condition": "CHF"}]', list_key="conditions")
    assert out == {"conditions": [{"condition": "CHF"}]}


def test_bare_array_in_fence_with_list_key():
    out = parse_json_object('```json\n[{"icd10": "E119"}]\n```', list_key="selected_codes")
    assert out["selected_codes"][0]["icd10"] == "E119"


def test_bare_array_without_list_key_is_an_error():
    parsed, err = try_parse_json_object('[{"condition": "CHF"}]')
    assert parsed is None and err


def test_braces_inside_strings_do_not_confuse_the_scanner():
    text = 'note: {"conditions": [{"condition": "uses {braces} in text"}]} done'
    assert parse_json_object(text)["conditions"][0]["condition"] == "uses {braces} in text"


def test_escaped_quote_inside_string():
    text = '{"conditions": [{"condition": "patient said \\"ouch\\""}]}'
    assert parse_json_object(text)["conditions"][0]["condition"] == 'patient said "ouch"'


def test_apostrophes_are_not_mangled():
    """We deliberately do NOT do single-quote repair — clinical text is full of these."""
    text = '{"conditions": [{"condition": "Crohn\'s disease"}]}'
    assert parse_json_object(text)["conditions"][0]["condition"] == "Crohn's disease"


def test_nested_objects():
    text = '{"selected_codes": [{"icd10": "E119", "meta": {"a": {"b": 1}}}]}'
    assert parse_json_object(text)["selected_codes"][0]["meta"]["a"]["b"] == 1


def test_empty_input_errors_cleanly():
    for bad in ("", "   ", None):
        parsed, err = try_parse_json_object(bad)
        assert parsed is None and err


def test_pure_prose_errors_cleanly():
    parsed, err = try_parse_json_object("I could not find any conditions in this note.")
    assert parsed is None
    assert "no JSON object found" in err


def test_truncated_json_errors_rather_than_half_parsing():
    """A `length` finish_reason mid-object must not silently yield a partial result."""
    parsed, err = try_parse_json_object('{"conditions": [{"condition": "CHF"')
    assert parsed is None and err


def test_try_parse_returns_error_string_not_exception():
    parsed, err = try_parse_json_object("not json at all")
    assert parsed is None and isinstance(err, str) and err


def test_extract_json_str_prefers_fence():
    assert extract_json_str('x ```json\n{"a":1}\n``` y') == '{"a":1}'


if __name__ == "__main__":
    failures = 0
    for fname, fn in sorted(globals().items()):
        if fname.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {fname}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {fname}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  ERROR {fname}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'OK'} - {failures} failure(s)")
    sys.exit(1 if failures else 0)
