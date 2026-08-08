"""
Tolerant JSON extraction for LLM output.

The pipeline asks models to "Return ONLY JSON". Claude obliges; smaller local
models routinely wrap the object in prose ("Sure! Here's the JSON: ...`), fence it,
leave a trailing comma, or emit a bare array instead of the requested object.

This module is the single implementation shared by `backend/app.py` and
`evaluation/common.py` (which puts `backend/` on sys.path). It replaces a
`_parse_json` helper that was duplicated in both files and only handled output
that was *entirely* a fenced block or *entirely* bare JSON.
"""
import json
import re
from typing import Optional, Tuple

# Matches a ```json ... ``` fence ANYWHERE in the text, not just as the whole string.
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Trailing comma before a closing brace/bracket — the single most common defect
# in small-model JSON.
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """Return the first balanced open_ch..close_ch span, ignoring delimiters in strings."""
    start = text.find(open_ch)
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair(s: str) -> str:
    """Conservative fixes only.

    Trailing commas are safe to strip. Single-quote-to-double-quote conversion is
    deliberately NOT attempted — clinical text is full of apostrophes and the
    false-positive rate is unacceptable.
    """
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def _try_loads(s: str):
    """json.loads with one repair pass. Returns the parsed value or None."""
    for candidate in (s, _repair(s)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def extract_json_str(text: str) -> Optional[str]:
    """Return the most plausible JSON substring in `text`, or None."""
    if not text:
        return None
    stripped = text.strip()

    fence = _FENCE_RE.search(stripped)
    if fence:
        inner = fence.group(1).strip()
        if inner:
            return inner

    return _balanced(stripped, '{', '}') or _balanced(stripped, '[', ']')


def parse_json_object(text: str, list_key: Optional[str] = None) -> dict:
    """Parse a JSON object out of `text`, tolerating fences, prose, and stray commas.

    `list_key` handles the common small-model failure of emitting a bare array
    instead of the requested wrapper object: a top-level list is returned as
    ``{list_key: [...]}``.

    Raises ValueError when nothing object-shaped can be recovered.
    """
    if not text or not text.strip():
        raise ValueError("empty model output")

    stripped = text.strip()

    # Pass 1 — whole text, then fence body. These are authoritative: if the model
    # produced a well-formed top-level value, honour it rather than digging inside.
    primary = [stripped]
    fence = _FENCE_RE.search(stripped)
    if fence and fence.group(1).strip():
        primary.append(fence.group(1).strip())

    saw_list = False
    for candidate in primary:
        parsed = _try_loads(candidate)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            if list_key:
                return {list_key: parsed}
            saw_list = True

    if saw_list:
        # Descending into a top-level array to grab its first element would look
        # like success and then silently drop every other entry.
        raise ValueError(
            "model returned a JSON array where an object was expected "
            f"(first 200 chars: {stripped[:200]!r})"
        )

    # Pass 2 — the model wrapped valid JSON in prose. Scan for a balanced span.
    for span in (_balanced(stripped, '{', '}'), _balanced(stripped, '[', ']')):
        if not span:
            continue
        parsed = _try_loads(span)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and list_key:
            return {list_key: parsed}

    raise ValueError(f"no JSON object found in model output (first 200 chars: {stripped[:200]!r})")


def try_parse_json_object(
    text: str, list_key: Optional[str] = None
) -> Tuple[Optional[dict], Optional[str]]:
    """Non-raising form: returns (parsed, None) or (None, error_message).

    Callers use the error string to record a structured parse failure rather than
    silently degrading to an empty result.
    """
    try:
        return parse_json_object(text, list_key=list_key), None
    except ValueError as exc:
        return None, str(exc)
