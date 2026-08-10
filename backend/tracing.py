"""
LangSmith tracing, written to local JSON files — one file per trace.

Every `/api/chat` request becomes a single LangSmith run tree: a root `chat` run
with child spans for the guardrail, each LLM step, the TF-IDF search, and the HCC
calculation. The completed tree is serialized to one JSON file per trace in
`LANGSMITH_TRACE_DIR`, named with the trace's UUID so filenames are unique.

Why local files rather than the LangSmith cloud by default
----------------------------------------------------------
This pipeline handles clinical notes. Traces are the highest-fidelity record of
what went through it, so shipping them to a third party by default would quietly
undo the PHI guardrail that is the point of `guardrails.py`. So:

  * Traces are written to disk locally. Nothing leaves the machine unless you set
    LANGSMITH_TRACE_UPLOAD=1 and provide LANGSMITH_API_KEY.
  * The root run records the **de-identified** note, not the raw one. The raw text
    is only captured if you explicitly set LANGSMITH_TRACE_RAW_INPUT=1, which logs
    a warning at startup.

Even so, a de-identified note is not anonymous data — treat the trace directory as
sensitive and keep it out of version control (it is in .gitignore).

Configuration
-------------
| Variable | Default | Purpose |
|---|---|---|
| `LANGSMITH_TRACE_DIR`       | `backend/traces` | Output folder (created on demand) |
| `LANGSMITH_TRACING`         | `0`              | Master on/off switch |
| `LANGSMITH_PROJECT`         | `cms-hcc-chatbot`| Project name recorded in each trace |
| `LANGSMITH_TRACE_RAW_INPUT` | `0`              | Record the pre-guardrail note (PHI) |
| `LANGSMITH_TRACE_MAX_FIELD` | `20000`          | Truncate any single string field |
| `LANGSMITH_TRACE_UPLOAD`    | `0`              | Also send to LangSmith cloud |

This module degrades to no-ops when tracing is off or `langsmith` is not installed,
so `app.py` never has to branch on whether tracing is enabled.
"""
import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


TRACE_DIR = os.environ.get("LANGSMITH_TRACE_DIR") or os.path.join(_HERE, "traces")
PROJECT = os.environ.get("LANGSMITH_PROJECT", "cms-hcc-chatbot")
ENABLED = _flag("LANGSMITH_TRACING")
RAW_INPUT = _flag("LANGSMITH_TRACE_RAW_INPUT")
UPLOAD = _flag("LANGSMITH_TRACE_UPLOAD")
MAX_FIELD = int(os.environ.get("LANGSMITH_TRACE_MAX_FIELD", "20000"))

SCHEMA_VERSION = 1

_available = False
_error: Optional[str] = None
_write_lock = threading.Lock()
_written = 0

if ENABLED:
    try:
        from langsmith.run_trees import RunTree  # noqa: F401
        _available = True
    except ImportError as exc:
        _error = f"langsmith is not installed ({exc}); tracing disabled. pip install langsmith"
        logger.warning(_error)

if _available:
    if RAW_INPUT:
        logger.warning(
            "LANGSMITH_TRACE_RAW_INPUT=1 — traces in %s will contain the ORIGINAL, "
            "un-de-identified clinical note. Only do this on de-identified test data.",
            TRACE_DIR,
        )
    logger.info("LangSmith tracing on — one JSON per trace in %s (upload=%s)", TRACE_DIR, UPLOAD)


# ── Serialization ────────────────────────────────────────────────────────────

def _truncate(value: Any) -> Any:
    """Cap oversized strings so one huge prompt cannot produce a 50 MB trace."""
    if isinstance(value, str) and len(value) > MAX_FIELD:
        return value[:MAX_FIELD] + f"... [truncated {len(value) - MAX_FIELD} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(v) for v in value]
    return value


def _safe(value: Any) -> Any:
    """Best-effort JSON-safe conversion; never raise from the tracing path."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt else None


def _latency_ms(run) -> Optional[float]:
    if run.start_time and run.end_time:
        return round((run.end_time - run.start_time).total_seconds() * 1000, 1)
    return None


def _serialize(run) -> Dict:
    """RunTree -> plain dict, recursively.

    Hand-rolled rather than `model_dump()` because a RunTree holds a client handle
    and a parent back-reference, which are neither serializable nor wanted here.
    """
    return {
        "id": str(run.id),
        "name": run.name,
        "run_type": run.run_type,
        "start_time": _iso(run.start_time),
        "end_time": _iso(run.end_time),
        "latency_ms": _latency_ms(run),
        "inputs": _safe(_truncate(run.inputs or {})),
        "outputs": _safe(_truncate(run.outputs or {})),
        "error": run.error,
        "tags": list(run.tags or []),
        "metadata": _safe((run.extra or {}).get("metadata", {})),
        "child_runs": [_serialize(c) for c in (run.child_runs or [])],
    }


def _filename(run) -> str:
    """`<UTC timestamp>__<trace uuid>.json` — unique, and sorts chronologically."""
    stamp = (run.start_time or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%f")[:-3]
    return f"{stamp}__{run.trace_id}.json"


def _write(run) -> Optional[str]:
    global _written
    try:
        os.makedirs(TRACE_DIR, exist_ok=True)
        path = os.path.join(TRACE_DIR, _filename(run))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "trace_id": str(run.trace_id),
            "project": PROJECT,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "redacted": not RAW_INPUT,
            "run": _serialize(run),
        }
        # Write-then-rename so a reader never sees a half-written trace.
        tmp = path + ".partial"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        with _write_lock:
            _written += 1
        logger.info("Wrote trace %s", path)
        return path
    except Exception as exc:
        # Observability must never take down the request it is observing.
        logger.error("Failed to write trace: %s", exc, exc_info=True)
        return None


# ── No-op implementations (tracing off / langsmith missing) ──────────────────

class _NullSpan:
    """Same surface as _Span, does nothing. Lets app.py stay branch-free."""
    id = None

    def output(self, **kwargs): pass
    def metadata(self, **kwargs): pass
    def error(self, exc): pass

    @contextmanager
    def span(self, *a, **k):
        yield self

    def __enter__(self): return self
    def __exit__(self, *exc): return False


_NULL = _NullSpan()


# ── Real implementation ──────────────────────────────────────────────────────

class _Span:
    """Wraps a RunTree node with a small, forgiving API."""

    def __init__(self, run):
        self._run = run
        self._outputs: Dict[str, Any] = {}
        self._error: Optional[str] = None

    @property
    def id(self):
        return str(self._run.id)

    def output(self, **kwargs):
        self._outputs.update(kwargs)

    def metadata(self, **kwargs):
        try:
            self._run.add_metadata(kwargs)
        except Exception:
            pass

    def error(self, exc):
        self._error = f"{type(exc).__name__}: {exc}"

    @contextmanager
    def span(self, name: str, run_type: str = "chain", **inputs):
        child = self._run.create_child(
            name=name, run_type=run_type,
            inputs=_safe(_truncate(inputs)),
            start_time=datetime.now(timezone.utc),
        )
        sp = _Span(child)
        try:
            yield sp
        except Exception as exc:
            sp.error(exc)
            sp._finish()
            raise
        sp._finish()

    def _finish(self):
        try:
            self._run.end(
                outputs=_safe(_truncate(self._outputs)) or None,
                error=self._error,
                end_time=datetime.now(timezone.utc),
            )
        except Exception:
            pass


@contextmanager
def trace(name: str = "chat", run_type: str = "chain", metadata: Optional[Dict] = None, **inputs):
    """Root span for one request. Writes exactly one JSON file on exit.

    Yields a no-op object when tracing is disabled, so callers need no guards:

        with tracing.trace("chat", message=text) as tr:
            with tr.span("llm.extract", run_type="llm", prompt=p) as sp:
                sp.output(text=result)
    """
    if not _available:
        yield _NULL
        return

    from langsmith.run_trees import RunTree

    root = None
    try:
        root = RunTree(
            name=name,
            run_type=run_type,
            inputs=_safe(_truncate(inputs)),
            project_name=PROJECT,
            extra={"metadata": _safe(metadata or {})},
            start_time=datetime.now(timezone.utc),
            client=None if UPLOAD else _null_client(),
        )
    except Exception as exc:
        logger.error("Could not start trace: %s", exc, exc_info=True)
        yield _NULL
        return

    sp = _Span(root)
    try:
        yield sp
    except Exception as exc:
        sp.error(exc)
        raise
    finally:
        sp._finish()
        _write(root)
        if UPLOAD:
            try:
                root.post(exclude_child_runs=False)
            except Exception as exc:
                logger.warning("LangSmith upload failed (trace still saved locally): %s", exc)


_NULL_CLIENT = None


def _null_client():
    """A client that swallows posts, so local-only tracing makes no network calls.

    RunTree posts to its client as spans complete; without this it would try to
    reach api.smith.langchain.com and stall on retries with no API key.
    """
    global _NULL_CLIENT
    if _NULL_CLIENT is None:
        class _NullClient:
            otel_exporter = None
            tracing_queue = None

            def create_run(self, *a, **k): pass
            def update_run(self, *a, **k): pass
            def flush(self, *a, **k): pass

        _NULL_CLIENT = _NullClient()
    return _NULL_CLIENT


def status() -> Dict:
    """Serializable tracing state for /api/status."""
    return {
        "enabled": _available,
        "requested": ENABLED,
        "error": _error,
        "dir": TRACE_DIR if _available else None,
        "project": PROJECT,
        "redacted": not RAW_INPUT,
        "upload": UPLOAD,
        "traces_written": _written,
    }
