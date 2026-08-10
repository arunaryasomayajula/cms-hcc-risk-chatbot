"""
Trace-capture tests.

The important ones are the PHI defaults: a trace is the highest-fidelity record of
what went through the pipeline, so it must not quietly become the place where
un-de-identified notes get written to disk.

Run standalone:  python tests/test_tracing.py
Or with pytest:  pytest tests/test_tracing.py
"""
import glob
import importlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

_TRACE_ENV = ("LANGSMITH_TRACING", "LANGSMITH_TRACE_DIR", "LANGSMITH_TRACE_RAW_INPUT",
              "LANGSMITH_TRACE_UPLOAD", "LANGSMITH_TRACE_MAX_FIELD", "LANGSMITH_PROJECT")


def load_tracing(tmpdir, **env):
    """Re-import `tracing` with a controlled environment (config is read at import)."""
    saved = {k: os.environ.get(k) for k in _TRACE_ENV}
    try:
        for k in _TRACE_ENV:
            os.environ.pop(k, None)
        os.environ["LANGSMITH_TRACING"] = "1"
        os.environ["LANGSMITH_TRACE_DIR"] = tmpdir
        for k, v in env.items():
            os.environ[k] = str(v)
        import tracing
        return importlib.reload(tracing)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def files(tmpdir):
    return sorted(glob.glob(os.path.join(tmpdir, "*.json")))


def test_writes_one_file_per_trace():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d)
        assert t.status()["enabled"], "langsmith should be importable"
        for i in range(3):
            with t.trace("chat", message=f"note {i}") as tr:
                with tr.span("llm.extract", run_type="llm", system="s"):
                    pass
        assert len(files(d)) == 3


def test_filenames_are_unique_and_contain_trace_id():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d)
        ids = []
        for _ in range(5):
            with t.trace("chat", message="x"):
                pass
        names = [os.path.basename(f) for f in files(d)]
        assert len(set(names)) == 5, "filenames must be unique"
        for f in files(d):
            payload = json.load(open(f, encoding="utf-8"))
            assert payload["trace_id"] in os.path.basename(f)
            ids.append(payload["trace_id"])
        assert len(set(ids)) == 5, "each trace needs its own id"


def test_nested_spans_are_captured_in_order():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d)
        with t.trace("chat", message="note") as tr:
            with tr.span("guardrail.deidentify", run_type="tool", chars_in=4) as sp:
                sp.output(redacted=True)
            with tr.span("llm.extract", run_type="llm", system="s") as sp:
                sp.metadata(provider="ollama", model="qwen2.5:3b")
                sp.output(text='{"conditions": []}')
            with tr.span("rag.search", run_type="retriever") as sp:
                sp.output(candidate_count=7)

        run = json.load(open(files(d)[0], encoding="utf-8"))["run"]
        assert run["name"] == "chat" and run["run_type"] == "chain"
        kids = run["child_runs"]
        assert [k["name"] for k in kids] == ["guardrail.deidentify", "llm.extract", "rag.search"]
        assert [k["run_type"] for k in kids] == ["tool", "llm", "retriever"]
        assert kids[1]["metadata"]["provider"] == "ollama"
        assert kids[2]["outputs"]["candidate_count"] == 7
        assert run["latency_ms"] is not None


def test_sanitized_text_is_what_gets_written_by_default():
    """The whole point: PHI must not land in the trace directory by accident."""
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d)
        assert t.RAW_INPUT is False, "raw capture must be opt-in"
        raw = "John Smith MRN 12345 has CHF"
        sanitized = "[REDACTED] MRN [REDACTED] has CHF"
        with t.trace("chat", message=sanitized) as tr:
            tr.metadata(guardrail_redacted=True)

        blob = open(files(d)[0], encoding="utf-8").read()
        assert "John Smith" not in blob
        assert "12345" not in blob
        assert "CHF" in blob, "the clinical content itself should still be traced"
        assert json.loads(blob)["redacted"] is True


def test_raw_input_opt_in_is_recorded_in_the_payload():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d, LANGSMITH_TRACE_RAW_INPUT="1")
        assert t.RAW_INPUT is True
        with t.trace("chat", message="John Smith has CHF"):
            pass
        payload = json.load(open(files(d)[0], encoding="utf-8"))
        # Consumers must be able to tell a redacted trace from a raw one.
        assert payload["redacted"] is False


def test_errors_are_recorded_and_reraised():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d)
        try:
            with t.trace("chat", message="x") as tr:
                with tr.span("llm.extract", run_type="llm"):
                    raise RuntimeError("provider exploded")
        except RuntimeError:
            pass
        else:
            raise AssertionError("exception must propagate to the caller")

        run = json.load(open(files(d)[0], encoding="utf-8"))["run"]
        assert "provider exploded" in (run["error"] or "")
        assert "provider exploded" in (run["child_runs"][0]["error"] or "")


def test_oversized_fields_are_truncated():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d, LANGSMITH_TRACE_MAX_FIELD="100")
        with t.trace("chat", message="x" * 5000):
            pass
        blob = open(files(d)[0], encoding="utf-8").read()
        assert "truncated" in blob
        assert len(blob) < 4000, "a huge prompt must not produce a huge trace"


def test_disabled_tracing_writes_nothing_and_still_works():
    with tempfile.TemporaryDirectory() as d:
        saved = {k: os.environ.get(k) for k in _TRACE_ENV}
        try:
            for k in _TRACE_ENV:
                os.environ.pop(k, None)
            os.environ["LANGSMITH_TRACE_DIR"] = d  # set, but tracing off
            import tracing
            t = importlib.reload(tracing)
            assert t.status()["enabled"] is False
            with t.trace("chat", message="x") as tr:
                with tr.span("llm.extract", run_type="llm") as sp:
                    sp.output(text="hi")
                    sp.metadata(provider="claude")
            assert files(d) == [], "nothing should be written when disabled"
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


def test_no_partial_files_left_behind():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d)
        with t.trace("chat", message="x"):
            pass
        assert glob.glob(os.path.join(d, "*.partial")) == []


def test_status_reports_config():
    with tempfile.TemporaryDirectory() as d:
        t = load_tracing(d, LANGSMITH_PROJECT="unit-test")
        with t.trace("chat", message="x"):
            pass
        st = t.status()
        assert st["enabled"] and st["project"] == "unit-test"
        assert st["dir"] == d and st["redacted"] is True
        assert st["upload"] is False, "must not upload anywhere by default"
        assert st["traces_written"] == 1


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
