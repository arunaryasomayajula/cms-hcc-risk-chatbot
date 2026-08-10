# Tracing with LangSmith (local JSON files)

Every `/api/chat` request can be captured as a **LangSmith run tree** and written to
**one JSON file per trace** in a folder you choose. Instrumentation lives in
[`backend/tracing.py`](../backend/tracing.py).

Traces answer the questions logs can't: which step was slow, what exactly was sent to
which model, what came back before parsing, and where a bad answer originated.

## 1. Turn it on

```powershell
$env:LANGSMITH_TRACING   = "1"
$env:LANGSMITH_TRACE_DIR = "C:\traces\cms-hcc"   # any folder; created on demand
python backend/app.py
```

```bash
export LANGSMITH_TRACING=1
export LANGSMITH_TRACE_DIR=/var/log/cms-hcc-traces
python backend/app.py
```

`langsmith` is in `backend/requirements.txt`. If it is missing, tracing disables
itself with a warning rather than breaking the app, and `GET /api/status` reports why:

```json
{"tracing": {"enabled": true, "dir": "C:\\traces\\cms-hcc", "project": "cms-hcc-chatbot",
             "redacted": true, "upload": false, "traces_written": 12}}
```

## 2. PHI: read this before enabling in production

A trace is the highest-fidelity record of what went through the pipeline. Left
unguarded it becomes the one place where un-de-identified clinical notes get written
to disk in plaintext — quietly undoing [the PHI guardrail](../backend/guardrails.py)
that runs before every LLM call. Two defaults follow from that:

- **The note recorded in the trace is the de-identified one.** The `guardrail`
  span records *what* was removed (entity types and counts), never the identifiers.
- **Nothing leaves the machine.** Traces are local files. The LangSmith cloud is used
  only if you set `LANGSMITH_TRACE_UPLOAD=1` *and* supply `LANGSMITH_API_KEY`.

Verified by test: with a note containing a name, MRN, DOB, email, and phone number,
none of the five appear anywhere in the trace file, while the clinical content
("chronic kidney disease") is retained. See
[`tests/test_tracing.py`](../tests/test_tracing.py).

Setting `LANGSMITH_TRACE_RAW_INPUT=1` captures the original text and logs a warning at
startup. Only do that on synthetic or already-de-identified data. Either way the
payload carries `"redacted": true|false`, so a consumer can always tell which it has.

> A de-identified note is still not anonymous. Treat the trace folder as sensitive:
> it is in `.gitignore`, and it deserves the same disk encryption and retention policy
> as any other clinical artifact.

## 3. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LANGSMITH_TRACING` | `0` | Master on/off switch |
| `LANGSMITH_TRACE_DIR` | `backend/traces` | Output folder |
| `LANGSMITH_PROJECT` | `cms-hcc-chatbot` | Recorded in each trace |
| `LANGSMITH_TRACE_RAW_INPUT` | `0` | Capture the pre-guardrail note (PHI) |
| `LANGSMITH_TRACE_MAX_FIELD` | `20000` | Truncate any single string field |
| `LANGSMITH_TRACE_UPLOAD` | `0` | Also send to LangSmith cloud (needs `LANGSMITH_API_KEY`) |

## 4. File layout

One file per trace:

```
20260810T035538041__019fe9cf-ceb9-7693-b542-96715be2f84d.json
└─ UTC timestamp ──┘  └─ trace UUID (the unique identifier) ─┘
```

The timestamp prefix makes the folder sort chronologically; the UUID guarantees
uniqueness. Files are written to `.partial` and atomically renamed, so a tailing
process never reads a half-written trace.

```jsonc
{
  "schema_version": 1,
  "trace_id": "019fe9cf-ceb9-7693-b542-96715be2f84d",
  "project": "cms-hcc-chatbot",
  "captured_at": "2026-08-10T03:57:58Z",
  "redacted": true,                     // false only if RAW_INPUT was set
  "run": {
    "name": "chat", "run_type": "chain", "latency_ms": 140254.2,
    "inputs":  { "message": "...", "demographics": {...}, "history_turns": 0 },
    "outputs": { "response": "...", "icd10_codes": [...], "status": "success" },
    "metadata": { "providers": {...}, "models": {...}, "is_clinical": true },
    "child_runs": [ /* one per span, same shape, nested */ ]
  }
}
```

Spans emitted per request:

| Span | Type | Captures |
|---|---|---|
| `chat` | chain | Root: sanitized note, demographics, providers/models, final status |
| `guardrail.deidentify` | tool | Entity types and counts removed, chars in/out |
| `llm.extract` | llm | System + messages, raw model output, provider/model/max_tokens |
| `rag.search` | retriever | Search queries, candidate count, top candidates |
| `llm.select` | llm | As above, for code selection |
| `hcc.calculate` | tool | Input codes, active HCCs, applicable model and score |
| `llm.explain` | llm | As above, for the explanation |

Each span carries its own `latency_ms`, which is the quickest way to see where a slow
request went. A real example: `llm.explain` 68.0 s, `llm.select` 52.3 s,
`llm.extract` 19.9 s, `rag.search` 14.9 ms, `hcc.calculate` 65.4 ms — the LLM calls
dominate by three orders of magnitude, and the local pipeline is essentially free.

## 5. Reading traces

They are plain JSON, so `jq` works. Slowest step per trace:

```bash
jq -r '[.trace_id, (.run.child_runs | max_by(.latency_ms) | .name)] | @tsv' traces/*.json
```

Every trace where a model returned unparseable JSON:

```bash
jq -r 'select(.run.outputs.status == "partial") | .trace_id' traces/*.json
```

Mean latency by step across a run of traces:

```bash
jq -s '[.[].run.child_runs[]] | group_by(.name)
       | map({step: .[0].name, mean_ms: (map(.latency_ms) | add / length)})' traces/*.json
```

## 6. Sending to the LangSmith cloud

For the hosted UI (waterfall views, datasets, evaluation):

```bash
export LANGSMITH_TRACE_UPLOAD=1
export LANGSMITH_API_KEY=lsv2_...
```

Local files are still written, so upload failures never lose a trace. **Do not enable
this with real patient data** unless your BAA and data-processing agreements cover
LangSmith — de-identified is not the same as anonymous.
