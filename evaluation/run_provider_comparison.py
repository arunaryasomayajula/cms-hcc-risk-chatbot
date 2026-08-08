"""
Cross-provider / cross-model comparison.

Runs every gold clinical case through the full pipeline once per spec (applied to
all three steps), then diffs the selected ICD-10 codes, resulting HCC set,
applicable risk score, and per-step latency. Also scores each spec's ICD-10
selection against the gold `expected_icd10`.

A spec is `provider` or `provider=model`, so two models of the same provider are
directly comparable:

    python run_provider_comparison.py --providers claude ollama=qwen2.5:7b-instruct
    python run_provider_comparison.py --providers claude=claude-haiku-4-5 claude=claude-opus-4-1

Requires whatever the chosen specs need: ANTHROPIC_API_KEY for 'claude', a running
vLLM endpoint for 'medgemma', the AirLLM sidecar for 'airllm', Ollama for 'ollama'.
Specs that fail are reported per case and skipped.
"""
import os
import csv
import json
import argparse

from common import load_jsonl, run_pipeline, DATASETS_DIR, ensure_results_dir


def parse_spec(spec: str):
    """'ollama=qwen2.5:7b' -> ('ollama', 'qwen2.5:7b').

    Split on '=' rather than ':' — Ollama model ids contain colons (name:tag).
    """
    provider, _, model = spec.partition("=")
    return provider.strip(), (model.strip() or None)


def _codes(result):
    return sorted({c.upper().replace(".", "") for c in result.get("icd10_codes", [])})


def _hccs(result):
    hcc = result.get("hcc_result") or {}
    return sorted({h["hcc"] for h in hcc.get("active_hccs", [])})


def _score(result):
    hcc = result.get("hcc_result") or {}
    return hcc.get("applicable_score")


def prf(predicted, expected):
    p, e = set(predicted), set(expected)
    tp = len(p & e)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(e) if e else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def run_for_spec(cases, spec):
    provider, model = parse_spec(spec)
    rows = []
    for case in cases:
        expected = [c.upper().replace(".", "") for c in case.get("expected_icd10", [])]
        try:
            result = run_pipeline(
                case["note"],
                case.get("demographics", {}),
                providers={"extract": provider, "select": provider, "explain": provider},
                models=({"extract": model, "select": model, "explain": model} if model else None),
                deidentify=False,
            )
        except Exception as exc:
            print(f"  ! {spec} failed on {case.get('id')}: {exc}")
            continue
        codes = _codes(result)
        p, r, f1 = prf(codes, expected)
        timings = result.get("timings", {})
        failures = result.get("parse_failures", [])
        if failures:
            # Otherwise a model that never emits valid JSON just looks like a low
            # F1 score, rather than a broken one.
            print(f"    ! {spec} parse failure on {case.get('id')}: "
                  f"{', '.join(f['step'] for f in failures)}")
        rows.append({
            "id": case.get("id"),
            "spec": spec,
            "provider": provider,
            "model": (result.get("models_used") or {}).get("extract", model or ""),
            "codes": codes,
            "hccs": _hccs(result),
            "score": _score(result),
            "code_precision": p,
            "code_recall": r,
            "code_f1": f1,
            "parse_failures": len(failures),
            "latency_total": round(sum(timings.values()), 3),
            "timings": timings,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(DATASETS_DIR, "clinical_cases.jsonl"))
    ap.add_argument("--providers", nargs="+", default=["claude"],
                    metavar="PROVIDER[=MODEL]",
                    help="e.g. claude ollama=qwen2.5:7b-instruct airllm=Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    cases = load_jsonl(args.dataset)
    all_rows = {}
    for spec in args.providers:
        print(f"\n── Running: {spec} ──")
        all_rows[spec] = run_for_spec(cases, spec)

    # Record the machine — local-provider latencies are meaningless without it.
    try:
        import hardware
        print(f"\n  Hardware: {hardware.summary()}")
    except Exception:
        pass

    # Side-by-side diff
    print("\n── Comparison ───────────────────────────────────────────")
    by_case = {}
    for spec, rows in all_rows.items():
        for row in rows:
            by_case.setdefault(row["id"], {})[spec] = row

    width = max((len(s) for s in all_rows), default=9)
    for cid, specmap in by_case.items():
        print(f"\n  {cid}")
        for spec, row in specmap.items():
            print(f"    {spec:<{width}} f1={row['code_f1']} "
                  f"score={row['score']} latency={row['latency_total']}s codes={row['codes']}")

        # Baseline-vs-rest keeps output linear in the number of specs while still
        # surfacing every disagreement. The old `len(provs) == 2` special case
        # silently printed nothing at all for three or more.
        specs = list(specmap)
        if len(specs) >= 2:
            base = specs[0]
            base_codes = set(specmap[base]["codes"])
            for other in specs[1:]:
                diff = base_codes ^ set(specmap[other]["codes"])
                if diff:
                    print(f"    Δ {base} vs {other}: {sorted(diff)}")
            if len(specs) > 2:
                sets = [set(specmap[s]["codes"]) for s in specs]
                print(f"    ∩ all: {sorted(set.intersection(*sets))}")
                print(f"    ∪ all: {sorted(set.union(*sets))}")

    # Aggregate per spec
    print("\n── Aggregate ────────────────────────────────────────────")
    for spec, rows in all_rows.items():
        if not rows:
            print(f"  {spec}: (no successful runs)")
            continue
        mean_f1 = round(sum(r["code_f1"] for r in rows) / len(rows), 3)
        mean_lat = round(sum(r["latency_total"] for r in rows) / len(rows), 3)
        bad = sum(1 for r in rows if r["parse_failures"])
        suffix = f"  parse-failures={bad}/{len(rows)}" if bad else ""
        print(f"  {spec}: mean code-F1={mean_f1}  mean latency={mean_lat}s  (n={len(rows)}){suffix}")

    # Write CSV + JSON
    results_dir = ensure_results_dir()
    csv_path = os.path.join(results_dir, "provider_comparison.csv")
    flat = [r for rows in all_rows.values() for r in rows]
    if flat:
        fields = ["id", "spec", "provider", "model", "code_precision", "code_recall", "code_f1",
                  "score", "parse_failures", "latency_total", "codes", "hccs"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in flat:
                w.writerow({**row, "codes": " ".join(row["codes"]), "hccs": " ".join(row["hccs"])})
        print(f"\n  Wrote {csv_path}")
    json_path = os.path.join(results_dir, "provider_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"  Wrote {json_path}")


if __name__ == "__main__":
    main()
