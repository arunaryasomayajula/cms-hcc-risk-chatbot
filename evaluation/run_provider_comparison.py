"""
Claude vs MedGemma provider comparison.

Runs every gold clinical case through the full pipeline once per provider (applied
to all three steps), then diffs the selected ICD-10 codes, resulting HCC set,
applicable risk score, and per-step latency. Also scores each provider's ICD-10
selection against the gold `expected_icd10`.

Requires:
- ANTHROPIC_API_KEY for the 'claude' provider.
- A running vLLM endpoint (VLLM_BASE_URL) serving MedGemma for the 'medgemma' provider.
  Providers that are unavailable are skipped with a warning.

Usage:  python run_provider_comparison.py [--providers claude medgemma]
"""
import os
import csv
import json
import argparse

from common import load_jsonl, run_pipeline, DATASETS_DIR, ensure_results_dir


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


def run_for_provider(cases, provider):
    rows = []
    for case in cases:
        expected = [c.upper().replace(".", "") for c in case.get("expected_icd10", [])]
        try:
            result = run_pipeline(
                case["note"],
                case.get("demographics", {}),
                providers={"extract": provider, "select": provider, "explain": provider},
                deidentify=False,
            )
        except Exception as exc:
            print(f"  ! {provider} failed on {case.get('id')}: {exc}")
            continue
        codes = _codes(result)
        p, r, f1 = prf(codes, expected)
        timings = result.get("timings", {})
        rows.append({
            "id": case.get("id"),
            "provider": provider,
            "codes": codes,
            "hccs": _hccs(result),
            "score": _score(result),
            "code_precision": p,
            "code_recall": r,
            "code_f1": f1,
            "latency_total": round(sum(timings.values()), 3),
            "timings": timings,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(DATASETS_DIR, "clinical_cases.jsonl"))
    ap.add_argument("--providers", nargs="+", default=["claude", "medgemma"])
    args = ap.parse_args()

    cases = load_jsonl(args.dataset)
    all_rows = {}
    for provider in args.providers:
        print(f"\n── Running provider: {provider} ──")
        all_rows[provider] = run_for_provider(cases, provider)

    # Side-by-side diff
    print("\n── Provider comparison ──────────────────────────────────")
    by_case = {}
    for provider, rows in all_rows.items():
        for row in rows:
            by_case.setdefault(row["id"], {})[provider] = row

    for cid, provmap in by_case.items():
        print(f"\n  {cid}")
        for provider, row in provmap.items():
            print(f"    {provider:<9} f1={row['code_f1']} "
                  f"score={row['score']} latency={row['latency_total']}s codes={row['codes']}")
        provs = list(provmap)
        if len(provs) == 2:
            a, b = provmap[provs[0]], provmap[provs[1]]
            code_diff = set(a["codes"]) ^ set(b["codes"])
            if code_diff:
                print(f"    Δ codes: {sorted(code_diff)}")

    # Aggregate F1 per provider
    print("\n── Aggregate ────────────────────────────────────────────")
    for provider, rows in all_rows.items():
        if not rows:
            print(f"  {provider}: (no successful runs)")
            continue
        mean_f1 = round(sum(r["code_f1"] for r in rows) / len(rows), 3)
        mean_lat = round(sum(r["latency_total"] for r in rows) / len(rows), 3)
        print(f"  {provider}: mean code-F1={mean_f1}  mean latency={mean_lat}s  (n={len(rows)})")

    # Write CSV + JSON
    results_dir = ensure_results_dir()
    csv_path = os.path.join(results_dir, "provider_comparison.csv")
    flat = [r for rows in all_rows.values() for r in rows]
    if flat:
        fields = ["id", "provider", "code_precision", "code_recall", "code_f1",
                  "score", "latency_total", "codes", "hccs"]
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
