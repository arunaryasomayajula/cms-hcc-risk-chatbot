"""
Guardrail (PHI de-identification) evaluation.

Runs `backend/guardrails.deidentify` over labeled PHI cases and measures:
- entity-type recall/precision : did we redact the entity types we expected, and
  did we avoid inventing types that weren't there?
- clinical-term survival        : the diagnoses/terms in `must_survive` must still
  be present verbatim in the sanitized text (no over-redaction of clinical content).

No API key required — the guardrail runs entirely locally (LLM Guard + spaCy).

Usage:  python run_guardrail_eval.py
"""
import os
import json
import argparse

from common import load_jsonl, DATASETS_DIR, ensure_results_dir


def evaluate(dataset_path):
    import guardrails as guardrails_mod

    status = guardrails_mod.status()
    if not status.get("available"):
        print(f"  ! Guardrail unavailable: {status.get('error')}")
        print("    Install llm-guard and run: python -m spacy download en_core_web_lg")
        return None, status

    cases = load_jsonl(dataset_path)
    per_case = []
    tot_tp = tot_fp = tot_fn = 0
    survival_ok = survival_total = 0

    for case in cases:
        res = guardrails_mod.deidentify(case["note"])
        found_types = {f["entity_type"] for f in res.findings}
        expected_types = set(case.get("expected_entities", []))

        tp = len(found_types & expected_types)
        fp = len(found_types - expected_types)
        fn = len(expected_types - found_types)
        tot_tp += tp; tot_fp += fp; tot_fn += fn

        must = case.get("must_survive", [])
        survived = [t for t in must if t.lower() in res.sanitized_text.lower()]
        survival_total += len(must)
        survival_ok += len(survived)
        corrupted = [t for t in must if t not in survived]

        per_case.append({
            "id": case.get("id"),
            "expected_entities": sorted(expected_types),
            "found_entities": sorted(found_types),
            "missed": sorted(expected_types - found_types),
            "extra": sorted(found_types - expected_types),
            "clinical_terms_corrupted": corrupted,
            "sanitized_preview": res.sanitized_text[:160],
        })

    precision = round(tot_tp / (tot_tp + tot_fp), 3) if (tot_tp + tot_fp) else 1.0
    recall = round(tot_tp / (tot_tp + tot_fn), 3) if (tot_tp + tot_fn) else 1.0
    f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0
    survival = round(survival_ok / survival_total, 3) if survival_total else 1.0

    agg = {
        "n_cases": len(per_case),
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "clinical_survival_rate": survival,
    }
    return {"per_case": per_case, "aggregate": agg}, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(DATASETS_DIR, "phi_cases.jsonl"))
    args = ap.parse_args()

    report, status = evaluate(args.dataset)
    if report is None:
        return

    print("\n── Guardrail (PHI de-identification) evaluation ─────────")
    for r in report["per_case"]:
        flags = []
        if r["missed"]:
            flags.append(f"MISSED={r['missed']}")
        if r["clinical_terms_corrupted"]:
            flags.append(f"CORRUPTED={r['clinical_terms_corrupted']}")
        status_str = "  ".join(flags) if flags else "ok"
        print(f"  {r['id']:<10} found={r['found_entities']}  {status_str}")
    a = report["aggregate"]
    print("─────────────────────────────────────────────────────────")
    print(f"  entity precision={a['entity_precision']} recall={a['entity_recall']} "
          f"f1={a['entity_f1']}")
    print(f"  clinical-term survival rate={a['clinical_survival_rate']}")

    out = os.path.join(ensure_results_dir(), "guardrail_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
