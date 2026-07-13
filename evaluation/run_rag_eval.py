"""
RAG retrieval evaluation.

Runs each gold case's condition search queries through the real TF-IDF index
(`ICD10RAG.search`) and measures how well the expected ICD-10 codes are retrieved.
No LLM / API key required — this isolates the retrieval quality of the RAG step.

Metrics (per case and aggregate):
- recall@k         : fraction of expected codes present in the top-k candidate pool
- MRR              : mean reciprocal rank of the first expected code found
- precision_pool   : expected ∩ retrieved / retrieved   (pool-level, informational)

Usage:  python run_rag_eval.py [--k 6 10] [--dataset datasets/clinical_cases.jsonl]
"""
import os
import json
import argparse

from common import load_jsonl, get_rag, DATASETS_DIR, ensure_results_dir


def recall_at_k(expected, ranked_codes, k):
    topk = ranked_codes[:k]
    if not expected:
        return 1.0
    hit = sum(1 for c in expected if c in topk)
    return hit / len(expected)


def mrr(expected, ranked_codes):
    exp = set(expected)
    for i, code in enumerate(ranked_codes, start=1):
        if code in exp:
            return 1.0 / i
    return 0.0


def evaluate(dataset_path, ks):
    rag = get_rag()
    cases = load_jsonl(dataset_path)
    per_case = []

    for case in cases:
        queries = case.get("search_queries") or [
            c for c in [case.get("note", "")]
        ]
        # Build a ranked candidate pool preserving best similarity order.
        pool, seen = [], set()
        for q in queries:
            for hit in rag.search(q, n_results=max(ks) if ks else 10):
                code = hit["icd10"]
                if code not in seen:
                    seen.add(code)
                    pool.append((code, hit.get("similarity", 0.0)))
        pool.sort(key=lambda x: x[1], reverse=True)
        ranked = [c for c, _ in pool]

        expected = [c.upper().replace(".", "") for c in case.get("expected_icd10", [])]
        row = {
            "id": case.get("id"),
            "expected": expected,
            "n_retrieved": len(ranked),
            "mrr": round(mrr(expected, ranked), 3),
        }
        for k in ks:
            row[f"recall@{k}"] = round(recall_at_k(expected, ranked, k), 3)
        row["precision_pool"] = round(
            (len(set(expected) & set(ranked)) / len(ranked)) if ranked else 0.0, 3
        )
        row["missing"] = [c for c in expected if c not in ranked]
        per_case.append(row)

    # Aggregate
    agg = {"n_cases": len(per_case), "mrr": _mean(per_case, "mrr")}
    for k in ks:
        agg[f"recall@{k}"] = _mean(per_case, f"recall@{k}")
    return per_case, agg


def _mean(rows, key):
    vals = [r[key] for r in rows if key in r]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(DATASETS_DIR, "clinical_cases.jsonl"))
    ap.add_argument("--k", type=int, nargs="+", default=[6, 10])
    args = ap.parse_args()

    per_case, agg = evaluate(args.dataset, args.k)

    print("\n── RAG retrieval evaluation ─────────────────────────────")
    for r in per_case:
        recalls = " ".join(f"r@{k}={r[f'recall@{k}']}" for k in args.k)
        miss = f"  missing={r['missing']}" if r["missing"] else ""
        print(f"  {r['id']:<32} {recalls}  mrr={r['mrr']}{miss}")
    print("─────────────────────────────────────────────────────────")
    print(f"  AGGREGATE ({agg['n_cases']} cases): "
          + " ".join(f"recall@{k}={agg[f'recall@{k}']}" for k in args.k)
          + f"  mrr={agg['mrr']}")

    out = os.path.join(ensure_results_dir(), "rag_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_case": per_case, "aggregate": agg}, f, indent=2)
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    main()
