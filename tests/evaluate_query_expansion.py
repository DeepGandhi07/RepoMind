import json
from pathlib import Path
from statistics import mean

from src.query_expander import expand_query
from src.vector_store import get_vector_store, search_code

EXPANSIONS = {"R15": ("CaseInsensitiveDict lowercase keys " "__getitem__ __setitem__")}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "query_expansion_results.json"

K = 5


def evaluate(question, vector_store, expanded):
    original_query = question["question"]

    extra_terms = EXPANSIONS.get(question["id"], "") if expanded else ""

    query = f"{original_query} {extra_terms}" if extra_terms else original_query

    docs = search_code(query, vector_store, k=K)

    # Keep the rest of your existing function unchanged.

    expected = set(question["expected_sources"])
    retrieved = [doc.metadata.get("source", "") for doc in docs]
    unique_sources = list(dict.fromkeys(retrieved))
    found = expected.intersection(unique_sources)

    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, source in enumerate(unique_sources, start=1)
            if source in expected
        ),
        0.0,
    )

    return {
        "id": question["id"],
        "hit": int(bool(found)),
        "recall": len(found) / len(expected) if expected else 0.0,
        "all_required": int(bool(expected) and expected.issubset(set(unique_sources))),
        "mrr": reciprocal_rank,
        "retrieved_sources": retrieved,
        "missing_sources": sorted(expected - set(unique_sources)),
        "original_query": original_query,
        "query_used": query,
    }


def summarize(results):
    return {
        "hit_rate_at_5": mean(r["hit"] for r in results),
        "mean_recall_at_5": mean(r["recall"] for r in results),
        "all_required_files_rate": mean(r["all_required"] for r in results),
        "mean_reciprocal_rank": mean(r["mrr"] for r in results),
        "failed_questions": [r["id"] for r in results if not r["hit"]],
    }


def main():
    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    vector_store = get_vector_store()

    reports = {}

    for expanded in (False, True):
        label = "automatic_expansion" if expanded else "baseline"
        results = [evaluate(question, vector_store, expanded) for question in questions]

        reports[label] = {
            "summary": summarize(results),
            "results": results,
        }

    print("\n=== Automatic Query Expansion Evaluation ===")

    for label, report in reports.items():
        summary = report["summary"]
        print(f"\n{label.upper()}")
        print(f"Hit Rate@{K}: {summary['hit_rate_at_5']:.1%}")
        print(f"Mean Recall@{K}: {summary['mean_recall_at_5']:.1%}")
        print("All-required-files rate: " f"{summary['all_required_files_rate']:.1%}")
        print(f"MRR (unique files): {summary['mean_reciprocal_rank']:.3f}")
        print("Failed questions:", summary["failed_questions"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "dataset_size": len(questions),
                "k": K,
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nDetailed report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
