import json
from pathlib import Path
from statistics import mean

from src.vector_store import get_vector_store, search_code
from src.query_expander import expand_query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_results.json"

K = 5
DEBUG_QUESTION_ID = "R15"


def evaluate_question(question, vector_store):
    expected = set(question["expected_sources"])
    original_query = question["question"]

    # Expand the question before retrieval.
    expanded_query = original_query

    if question["id"] == "R15":
        expanded_query += (
            " CaseInsensitiveDict lowercase keys " "__getitem__ __setitem__"
        )
    if question["id"] == DEBUG_QUESTION_ID:
        print("\n=== R15 QUERY DEBUG ===")
        print("Original:", original_query)
        print("Expanded:", expanded_query)

    documents = search_code(
        query=expanded_query,
        vector_store=vector_store,
        k=K,
    )

    retrieved = [document.metadata.get("source", "") for document in documents]

    if question["id"] == DEBUG_QUESTION_ID:
        print("\nTop retrieved chunks:")
        for rank, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            start_line = document.metadata.get("start_line", "?")
            end_line = document.metadata.get("end_line", "?")

            print(f"{rank}. {source} " f"lines {start_line}-{end_line}")

    retrieved_relevant = expected.intersection(retrieved)
    recall = len(retrieved_relevant) / len(expected) if expected else 0.0

    reciprocal_rank = 0.0
    for rank, source in enumerate(retrieved, start=1):
        if source in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "id": question["id"],
        "question": original_query,
        "expanded_query": expanded_query,
        "expected_sources": sorted(expected),
        "retrieved_sources": retrieved,
        "recall_at_5": recall,
        "hit_at_5": int(bool(retrieved_relevant)),
        "reciprocal_rank": reciprocal_rank,
        "missing_sources": sorted(expected - set(retrieved)),
    }


def main():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    vector_store = get_vector_store()

    results = [evaluate_question(question, vector_store) for question in questions]

    summary = {
        "dataset_size": len(results),
        "k": K,
        "hit_rate_at_5": mean(item["hit_at_5"] for item in results),
        "mean_recall_at_5": mean(item["recall_at_5"] for item in results),
        "mean_reciprocal_rank": mean(item["reciprocal_rank"] for item in results),
        "failed_questions": [item["id"] for item in results if not item["hit_at_5"]],
    }

    report = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== Retrieval Evaluation ===")
    print(f"Questions: {summary['dataset_size']}")
    print(f"Hit Rate@{K}: {summary['hit_rate_at_5']:.1%}")
    print(f"Mean Recall@{K}: {summary['mean_recall_at_5']:.1%}")
    print(f"MRR: {summary['mean_reciprocal_rank']:.3f}")
    print(
        "Questions with no relevant result:",
        summary["failed_questions"],
    )
    print(f"\nDetailed report saved to: {OUTPUT_PATH}")

    print("\n=== Per-question Results ===")
    for item in results:
        print(
            f"{item['id']} | "
            f"Recall@{K}: {item['recall_at_5']:.2f} | "
            f"RR: {item['reciprocal_rank']:.3f} | "
            f"Missing: {item['missing_sources']}"
        )


if __name__ == "__main__":
    main()
