import argparse
import json
from pathlib import Path
from statistics import mean

from src.vector_store import get_vector_store, search_code
from src.query_expander import (
    expand_query,
    expand_query_from_cache,
    should_expand_query,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_results.json"
AUTO_OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_results_auto.json"
SELECTIVE_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_results_selective.json"
)

K = 5


def evaluate_question(question, vector_store, query, documents=None):
    expected = set(question["expected_sources"])

    if documents is None:
        documents = search_code(
            query=query,
            vector_store=vector_store,
            k=K,
        )

    retrieved = [doc.metadata.get("source", "") for doc in documents]

    relevant = expected.intersection(retrieved)
    recall = len(relevant) / len(expected) if expected else 0.0

    reciprocal_rank = 0.0
    for rank, source in enumerate(retrieved, start=1):
        if source in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "id": question["id"],
        "question": question["question"],
        "query": query,
        "expected_sources": sorted(expected),
        "retrieved_sources": retrieved,
        "recall_at_5": recall,
        "hit_at_5": int(bool(relevant)),
        "reciprocal_rank": reciprocal_rank,
        "missing_sources": sorted(expected - set(retrieved)),
    }


def summarize(results):
    if not results:
        return {
            "dataset_size": 0,
            "k": K,
            "hit_rate_at_5": 0.0,
            "mean_recall_at_5": 0.0,
            "mean_reciprocal_rank": 0.0,
            "failed_questions": [],
        }

    return {
        "dataset_size": len(results),
        "k": K,
        "hit_rate_at_5": mean(r["hit_at_5"] for r in results),
        "mean_recall_at_5": mean(r["recall_at_5"] for r in results),
        "mean_reciprocal_rank": mean(r["reciprocal_rank"] for r in results),
        "failed_questions": [r["id"] for r in results if not r["hit_at_5"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selective-expand",
        action="store_true",
        help="Evaluate selective expansion using cache only; never call Gemini.",
    )
    parser.add_argument(
        "--auto-expand",
        action="store_true",
        help="Run the optional Gemini expansion experiment.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum questions in the automatic experiment (default: 3).",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    vector_store = get_vector_store()

    if args.selective_expand:
        selective_results = []

        for index, question in enumerate(questions, start=1):
            original_query = question["question"]

            print(
                f"\n[{index}/{len(questions)}] " f"{question['id']}: {original_query}"
            )

            # First, perform ordinary retrieval.
            baseline_documents = search_code(
                query=original_query,
                vector_store=vector_store,
                k=K,
            )

            documents = baseline_documents
            final_query = original_query
            expansion_triggered = should_expand_query(
                original_query,
                baseline_documents,
            )
            expansion_used = False

            # Expand only when overlap is weak, using cache only.
            if expansion_triggered:
                cached_query = expand_query_from_cache(original_query)

                if cached_query != original_query:
                    expanded_documents = search_code(
                        query=cached_query,
                        vector_store=vector_store,
                        k=K,
                    )

                    # Match the production pipeline: use expanded
                    # results only when retrieval returns documents.
                    if expanded_documents:
                        documents = expanded_documents
                        final_query = cached_query
                        expansion_used = True

            result = evaluate_question(
                question=question,
                vector_store=vector_store,
                query=final_query,
                documents=documents,
            )

            result["expansion_triggered"] = expansion_triggered
            result["cached_expansion_used"] = expansion_used

            selective_results.append(result)

            print(
                f"  Triggered: {expansion_triggered}; "
                f"cached expansion used: {expansion_used}; "
                f"Hit@5: {result['hit_at_5']}; "
                f"Recall@5: {result['recall_at_5']:.3f}"
            )

        report = {
            "mode": "selective_cached_expansion",
            "api_calls_for_expansion": 0,
            "questions_evaluated": len(selective_results),
            "summary": summarize(selective_results),
            "results": selective_results,
        }

        SELECTIVE_OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        SELECTIVE_OUTPUT_PATH.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("\n=== Selective Expansion Results (Cache Only) ===")
        print(json.dumps(report["summary"], indent=2))
        print(f"\nReport saved to: {SELECTIVE_OUTPUT_PATH}")
        return

    # Keep the existing baseline/manual experiment unchanged.
    if not args.auto_expand:
        baseline_results = []
        expanded_results = []

        for question in questions:
            original_query = question["question"]
            extra_terms = question.get("expansion_terms", "").strip()

            baseline_results.append(
                evaluate_question(question, vector_store, original_query)
            )

            manual_query = (
                f"{original_query} {extra_terms}".strip()
                if extra_terms
                else original_query
            )
            expanded_results.append(
                evaluate_question(question, vector_store, manual_query)
            )

        report = {
            "baseline": {
                "summary": summarize(baseline_results),
                "results": baseline_results,
            },
            "expanded": {
                "summary": summarize(expanded_results),
                "results": expanded_results,
            },
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("\n=== Baseline Retrieval ===")
        print(json.dumps(report["baseline"]["summary"], indent=2))
        print("\n=== Manual Expansion ===")
        print(json.dumps(report["expanded"]["summary"], indent=2))
        print(f"\nReport saved to: {OUTPUT_PATH}")
        return

    # Explicit opt-in: this is the only mode that calls Gemini.
    selected_questions = questions[: args.limit]
    auto_results = []

    for index, question in enumerate(selected_questions, start=1):
        original_query = question["question"]
        print(
            f"\n[{index}/{len(selected_questions)}] "
            f"{question['id']}: {original_query}"
        )

        # expand_query uses its disk cache when called without a client.
        expanded_query = expand_query(original_query)

        if expanded_query == original_query:
            print("No expansion returned; evaluating original query.")
        else:
            print(f"Expanded query: {expanded_query}")

        auto_results.append(evaluate_question(question, vector_store, expanded_query))

    report = {
        "mode": "automatic_gemini_expansion",
        "questions_evaluated": len(selected_questions),
        "summary": summarize(auto_results),
        "results": auto_results,
    }

    AUTO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Automatic Expansion Results ===")
    print(json.dumps(report["summary"], indent=2))
    print(f"\nSeparate report saved to: {AUTO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
