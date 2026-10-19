import json
import re
from pathlib import Path
from statistics import mean

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.vector_store import get_vector_store, search_code

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"

K = 5
RRF_CONSTANT = 60
WEIGHTS = [1.0, 2.0, 3.0]


def tokenize(text):
    return re.findall(r"[a-z_][a-z0-9_]*|\d+", text.lower())


def doc_key(doc):
    return (
        doc.metadata.get("source", ""),
        doc.metadata.get("start_index", -1),
    )


def weighted_search(query, vector_store, semantic, corpus, bm25, weight):
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    lexical = [
        corpus[i]
        for i in sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )
        if bm25_scores[i] > 0
    ]

    scores = {}
    documents = {}

    for results, result_weight in (
        (semantic, 1.0),
        (lexical, weight),
    ):
        for rank, doc in enumerate(results, start=1):
            key = doc_key(doc)
            documents[key] = doc
            scores[key] = scores.get(key, 0.0) + (result_weight / (RRF_CONSTANT + rank))

    asks_about_tests = any(
        term in query.lower()
        for term in ("test", "tests", "testing", "pytest", "unit test")
    )

    def final_score(key):
        source = documents[key].metadata.get("source", "").lower()
        score = scores[key]

        if not asks_about_tests:
            if source.startswith("src/"):
                score += 0.01
            elif source.startswith("tests/"):
                score -= 0.005

        return score

    ranked = sorted(scores, key=final_score, reverse=True)
    return [documents[key] for key in ranked[:K]]


def measure(question, retrieved_docs):
    expected = set(question["expected_sources"])
    retrieved = [doc.metadata.get("source", "") for doc in retrieved_docs]
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
        "hit": int(bool(found)),
        "recall": len(found) / len(expected) if expected else 0.0,
        "all_required": int(bool(expected) and expected.issubset(set(unique_sources))),
        "mrr": reciprocal_rank,
        "missing": sorted(expected - set(unique_sources)),
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
    vs = get_vector_store()

    stored = vs.get(include=["documents", "metadatas"])
    corpus = [
        Document(page_content=d, metadata=m)
        for d, m in zip(stored["documents"], stored["metadatas"])
    ]
    bm25 = BM25Okapi([tokenize(doc.page_content) for doc in corpus])

    reports = {}

    for weight in WEIGHTS:
        evaluated = []

        for question in questions:
            query = question["question"]
            semantic = vs.similarity_search(query, k=max(K * 4, 20))

            if weight == 1.0:
                # Actual production baseline, including its current fusion logic.
                docs = search_code(query, vs, k=K)
            else:
                docs = weighted_search(query, vs, semantic, corpus, bm25, weight)

            metrics = measure(question, docs)
            evaluated.append({"id": question["id"], **metrics})

        reports[str(weight)] = {
            "summary": summarize(evaluated),
            "results": evaluated,
        }

    print("\n=== Retrieval Comparison ===")
    for weight, report in reports.items():
        s = report["summary"]
        print(f"\nBM25 weight {weight}")
        print(f"Hit Rate@{K}: {s['hit_rate_at_5']:.1%}")
        print(f"Mean Recall@{K}: {s['mean_recall_at_5']:.1%}")
        print(f"All-required-files rate: {s['all_required_files_rate']:.1%}")
        print(f"MRR (unique files): {s['mean_reciprocal_rank']:.3f}")
        print(f"Failed questions: {s['failed_questions']}")

    output = PROJECT_ROOT / "data" / "evaluation" / "weighted_retrieval_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nDetailed report saved to: {output}")


if __name__ == "__main__":
    main()
