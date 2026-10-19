from src.vector_store import get_vector_store, search_code
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import re

QUERY = "How does Requests represent case-insensitive HTTP headers?"
TARGET = "src/requests/structures.py"
K = 5
RRF_CONSTANT = 60


def tokenize(text):
    return re.findall(r"[a-z_][a-z0-9_]*|\d+", text.lower())


def key(doc):
    return (
        doc.metadata.get("source", ""),
        doc.metadata.get("start_index", -1),
    )


vs = get_vector_store()

semantic = vs.similarity_search(QUERY, k=20)

stored = vs.get(include=["documents", "metadatas"])
corpus = [
    Document(page_content=d, metadata=m)
    for d, m in zip(stored["documents"], stored["metadatas"])
]

bm25 = BM25Okapi([tokenize(d.page_content) for d in corpus])
scores = bm25.get_scores(tokenize(QUERY))
bm25_results = [
    corpus[i]
    for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if scores[i] > 0
]


def weighted_rrf(bm25_weight):
    scores = {}
    docs = {}

    for results, weight in (
        (semantic, 1.0),
        (bm25_results, bm25_weight),
    ):
        for rank, doc in enumerate(results, start=1):
            k = key(doc)
            docs[k] = doc
            scores[k] = scores.get(k, 0.0) + weight / (RRF_CONSTANT + rank)

    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)

    print(f"\nBM25 weight: {bm25_weight}")
    for rank, k in enumerate(ranked[:K], start=1):
        marker = " <-- TARGET" if k[0] == TARGET else ""
        print(f"{rank}. {k[0]} (start_index={k[1]}){marker}")


print("Current implementation:")
for rank, doc in enumerate(search_code(QUERY, vs, k=K), start=1):
    marker = " <-- TARGET" if doc.metadata.get("source") == TARGET else ""
    print(f"{rank}. {doc.metadata.get('source')}{marker}")

weighted_rrf(1.0)
weighted_rrf(2.0)
weighted_rrf(3.0)
