from src.vector_store import get_vector_store
from rank_bm25 import BM25Okapi
import re

QUERY = "How does Requests represent case-insensitive HTTP headers?"
TARGET = "src/requests/structures.py"

TOKEN_PATTERN = re.compile(r"[a-z_][a-z0-9_]*|\d+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def show_results(title, results):
    print(f"\n=== {title} ===")

    for rank, item in enumerate(results[:10], start=1):
        if isinstance(item, tuple):
            source, start_index, content = item
        else:
            source = item.metadata.get("source", "")
            start_index = item.metadata.get("start_index", -1)
            content = item.page_content

        marker = " <-- TARGET" if source == TARGET else ""
        print(f"{rank}. {source} (start_index={start_index}){marker}")
        print(content[:180].replace("\n", " "))


vs = get_vector_store()

# Inspect the collection's indexed chunks.
data = vs.get(include=["documents", "metadatas"])
documents = data["documents"]
metadatas = data["metadatas"]

corpus = [
    (m.get("source", ""), m.get("start_index", -1), d)
    for d, m in zip(documents, metadatas)
]

# Stage 1: Semantic search.
semantic = vs.similarity_search(QUERY, k=20)
show_results("SEMANTIC SEARCH", semantic)

# Stage 2: BM25 search over all indexed chunks.
bm25 = BM25Okapi([tokenize(d) for d in documents])
scores = bm25.get_scores(tokenize(QUERY))
bm25_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True,
)
bm25_results = [corpus[i] for i in bm25_indices[:20]]
show_results("BM25 SEARCH", bm25_results)

# Stage 3: Current hybrid search.
from src.vector_store import search_code

hybrid = search_code(QUERY, vs, k=10)
show_results("HYBRID SEARCH", hybrid)
