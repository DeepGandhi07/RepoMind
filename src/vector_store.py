import re
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# All paths are relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "repomind_code"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(
    embedding_model: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Open or create the persistent ChromaDB collection."""

    if embedding_model is None:
        embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )


def index_code_chunks(
    chunks: list[Document],
    vector_store: Chroma,
    current_sources: set[str] | None = None,
) -> int:
    """Synchronize indexed chunks with the current repository snapshot.

    Adds new chunks, updates changed chunks, and deletes obsolete chunks.
    Pass every current source path in current_sources to remove deleted files.
    Returns the number of chunks added or updated.
    """
    from collections import defaultdict

    chunks_by_source: dict[str, list[tuple[Document, str]]] = defaultdict(list)

    for chunk in chunks:
        source = chunk.metadata.get("source")
        if not source:
            raise ValueError("Every chunk must have a source path.")

        start_index = chunk.metadata.get("start_index", 0)
        chunk_id = f"{source}:{start_index}"
        chunks_by_source[source].append((chunk, chunk_id))

    # If supplied, current_sources must represent the complete repository
    # snapshot, not just files that happened to produce chunks.
    if current_sources is not None:
        missing_paths = {
            source for source in chunks_by_source if source not in current_sources
        }
        if missing_paths:
            raise ValueError(
                "Chunks contain source paths absent from current_sources: "
                + ", ".join(sorted(missing_paths))
            )

    existing = vector_store.get(include=["documents", "metadatas"])

    existing_by_id = {
        chunk_id: {
            "document": document,
            "metadata": metadata or {},
        }
        for chunk_id, document, metadata in zip(
            existing["ids"],
            existing["documents"],
            existing["metadatas"],
        )
    }

    desired_by_id = {
        chunk_id: chunk
        for source_chunks in chunks_by_source.values()
        for chunk, chunk_id in source_chunks
    }

    changed_chunks = []
    changed_ids = []

    for chunk_id, chunk in desired_by_id.items():
        old_chunk = existing_by_id.get(chunk_id)

        if old_chunk is None or (
            old_chunk["document"] != chunk.page_content
            or old_chunk["metadata"] != chunk.metadata
        ):
            changed_chunks.append(chunk)
            changed_ids.append(chunk_id)

    if changed_chunks:
        vector_store.add_documents(
            documents=changed_chunks,
            ids=changed_ids,
        )

    stale_ids = []

    for chunk_id, old_chunk in existing_by_id.items():
        source = old_chunk["metadata"].get("source", "")

        if current_sources is not None:
            # Full snapshot: delete anything not present in the desired index.
            if source not in current_sources or chunk_id not in desired_by_id:
                stale_ids.append(chunk_id)
        elif source in chunks_by_source and chunk_id not in desired_by_id:
            # Partial indexing run: only clean up files explicitly processed.
            stale_ids.append(chunk_id)

    if stale_ids:
        vector_store.delete(ids=stale_ids)

    return len(changed_chunks)


def search_code(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> list[Document]:
    """Combine semantic search and BM25 using Reciprocal Rank Fusion."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    def tokenize(text: str) -> list[str]:
        # Preserve identifiers such as HTTPDigestAuth and HTTP_401_ERROR.
        return re.findall(r"[a-z_][a-z0-9_]*|\d+", text.lower())

    # 1. Semantic retrieval from ChromaDB.
    semantic_results = vector_store.similarity_search(
        query,
        k=max(k * 4, 20),
    )

    # 2. Load indexed chunks for lexical retrieval.
    # This is suitable for the current small repository. We can paginate
    # or cache the corpus later as the index grows.
    stored = vector_store.get(include=["documents", "metadatas"])

    corpus_documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(
            stored["documents"],
            stored["metadatas"],
        )
    ]

    # Stable key used to merge the same chunk from both retrieval methods.
    def document_key(document: Document) -> tuple[str, int, str]:
        return (
            document.metadata.get("source", ""),
            document.metadata.get("start_index", 0),
            document.page_content,
        )

    # Keep one document per key.
    documents_by_key = {
        document_key(document): document for document in corpus_documents
    }

    for document in semantic_results:
        documents_by_key.setdefault(document_key(document), document)

    # 3. Build the BM25 index over the complete indexed corpus.
    bm25_results = []

    if corpus_documents:
        tokenized_corpus = [
            tokenize(document.page_content) for document in corpus_documents
        ]

        if any(tokenized_corpus):
            bm25 = BM25Okapi(tokenized_corpus)
            query_tokens = tokenize(query)
            scores = bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )

            bm25_results = [
                corpus_documents[index] for index in ranked_indices if scores[index] > 0
            ]

    # 4. Fuse rankings with Reciprocal Rank Fusion.
    # RRF uses rank positions rather than comparing incompatible raw scores.
    rrf_scores: dict[tuple[str, int, str], float] = {}
    rrf_constant = 60

    for results in (semantic_results, bm25_results):
        for rank, document in enumerate(results, start=1):
            key = document_key(document)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_constant + rank)

    # Prefer implementation code for general "how does it work?"
    # questions, but retain test results when the question asks about tests.
    query_lower = query.lower()
    asks_about_tests = any(
        term in query_lower
        for term in ("test", "tests", "testing", "pytest", "unit test")
    )

    def final_score(key):
        document = documents_by_key[key]
        source = document.metadata.get("source", "").lower()
        score = rrf_scores[key]

        if not asks_about_tests:
            if source.startswith("src/"):
                score += 0.01
            elif source.startswith("tests/"):
                score -= 0.005

        return score

    ranked_keys = sorted(
        rrf_scores,
        key=final_score,
        reverse=True,
    )

    return [documents_by_key[key] for key in ranked_keys[:k]]
