import re
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

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
) -> int:
    """Insert new chunks and update existing chunks when their content changes."""

    if not chunks:
        return 0

    ids = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        start_index = chunk.metadata.get("start_index", 0)
        ids.append(f"{source}:{start_index}")

    existing = vector_store.get(
        ids=ids,
        include=["documents", "metadatas"],
    )

    existing_by_id = {
        chunk_id: {
            "document": document,
            "metadata": metadata,
        }
        for chunk_id, document, metadata in zip(
            existing["ids"],
            existing["documents"],
            existing["metadatas"],
        )
    }

    changed_chunks = []
    changed_ids = []

    for chunk, chunk_id in zip(chunks, ids):
        existing_chunk = existing_by_id.get(chunk_id)

        if existing_chunk is None:
            changed_chunks.append(chunk)
            changed_ids.append(chunk_id)
            continue

        old_document = existing_chunk["document"]
        old_metadata = existing_chunk["metadata"] or {}

        if old_document != chunk.page_content or old_metadata != chunk.metadata:
            changed_chunks.append(chunk)
            changed_ids.append(chunk_id)

    if changed_chunks:
        vector_store.add_documents(
            documents=changed_chunks,
            ids=changed_ids,
        )

    return len(changed_chunks)


def search_code(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> list[Document]:
    """Retrieve semantic candidates and rerank them using code identifiers."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    # Retrieve a wider candidate pool than the final result count.
    candidates = vector_store.similarity_search(query, k=max(k * 4, 20))

    # Extract meaningful identifier-like tokens from the question.
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
    query_terms = {token.lower() for token in tokens if len(token) > 2}

    def score(document: Document) -> int:
        content = document.page_content
        content_lower = content.lower()
        source = document.metadata.get("source", "").lower()

        # Prefer exact identifiers and matching source paths.
        identifier_matches = sum(1 for token in query_terms if token in content_lower)
        path_matches = sum(1 for token in query_terms if token in source)

        return identifier_matches + path_matches * 3

    ranked = sorted(candidates, key=score, reverse=True)

    return ranked[:k]
