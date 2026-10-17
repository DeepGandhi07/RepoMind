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
    """Index chunks without adding duplicate IDs on repeated runs."""

    if not chunks:
        return 0

    # Use source path and character offset to identify each chunk.
    ids = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        start_index = chunk.metadata.get("start_index", 0)
        ids.append(f"{source}:{start_index}")

    # Only insert IDs that are not already present.
    existing = vector_store.get(ids=ids, include=[])

    existing_ids = set(existing["ids"])

    new_documents = []
    new_ids = []

    for chunk, chunk_id in zip(chunks, ids):
        if chunk_id not in existing_ids:
            new_documents.append(chunk)
            new_ids.append(chunk_id)

    if new_documents:
        vector_store.add_documents(
            documents=new_documents,
            ids=new_ids,
        )

    return len(new_documents)


def search_code(
    query: str,
    vector_store: Chroma,
    k: int = 5,
) -> list[Document]:
    """Find the code chunks most relevant to a question."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    return vector_store.similarity_search(query, k=k)
