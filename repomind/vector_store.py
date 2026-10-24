import hashlib
import re
from collections import OrderedDict, defaultdict
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from repomind.config import CHROMA_DIR

COLLECTION_NAME = "repomind_code"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

_LEXICAL_CACHE: "OrderedDict[str, tuple[list[Document], BM25Okapi | None]]" = OrderedDict()
_LEXICAL_CACHE_SIZE = 4


def tokenize(text: str) -> list[str]:
    """Split source text into lexical tokens that work across programming languages.

    Each identifier is kept whole and also split into parts. `HTTPBasicAuth`
    yields `httpbasicauth`, `http`, `basic`, and `auth`. `case_insensitive`
    yields `case_insensitive`, `case`, and `insensitive`. This is based only on
    identifier shape, so it works for any language.
    """
    tokens: list[str] = []
    for identifier in _IDENTIFIER.findall(text):
        tokens.append(identifier.lower())
        parts: list[str] = []
        for chunk in identifier.split("_"):
            if chunk:
                parts.extend(_CAMEL_PART.findall(chunk) or [chunk])
        if parts != [identifier]:
            tokens.extend(part.lower() for part in parts)
    return tokens


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _shared_embedding_model() -> HuggingFaceEmbeddings:
    return get_embedding_model()


def get_vector_store(
    embedding_model: HuggingFaceEmbeddings | None = None,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Open or create a persistent ChromaDB collection."""

    if embedding_model is None:
        embedding_model = _shared_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )


def delete_collection(collection_name: str) -> None:
    """Remove a collection and its vectors. Missing collections are ignored."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


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

    chunks_by_source: dict[str, list[tuple[Document, str]]] = defaultdict(list)

    for chunk in chunks:
        source = chunk.metadata.get("source")
        if not source:
            raise ValueError("Every chunk must have a source path.")

        start_index = chunk.metadata.get("start_index", 0)
        chunk_id = f"{source}:{start_index}"
        chunks_by_source[source].append((chunk, chunk_id))

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
            if source not in current_sources or chunk_id not in desired_by_id:
                stale_ids.append(chunk_id)
        elif source in chunks_by_source and chunk_id not in desired_by_id:
            stale_ids.append(chunk_id)

    if stale_ids:
        vector_store.delete(ids=stale_ids)

    return len(changed_chunks)


def _lexical_index(corpus: list[Document]) -> BM25Okapi | None:
    """Build BM25 once per distinct corpus. Content hashing keeps the cache correct."""
    digest = hashlib.sha256()
    for document in corpus:
        digest.update(document.page_content.encode("utf-8", errors="replace"))
        digest.update(repr(sorted(document.metadata.items())).encode("utf-8"))
        digest.update(b"\x1e")
    key = digest.hexdigest()

    if key in _LEXICAL_CACHE:
        _LEXICAL_CACHE.move_to_end(key)
        return _LEXICAL_CACHE[key][1]

    tokenized = [tokenize(document.page_content) for document in corpus]
    bm25 = BM25Okapi(tokenized) if any(tokenized) else None

    _LEXICAL_CACHE[key] = (corpus, bm25)
    while len(_LEXICAL_CACHE) > _LEXICAL_CACHE_SIZE:
        _LEXICAL_CACHE.popitem(last=False)
    return bm25


def search_code(
    query: str,
    vector_store: Chroma,
    k: int = 5,
    extra_queries: list[str] | None = None,
) -> list[Document]:
    """Combine semantic and BM25 rankings for one or more queries with Reciprocal Rank Fusion.

    `extra_queries` (for example, an expanded query) contribute their own rankings,
    which are fused with the original query's rankings.
    """

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k < 1:
        raise ValueError("k must be at least 1.")

    queries = [query] + [text for text in (extra_queries or []) if text and text.strip()]

    stored = vector_store.get(include=["documents", "metadatas"])
    corpus_documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(stored["documents"], stored["metadatas"])
    ]
    bm25 = _lexical_index(corpus_documents) if corpus_documents else None

    def document_key(document: Document) -> tuple[str, int, str]:
        return (
            document.metadata.get("source", ""),
            document.metadata.get("start_index", 0),
            document.page_content,
        )

    documents_by_key = {document_key(document): document for document in corpus_documents}

    rrf_scores: dict[tuple[str, int, str], float] = {}
    rrf_constant = 60

    def add_ranking(results: list[Document]) -> None:
        for rank, document in enumerate(results, start=1):
            key = document_key(document)
            documents_by_key.setdefault(key, document)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_constant + rank)

    for text in queries:
        semantic_results = vector_store.similarity_search(text, k=max(k * 4, 20))
        add_ranking(semantic_results)

        if bm25 is not None:
            scores = bm25.get_scores(tokenize(text))
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )
            add_ranking(
                [corpus_documents[index] for index in ranked_indices if scores[index] > 0]
            )

    query_lower = query.lower()
    asks_about_tests = any(
        term in query_lower
        for term in ("test", "tests", "testing", "pytest", "unit test")
    )

    def final_score(key):
        document = documents_by_key[key]
        kind = document.metadata.get("kind")
        source = document.metadata.get("source", "").lower()
        score = rrf_scores[key]

        if not asks_about_tests:
            if kind == "test" or source.startswith("tests/"):
                score -= 0.005
            elif source.startswith("src/"):
                score += 0.01

        return score

    ranked_keys = sorted(
        rrf_scores,
        key=final_score,
        reverse=True,
    )

    return [documents_by_key[key] for key in ranked_keys[:k]]
