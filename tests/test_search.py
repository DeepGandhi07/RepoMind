import pytest
from langchain_core.documents import Document

from src.vector_store import search_code


class FakeVectorStore:
    """A small mock that behaves like Chroma similarity search."""

    def __init__(self, documents):
        self.documents = documents
        self.last_k = None

    def similarity_search(self, query, k=4):
        self.last_k = k
        return self.documents[:k]

    def get(self, include=None):
        return {
            "ids": [str(i) for i in range(len(self.documents))],
            "documents": [doc.page_content for doc in self.documents],
            "metadatas": [doc.metadata for doc in self.documents],
        }


@pytest.fixture
def fake_store():
    return FakeVectorStore(
        [
            Document(
                page_content="class HTTPBasicAuth(AuthBase):",
                metadata={"source": "src/requests/auth.py"},
            ),
            Document(
                page_content="def get(url, **kwargs):",
                metadata={"source": "src/requests/api.py"},
            ),
        ]
    )


def test_empty_query_raises_error(fake_store):
    with pytest.raises(ValueError):
        search_code("   ", fake_store)


def test_invalid_k_raises_error(fake_store):
    with pytest.raises(ValueError):
        search_code("HTTP authentication", fake_store, k=0)


def test_returns_requested_number_of_results(fake_store):
    results = search_code("HTTP authentication", fake_store, k=1)
    assert len(results) == 1


def test_authentication_query_prefers_auth_file(fake_store):
    results = search_code("HTTPBasicAuth authentication", fake_store, k=2)
    assert results[0].metadata["source"] == "src/requests/auth.py"
