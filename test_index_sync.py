from langchain_core.documents import Document

from src.vector_store import index_code_chunks


class FakeVectorStore:
    def __init__(self):
        self.records = {}

    def get(self, where=None, include=None):
        records = self.records

        if where and "source" in where:
            source = where["source"]
            records = {
                key: value
                for key, value in records.items()
                if value["metadata"].get("source") == source
            }

        return {
            "ids": list(records.keys()),
            "documents": [v["document"] for v in records.values()],
            "metadatas": [v["metadata"] for v in records.values()],
        }

    def add_documents(self, documents, ids):
        for document, chunk_id in zip(documents, ids):
            self.records[chunk_id] = {
                "document": document.page_content,
                "metadata": document.metadata,
            }

    def delete(self, ids):
        for chunk_id in ids:
            self.records.pop(chunk_id, None)


def make_chunk(source, content, start_index=0):
    return Document(
        page_content=content,
        metadata={"source": source, "start_index": start_index},
    )


def test_deleted_file_chunks_are_removed():
    store = FakeVectorStore()

    # Simulate an existing index containing two files.
    index_code_chunks(
        [
            make_chunk("src/a.py", "print('a')"),
            make_chunk("src/deleted.py", "print('old file')"),
        ],
        store,
        current_sources={"src/a.py", "src/deleted.py"},
    )

    # The second file has disappeared from the repository.
    index_code_chunks(
        [make_chunk("src/a.py", "print('a')")],
        store,
        current_sources={"src/a.py"},
    )

    assert "src/a.py:0" in store.records
    assert "src/deleted.py:0" not in store.records


def test_changed_file_removes_obsolete_chunks():
    store = FakeVectorStore()

    index_code_chunks(
        [
            make_chunk("src/a.py", "old chunk", 0),
            make_chunk("src/a.py", "second old chunk", 100),
        ],
        store,
        current_sources={"src/a.py"},
    )

    index_code_chunks(
        [make_chunk("src/a.py", "updated chunk", 0)],
        store,
        current_sources={"src/a.py"},
    )

    assert store.records["src/a.py:0"]["document"] == "updated chunk"
    assert "src/a.py:100" not in store.records
