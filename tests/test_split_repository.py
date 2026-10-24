from repomind.code_processor import split_repository_documents
from repomind.ingest import SourceFile


def test_chunks_carry_language_kind_and_line_numbers():
    python_source = "\n".join(f"def function_{index}():\n    return {index}" for index in range(120))
    files = [
        SourceFile(path="pkg/module.py", content=python_source, language="python", kind="code"),
        SourceFile(path="notes.txt", content="plain text " * 300, language="text", kind="doc"),
    ]

    chunks = split_repository_documents(files)

    python_chunks = [chunk for chunk in chunks if chunk.metadata["source"] == "pkg/module.py"]
    text_chunks = [chunk for chunk in chunks if chunk.metadata["source"] == "notes.txt"]

    assert python_chunks and text_chunks
    assert all(chunk.metadata["language"] == "python" for chunk in python_chunks)
    assert all(chunk.metadata["kind"] == "doc" for chunk in text_chunks)
    assert all(chunk.metadata["start_line"] >= 1 for chunk in chunks)
    assert all(chunk.metadata["end_line"] >= chunk.metadata["start_line"] for chunk in chunks)
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)
