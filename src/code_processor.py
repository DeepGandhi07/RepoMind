from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)


def files_to_documents(files: list[dict]) -> list[Document]:
    """Convert GitHub loader results into LangChain Documents."""

    documents = []

    for file in files:
        document = Document(
            page_content=file["content"],
            metadata={
                "source": file["path"],
            },
        )

        documents.append(document)

    return documents


def split_code_documents(
    files: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split source files into searchable chunks with line numbers."""

    documents = files_to_documents(files)

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Keep the original source content so we can calculate line numbers.
    source_contents = {file["path"]: file["content"] for file in files}

    for chunk in chunks:
        source = chunk.metadata["source"]
        source_content = source_contents[source]

        start_index = chunk.metadata["start_index"]
        end_index = min(
            start_index + len(chunk.page_content),
            len(source_content),
        )

        # Character offsets are zero-based; line numbers are one-based.
        start_line = source_content.count("\n", 0, start_index) + 1
        end_line = source_content.count("\n", 0, end_index) + 1

        chunk.metadata["start_line"] = start_line
        chunk.metadata["end_line"] = end_line

    return chunks
