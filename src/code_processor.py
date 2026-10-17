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
    """Convert source files into smaller, searchable code chunks."""

    documents = files_to_documents(files)

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    return chunks
