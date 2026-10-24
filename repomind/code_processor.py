"""Split repository files into searchable chunks that keep their line ranges."""

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from repomind.ingest import SourceFile
from repomind.languages import SPLITTER_LANGUAGES

_SPLITTER_ENUM_NAMES = {
    "python": "PYTHON", "js": "JS", "ts": "TS", "java": "JAVA", "kotlin": "KOTLIN",
    "go": "GO", "rust": "RUST", "ruby": "RUBY", "php": "PHP", "c": "C", "cpp": "CPP",
    "csharp": "CSHARP", "scala": "SCALA", "swift": "SWIFT", "markdown": "MARKDOWN",
    "rst": "RST", "html": "HTML", "latex": "LATEX", "perl": "PERL", "lua": "LUA",
    "haskell": "HASKELL", "elixir": "ELIXIR", "sol": "SOL", "proto": "PROTO",
}

_GENERIC_SEPARATORS = ["\n\n", "\n", " ", ""]


def _splitter_for(language: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    enum_name = _SPLITTER_ENUM_NAMES.get(language) if language in SPLITTER_LANGUAGES else None
    member = getattr(Language, enum_name, None) if enum_name else None

    if member is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=member,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    return RecursiveCharacterTextSplitter(
        separators=_GENERIC_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def _add_line_numbers(chunk: Document, source_content: str) -> None:
    start_index = chunk.metadata["start_index"]
    end_index = min(start_index + len(chunk.page_content), len(source_content))
    chunk.metadata["start_line"] = source_content.count("\n", 0, start_index) + 1
    chunk.metadata["end_line"] = source_content.count("\n", 0, end_index) + 1


def split_repository_documents(
    files: list[SourceFile],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split source files into chunks with language, kind, and line metadata."""
    splitters: dict[str, RecursiveCharacterTextSplitter] = {}
    chunks: list[Document] = []

    for file in files:
        if file.language not in splitters:
            splitters[file.language] = _splitter_for(file.language, chunk_size, chunk_overlap)

        document = Document(
            page_content=file.content,
            metadata={"source": file.path, "language": file.language, "kind": file.kind},
        )
        for chunk in splitters[file.language].split_documents([document]):
            if not chunk.page_content.strip():
                continue
            _add_line_numbers(chunk, file.content)
            chunks.append(chunk)

    return chunks
