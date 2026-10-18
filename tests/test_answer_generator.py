from langchain_core.documents import Document

from src.answer_generator import build_context


def test_context_includes_file_and_line_range():
    document = Document(
        page_content="class HTTPBasicAuth(AuthBase):\n    pass",
        metadata={
            "source": "src/requests/auth.py",
            "start_line": 85,
            "end_line": 113,
            "start_index": 2592,
        },
    )

    context = build_context([document])

    assert "src/requests/auth.py" in context
    assert "lines 85-113" in context
    assert "class HTTPBasicAuth(AuthBase):" in context


def test_context_falls_back_when_line_metadata_missing():
    document = Document(
        page_content="def example():\n    pass",
        metadata={
            "source": "src/example.py",
            "start_index": 100,
        },
    )

    context = build_context([document])

    assert "src/example.py" in context
    assert "character offset 100" in context
    assert "def example():" in context


def test_context_handles_multiple_documents():
    documents = [
        Document(
            page_content="class HTTPBasicAuth:",
            metadata={
                "source": "src/requests/auth.py",
                "start_line": 85,
                "end_line": 113,
            },
        ),
        Document(
            page_content="auth = session.auth",
            metadata={
                "source": "src/requests/sessions.py",
                "start_line": 593,
                "end_line": 607,
            },
        ),
    ]

    context = build_context(documents)

    assert "Source 1:" in context
    assert "Source 2:" in context
    assert "src/requests/auth.py, lines 85-113" in context
    assert "src/requests/sessions.py, lines 593-607" in context
