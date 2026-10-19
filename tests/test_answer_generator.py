from langchain_core.documents import Document

from src.answer_generator import build_context

from types import SimpleNamespace

import src.answer_generator as answer_generator


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


def test_answer_question_uses_expanded_query_for_retrieval(monkeypatch):
    question = "How does Requests represent case-insensitive HTTP headers?"
    expanded_query = question + " CaseInsensitiveDict lowercase keys"
    retrieved_queries = []

    fake_document = SimpleNamespace(
        page_content="class CaseInsensitiveDict: pass",
        metadata={
            "source": "src/requests/structures.py",
            "start_line": 10,
            "end_line": 20,
            "start_index": 100,
        },
    )

    monkeypatch.setattr(
        answer_generator,
        "USE_QUERY_EXPANSION",
        True,
    )
    monkeypatch.setattr(
        answer_generator,
        "expand_query",
        lambda query: expanded_query,
    )

    def fake_search_code(query, vector_store, k):
        retrieved_queries.append(query)
        return [fake_document]

    class FakeModels:
        def generate_content(self, model, contents):
            return SimpleNamespace(
                text="Headers are case-insensitive [src/requests/structures.py:10-20]."
            )

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setattr(
        answer_generator,
        "search_code",
        fake_search_code,
    )
    monkeypatch.setattr(
        answer_generator.genai,
        "Client",
        FakeClient,
    )
    monkeypatch.setattr(
        answer_generator,
        "check_answer_grounding",
        lambda **kwargs: {"status": "supported"},
    )

    result = answer_generator.answer_question(
        question,
        vector_store=object(),
    )

    assert retrieved_queries == [question, expanded_query]
    assert result["sources"][0]["file"] == "src/requests/structures.py"


def test_answer_question_uses_original_query_when_expansion_disabled(monkeypatch):
    question = "How does Requests represent case-insensitive HTTP headers?"
    retrieved_queries = []

    monkeypatch.setattr(
        answer_generator,
        "USE_QUERY_EXPANSION",
        False,
    )

    def fake_search_code(query, vector_store, k):
        retrieved_queries.append(query)
        return []

    monkeypatch.setattr(
        answer_generator,
        "search_code",
        fake_search_code,
    )

    result = answer_generator.answer_question(
        question,
        vector_store=object(),
    )

    assert retrieved_queries == [question]
    assert result["sources"] == []


def test_answer_generation_receives_original_question(monkeypatch):
    question = "How does Requests represent case-insensitive HTTP headers?"
    expanded_query = question + " CaseInsensitiveDict lowercase keys"
    captured_prompts = []

    fake_document = SimpleNamespace(
        page_content="class CaseInsensitiveDict: pass",
        metadata={
            "source": "src/requests/structures.py",
            "start_line": 10,
            "end_line": 20,
            "start_index": 100,
        },
    )

    monkeypatch.setattr(
        answer_generator,
        "USE_QUERY_EXPANSION",
        True,
    )
    monkeypatch.setattr(
        answer_generator,
        "expand_query",
        lambda query: expanded_query,
    )
    monkeypatch.setattr(
        answer_generator,
        "search_code",
        lambda query, vector_store, k: [fake_document],
    )

    class FakeModels:
        def generate_content(self, model, contents):
            captured_prompts.append(contents)
            return SimpleNamespace(
                text="Headers use a case-insensitive mapping "
                "[src/requests/structures.py:10-20]."
            )

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    monkeypatch.setattr(
        answer_generator.genai,
        "Client",
        FakeClient,
    )
    monkeypatch.setattr(
        answer_generator,
        "check_answer_grounding",
        lambda **kwargs: {"status": "supported"},
    )

    answer_generator.answer_question(question, vector_store=object())

    assert len(captured_prompts) == 1
    assert f"User question:\n{question}" in captured_prompts[0]
    assert expanded_query not in captured_prompts[0]
