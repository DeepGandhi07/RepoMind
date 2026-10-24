from types import SimpleNamespace

import pytest
from google import genai as genai_sdk

import repomind.query_expander as query_expander
from repomind.query_expander import expand_query

QUESTION = "How does Requests represent case-insensitive HTTP headers?"


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, response_text=None, error=None):
        self.response_text = response_text
        self.error = error
        self.last_prompt = None
        self.last_model = None

    def generate_content(self, model, contents):
        self.last_model = model
        self.last_prompt = contents

        if self.error:
            raise self.error

        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, response_text=None, error=None):
        self.models = FakeModels(response_text, error)


def test_expands_query_and_preserves_original_question():
    client = FakeClient("CaseInsensitiveDict, lowercase keys, __getitem__")

    result = expand_query(QUESTION, client=client)

    assert QUESTION in result
    assert "CaseInsensitiveDict" in result
    assert "lowercase keys" in result
    assert client.models.last_prompt is not None


def test_returns_original_when_model_returns_none():
    client = FakeClient("NONE")

    result = expand_query(QUESTION, client=client)

    assert result == QUESTION


def test_returns_original_when_model_returns_empty_text():
    client = FakeClient("")

    result = expand_query(QUESTION, client=client)

    assert result == QUESTION


def test_returns_original_when_api_call_fails():
    client = FakeClient(error=RuntimeError("API unavailable"))

    result = expand_query(QUESTION, client=client)

    assert result == QUESTION


def test_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        expand_query("   ", client=FakeClient("NONE"))


def test_strips_markdown_code_fences():
    client = FakeClient("```text\nCaseInsensitiveDict, lowercase keys\n```")

    result = expand_query(QUESTION, client=client)

    assert QUESTION in result
    assert "CaseInsensitiveDict" in result
    assert "```" not in result


def test_removes_bullet_formatting():
    client = FakeClient("- CaseInsensitiveDict\n- lowercase keys")

    result = expand_query(QUESTION, client=client)

    assert QUESTION in result
    assert "CaseInsensitiveDict" in result
    assert "lowercase keys" in result
    assert "- CaseInsensitiveDict" not in result


def test_should_expand_query_skips_strong_lexical_match():
    question = "authentication token header"

    documents = [
        SimpleNamespace(
            page_content=("Authentication token header is used " "for API requests.")
        )
    ]

    assert (
        query_expander.should_expand_query(
            question,
            documents,
        )
        is False
    )


def test_should_expand_query_triggers_on_weak_match():
    question = "authentication token header"

    documents = [
        SimpleNamespace(page_content=("Network timeout retry session connection."))
    ]

    assert (
        query_expander.should_expand_query(
            question,
            documents,
        )
        is True
    )


def test_should_expand_query_triggers_when_no_documents():
    question = "authentication token header"

    assert (
        query_expander.should_expand_query(
            question,
            [],
        )
        is True
    )


def test_persistent_cache_reuses_expansion(
    monkeypatch,
    tmp_path,
):
    cache_path = tmp_path / "query_expansions.json"
    monkeypatch.setattr(
        query_expander,
        "CACHE_PATH",
        cache_path,
    )

    api_calls = []

    class FakeModels:
        def generate_content(self, model, contents):
            api_calls.append(contents)
            return SimpleNamespace(text="CaseInsensitiveDict, lowercase keys")

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    client_instances = []

    def fake_client_factory():
        client = FakeClient()
        client_instances.append(client)
        return client

    monkeypatch.setattr(
        genai_sdk,
        "Client",
        fake_client_factory,
    )

    question = "How are HTTP headers represented?"

    first_result = query_expander.expand_query(question)
    second_result = query_expander.expand_query(question)

    assert first_result == second_result
    assert "CaseInsensitiveDict" in first_result
    assert len(api_calls) == 1
    assert len(client_instances) == 1
    assert cache_path.exists()
