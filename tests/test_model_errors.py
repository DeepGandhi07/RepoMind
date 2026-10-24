import pytest

import repomind.llm as llm
from repomind.llm import ModelUnavailableError, generate_text


class ModelMissing(Exception):
    status_code = 404


class FakeCompletions:
    def __init__(self, error):
        self.error = error
        self.calls = []

    def create(self, model, messages, temperature):
        self.calls.append(model)
        raise self.error


class FakeGroq:
    def __init__(self, error):
        from types import SimpleNamespace
        self.completions = FakeCompletions(error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_missing_model_gives_actionable_error_without_retries(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")
    client = FakeGroq(ModelMissing("model_not_found: The model does not exist"))

    with pytest.raises(ModelUnavailableError, match="repomind models"):
        generate_text(client, "retired-model", "hi", retries=3, sleep=lambda _s: None)

    assert client.completions.calls == ["retired-model"]
