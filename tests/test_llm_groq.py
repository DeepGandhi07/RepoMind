from types import SimpleNamespace

import pytest

import repomind.llm as llm
from repomind.llm import LLMUnavailableError, generate_text


def no_sleep(_seconds):
    pass


class QuotaExceeded(Exception):
    status_code = 429

    def __init__(self, message):
        super().__init__(message)


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, model, messages, temperature):
        self.calls.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


class FakeGroq:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def groq_provider(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "groq")


def test_groq_responses_expose_text(groq_provider):
    client = FakeGroq(["grounded answer"])

    response = generate_text(client, "llama-test", "question", sleep=no_sleep)

    assert response.text == "grounded answer"
    assert client.completions.calls == ["llama-test"]


def test_groq_transient_errors_are_retried(groq_provider):
    class Busy(Exception):
        status_code = 503

    client = FakeGroq([Busy("busy"), "recovered"])

    response = generate_text(client, "llama-test", "question", retries=2, sleep=no_sleep)

    assert response.text == "recovered"


def test_daily_quota_is_not_retried(groq_provider):
    client = FakeGroq([QuotaExceeded("Rate limit reached: tokens per day (TPD) exceeded")])

    with pytest.raises(LLMUnavailableError, match="daily usage limit"):
        generate_text(client, "llama-test", "question", retries=3, sleep=no_sleep)

    assert client.completions.calls == ["llama-test"]
