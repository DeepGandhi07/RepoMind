import pytest

from repomind.llm import LLMUnavailableError, generate_text


class ServiceUnavailable(Exception):
    """Mimics an API error that exposes an HTTP-style status code."""

    code = 503


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, model, contents):
        self.calls.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def no_sleep(_seconds):
    pass


def test_retries_transient_error_then_succeeds():
    client = FakeClient([ServiceUnavailable("busy"), "ok"])

    result = generate_text(client, "model-a", "hi", retries=2, sleep=no_sleep)

    assert result == "ok"
    assert client.models.calls == ["model-a", "model-a"]


def test_falls_back_to_next_model_after_retries():
    client = FakeClient([ServiceUnavailable("busy"), ServiceUnavailable("busy"), "fallback answer"])

    result = generate_text(
        client,
        "primary",
        "hi",
        fallback_models=["backup"],
        retries=1,
        sleep=no_sleep,
    )

    assert result == "fallback answer"
    assert client.models.calls == ["primary", "primary", "backup"]


def test_raises_friendly_error_when_everything_is_unavailable():
    client = FakeClient([ServiceUnavailable("busy")] * 3)

    with pytest.raises(LLMUnavailableError):
        generate_text(client, "model-a", "hi", retries=2, sleep=no_sleep)


def test_non_transient_errors_are_not_retried():
    client = FakeClient([ValueError("bad request")])

    with pytest.raises(ValueError):
        generate_text(client, "model-a", "hi", sleep=no_sleep)

    assert client.models.calls == ["model-a"]
