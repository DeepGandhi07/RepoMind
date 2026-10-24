import json
from types import SimpleNamespace

import pytest

from repomind.grounding_checker import check_answer_grounding


class FakeModels:
    def __init__(self, response_text):
        self.response_text = response_text
        self.last_request = None

    def generate_content(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(text=self.response_text)


class FakeClient:
    def __init__(self, response_text):
        self.models = FakeModels(response_text)


def test_grounding_checker_returns_claim_assessments():
    payload = {
        "overall_status": "supported",
        "claims": [
            {
                "claim": "HTTPBasicAuth sets the Authorization header.",
                "status": "supported",
                "citations": ["src/requests/auth.py:110-113"],
                "evidence": "The method assigns the Authorization header.",
                "reason": "The cited code directly performs the assignment.",
            }
        ],
    }
    client = FakeClient(json.dumps(payload))

    result = check_answer_grounding(
        answer="HTTPBasicAuth sets the Authorization header.",
        context="r.headers['Authorization'] = value",
        client=client,
        model="test-model",
    )

    assert result["overall_status"] == "supported"
    assert result["claims"][0]["status"] == "supported"
    assert client.models.last_request["model"] == "test-model"


def test_grounding_checker_accepts_markdown_json_fences():
    payload = {
        "overall_status": "insufficient_evidence",
        "claims": [],
    }
    client = FakeClient(f"```json\n{json.dumps(payload)}\n```")

    result = check_answer_grounding(
        answer="The answer.",
        context="",
        client=client,
        model="test-model",
    )

    assert result["overall_status"] == "insufficient_evidence"


def test_grounding_checker_rejects_invalid_json():
    client = FakeClient("This is not JSON.")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        check_answer_grounding(
            answer="The answer.",
            context="Some code.",
            client=client,
            model="test-model",
        )


def test_grounding_checker_rejects_invalid_status():
    payload = {
        "overall_status": "probably correct",
        "claims": [],
    }
    client = FakeClient(json.dumps(payload))

    with pytest.raises(RuntimeError, match="invalid overall status"):
        check_answer_grounding(
            answer="The answer.",
            context="Some code.",
            client=client,
            model="test-model",
        )


def test_grounding_checker_accepts_supported_claim():
    payload = {
        "overall_status": "supported",
        "claims": [
            {
                "claim": "The handler sets the Authorization header.",
                "status": "supported",
                "citations": ["src/requests/auth.py:110-112"],
                "evidence": "The code assigns the Authorization header.",
                "reason": "The assignment directly supports the claim.",
            }
        ],
    }
    client = FakeClient(json.dumps(payload))

    result = check_answer_grounding(
        answer="The handler sets the Authorization header.",
        context="r.headers['Authorization'] = value",
        client=client,
        model="test-model",
    )

    assert result["overall_status"] == "supported"
    assert result["claims"][0]["status"] == "supported"


def test_grounding_checker_accepts_unsupported_claim_assessment():
    payload = {
        "overall_status": "unsupported",
        "claims": [
            {
                "claim": "The handler sets the Proxy-Authorization header.",
                "status": "unsupported",
                "citations": ["src/requests/auth.py:110-112"],
                "evidence": "The code sets the Authorization header instead.",
                "reason": "The provided code does not support the claimed header.",
            }
        ],
    }
    client = FakeClient(json.dumps(payload))

    result = check_answer_grounding(
        answer="The handler sets the Proxy-Authorization header.",
        context="r.headers['Authorization'] = value",
        client=client,
        model="test-model",
    )

    assert result["overall_status"] == "unsupported"
    assert result["claims"][0]["status"] == "unsupported"


def test_grounding_checker_accepts_insufficient_evidence_assessment():
    payload = {
        "overall_status": "insufficient_evidence",
        "claims": [
            {
                "claim": "The handler retries authentication after a 401 response.",
                "status": "insufficient_evidence",
                "citations": [],
                "evidence": "The supplied context only shows header assignment.",
                "reason": "The context does not show response handling or retries.",
            }
        ],
    }
    client = FakeClient(json.dumps(payload))

    result = check_answer_grounding(
        answer="The handler retries authentication after a 401 response.",
        context="r.headers['Authorization'] = value",
        client=client,
        model="test-model",
    )

    assert result["overall_status"] == "insufficient_evidence"
    assert result["claims"][0]["status"] == "insufficient_evidence"
