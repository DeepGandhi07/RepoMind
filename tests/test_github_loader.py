import base64
from unittest.mock import Mock

import pytest
import requests

from src import github_loader
from src.github_loader import fetch_python_files, parse_github_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/psf/requests", ("psf", "requests")),
        ("https://github.com/psf/requests.git", ("psf", "requests")),
        ("https://www.github.com/psf/requests", ("psf", "requests")),
        ("https://github.com/psf/requests/tree/main", ("psf", "requests")),
    ],
)
def test_parse_github_url_valid(url, expected):
    assert parse_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://gitlab.com/psf/requests",
        "ftp://github.com/psf/requests",
        "https://github.com/psf",
        "not-a-url",
    ],
)
def test_parse_github_url_invalid(url):
    with pytest.raises(ValueError):
        parse_github_url(url)


def make_response(data):
    response = Mock()
    response.json.return_value = data
    response.raise_for_status.return_value = None
    return response


def test_fetch_python_files_returns_decoded_python_files(monkeypatch):
    source = "def hello():\n    return 'world'\n"
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")

    responses = [
        make_response({"default_branch": "main"}),
        make_response(
            {
                "truncated": False,
                "tree": [
                    {"type": "blob", "path": "src/hello.py", "size": 100},
                    {"type": "blob", "path": "README.md", "size": 50},
                ],
            }
        ),
        make_response({"encoding": "base64", "content": encoded}),
    ]
    get_mock = Mock(side_effect=responses)
    monkeypatch.setattr(github_loader.requests, "get", get_mock)

    files = fetch_python_files("https://github.com/example/project")

    assert files == [{"path": "src/hello.py", "content": source}]
    assert get_mock.call_count == 3


def test_fetch_python_files_skips_ignored_and_oversized_files(monkeypatch):
    responses = [
        make_response({"default_branch": "main"}),
        make_response(
            {
                "truncated": False,
                "tree": [
                    {"type": "blob", "path": "venv/lib.py", "size": 100},
                    {"type": "blob", "path": "src/large.py", "size": 200_001},
                    {"type": "blob", "path": "src/keep.py", "size": 100},
                ],
            }
        ),
        make_response(
            {
                "encoding": "base64",
                "content": base64.b64encode(b"print('kept')").decode("ascii"),
            }
        ),
    ]
    get_mock = Mock(side_effect=responses)
    monkeypatch.setattr(github_loader.requests, "get", get_mock)

    files = fetch_python_files("https://github.com/example/project")

    assert [file["path"] for file in files] == ["src/keep.py"]
    assert get_mock.call_count == 3


def test_fetch_python_files_skips_non_base64_content(monkeypatch):
    responses = [
        make_response({"default_branch": "main"}),
        make_response(
            {
                "truncated": False,
                "tree": [
                    {"type": "blob", "path": "src/file.py", "size": 10},
                ],
            }
        ),
        make_response({"encoding": "none", "content": ""}),
    ]
    monkeypatch.setattr(github_loader.requests, "get", Mock(side_effect=responses))

    files = fetch_python_files("https://github.com/example/project")

    assert files == []


def test_fetch_python_files_rejects_truncated_tree(monkeypatch):
    responses = [
        make_response({"default_branch": "main"}),
        make_response({"truncated": True, "tree": []}),
    ]
    get_mock = Mock(side_effect=responses)
    monkeypatch.setattr(github_loader.requests, "get", get_mock)

    with pytest.raises(ValueError, match="too large"):
        fetch_python_files("https://github.com/example/project")

    assert get_mock.call_count == 2


def test_fetch_python_files_propagates_api_errors(monkeypatch):
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("Not found")
    monkeypatch.setattr(github_loader.requests, "get", Mock(return_value=response))

    with pytest.raises(requests.HTTPError, match="Not found"):
        fetch_python_files("https://github.com/example/project")
