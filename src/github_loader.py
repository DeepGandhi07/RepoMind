from urllib.parse import urlparse
import base64

import requests

import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_API = "https://api.github.com"


def parse_github_url(repo_url: str) -> tuple[str, str]:
    """Extract the owner and repository name from a GitHub URL."""

    parsed = urlparse(repo_url.strip())

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Please provide a valid GitHub HTTPS URL.")

    if parsed.hostname not in {"github.com", "www.github.com"}:
        raise ValueError("Only github.com repository URLs are supported.")

    parts = parsed.path.strip("/").split("/")

    if len(parts) < 2:
        raise ValueError("URL must contain an owner and repository name.")

    owner = parts[0]
    repo = parts[1].removesuffix(".git")

    if not owner or not repo:
        raise ValueError("Invalid GitHub repository URL.")

    return owner, repo


def fetch_python_files(repo_url: str) -> list[dict]:
    """
    Fetch Python source files from a public GitHub repository.
    Returns a list of dictionaries containing file paths and source code.
    """

    token = os.getenv("GITHUB_TOKEN")

    owner, repo = parse_github_url(repo_url)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    repo_endpoint = f"{GITHUB_API}/repos/{owner}/{repo}"
    response = requests.get(
        repo_endpoint,
        headers=headers,
        timeout=20,
    )

    response.raise_for_status()
    repository = response.json()
    default_branch = repository["default_branch"]
    tree_endpoint = f"{GITHUB_API}/repos/{owner}/{repo}" f"/git/trees/{default_branch}"
    response = requests.get(
        tree_endpoint,
        headers=headers,
        params={"recursive": "1"},
        timeout=30,
    )

    response.raise_for_status()
    tree_data = response.json()

    if tree_data.get("truncated"):
        raise ValueError(
            "This repository is too large for the initial loader. "
            "Try a smaller repository."
        )

    files = []
    ignored_directories = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "site-packages",
        "dist",
        "build",
    }

    for item in tree_data.get("tree", []):
        if item.get("type") != "blob":
            continue

        path = item["path"]
        parts = path.split("/")

        if not path.endswith(".py"):
            continue

        if any(part in ignored_directories for part in parts):
            continue

        if item.get("size", 0) > 200_000:
            continue

        content_url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

        content_response = requests.get(
            content_url,
            headers=headers,
            params={"ref": default_branch},
            timeout=20,
        )

        content_response.raise_for_status()
        content_data = content_response.json()

        if content_data.get("encoding") != "base64":
            continue

        content = base64.b64decode(content_data["content"]).decode(
            "utf-8", errors="replace"
        )
        files.append(
            {
                "path": path,
                "content": content,
            }
        )

    return files
