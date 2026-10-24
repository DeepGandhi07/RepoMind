"""Repository-level context (README and package manifests) for overview questions."""

import re

from repomind.ingest import SourceFile

README_NAMES = {"readme.md", "readme.rst", "readme.txt", "readme"}
MANIFEST_NAMES = {
    "package.json", "pyproject.toml", "setup.py", "setup.cfg", "cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "composer.json", "gemfile",
}

_OVERVIEW_PATTERN = re.compile(
    r"\b(what (does|is) (this|the) (project|repo|repository|codebase|app)"
    r"|overview|purpose|architecture|about (this|the) (project|repo|repository|codebase)"
    r"|explain (the )?(project|repo|repository|codebase))\b",
    re.IGNORECASE,
)


def is_overview_question(question: str) -> bool:
    return bool(_OVERVIEW_PATTERN.search(question))


def build_overview(files: list[SourceFile], limit: int = 6000) -> str:
    """Summarize the repository's layout, README, and package manifests for the model."""
    top_level = sorted({file.path.split("/")[0] for file in files})[:40]
    parts = [f"Top-level layout: {', '.join(top_level)}"]

    readmes = sorted(
        (file for file in files if file.path.lower().split("/")[-1] in README_NAMES),
        key=lambda file: file.path.count("/"),
    )
    if readmes:
        parts.append(f"README ({readmes[0].path}):\n{readmes[0].content[:3000]}")

    for file in files:
        if "/" not in file.path and file.path.lower() in MANIFEST_NAMES:
            parts.append(f"{file.path}:\n{file.content[:1500]}")

    return "\n\n".join(parts)[:limit]
