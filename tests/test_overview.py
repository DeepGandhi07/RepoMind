from repomind.ingest import SourceFile
from repomind.overview import build_overview, is_overview_question


def test_overview_questions_are_detected():
    assert is_overview_question("What does this project do?")
    assert is_overview_question("Explain the overall architecture.")
    assert not is_overview_question("Where is retry logic implemented?")


def test_overview_includes_readme_and_manifest():
    files = [
        SourceFile(path="README.md", content="# Nexora\nA backend and UI kit.", language="markdown", kind="doc"),
        SourceFile(path="package.json", content='{"name": "nexora"}', language="config", kind="config"),
        SourceFile(path="packages/ui/src/app.tsx", content="export {}", language="ts", kind="code"),
    ]

    overview = build_overview(files)

    assert "A backend and UI kit." in overview
    assert '"name": "nexora"' in overview
    assert "packages" in overview
