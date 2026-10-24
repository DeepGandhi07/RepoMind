from repomind.languages import (
    is_binary_content,
    kind_for,
    language_for,
    looks_minified,
    should_skip_name,
)


def test_languages_are_detected_for_many_ecosystems():
    assert language_for("src/main.rs") == "rust"
    assert language_for("web/App.tsx") == "ts"
    assert language_for("cmd/server.go") == "go"
    assert language_for("Dockerfile") == "dockerfile"
    assert language_for("docs/guide.md") == "markdown"
    assert language_for("unknown.xyz") == "text"


def test_kinds_distinguish_code_tests_docs_and_config():
    assert kind_for("src/lib/parser.py") == "code"
    assert kind_for("tests/test_parser.py") == "test"
    assert kind_for("pkg/parser_test.go") == "test"
    assert kind_for("README.md") == "doc"
    assert kind_for("config/settings.yaml") == "config"


def test_secrets_and_lockfiles_are_skipped():
    assert should_skip_name(".env")
    assert should_skip_name("server.pem")
    assert should_skip_name("id_ed25519")
    assert should_skip_name("package-lock.json")
    assert not should_skip_name(".env.example")
    assert not should_skip_name("main.py")


def test_binary_and_minified_content_is_detected():
    assert is_binary_content(b"\x89PNG\x00\x00")
    assert not is_binary_content(b"fn main() {}")
    assert looks_minified("x" * 5000)
    assert not looks_minified("def f():\n    return 1\n" * 50)
