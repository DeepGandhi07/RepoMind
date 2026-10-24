import pytest

from repomind.ingest import Limits, RepositoryError, collect_files, normalize_remote, repository_name


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def test_collect_files_indexes_any_language_and_skips_noise(tmp_path):
    _write(tmp_path / "src" / "main.rs", "fn main() {}\n")
    _write(tmp_path / "web" / "App.tsx", "export const App = () => null;\n")
    _write(tmp_path / "README.md", "# Project\nDocs here.\n")
    _write(tmp_path / "node_modules" / "dep" / "index.js", "module.exports = 1;\n")
    _write(tmp_path / ".env", "SECRET=1\n")
    _write(tmp_path / "package-lock.json", "{}\n")
    _write(tmp_path / "logo.png", b"\x89PNG\x00\x00binary")
    _write(tmp_path / "empty.py", "")

    collection = collect_files(tmp_path, Limits())
    paths = {file.path for file in collection.files}

    assert paths == {"src/main.rs", "web/App.tsx", "README.md"}
    assert {file.language for file in collection.files} == {"rust", "ts", "markdown"}
    assert collection.skipped["lockfile or secret"] == 2
    assert collection.skipped["binary"] == 1


def test_collect_files_enforces_file_count_limit(tmp_path):
    for index in range(3):
        _write(tmp_path / f"f{index}.txt", f"content {index}\n")

    with pytest.raises(RepositoryError):
        collect_files(tmp_path, Limits(max_files=2))


def test_collect_files_skips_oversized_files(tmp_path):
    _write(tmp_path / "big.txt", "x" * 1000)
    _write(tmp_path / "small.txt", "ok\n")

    collection = collect_files(tmp_path, Limits(max_file_bytes=100))

    assert [file.path for file in collection.files] == ["small.txt"]
    assert collection.skipped["too large"] == 1


def test_remote_urls_are_validated():
    assert normalize_remote("https://github.com/psf/requests.git") == ("github.com", "psf/requests")
    assert normalize_remote("https://gitlab.com/group/sub/project") == ("gitlab.com", "group/sub/project")

    with pytest.raises(RepositoryError):
        normalize_remote("http://github.com/psf/requests")
    with pytest.raises(RepositoryError):
        normalize_remote("https://user:secret@github.com/psf/requests")


def test_repository_name_for_remote_urls():
    assert repository_name("https://github.com/psf/requests") == "psf/requests"
