import pytest

import repomind.registry as registry
from repomind.registry import (
    RepoRecord,
    collection_name,
    find_record,
    list_records,
    remove_record,
    repo_key,
    save_record,
)


@pytest.fixture
def temp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "repositories.json")
    return tmp_path


def _record(source, name, indexed_at):
    key = repo_key(source)
    return RepoRecord(
        key=key,
        name=name,
        source=source,
        collection=collection_name(key, name),
        commit=None,
        branch=None,
        files=1,
        chunks=1,
        indexed_at=indexed_at,
    )


def test_repo_keys_are_stable_and_case_insensitive_for_remotes():
    first = repo_key("https://github.com/Psf/Requests")
    second = repo_key("https://github.com/psf/requests.git")
    assert first == second
    assert first != repo_key("https://github.com/psf/httpx")


def test_collection_names_meet_chroma_rules():
    name = collection_name(repo_key("https://github.com/psf/requests"), "psf/requests with a very long name that goes on and on")
    assert 3 <= len(name) <= 63
    assert name[0].isalnum() and name[-1].isalnum()
    assert all(ch.isalnum() or ch in "-_." for ch in name)


def test_records_can_be_saved_listed_and_found(temp_registry):
    older = _record("https://github.com/psf/requests", "psf/requests", "2026-01-01T00:00:00+00:00")
    newer = _record("https://github.com/pallets/flask", "pallets/flask", "2026-02-01T00:00:00+00:00")
    save_record(older)
    save_record(newer)

    assert [record.name for record in list_records()] == ["pallets/flask", "psf/requests"]
    assert find_record("psf/requests").key == older.key
    assert find_record("https://github.com/pallets/flask").key == newer.key
    assert find_record("nothing-here") is None


def test_records_can_be_removed(temp_registry):
    record = _record("https://github.com/psf/requests", "psf/requests", "2026-01-01T00:00:00+00:00")
    save_record(record)
    remove_record(record.key)
    assert list_records() == []
