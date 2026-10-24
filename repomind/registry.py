"""Tracks every indexed repository and the vector collection that holds it."""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from repomind.config import REGISTRY_PATH
from repomind.ingest import is_local_source, normalize_remote


@dataclass
class RepoRecord:
    key: str
    name: str
    source: str
    collection: str
    commit: str | None
    branch: str | None
    files: int
    chunks: int
    skipped: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    indexed_at: str = ""
    overview: str = ""


def repo_key(source: str) -> str:
    """Stable identifier for a repository, derived from its URL or absolute path."""
    if is_local_source(source):
        identity = "local:" + os.path.normcase(str(Path(os.path.expanduser(source.strip())).resolve()))
    else:
        host, path = normalize_remote(source)
        identity = f"remote:{host}/{path.lower()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def collection_name(key: str, name: str) -> str:
    """Chroma collection names must be 3-63 chars, alphanumeric at both ends."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40].strip("-") or "repo"
    return f"rm-{slug}-{key}"


def _read_all() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_all(data: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REGISTRY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(REGISTRY_PATH)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_record(record: RepoRecord) -> None:
    data = _read_all()
    data[record.key] = asdict(record)
    _write_all(data)


def remove_record(key: str) -> None:
    data = _read_all()
    data.pop(key, None)
    _write_all(data)


def list_records() -> list[RepoRecord]:
    records = [RepoRecord(**value) for value in _read_all().values()]
    return sorted(records, key=lambda record: record.indexed_at, reverse=True)


def find_record(reference: str) -> RepoRecord | None:
    """Find a repository by key, by 'owner/repo' name, or by its source URL or path."""
    wanted = reference.strip()
    if not wanted:
        return None
    records = list_records()
    for record in records:
        if record.key == wanted or record.name.lower() == wanted.lower() or record.source == wanted:
            return record
    try:
        key = repo_key(wanted)
    except ValueError:
        return None
    return next((record for record in records if record.key == key), None)
