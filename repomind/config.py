"""Filesystem locations and shared settings."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data"
    if (PROJECT_ROOT / "pyproject.toml").exists()
    else Path.home() / ".repomind"
)
DATA_DIR = Path(os.getenv("REPOMIND_DATA_DIR", _DEFAULT_DATA_DIR))
CHROMA_DIR = DATA_DIR / "chroma"
REGISTRY_PATH = DATA_DIR / "repositories.json"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
