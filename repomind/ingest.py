"""Turn a repository into indexable source files.

Remote repositories are fetched with a shallow `git clone`, which works for any
host (GitHub, GitLab, Bitbucket, self-hosted) and for private repositories when
a token is provided. Local folders are read in place. Both paths go through the
same filtering, so the results do not depend on where the code came from.
"""

import base64
import os
import shutil
import stat
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from repomind.config import env_int
from repomind.languages import (
    EXCLUDED_DIRS,
    is_binary_content,
    is_binary_extension,
    kind_for,
    language_for,
    looks_minified,
    should_skip_name,
)


class RepositoryError(ValueError):
    """A repository cannot be fetched or indexed. The message is safe to show users."""


@dataclass(frozen=True)
class Limits:
    max_files: int = 5000
    max_file_bytes: int = 300_000
    max_total_bytes: int = 30_000_000
    max_chunks: int = 40_000
    clone_timeout: int = 300

    @classmethod
    def from_env(cls) -> "Limits":
        return cls(
            max_files=env_int("REPOMIND_MAX_FILES", 5000),
            max_file_bytes=env_int("REPOMIND_MAX_FILE_BYTES", 300_000),
            max_total_bytes=env_int("REPOMIND_MAX_TOTAL_BYTES", 30_000_000),
            max_chunks=env_int("REPOMIND_MAX_CHUNKS", 40_000),
            clone_timeout=env_int("REPOMIND_CLONE_TIMEOUT", 300),
        )


@dataclass(frozen=True)
class SourceFile:
    path: str
    content: str
    language: str
    kind: str


@dataclass(frozen=True)
class Collection:
    files: list[SourceFile]
    skipped: dict[str, int]


def is_local_source(source: str) -> bool:
    return os.path.isdir(os.path.expanduser(source.strip()))


def normalize_remote(source: str) -> tuple[str, str]:
    """Validate a remote URL and return (host, path)."""
    parsed = urlparse(source.strip())
    if parsed.scheme != "https":
        raise RepositoryError("Only https:// repository URLs are supported.")
    if parsed.username or parsed.password:
        raise RepositoryError(
            "Do not put credentials in the URL. Set GITHUB_TOKEN or GITLAB_TOKEN instead."
        )
    host = (parsed.hostname or "").lower()
    path = parsed.path.strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not host or not path:
        raise RepositoryError("The URL must include a host and a repository path.")
    return host, path


def repository_name(source: str) -> str:
    """Human-readable name: 'owner/repo' for remotes, the folder name for local paths."""
    if is_local_source(source):
        return Path(os.path.expanduser(source.strip())).resolve().name
    _, path = normalize_remote(source)
    return path


def _default_token(host: str) -> str | None:
    if host == "github.com":
        return os.getenv("GITHUB_TOKEN") or None
    if host == "gitlab.com":
        return os.getenv("GITLAB_TOKEN") or None
    return None


def _auth_environment(host: str, token: str | None) -> dict[str, str]:
    """Pass the token through git config env vars so it never appears on the command line."""
    if not token:
        return {}
    if host == "github.com":
        credentials = f"x-access-token:{token}"
    elif host == "gitlab.com":
        credentials = f"oauth2:{token}"
    else:
        return {}
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
    return {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {encoded}",
    }


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _remove_readonly(function, path, _excinfo):
    os.chmod(path, stat.S_IWRITE)
    function(path)


@contextmanager
def checkout(source: str, limits: Limits, token: str | None = None, on_step=None):
    """Yield (root_path, commit, branch). Remote checkouts are removed on exit."""
    notify = on_step or (lambda message: None)

    if is_local_source(source):
        root = Path(os.path.expanduser(source.strip())).resolve()
        commit = _git_output(["git", "-C", str(root), "rev-parse", "HEAD"])
        yield root, commit, None
        return

    host, path = normalize_remote(source)
    if shutil.which("git") is None:
        raise RepositoryError("Git is required to analyze remote repositories. Install git and try again.")

    clone_url = urlunparse(("https", host, "/" + path, "", "", ""))
    token = token or _default_token(host)
    environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0", **_auth_environment(host, token)}

    workspace = tempfile.mkdtemp(prefix="repomind-")
    try:
        destination = Path(workspace) / "repo"
        notify("Cloning repository (shallow copy)…")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", clone_url, str(destination)],
                capture_output=True,
                text=True,
                timeout=limits.clone_timeout,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RepositoryError(
                f"Cloning took longer than {limits.clone_timeout} seconds. "
                "Raise REPOMIND_CLONE_TIMEOUT to wait longer."
            ) from exc

        if result.returncode != 0:
            lines = (result.stderr or "").strip().splitlines()
            reason = lines[-1] if lines else "unknown error"
            raise RepositoryError(
                f"Could not clone {clone_url}: {reason}. "
                "If the repository is private, set GITHUB_TOKEN or GITLAB_TOKEN."
            )

        commit = _git_output(["git", "-C", str(destination), "rev-parse", "HEAD"])
        branch = _git_output(["git", "-C", str(destination), "rev-parse", "--abbrev-ref", "HEAD"])
        yield destination, commit, branch
    finally:
        shutil.rmtree(workspace, onerror=_remove_readonly)


def collect_files(root: Path, limits: Limits) -> Collection:
    """Walk a checkout and return indexable text files plus skip statistics."""
    root = Path(root)
    files: list[SourceFile] = []
    skipped: Counter[str] = Counter()
    total_bytes = 0

    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in EXCLUDED_DIRS and not os.path.islink(os.path.join(directory, name))
        )

        for name in sorted(filenames):
            full = Path(directory) / name
            relative = full.relative_to(root).as_posix()

            if full.is_symlink():
                skipped["symlink"] += 1
                continue
            if should_skip_name(name):
                skipped["lockfile or secret"] += 1
                continue
            if is_binary_extension(name):
                skipped["binary"] += 1
                continue

            try:
                size = full.stat().st_size
            except OSError:
                skipped["unreadable"] += 1
                continue
            if size == 0:
                continue
            if size > limits.max_file_bytes:
                skipped["too large"] += 1
                continue

            data = full.read_bytes()
            if is_binary_content(data):
                skipped["binary"] += 1
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                skipped["not UTF-8"] += 1
                continue
            if not text.strip():
                continue
            if looks_minified(text):
                skipped["minified"] += 1
                continue

            total_bytes += len(data)
            if total_bytes > limits.max_total_bytes:
                raise RepositoryError(
                    f"The repository holds more than {limits.max_total_bytes // 1_000_000} MB of "
                    "text. Raise REPOMIND_MAX_TOTAL_BYTES to index it anyway."
                )
            if len(files) >= limits.max_files:
                raise RepositoryError(
                    f"The repository has more than {limits.max_files} indexable files. "
                    "Raise REPOMIND_MAX_FILES to index it anyway."
                )

            files.append(
                SourceFile(
                    path=relative,
                    content=text,
                    language=language_for(relative),
                    kind=kind_for(relative),
                )
            )

    return Collection(files=files, skipped=dict(skipped))
