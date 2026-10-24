"""Rules that decide which files get indexed and how they are labeled.

These rules are language-agnostic. They skip generated, binary, vendored, and
secret files, and they label each remaining file with a language and a kind
(code, test, doc, or config) so retrieval and citations can use that metadata.
"""

import fnmatch
from pathlib import PurePosixPath

EXTENSION_LANGUAGES = {
    ".py": "python", ".pyi": "python",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cs": "csharp",
    ".scala": "scala",
    ".swift": "swift",
    ".dart": "dart",
    ".pl": "perl", ".pm": "perl",
    ".lua": "lua",
    ".hs": "haskell",
    ".ex": "elixir", ".exs": "elixir",
    ".sol": "sol",
    ".proto": "proto",
    ".m": "objc", ".mm": "objc",
    ".r": "r",
    ".jl": "julia",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".css": "css", ".scss": "css", ".less": "css",
    ".html": "html", ".htm": "html", ".vue": "html", ".svelte": "html",
    ".tex": "latex",
    ".md": "markdown", ".markdown": "markdown",
    ".rst": "rst",
    ".adoc": "text", ".txt": "text",
    ".yaml": "config", ".yml": "config", ".toml": "config", ".json": "config",
    ".ini": "config", ".cfg": "config", ".xml": "config", ".gradle": "config",
    ".tf": "config", ".hcl": "config", ".graphql": "config", ".gql": "config",
}

FILENAME_LANGUAGES = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "cmakelists.txt": "cmake",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "jenkinsfile": "groovy",
}

SPLITTER_LANGUAGES = {
    "python", "js", "ts", "java", "kotlin", "go", "rust", "ruby", "php", "c",
    "cpp", "csharp", "scala", "swift", "markdown", "rst", "html", "latex",
    "perl", "lua", "haskell", "elixir", "sol", "proto",
}

DOC_LANGUAGES = {"markdown", "rst", "text"}
CONFIG_LANGUAGES = {"config"}

EXCLUDED_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "bower_components", "vendor", "third_party", "third-party",
    "venv", ".venv", "__pycache__", ".tox", ".nox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".idea", ".vscode", ".next", ".nuxt", ".gradle", ".dart_tool",
    "dist", "build", "target", "out", "coverage", "htmlcov", "Pods", "Carthage",
    ".cache", ".terraform", ".yarn",
}

LOCKFILE_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "poetry.lock", "pipfile.lock", "uv.lock", "cargo.lock", "go.sum",
    "composer.lock", "gemfile.lock", "mix.lock", "pubspec.lock",
    "packages.lock.json",
}

SENSITIVE_PATTERNS = [
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.keystore", "*.jks",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "*.kdbx", "credentials",
    "credentials.*", "*.tfstate", "*.tfstate.backup", ".netrc", ".npmrc", ".pypirc",
]
SAFE_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template", ".env.dist"}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".jar", ".war", ".class", ".so", ".dll", ".dylib", ".exe", ".o", ".obj",
    ".a", ".lib", ".pyc", ".pyd", ".wasm",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".mp3", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".ogg", ".flac", ".webm",
    ".sqlite", ".db", ".psd", ".ai", ".sketch",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods",
    ".bin", ".dat", ".parquet", ".pkl", ".npy", ".npz", ".h5", ".onnx",
    ".pt", ".safetensors", ".ckpt",
}

_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs"}


def should_skip_name(name: str) -> bool:
    """True for lockfiles and sensitive files, judged by file name alone."""
    lowered = name.lower()
    if lowered in LOCKFILE_NAMES:
        return True
    if lowered in SAFE_ENV_TEMPLATES:
        return False
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in SENSITIVE_PATTERNS)


def language_for(path: str) -> str:
    pure = PurePosixPath(path)
    name = pure.name.lower()
    if name in FILENAME_LANGUAGES:
        return FILENAME_LANGUAGES[name]
    return EXTENSION_LANGUAGES.get(pure.suffix.lower(), "text")


def kind_for(path: str) -> str:
    """Classify a file as code, test, doc, or config."""
    pure = PurePosixPath(path)
    name = pure.name.lower()
    language = language_for(path)

    if language in DOC_LANGUAGES or name in {"readme", "license", "changelog", "contributing"}:
        return "doc"

    parts = {part.lower() for part in pure.parts[:-1]}
    is_test = (
        bool(parts & _TEST_DIRS)
        or name.startswith("test_")
        or ".test." in name
        or ".spec." in name
        or name.endswith(("_test.go", "_test.py", "test.java", "test.kt"))
    )
    if is_test:
        return "test"

    if language in CONFIG_LANGUAGES:
        return "config"
    return "code"


def is_binary_extension(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in BINARY_EXTENSIONS


def is_binary_content(data: bytes) -> bool:
    """Heuristic: a NUL byte in the first 8 KB means the file is not text."""
    return b"\x00" in data[:8192]


def looks_minified(text: str) -> bool:
    """Minified bundles have very long lines and are not useful to index."""
    lines = text.split("\n")
    longest = max((len(line) for line in lines), default=0)
    average = len(text) / max(len(lines), 1)
    return longest > 2000 and average > 300
