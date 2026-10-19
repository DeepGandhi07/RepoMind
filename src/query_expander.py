import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "query_expansions.json"


def _cache_key(question: str, model: str) -> str:
    normalized = " ".join(question.lower().split())
    value = f"{model}:{normalized}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_cache() -> dict:
    try:
        if CACHE_PATH.exists():
            with CACHE_PATH.open("r", encoding="utf-8") as file:
                data = json.load(file)
                return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read query cache: {exc}")

    return {}


def _write_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = CACHE_PATH.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(cache, file, indent=2)

        temporary_path.replace(CACHE_PATH)
    except OSError as exc:
        print(f"Could not save query cache: {exc}")


def expand_query(
    question: str,
    client=None,
    model: str = GEMINI_MODEL,
) -> str:
    """Expand a question, caching successful results on disk."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    use_cache = client is None
    cache = _read_cache() if use_cache else {}
    key = _cache_key(question, model)

    cached_terms = cache.get(key)
    if isinstance(cached_terms, str) and cached_terms.strip():
        return f"{question}\nRelated terminology: {cached_terms}"

    try:
        if client is None:
            client = genai.Client()

        prompt = f"""
You improve source-code retrieval for an AI GitHub repository assistant.

Generate up to 8 concise technical search terms, identifiers,
class names, function names, or implementation concepts relevant
to the user's question.

Rules:
- Preserve the meaning of the question.
- Prefer specific technical terms over generic synonyms.
- Do not answer the question.
- Do not claim that an identifier exists in the repository.
- Return only comma-separated search terms.
- If no useful terms can be inferred, return NONE.

User question:
{question}
"""

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = (response.text or "").strip()
        text = re.sub(r"^\s*```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = re.sub(
            r"^\s*[-*]\s+",
            "",
            text,
            flags=re.MULTILINE,
        )
        terms = re.sub(r"\s+", " ", text.replace("\n", ", "))
        terms = terms[:500].strip(" ,")

        if not terms or terms.upper() == "NONE":
            return question

        if use_cache:
            cache[key] = terms
            _write_cache(cache)

        return f"{question}\nRelated terminology: {terms}"

    except Exception as exc:
        print(f"Query expansion failed; using original query: {exc}")
        return question


def should_expand_query(
    question: str,
    documents: list,
    min_overlap: float = 0.4,
) -> bool:
    """Return True when retrieved chunks have weak lexical overlap."""

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def tokens(text: str) -> set[str]:
        found = re.findall(r"[a-z_][a-z0-9_]*|\d+", text.lower())
        return {token for token in found if token not in stop_words}

    question_tokens = tokens(question)

    if not question_tokens or not documents:
        return True

    best_overlap = 0.0

    for document in documents:
        content = document.page_content
        document_tokens = tokens(content)

        overlap = len(question_tokens & document_tokens) / len(question_tokens)
        best_overlap = max(best_overlap, overlap)

    return best_overlap < min_overlap
