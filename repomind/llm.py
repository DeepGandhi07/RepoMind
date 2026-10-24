"""Language-model access: provider selection, per-user API keys, retries, and fallbacks.

Each request can carry its own LLMConfig (provider, model, and API key). It is
stored in a context variable, so the answer pipeline picks it up without every
function needing an extra parameter. Without an explicit config, the values come
from the environment.
"""

import os
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from dotenv import load_dotenv

from repomind.config import PROJECT_ROOT, env_bool

load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED_PROVIDERS = ("gemini", "groq")
PROVIDER_NAMES = {"gemini": "Google Gemini", "groq": "Groq"}
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "groq": "llama-3.3-70b-versatile"}
API_KEY_VARIABLES = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}
NON_CHAT_MODEL_MARKERS = ("whisper", "orpheus", "guard", "embed", "tts")
TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
DEFAULT_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.5"))


def _clean(value: str | None) -> str:
    return (value or "").strip().strip('"').strip("'").strip()


def _env_provider() -> str:
    provider = _clean(os.getenv("LLM_PROVIDER")).lower() or "gemini"
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of: {', '.join(SUPPORTED_PROVIDERS)}.")
    return provider


LLM_PROVIDER = _env_provider()
LLM_MODEL = (
    _clean(os.getenv("LLM_MODEL"))
    or _clean(os.getenv("GROQ_MODEL" if LLM_PROVIDER == "groq" else "GEMINI_MODEL"))
    or DEFAULT_MODELS[LLM_PROVIDER]
)


class LLMUnavailableError(RuntimeError):
    """The provider stayed busy, or the daily quota is used up."""


class ModelUnavailableError(RuntimeError):
    """The selected model does not exist or this key cannot use it."""


class _TextResponse:
    def __init__(self, text: str):
        self.text = text


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    fallback_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {self.provider}.")
        if not self.model.strip():
            raise ValueError("A model name is required.")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            fallback_models=tuple(fallback_models_from_env()),
        )


_active_config: ContextVar[LLMConfig | None] = ContextVar("repomind_llm_config", default=None)


def current_config() -> LLMConfig:
    return _active_config.get() or LLMConfig.from_env()


@contextmanager
def using_config(config: LLMConfig | None):
    if config is None:
        yield
        return
    token = _active_config.set(config)
    try:
        yield
    finally:
        _active_config.reset(token)


def server_key_configured() -> bool:
    """True when the server itself has a key. Hosted deployments should disable this."""
    if not env_bool("REPOMIND_ALLOW_SERVER_KEY", True):
        return False
    return bool(_clean(os.getenv(API_KEY_VARIABLES[LLM_PROVIDER])))


def create_client(config: LLMConfig | None = None):
    config = config or current_config()

    if config.provider == "groq":
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError('The Groq SDK is not installed. Run: pip install -e ".[groq]"') from exc
        api_key = config.api_key or _clean(os.getenv(API_KEY_VARIABLES["groq"]))
        if not api_key:
            raise RuntimeError("Add a Groq API key to continue.")
        return Groq(api_key=api_key, max_retries=0)

    from google import genai

    if config.api_key:
        return genai.Client(api_key=config.api_key)
    return genai.Client()


def fallback_models_from_env() -> list[str]:
    raw = os.getenv("LLM_FALLBACK_MODELS") or os.getenv("GEMINI_FALLBACK_MODELS", "")
    return [name.strip() for name in raw.split(",") if name.strip()]


def _status_attribute(exc: Exception) -> int | None:
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _status_code(exc: Exception) -> int | None:
    value = _status_attribute(exc)
    if value is not None:
        return value
    text = str(exc)
    for code in sorted(TRANSIENT_STATUS_CODES):
        if str(code) in text:
            return code
    if "UNAVAILABLE" in text or "RESOURCE_EXHAUSTED" in text:
        return 503
    return None


def _is_model_missing(exc: Exception) -> bool:
    text = str(exc).lower()
    return _status_attribute(exc) == 404 or "model_not_found" in text or "does not exist" in text


def _is_daily_quota(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in ("per day", "tokens per day", "requests per day", "daily", " tpd", "rpd"))


def list_models(config: LLMConfig | None = None) -> list[str]:
    """Return the chat-capable model IDs this key can use."""
    config = config or current_config()
    client = create_client(config)

    if config.provider == "groq":
        response = client.models.list()
        identifiers = [getattr(item, "id", None) for item in (getattr(response, "data", None) or [])]
        return sorted(
            identifier
            for identifier in identifiers
            if identifier and not any(marker in identifier.lower() for marker in NON_CHAT_MODEL_MARKERS)
        )

    return sorted(
        model.name.removeprefix("models/")
        for model in client.models.list()
        if "generateContent" in (getattr(model, "supported_actions", None) or ["generateContent"])
    )


def _call_provider(client, model: str, contents: str):
    if current_config().provider == "groq":
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": contents}],
            temperature=0.2,
            max_tokens=2048,
        )
        return _TextResponse(completion.choices[0].message.content or "")
    return client.models.generate_content(model=model, contents=contents)


def generate_text(
    client,
    model: str,
    contents: str,
    *,
    fallback_models: list[str] | None = None,
    retries: int = DEFAULT_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    sleep=time.sleep,
):
    """Call the active provider with retries. Returns an object with a `.text` attribute."""
    provider_name = PROVIDER_NAMES[current_config().provider]
    candidates = [model] + [name for name in (fallback_models or []) if name != model]
    last_error: Exception | None = None

    for candidate in candidates:
        for attempt in range(retries + 1):
            try:
                return _call_provider(client, candidate, contents)
            except Exception as exc:
                if _is_model_missing(exc):
                    raise ModelUnavailableError(
                        f"The {provider_name} model '{candidate}' is not available for this API key. "
                        "Choose another model, or run `repomind models` to see the options."
                    ) from exc
                if _status_code(exc) not in TRANSIENT_STATUS_CODES:
                    raise
                if _is_daily_quota(exc):
                    raise LLMUnavailableError(
                        f"The {provider_name} daily usage limit is reached for this key. "
                        "Wait for it to reset, or use a different key or provider."
                    ) from exc
                last_error = exc
                if attempt < retries:
                    sleep(base_delay * (2**attempt) + random.uniform(0, 0.5))

    tried = ", ".join(candidates)
    raise LLMUnavailableError(
        f"{provider_name} is busy right now. RepoMind retried {retries + 1} times per model "
        f"({tried}) without success. Please try again in a minute."
    ) from last_error
