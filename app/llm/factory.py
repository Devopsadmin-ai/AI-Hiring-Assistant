from .fallback import FallbackLLMClient
from app.core.config import settings
from .gemini import GeminiLLMClient
from functools import lru_cache
from .groq import GroqLLMClient
from .base import LLMClient


def _build_primary() -> LLMClient:
    provider = settings.LLM_PROVIDER.lower()

    if provider == "gemini":
        return GeminiLLMClient()

    raise ValueError(f"Unsupported LLM_PROVIDER : {provider}")


def _build_fallback() -> LLMClient | None:
    if not settings.GROQ_API_KEY:
        return None

    if settings.LLM_PROVIDER.lower() == "groq":
        return None

    try:
        return GroqLLMClient()

    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_client() -> LLMClient:
    primary = _build_primary()
    fallback = _build_fallback()

    if fallback is not None:
        return FallbackLLMClient(primary, fallback)

    return primary


def get_llm_client() -> LLMClient:
    return _get_client()
