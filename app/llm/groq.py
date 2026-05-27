from app.core.config import settings
from .base import LLMClient
from groq import AsyncGroq


class GroqLLMClient(LLMClient):
    def __init__(self) -> None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set in .env")

        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._model = settings.GROQ_MODEL

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        json_mode: bool = False
    ) -> str:
        kwargs: dict = dict(model=self._model, messages=messages, temperature=temperature, max_tokens=max_tokens)

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)

        return resp.choices[0].message.content or ""
