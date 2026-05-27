import base64
import asyncio
from app.core.logger import setup_logger
from app.core.config import settings
from google.genai import types
from .base import LLMClient
from google import genai

logger = setup_logger(__name__)


def _extract_text(resp) -> str:
    if resp.text:
        return resp.text

    try:
        parts = resp.candidates[0].content.parts
        texts = [p.text for p in parts if not getattr(p, "thought", False) and hasattr(p, "text") and p.text]
        return "".join(texts)

    except Exception as e:
        logger.error(f"Text extraction failed : {e}")
        return ""


def _build_parts(content) -> list:
    if isinstance(content, str):
        return [types.Part(text=content)]

    parts = []

    for block in content:
        if block["type"] == "text":
            parts.append(types.Part(text=block["text"]))

        elif block["type"] == "file":
            b64 = base64.b64encode(block["data"]).decode()
            parts.append(types.Part(inline_data=types.Blob(data=b64, mime_type=block["mime_type"])))

    return parts


class GeminiLLMClient(LLMClient):
    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in .env")

        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = settings.GEMINI_MODEL

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        json_mode: bool = False
    ) -> str:
        system_instruction = None
        contents: list[types.Content] = []

        for msg in messages:
            role, content = msg["role"], msg["content"]

            if role == "system":
                system_instruction = (content if isinstance(content, str) else str(content))

            elif role == "user":
                contents.append(types.Content(role="user", parts=_build_parts(content)))

            elif role in ("assistant", "model"):
                contents.append(types.Content(role="model", parts=_build_parts(content)))

        cfg: dict = dict(temperature=temperature, max_output_tokens=max_tokens)

        if json_mode:
            cfg["response_mime_type"] = "application/json"

            try:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            except Exception:
                pass

        if system_instruction:
            cfg["system_instruction"] = system_instruction

        loop = asyncio.get_running_loop()

        try:
            resp = await loop.run_in_executor(None, lambda: self._client.models.generate_content(model=self._model, contents=contents, config=types.GenerateContentConfig(**cfg)))

        except Exception as e:
            logger.error(f"Gemini API error : {e}")
            raise

        text = _extract_text(resp)

        return text
