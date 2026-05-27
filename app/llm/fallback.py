import fitz
from app.core.logger import setup_logger
from .base import LLMClient

logger = setup_logger(__name__)


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        text = []

        for page in doc:
            text.append(page.get_text())

        return "\n".join(text).strip()

    except Exception as e:
        logger.error(f"PDF extraction failed : {e}")
        return ""


def _convert_file_parts_to_text(messages: list[dict]) -> list[dict]:
    converted = []

    for msg in messages:
        content = msg.get("content")

        if isinstance(content, str):
            converted.append(msg)
            continue

        if isinstance(content, list):
            text_parts = []

            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

                elif part.get("type") == "file":
                    data = part.get("data")
                    mime = part.get("mime_type", "")

                    extracted_text = ""

                    if mime == "application/pdf" and isinstance(data, (bytes, bytearray)):
                        extracted_text = _extract_text_from_pdf_bytes(data)

                    elif mime.startswith("text/") and isinstance(data, (bytes, bytearray)):
                        try:
                            extracted_text = data.decode("utf-8", errors="ignore")

                        except Exception as e:
                            logger.error(f"Text decode failed : {e}")

                    elif mime == "application/json" and isinstance(data, (bytes, bytearray)):
                        try:
                            extracted_text = data.decode("utf-8", errors="ignore")

                        except Exception as e:
                            logger.error(f"JSON decode failed : {e}")

                    if extracted_text:
                        text_parts.append(extracted_text)

                    else:
                        logger.error(f"Could not extract file (mime={mime})")

            content_str = "\n".join(text_parts).strip()

            if not content_str:
                content_str = "[No usable text content extracted from input]"

            MAX_CHARS = 12000

            if len(content_str) > MAX_CHARS:
                logger.warning("Truncating extracted content for token safety")
                content_str = content_str[:MAX_CHARS]

            converted.append(
                {
                    "role": msg.get("role", "user"),
                    "content": content_str
                }
            )

        else:
            converted.append(
                {
                    "role": msg.get("role", "user"),
                    "content": str(content)
                }
            )

    return converted


class FallbackLLMClient(LLMClient):
    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self._primary = primary
        self._fallback = fallback
        primary_name = type(primary).__name__
        fallback_name = type(fallback).__name__
        logger.info(f"Fallback model used : primary={primary_name}, fallback={fallback_name}")

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        json_mode: bool = False
    ) -> str:
        primary_name = type(self._primary).__name__
        fallback_name = type(self._fallback).__name__

        try:
            result = await self._primary.complete(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
            logger.info(f"{primary_name} model ran successfully")
            return result

        except Exception as primary_exc:
            logger.warning(f"{primary_name} model failed ({type(primary_exc).__name__} : {primary_exc}), falling back to {fallback_name} model.")

        safe_messages = _convert_file_parts_to_text(messages)

        try:
            result = await self._fallback.complete(safe_messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
            logger.info(f"{fallback_name} model ran successfully")
            return result

        except Exception as fallback_exc:
            logger.error(f"{fallback_name} model also failed ({type(fallback_exc).__name__} : {fallback_exc})")
            raise
