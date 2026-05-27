from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        json_mode: bool = False
    ) -> str:
        ...

    async def complete_json(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> str:
        return await self.complete(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True)
