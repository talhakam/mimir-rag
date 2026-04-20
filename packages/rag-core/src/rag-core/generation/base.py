# packages/rag-core/src/rag_core/generation/base.py
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        ...

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        ...