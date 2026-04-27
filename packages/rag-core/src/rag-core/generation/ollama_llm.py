# packages/rag-core/src/rag_core/generation/ollama_llm.py
from collections.abc import AsyncIterator
import httpx
import json

from rag_core.generation.base import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3.1:8b", temperature: float = 0.1,
                 max_tokens: int = 1024):
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, prompt: str, **kwargs) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self._temperature),
                        "num_predict": kwargs.get("max_tokens", self._max_tokens),
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": kwargs.get("temperature", self._temperature),
                        "num_predict": kwargs.get("max_tokens", self._max_tokens),
                    },
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if not data.get("done", False):
                            yield data.get("response", "")