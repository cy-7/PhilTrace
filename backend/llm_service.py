"""
LLM service integration with local Ollama.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import httpx

from backend.config import settings
from backend.logging_utils import get_logger
from backend.philosophy_prompts import get_philosophy_prompt


logger = get_logger(__name__)

OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_API_TAGS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_API_CHAT = f"{OLLAMA_BASE_URL}/api/chat"

DEFAULT_PHILOSOPHY_PROMPT = get_philosophy_prompt(enable=True)


class OllamaServiceError(RuntimeError):
    """Raised when Ollama integration fails."""


@dataclass
class OllamaModel:
    name: str
    size: int
    digest: str
    modified_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaModel":
        return cls(
            name=data.get("name", ""),
            size=int(data.get("size", 0) or 0),
            digest=data.get("digest", ""),
            modified_at=data.get("modified_at", ""),
        )

    def to_summary_dict(self) -> Dict[str, str]:
        size_mb = self.size / (1024 * 1024)
        return {
            "id": self.name.split(":")[0] if ":" in self.name else self.name,
            "name": self.name,
            "size": f"{size_mb:.1f} MB",
            "modified": self.modified_at[:10] if self.modified_at else "unknown",
        }


@dataclass
class ChatMessage:
    role: str
    content: str


class OllamaService:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = float(settings.api_timeout_seconds)):
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OllamaService":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def list_models(self) -> List[OllamaModel]:
        try:
            response = self.client.get(OLLAMA_API_TAGS)
            response.raise_for_status()
            data = response.json()
            return [OllamaModel.from_dict(m) for m in data.get("models", [])]
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise OllamaServiceError(f"Cannot connect to Ollama at {self.base_url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OllamaServiceError(f"Ollama HTTP error while listing models: {exc}") from exc

    def chat_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        if not model:
            raise OllamaServiceError("Model name cannot be empty")

        ollama_messages: List[Dict[str, str]] = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        ollama_messages.extend({"role": m.role, "content": m.content} for m in messages)

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 2048},
        }

        try:
            with self.client.stream("POST", OLLAMA_API_CHAT, json=payload, timeout=self.timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="ignore")
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping invalid streaming chunk")
                        continue

                    if data.get("done", False):
                        break

                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise OllamaServiceError(f"Chat connection failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OllamaServiceError(f"Chat request failed: {exc}") from exc

    def chat(self, model: str, message: str, system_prompt: Optional[str] = None) -> str:
        content = []
        for chunk in self.chat_stream(model=model, messages=[ChatMessage(role="user", content=message)], system_prompt=system_prompt, stream=True):
            content.append(chunk)
        return "".join(content)


def check_ollama_status() -> Dict[str, Any]:
    with OllamaService() as service:
        try:
            models = service.list_models()
            return {
                "available": True,
                "models": [m.to_summary_dict() for m in models],
                "count": len(models),
            }
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc), "models": [], "count": 0}


def get_philosophy_system_prompt(enable: bool = True) -> Optional[str]:
    if not enable:
        return None
    return DEFAULT_PHILOSOPHY_PROMPT
