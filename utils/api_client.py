"""
API client for PhilTrace frontend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 600


@dataclass
class AnalysisResult:
    concept: str
    periods: List[str]
    definitions: Dict[str, str]
    period_years: Dict[str, str]
    scatter_data: pd.DataFrame
    drift_data: pd.DataFrame
    cluster_stats: List[Dict[str, Any]]
    quotations: List[Dict[str, Any]]
    confidence_score: float
    analysis_timestamp: str
    processing_time_ms: int
    model_info: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        return cls(
            concept=data["concept"],
            periods=data["periods"],
            definitions=data["definitions"],
            period_years=data["period_years"],
            scatter_data=pd.DataFrame(data.get("scatter_data", [])),
            drift_data=pd.DataFrame(data.get("drift_data", [])),
            cluster_stats=data.get("cluster_stats", []),
            quotations=data.get("quotations", []),
            confidence_score=float(data.get("confidence_score", 0.0)),
            analysis_timestamp=str(data.get("analysis_timestamp", "")),
            processing_time_ms=int(data.get("processing_time_ms", 0)),
            model_info=data.get("model_info"),
        )


@dataclass
class ChatResult:
    response: str
    model: str
    adapter: Optional[str]
    token_count: int
    response_time_ms: int


class PhilTraceClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "Accept": "application/json"})

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method=method, url=url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except ConnectionError as exc:
            raise ConnectionError(
                f"Cannot connect to backend: {self.base_url}. "
                f"Please run: uvicorn backend.main:app --reload"
            ) from exc
        except Timeout as exc:
            raise Timeout(f"Request timeout ({self.timeout}s): {endpoint}") from exc
        except requests.HTTPError as exc:
            raise requests.HTTPError(f"HTTP error: {exc.response.status_code} - {exc.response.text}") from exc

    def health_check(self) -> Dict[str, Any]:
        return self._request("GET", "/health").json()

    def get_concept_analysis(
        self,
        word: str,
        periods: Optional[List[str]] = None,
        include_examples: bool = True,
        umap_params: Optional[Dict[str, Any]] = None,
        hdbscan_params: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False,
    ) -> AnalysisResult:
        if not word or not word.strip():
            raise ValueError("word cannot be empty")

        payload: Dict[str, Any] = {"word": word.strip(), "force_recompute": force_recompute}
        if periods is not None:
            payload["periods"] = periods
        if include_examples is not None:
            payload["include_examples"] = include_examples
        if umap_params is not None:
            payload["umap_params"] = umap_params
        if hdbscan_params is not None:
            payload["hdbscan_params"] = hdbscan_params

        started = time.time()
        response = self._request("POST", "/analyze", json=payload)
        elapsed = time.time() - started
        print(f"[API Client] analyze('{word}') completed in {elapsed:.2f}s")

        return AnalysisResult.from_dict(response.json())

    def send_chat_message(
        self,
        message: str,
        model: str = "qwen2:7b",
        adapter: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
    ) -> ChatResult:
        if not message or not message.strip():
            raise ValueError("message cannot be empty")

        payload: Dict[str, Any] = {
            "message": message.strip(),
            "model": model,
            "adapter": adapter,
            "stream": stream,
        }
        if history:
            payload["history"] = history

        response = self._request("POST", "/chat/simple", json=payload)
        data = response.json()
        return ChatResult(
            response=data["response"],
            model=data["model"],
            adapter=data.get("adapter"),
            token_count=int(data.get("token_count", 0)),
            response_time_ms=int(data.get("response_time_ms", 0)),
        )

    def chat_streaming(
        self,
        message: str,
        model: str = "qwen2:7b",
        adapter: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        # Fallback implementation: use non-stream endpoint and yield once.
        result = self.send_chat_message(message=message, model=model, adapter=adapter, history=history, stream=False)
        yield result.response

    def get_available_models(self) -> List[Dict[str, Any]]:
        return self._request("GET", "/models").json().get("models", [])

    def get_available_adapters(self) -> List[Dict[str, Any]]:
        response = self._request("GET", "/adapters")
        return response.json().get("adapters", [])

    def get_available_periods(self) -> List[Dict[str, Any]]:
        response = self._request("GET", "/periods")
        return response.json().get("periods", [])

    def get_ollama_models(self) -> List[Dict[str, Any]]:
        try:
            data = self._request("GET", "/models").json()
            if "error" in data:
                return []
            return data.get("models", [])
        except Exception:
            return []


def get_client(base_url: str = DEFAULT_BASE_URL) -> PhilTraceClient:
    return PhilTraceClient(base_url=base_url)


def analyze_concept(word: str, base_url: str = DEFAULT_BASE_URL) -> Optional[AnalysisResult]:
    try:
        return PhilTraceClient(base_url=base_url).get_concept_analysis(word)
    except ConnectionError:
        return None


class MockAPIClient:
    def __init__(self):
        from utils.api_mock import (
            mock_chat_response,
            mock_get_available_loras,
            mock_get_available_models,
            mock_get_concept_analysis,
        )

        self._analyze = mock_get_concept_analysis
        self._chat = mock_chat_response
        self._models = mock_get_available_models
        self._loras = mock_get_available_loras

    def get_concept_analysis(self, word: str, periods: Optional[List[str]] = None, **kwargs) -> AnalysisResult:
        data = self._analyze(word, periods)
        data.setdefault("processing_time_ms", 0)
        data.setdefault("cluster_stats", [])
        return AnalysisResult.from_dict(data)

    def send_chat_message(self, message: str, model: str = "qwen2:7b", adapter: Optional[str] = None, **kwargs) -> ChatResult:
        data = self._chat(message, model, adapter)
        return ChatResult(
            response=data["response"],
            model=data["model"],
            adapter=data.get("adapter"),
            token_count=int(data.get("token_count", 0)),
            response_time_ms=int(float(data.get("response_time", 1.0)) * 1000),
        )


def get_client_with_fallback(base_url: str = DEFAULT_BASE_URL):
    try:
        client = PhilTraceClient(base_url=base_url)
        client.health_check()
        return client
    except ConnectionError:
        print("[API Client] backend unavailable, using mock data")
        return MockAPIClient()
