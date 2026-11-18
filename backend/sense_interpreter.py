"""
LLM-based Sense Interpretation Module for PhilTrace.

Provides abstract base class and concrete implementations for generating
philosophical interpretations of semantic clusters using LLMs.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.config import settings
from backend.logging_utils import get_logger
from backend.pipeline_config import InterpretationConfig, get_pipeline_config

logger = get_logger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SenseInterpretation:
    """Result of LLM interpretation for a single semantic cluster."""
    sense_id: int
    target_word: str
    description: str
    keywords: List[str]
    tradition: str
    period: str  # dominant period of the cluster
    n_samples_used: int
    model_used: str
    timestamp: str
    raw_response: Optional[str] = None
    is_fallback: bool = False  # True if LLM failed and we used retrieval fallback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sense_id": self.sense_id,
            "target_word": self.target_word,
            "description": self.description,
            "keywords": self.keywords,
            "tradition": self.tradition,
            "period": self.period,
            "n_samples_used": self.n_samples_used,
            "model_used": self.model_used,
            "timestamp": self.timestamp,
            "is_fallback": self.is_fallback,
        }


# ============================================================================
# Abstract base class
# ============================================================================

class SenseInterpreter(ABC):
    """
    Abstract base class for sense interpretation providers.

    Subclass this to add new LLM backends (Ollama, OpenAI, lab models, etc.).
    """

    def __init__(self, config: Optional[InterpretationConfig] = None):
        self.config = config or get_pipeline_config().interpretation
        self._prompt_template: Optional[str] = None
        self._cache_dir = Path(__file__).parent.parent / "data" / "cache" / "interpretations"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def prompt_template(self) -> str:
        """Load prompt template from file (cached after first load)."""
        if self._prompt_template is None:
            template_path = Path(__file__).parent.parent / self.config.prompt_template_path
            if template_path.exists():
                self._prompt_template = template_path.read_text(encoding="utf-8")
            else:
                logger.warning("Prompt template not found at %s, using default", template_path)
                self._prompt_template = self._default_prompt_template()
        return self._prompt_template

    @staticmethod
    def _default_prompt_template() -> str:
        return (
            '下面是哲学语料中"{target_word}"一词在 {period} 时期的若干使用实例，'
            '这些实例被算法判定为属于同一个语义簇。请仔细阅读后完成：\n\n'
            '1. 用 1-2 句话概括此语义簇代表的"{target_word}"的具体义项。\n'
            '2. 给出 3-5 个能精确刻画此义项的关键词。\n'
            '3. 如此义项可关联到具体哲学传统，请指出；不确定时留空。\n\n'
            '严格 JSON 输出：{"description": "...", "keywords": [...], "tradition": "..."}\n\n'
            '实例如下：\n{samples}'
        )

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Send prompt to LLM and return raw response text."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier for caching and logging."""
        ...

    def interpret_cluster(
        self,
        target_word: str,
        sense_id: int,
        samples: List[Dict[str, Any]],
        dominant_period: str,
    ) -> SenseInterpretation:
        """
        Generate interpretation for a single semantic cluster.

        Args:
            target_word: The concept being analyzed.
            sense_id: Cluster ID.
            samples: List of sample dicts with keys: text, period, source.
            dominant_period: The most common period in this cluster.

        Returns:
            SenseInterpretation with LLM-generated description.
        """
        # Check cache first
        cached = self._load_from_cache(target_word, sense_id, samples)
        if cached is not None:
            logger.info("Cache hit for sense_id=%d of '%s'", sense_id, target_word)
            return cached

        # Build prompt
        prompt = self._build_prompt(target_word, samples, dominant_period)

        # Call LLM with retry
        raw_response = self._call_with_retry(prompt)

        if raw_response is None:
            # Fallback to retrieval-based interpretation
            logger.warning(
                "LLM failed for sense_id=%d of '%s', using retrieval fallback",
                sense_id, target_word
            )
            return self._fallback_interpretation(target_word, sense_id, samples, dominant_period)

        # Parse JSON response
        interpretation = self._parse_response(raw_response, target_word, sense_id, samples, dominant_period)

        # Cache the result
        self._save_to_cache(target_word, sense_id, samples, interpretation)

        return interpretation

    def interpret_all_clusters(
        self,
        target_word: str,
        clustered_df: pd.DataFrame,
    ) -> List[SenseInterpretation]:
        """
        Interpret all non-noise clusters in a clustered DataFrame.

        Args:
            target_word: The concept being analyzed.
            clustered_df: DataFrame with columns [text, period, source, label].

        Returns:
            List of SenseInterpretation for each valid cluster.
        """
        interpretations = []
        labels = sorted(set(int(v) for v in clustered_df["label"].unique() if int(v) != -1))

        for sense_id in labels:
            cluster_df = clustered_df[clustered_df["label"] == sense_id]

            # Determine dominant period
            period_counts = cluster_df["period"].value_counts()
            dominant_period = period_counts.index[0] if not period_counts.empty else "Unknown"

            # Sample representative examples
            samples = self._sample_representatives(cluster_df, target_word)

            interpretation = self.interpret_cluster(
                target_word=target_word,
                sense_id=sense_id,
                samples=samples,
                dominant_period=dominant_period,
            )
            interpretations.append(interpretation)

        return interpretations

    def _sample_representatives(
        self, cluster_df: pd.DataFrame, target_word: str
    ) -> List[Dict[str, Any]]:
        """
        Sample N representative examples from a cluster.

        Strategy: centroid_nearest (default) - compute centroid, rank by cosine similarity.
        """
        n_samples = min(self.config.n_samples, len(cluster_df))

        if self.config.sampling_strategy == "random":
            sampled = cluster_df.sample(n=n_samples, random_state=42)
        elif self.config.sampling_strategy == "diverse":
            # Take samples spread across the cluster
            sampled = cluster_df.iloc[:: max(1, len(cluster_df) // n_samples)][:n_samples]
        else:
            # Default: centroid_nearest
            if "vector" in cluster_df.columns:
                vectors = np.vstack(cluster_df["vector"].values)
                centroid = vectors.mean(axis=0)
                # Cosine similarity to centroid
                norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
                centroid_norm = np.linalg.norm(centroid) + 1e-8
                similarities = (vectors @ centroid) / (norms.squeeze() * centroid_norm)
                top_indices = np.argsort(similarities)[-n_samples:][::-1]
                sampled = cluster_df.iloc[top_indices]
            else:
                sampled = cluster_df.head(n_samples)

        return [
            {
                "text": row["text"],
                "period": row["period"],
                "source": row["source"],
            }
            for _, row in sampled.iterrows()
        ]

    def _build_prompt(
        self, target_word: str, samples: List[Dict[str, Any]], period: str
    ) -> str:
        """Build the prompt from template and samples."""
        samples_text = "\n\n".join(
            f"[{i+1}] ({s['period']}, {s['source']})\n\"{s['text']}\""
            for i, s in enumerate(samples)
        )

        # Use simple string replacement instead of .format() to avoid
        # conflicts with JSON curly braces in the template
        prompt = self.prompt_template
        prompt = prompt.replace("{target_word}", target_word)
        prompt = prompt.replace("{period}", period)
        prompt = prompt.replace("{samples}", samples_text)
        return prompt

    def _call_with_retry(self, prompt: str) -> Optional[str]:
        """Call LLM with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_llm(prompt)
                if response and response.strip():
                    return response
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1, self.config.max_retries, exc
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)

        return None

    def _parse_response(
        self,
        raw_response: str,
        target_word: str,
        sense_id: int,
        samples: List[Dict[str, Any]],
        dominant_period: str,
    ) -> SenseInterpretation:
        """Parse LLM JSON response into SenseInterpretation."""
        # Try to extract JSON from response (LLM might add extra text)
        json_match = re.search(r"\{[^{}]*\}", raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw_response.strip()

        try:
            data = json.loads(json_str)
            description = str(data.get("description", ""))
            keywords = data.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]
            tradition = str(data.get("tradition", ""))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse LLM JSON response: %s", exc)
            # Use raw response as description
            description = raw_response.strip()[:500]
            keywords = []
            tradition = ""

        return SenseInterpretation(
            sense_id=sense_id,
            target_word=target_word,
            description=description,
            keywords=keywords[:5],
            tradition=tradition,
            period=dominant_period,
            n_samples_used=len(samples),
            model_used=self.get_model_name(),
            timestamp=pd.Timestamp.now().isoformat(),
            raw_response=raw_response,
            is_fallback=False,
        )

    def _fallback_interpretation(
        self,
        target_word: str,
        sense_id: int,
        samples: List[Dict[str, Any]],
        dominant_period: str,
    ) -> SenseInterpretation:
        """Retrieval-based fallback when LLM is unavailable."""
        # Use the first sample as a basic description
        if samples:
            description = f"[Fallback] Representative usage: \"{samples[0]['text'][:200]}\""
        else:
            description = "[Fallback] No samples available for interpretation."

        return SenseInterpretation(
            sense_id=sense_id,
            target_word=target_word,
            description=description,
            keywords=[],
            tradition="",
            period=dominant_period,
            n_samples_used=len(samples),
            model_used="fallback_retrieval",
            timestamp=pd.Timestamp.now().isoformat(),
            raw_response=None,
            is_fallback=True,
        )

    # ========================================================================
    # Caching
    # ========================================================================

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Sanitize a string for use in filenames (remove :, /, \\, etc.)."""
        return re.sub(r'[<>:"/\\|?*]', '_', name)

    def _cache_key(self, target_word: str, sense_id: int, samples: List[Dict[str, Any]]) -> str:
        """Generate cache key from sense_id + model + sample hash."""
        sample_hash = hashlib.md5(
            json.dumps(samples, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]  # noqa: S324
        model_name = self._safe_filename(self.get_model_name())
        return f"{target_word}_s{sense_id}_{model_name}_{sample_hash}"

    def _cache_file(self, key: str) -> Path:
        return self._cache_dir / f"{key}.pkl"

    def _load_from_cache(
        self, target_word: str, sense_id: int, samples: List[Dict[str, Any]]
    ) -> Optional[SenseInterpretation]:
        """Load cached interpretation if available."""
        if not self.config.enable_cache:
            return None

        key = self._cache_key(target_word, sense_id, samples)
        cache_file = self._cache_file(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:  # noqa: BLE001
            logger.warning("Corrupted interpretation cache: %s", cache_file)
            cache_file.unlink(missing_ok=True)
            return None

    def _save_to_cache(
        self, target_word: str, sense_id: int, samples: List[Dict[str, Any]], result: SenseInterpretation
    ) -> None:
        """Save interpretation to cache."""
        if not self.config.enable_cache:
            return

        key = self._cache_key(target_word, sense_id, samples)
        cache_file = self._cache_file(key)

        try:
            tmp = cache_file.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(result, f)
            tmp.replace(cache_file)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to cache interpretation: %s", exc)


# ============================================================================
# Ollama Implementation
# ============================================================================

class OllamaInterpreter(SenseInterpreter):
    """Sense interpreter using local Ollama LLM."""

    def __init__(self, config: Optional[InterpretationConfig] = None):
        super().__init__(config)
        self._ollama_url = settings.ollama_base_url

    def get_model_name(self) -> str:
        return self.config.ollama_model

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama chat API."""
        import httpx

        payload = {
            "model": self.config.ollama_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more consistent JSON
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                f"{self._ollama_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")


# ============================================================================
# OpenAI-compatible Implementation
# ============================================================================

class OpenAIInterpreter(SenseInterpreter):
    """Sense interpreter using OpenAI-compatible API (OpenAI, DeepSeek, etc.)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        config: Optional[InterpretationConfig] = None,
    ):
        super().__init__(config)
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    def get_model_name(self) -> str:
        return self._model

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI-compatible chat API."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


# ============================================================================
# DeepSeek Implementation
# ============================================================================

class DeepSeekInterpreter(SenseInterpreter):
    """Sense interpreter using DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[InterpretationConfig] = None,
    ):
        super().__init__(config)
        import os
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._model = model or self.config.deepseek_model
        self._base_url = base_url or self.config.deepseek_base_url

        if not self._api_key:
            logger.warning("DEEPSEEK_API_KEY not set. DeepSeek calls will fail.")

    def get_model_name(self) -> str:
        return f"deepseek/{self._model}"

    def _call_llm(self, prompt: str) -> str:
        """Call DeepSeek chat API."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def chat_stream(self, message: str, system_prompt: Optional[str] = None):
        """Stream chat response from DeepSeek API. Yields text chunks."""
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4096,
            "stream": True,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue


# ============================================================================
# Lab Philosophy LLM Implementation (placeholder for future)
# ============================================================================

class LabPhilLLMInterpreter(SenseInterpreter):
    """
    Placeholder for the lab's custom philosophy LLM.

    To integrate: implement _call_llm with the lab's API endpoint.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:8080/v1/chat",
        model: str = "phil-llm-v1",
        config: Optional[InterpretationConfig] = None,
    ):
        super().__init__(config)
        self._endpoint = endpoint
        self._model = model

    def get_model_name(self) -> str:
        return self._model

    def _call_llm(self, prompt: str) -> str:
        """Call lab LLM API."""
        import httpx

        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
        }

        with httpx.Client(timeout=self.config.timeout) as client:
            response = client.post(self._endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            # Adapt to lab API response format
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            elif "message" in data:
                return data["message"].get("content", "")
            elif "response" in data:
                return data["response"]
            return str(data)


# ============================================================================
# Factory function
# ============================================================================

def create_interpreter(
    provider: Optional[str] = None,
    config: Optional[InterpretationConfig] = None,
    **kwargs,
) -> SenseInterpreter:
    """
    Factory function to create the appropriate interpreter.

    Args:
        provider: Override provider from config. Options: "ollama", "deepseek", "openai", "lab_phil_llm".
        config: Optional config override.
        **kwargs: Additional arguments passed to the interpreter constructor.

    Returns:
        Configured SenseInterpreter instance.
    """
    cfg = config or get_pipeline_config().interpretation
    provider = provider or cfg.provider

    if provider == "ollama":
        return OllamaInterpreter(config=cfg)
    elif provider == "deepseek":
        api_key = kwargs.get("api_key")
        model = kwargs.get("model")
        base_url = kwargs.get("base_url")
        return DeepSeekInterpreter(api_key=api_key, model=model, base_url=base_url, config=cfg)
    elif provider == "openai":
        api_key = kwargs.get("api_key", "")
        model = kwargs.get("model", "gpt-4o")
        base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        return OpenAIInterpreter(api_key=api_key, model=model, base_url=base_url, config=cfg)
    elif provider == "lab_phil_llm":
        endpoint = kwargs.get("endpoint", "http://localhost:8080/v1/chat")
        model = kwargs.get("model", "phil-llm-v1")
        return LabPhilLLMInterpreter(endpoint=endpoint, model=model, config=cfg)
    else:
        logger.warning("Unknown provider '%s', defaulting to Ollama", provider)
        return OllamaInterpreter(config=cfg)


# ============================================================================
# Output persistence
# ============================================================================

def save_interpretations(
    target_word: str,
    interpretations: List[SenseInterpretation],
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save interpretations to JSON and optionally feather format.

    Returns the path to the saved JSON file.
    """
    cfg = get_pipeline_config().output
    output_dir = output_dir or Path(__file__).parent.parent / cfg.interpretations_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    data = {
        "target_word": target_word,
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_senses": len(interpretations),
        "interpretations": [interp.to_dict() for interp in interpretations],
    }

    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", target_word)[:40]
    json_path = output_dir / f"{slug}_interpretations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Feather output
    if cfg.save_feather:
        df = pd.DataFrame([interp.to_dict() for interp in interpretations])
        feather_path = output_dir / f"{slug}_interpretations.feather"
        df.to_feather(feather_path)
        logger.info("Saved interpretations feather: %s", feather_path)

    logger.info("Saved %d interpretations for '%s' to %s", len(interpretations), target_word, json_path)
    return json_path
