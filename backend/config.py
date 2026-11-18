"""
Backend runtime configuration for PhilTrace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    log_level: str = os.getenv("PHILTRACE_LOG_LEVEL", "INFO").upper()
    bootstrap_engine_on_startup: bool = _env_bool("PHILTRACE_BOOTSTRAP_ENGINE", True)
    enable_cache: bool = _env_bool("PHILTRACE_ENABLE_CACHE", True)
    cache_ttl_seconds: int = _env_int("PHILTRACE_CACHE_TTL_SECONDS", 7 * 24 * 60 * 60)
    api_timeout_seconds: int = _env_int("PHILTRACE_API_TIMEOUT_SECONDS", 120)
    ollama_base_url: str = os.getenv("PHILTRACE_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    host: str = os.getenv("PHILTRACE_HOST", "0.0.0.0")
    port: int = _env_int("PHILTRACE_PORT", 8000)
    reload: bool = _env_bool("PHILTRACE_RELOAD", True)


settings = Settings()
