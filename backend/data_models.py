"""
Data models for PhilTrace backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=100, description="Target concept to analyze")
    periods: Optional[List[str]] = Field(default=None, description="Optional period subset")
    include_examples: bool = Field(default=True, description="Whether to include example quotations")
    umap_params: Optional[Dict[str, Any]] = Field(default=None, description="UMAP overrides")
    hdbscan_params: Optional[Dict[str, Any]] = Field(default=None, description="HDBSCAN overrides")
    force_recompute: bool = Field(default=False, description="Bypass vector/clustering cache")


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    model: str = Field(default="qwen2:7b")
    adapter: Optional[str] = Field(default=None)
    history: Optional[List[ChatMessage]] = Field(default=None)
    stream: bool = Field(default=False)


class ScatterPoint(BaseModel):
    x: float
    y: float
    period: str
    label: int
    text: str
    source: str
    word: Optional[str] = None
    cluster_keywords: Optional[str] = None


class DriftDataPoint(BaseModel):
    period: str
    semantic_stability: float
    concept_usage: float


class ClusterStat(BaseModel):
    cluster_id: int
    count: int
    dominant_period: str
    period_distribution: Dict[str, int]
    top_keywords: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    concept: str
    periods: List[str]
    definitions: Dict[str, str]
    period_years: Dict[str, str]
    scatter_data: List[ScatterPoint]
    drift_data: List[DriftDataPoint]
    cluster_stats: List[ClusterStat]
    quotations: List[Dict[str, str]]
    confidence_score: float
    analysis_timestamp: datetime
    processing_time_ms: int
    model_info: Optional[Dict[str, str]] = None


class ChatResponse(BaseModel):
    response: str
    model: str
    adapter: Optional[str] = None
    token_count: int
    response_time_ms: int


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    model_loaded: bool
    corpus_loaded: bool = False
    extra: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
