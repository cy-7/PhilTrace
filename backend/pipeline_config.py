"""
Pipeline configuration loader for PhilTrace.

Reads config/pipeline.yaml and provides typed access to all pipeline parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline.yaml"


@dataclass
class EncoderConfig:
    model_name: str = "answerdotai/ModernBERT-large"
    fallback_model_name: str = "answerdotai/ModernBERT-base"
    subword_pooling: str = "mean"  # "mean" | "first"
    hidden_layer_index: int = -1
    max_length: int = 8192
    long_text_strategy: str = "truncate"  # "truncate" | "sliding_window"
    sliding_window_size: int = 4096
    sliding_window_stride: int = 2048
    batch_size: int = 16


@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "cosine"
    random_state: int = 42


@dataclass
class HDBSCANConfig:
    min_cluster_size: int = 15
    min_samples: int = 5
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"


@dataclass
class InterpretationConfig:
    provider: str = "ollama"
    ollama_model: str = "qwen2.5:72b"
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    n_samples: int = 8
    sampling_strategy: str = "centroid_nearest"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: int = 2
    enable_cache: bool = True
    prompt_template_path: str = "prompts/sense_interpretation.md"


@dataclass
class CacheConfig:
    include_model_in_key: bool = True
    ttl_seconds: int = 604800


@dataclass
class OutputConfig:
    interpretations_dir: str = "data/interpretations"
    save_feather: bool = True
    save_usages_feather: bool = True


@dataclass
class PipelineConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    interpretation: InterpretationConfig = field(default_factory=InterpretationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "PipelineConfig":
        """Load configuration from YAML file."""
        config_path = path or CONFIG_PATH
        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            encoder=_build_dataclass(EncoderConfig, raw.get("encoder", {})),
            umap=_build_dataclass(UMAPConfig, raw.get("umap", {})),
            hdbscan=_build_dataclass(HDBSCANConfig, raw.get("hdbscan", {})),
            interpretation=_build_dataclass(InterpretationConfig, raw.get("interpretation", {})),
            cache=_build_dataclass(CacheConfig, raw.get("cache", {})),
            output=_build_dataclass(OutputConfig, raw.get("output", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for logging/debugging."""
        import dataclasses
        return dataclasses.asdict(self)


def _build_dataclass(cls, data: Dict[str, Any]):
    """Build a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


# Module-level singleton
_pipeline_config: Optional[PipelineConfig] = None


def get_pipeline_config(force_reload: bool = False) -> PipelineConfig:
    """Get the pipeline configuration singleton."""
    global _pipeline_config
    if _pipeline_config is None or force_reload:
        _pipeline_config = PipelineConfig.from_yaml()
    return _pipeline_config
