"""
Analysis engine for concept drift analysis.

Orchestrates the full pipeline: corpus extraction → embedding → UMAP → HDBSCAN → results.
Now uses ModernBERTEncoder for token-level contextual embeddings.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import settings
from backend.logging_utils import get_logger
from backend.modernbert_encoder import ModernBERTEncoder
from backend.pipeline_config import get_pipeline_config

logger = get_logger(__name__)

try:
    import umap
except Exception:  # noqa: BLE001
    umap = None

try:
    import hdbscan
except Exception:  # noqa: BLE001
    hdbscan = None


CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = ["Ancient", "Medieval", "Modern", "Contemporary"]
PERIOD_YEARS = {
    "Ancient": "800 BCE - 500 CE",
    "Medieval": "500 CE - 1500 CE",
    "Modern": "1500 CE - 1900 CE",
    "Contemporary": "1900 CE - Present",
}


@dataclass
class Sentence:
    text: str
    period: str
    source: str


@dataclass
class ClusterInfo:
    cluster_id: int
    count: int
    dominant_period: str
    period_distribution: Dict[str, int]
    top_keywords: List[str]


class CTFIDFExtractor:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self._idf: Dict[str, float] = {}
        self._cluster_tfs: Dict[int, Dict[str, int]] = {}

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def fit(self, clusters: Dict[int, List[str]]) -> "CTFIDFExtractor":
        if not clusters:
            return self

        cluster_docs = {cid: self._clean_text(" ".join(texts)) for cid, texts in clusters.items()}

        self._cluster_tfs = {}
        for cid, doc in cluster_docs.items():
            tf: Dict[str, int] = {}
            for word in doc.split():
                if len(word) > 2:
                    tf[word] = tf.get(word, 0) + 1
            self._cluster_tfs[cid] = tf

        n_clusters = len(cluster_docs)
        doc_freqs: Dict[str, int] = {}
        for tf in self._cluster_tfs.values():
            for word in tf.keys():
                doc_freqs[word] = doc_freqs.get(word, 0) + 1

        self._idf = {word: np.log(n_clusters / (df + 1)) + 1 for word, df in doc_freqs.items()}
        return self

    def get_top_keywords(self, cluster_id: int) -> List[str]:
        if cluster_id == -1 or cluster_id not in self._cluster_tfs:
            return []

        stopwords = {
            "the", "and", "for", "that", "with", "this", "are", "was", "have", "from",
            "they", "will", "would", "could", "should", "been", "being", "were", "has",
            "had", "not", "but", "which", "also", "into", "more", "some", "than", "then",
            "what", "when", "where", "who", "how", "why", "all", "each", "other", "only",
            "own", "same", "can", "our", "its", "his", "her", "their", "may", "any", "such",
            "these", "those",
        }

        tf = self._cluster_tfs[cluster_id]
        scores = {word: freq * self._idf.get(word, 0.0) for word, freq in tf.items()}
        sorted_words = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in sorted_words if word not in stopwords][: self.top_k]


def _safe_word_for_cache(word: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", word.strip())[:40] or "concept"
    digest = hashlib.md5(word.encode("utf-8")).hexdigest()[:8]  # noqa: S324
    return f"{slug}_{digest}"


def _cache_path(word: str, suffix: str, model_name: Optional[str] = None) -> Path:
    """Generate cache path, optionally including model name for isolation."""
    pipeline_cfg = get_pipeline_config()
    if pipeline_cfg.cache.include_model_in_key and model_name:
        model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:30]
        return CACHE_DIR / f"{_safe_word_for_cache(word)}_{model_slug}.{suffix}"
    return CACHE_DIR / f"{_safe_word_for_cache(word)}.{suffix}"


def _atomic_pickle_dump(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


def _is_cache_fresh(path: Path) -> bool:
    ttl = max(0, int(settings.cache_ttl_seconds))
    if ttl == 0:
        return True
    age = time.time() - path.stat().st_mtime
    return age <= ttl


def save_vectors_cache(word: str, df: pd.DataFrame, model_name: Optional[str] = None) -> None:
    if not settings.enable_cache:
        return
    path = _cache_path(word, "vec.pkl", model_name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_pickle(tmp)
    tmp.replace(path)


def load_vectors_cache(word: str, model_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not settings.enable_cache:
        return None

    path = _cache_path(word, "vec.pkl", model_name)
    if not path.exists() or not _is_cache_fresh(path):
        return None

    try:
        result = pd.read_pickle(path)
    except Exception:  # noqa: BLE001
        logger.warning("Corrupted vector cache removed: %s", path)
        path.unlink(missing_ok=True)
        return None

    if isinstance(result, pd.DataFrame):
        return result
    return None


def save_clustering_cache(word: str, params: Dict[str, Any], result: Dict[str, Any], model_name: Optional[str] = None) -> None:
    if not settings.enable_cache:
        return

    path = _cache_path(word, "res.pkl", model_name)
    payload = {"params": params, "result": result, "timestamp": time.time()}
    _atomic_pickle_dump(path, payload)


def load_clustering_cache(word: str, params: Dict[str, Any], model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not settings.enable_cache:
        return None

    path = _cache_path(word, "res.pkl", model_name)
    if not path.exists() or not _is_cache_fresh(path):
        return None

    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception:  # noqa: BLE001
        logger.warning("Corrupted clustering cache removed: %s", path)
        path.unlink(missing_ok=True)
        return None

    if payload.get("params") == params:
        return payload.get("result")
    return None


def save_usages_feather(word: str, df: pd.DataFrame, model_name: Optional[str] = None) -> Optional[Path]:
    """Save usages DataFrame to feather format for persistence."""
    pipeline_cfg = get_pipeline_config()
    if not pipeline_cfg.output.save_usages_feather:
        return None

    output_dir = Path(__file__).parent.parent / "data" / "usages"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feather doesn't support object arrays (numpy vectors), so store as list
    save_df = df.copy()
    save_df["vector"] = save_df["vector"].apply(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)

    model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name or "unknown")[:30]
    path = output_dir / f"{_safe_word_for_cache(word)}_{model_slug}.feather"
    save_df.to_feather(path)
    logger.info("Saved usages feather: %s (%d rows)", path.name, len(df))
    return path


class AnalysisEngine:
    def __init__(self, use_cache: bool = True):
        self.pipeline_config = get_pipeline_config()
        self.use_cache = use_cache and settings.enable_cache
        self.corpus_files: Dict[str, List[Path]] = {}

        # Initialize the ModernBERT encoder
        self.encoder = ModernBERTEncoder(config=self.pipeline_config.encoder)
        self.hidden_size = self.encoder.hidden_size
        self.model_name = self.encoder.model_name_loaded or "fallback"

        self.load_corpus()

    @property
    def model(self):
        """Backward compatibility: check if encoder model is loaded."""
        return self.encoder.model

    def load_corpus(self, corpus_root: Optional[Path] = None) -> None:
        corpus_root = corpus_root or Path(__file__).parent.parent / "corpus"

        self.corpus_files = {period: [] for period in PERIODS}
        if not corpus_root.exists():
            logger.warning("Corpus directory not found: %s", corpus_root)
            return

        for period in PERIODS:
            period_dir = corpus_root / period.lower()
            if not period_dir.exists():
                period_dir = corpus_root / period
            if period_dir.exists():
                self.corpus_files[period] = list(period_dir.glob("*.txt"))

        total_files = sum(len(items) for items in self.corpus_files.values())
        logger.info("Loaded corpus index with %d files", total_files)

    def get_vectors(self, target_word: str, force_recompute: bool = False) -> pd.DataFrame:
        if self.use_cache and not force_recompute:
            cached = load_vectors_cache(target_word, self.model_name)
            if cached is not None:
                logger.info("Loaded vectors from cache for '%s'", target_word)
                return cached

        sentences = self._extract_sentences(target_word)
        if not sentences:
            return pd.DataFrame(columns=["text", "period", "source", "vector"])

        # Batch encode for efficiency
        texts = [s.text for s in sentences]
        vectors = self.encoder.encode_batch(texts, target_word)

        rows: List[Dict[str, Any]] = []
        for sentence, vector in zip(sentences, vectors):
            rows.append(
                {
                    "text": sentence.text,
                    "period": sentence.period,
                    "source": sentence.source,
                    "vector": vector,
                }
            )

        df = pd.DataFrame(rows)
        if self.use_cache and not df.empty:
            save_vectors_cache(target_word, df, self.model_name)

        # Also save as feather for persistence
        if not df.empty:
            save_usages_feather(target_word, df, self.model_name)

        return df

    def _extract_sentences(self, target_word: str) -> List[Sentence]:
        pattern = re.compile(r"\b" + re.escape(target_word) + r"\b", re.IGNORECASE)
        collected: List[Sentence] = []

        for period in PERIODS:
            for file_path in self.corpus_files.get(period, []):
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:  # noqa: BLE001
                    logger.warning("Cannot read corpus file: %s", file_path)
                    continue

                chunks = re.split(r"[。！？.!?\n]+", text)
                for chunk in chunks:
                    sentence = chunk.strip()
                    if len(sentence) < 10:
                        continue
                    if pattern.search(sentence):
                        collected.append(Sentence(text=sentence, period=period, source=file_path.stem))

        return collected

    def cluster(self, df: pd.DataFrame, umap_params: Optional[Dict[str, Any]] = None, hdbscan_params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        result_df = df.copy()
        vectors = np.vstack(result_df["vector"].values)
        n_samples = len(result_df)

        if n_samples < 3:
            result_df["x"] = np.arange(n_samples, dtype=float)
            result_df["y"] = np.zeros(n_samples, dtype=float)
            result_df["label"] = np.zeros(n_samples, dtype=int)
            return result_df

        # Get default params from pipeline config
        cfg_umap = self.pipeline_config.umap
        cfg_hdbscan = self.pipeline_config.hdbscan

        default_umap = {
            "n_neighbors": cfg_umap.n_neighbors,
            "min_dist": cfg_umap.min_dist,
            "n_components": cfg_umap.n_components,
            "metric": cfg_umap.metric,
        }
        default_hdbscan = {
            "min_cluster_size": cfg_hdbscan.min_cluster_size,
            "min_samples": cfg_hdbscan.min_samples,
            "metric": cfg_hdbscan.metric,
            "cluster_selection_method": cfg_hdbscan.cluster_selection_method,
        }

        # 2D projection with robust fallback.
        try:
            if umap is None:
                raise RuntimeError("umap not installed")
            u_params = {**default_umap, **(umap_params or {})}
            reducer = umap.UMAP(
                n_neighbors=min(int(u_params["n_neighbors"]), max(2, n_samples - 1)),
                min_dist=float(u_params["min_dist"]),
                n_components=2,
                metric=u_params["metric"],
                random_state=cfg_umap.random_state,
            )
            embeddings_2d = reducer.fit_transform(vectors)
        except Exception as exc:  # noqa: BLE001
            logger.warning("UMAP failed, falling back to first two dimensions: %s", exc)
            if vectors.shape[1] >= 2:
                embeddings_2d = vectors[:, :2]
            else:
                embeddings_2d = np.column_stack((vectors[:, 0], np.zeros(n_samples)))

        # clustering with robust fallback.
        try:
            if hdbscan is None:
                raise RuntimeError("hdbscan not installed")
            h_params = {**default_hdbscan, **(hdbscan_params or {})}
            min_cluster_size = min(max(2, int(h_params["min_cluster_size"])), n_samples)
            min_samples_val = min(max(1, int(h_params["min_samples"])), min_cluster_size)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples_val,
                metric=h_params["metric"],
                cluster_selection_method=h_params["cluster_selection_method"],
            )
            labels = clusterer.fit_predict(embeddings_2d)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HDBSCAN failed, assigning single-cluster labels: %s", exc)
            labels = np.zeros(n_samples, dtype=int)

        result_df["x"] = embeddings_2d[:, 0]
        result_df["y"] = embeddings_2d[:, 1]
        result_df["label"] = labels
        return result_df

    @staticmethod
    def extract_keywords(df: pd.DataFrame, top_k: int = 5) -> Dict[int, List[str]]:
        clusters: Dict[int, List[str]] = {}
        for _, row in df.iterrows():
            label = int(row["label"])
            clusters.setdefault(label, []).append(row["text"])

        extractor = CTFIDFExtractor(top_k=top_k)
        extractor.fit(clusters)
        return {cluster_id: extractor.get_top_keywords(cluster_id) for cluster_id in clusters.keys()}

    @staticmethod
    def get_cluster_info(df: pd.DataFrame, keywords: Dict[int, List[str]]) -> List[ClusterInfo]:
        infos: List[ClusterInfo] = []
        for label in sorted(int(v) for v in df["label"].unique() if int(v) != -1):
            subset = df[df["label"] == label]
            dist = subset["period"].value_counts().to_dict()
            dominant_period = max(dist, key=dist.get) if dist else "Unknown"
            infos.append(
                ClusterInfo(
                    cluster_id=label,
                    count=len(subset),
                    dominant_period=dominant_period,
                    period_distribution={k: int(v) for k, v in dist.items()},
                    top_keywords=keywords.get(label, []),
                )
            )
        infos.sort(key=lambda item: item.count, reverse=True)
        return infos

    def analyze(
        self,
        word: str,
        umap_params: Optional[Dict[str, Any]] = None,
        hdbscan_params: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        started = time.time()

        vectors_df = self.get_vectors(word, force_recompute=force_recompute)
        if vectors_df.empty:
            return self._create_empty_result(word)

        cache_params = {"umap": umap_params or {}, "hdbscan": hdbscan_params or {}}
        if self.use_cache and not force_recompute:
            cached = load_clustering_cache(word, cache_params, self.model_name)
            if cached is not None:
                cached["processing_time_ms"] = int((time.time() - started) * 1000)
                return cached

        clustered_df = self.cluster(vectors_df, umap_params, hdbscan_params)
        keywords = self.extract_keywords(clustered_df)
        cluster_info = self.get_cluster_info(clustered_df, keywords)

        result = {
            "concept": word,
            "periods": PERIODS,
            "definitions": self._generate_definitions(clustered_df, keywords),
            "period_years": PERIOD_YEARS,
            "scatter_data": self._package_scatter_data(clustered_df, word, keywords),
            "drift_data": self._calculate_drift_data(clustered_df),
            "cluster_stats": [
                {
                    "cluster_id": c.cluster_id,
                    "count": c.count,
                    "dominant_period": c.dominant_period,
                    "period_distribution": c.period_distribution,
                    "top_keywords": c.top_keywords,
                }
                for c in cluster_info
            ],
            "quotations": self._collect_quotations(clustered_df),
            "confidence_score": self._calculate_confidence(clustered_df),
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "processing_time_ms": int((time.time() - started) * 1000),
            "model_info": self.encoder.get_model_info(),
            "algorithm_params": {
                "umap": umap_params or {
                    "n_neighbors": self.pipeline_config.umap.n_neighbors,
                    "min_dist": self.pipeline_config.umap.min_dist,
                    "n_components": self.pipeline_config.umap.n_components,
                    "metric": self.pipeline_config.umap.metric,
                },
                "hdbscan": hdbscan_params or {
                    "min_cluster_size": self.pipeline_config.hdbscan.min_cluster_size,
                    "min_samples": self.pipeline_config.hdbscan.min_samples,
                    "metric": self.pipeline_config.hdbscan.metric,
                    "cluster_selection_method": self.pipeline_config.hdbscan.cluster_selection_method,
                },
            },
        }

        if self.use_cache:
            save_clustering_cache(word, cache_params, result, self.model_name)

        return result

    def _generate_definitions(self, df: pd.DataFrame, keywords: Dict[int, List[str]]) -> Dict[str, str]:
        definitions: Dict[str, str] = {}
        for period in PERIODS:
            period_df = df[df["period"] == period]
            if period_df.empty:
                definitions[period] = "No related corpus found"
                continue

            label_counts = period_df["label"].value_counts()
            top_label = int(label_counts.index[0]) if not label_counts.empty else -1
            longest_row = period_df.loc[period_df["text"].str.len().idxmax()]
            if top_label == -1:
                definitions[period] = longest_row["text"][:220]
            else:
                keyword_text = ", ".join(keywords.get(top_label, [])[:5])
                definitions[period] = f"Keywords: {keyword_text}\n\n{longest_row['text'][:180]}..."
        return definitions

    @staticmethod
    def _calculate_drift_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        total = max(1, len(df))
        for period in PERIODS:
            period_df = df[df["period"] == period]
            if period_df.empty:
                payload.append({"period": period, "semantic_stability": 0.0, "concept_usage": 0.0})
                continue

            labels = period_df["label"].values
            non_noise = [int(label) for label in labels if int(label) != -1]
            unique_clusters = len(set(non_noise))
            stability = 0.9 if unique_clusters <= 1 else max(0.3, 1.0 / unique_clusters)
            payload.append(
                {
                    "period": period,
                    "semantic_stability": float(min(1.0, stability)),
                    "concept_usage": float(len(period_df) / total),
                }
            )
        return payload

    @staticmethod
    def _package_scatter_data(df: pd.DataFrame, word: str, keywords: Dict[int, List[str]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            label = int(row["label"])
            rows.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "period": row["period"],
                    "label": label,
                    "text": str(row["text"])[:160],
                    "source": row["source"],
                    "word": word,
                    "cluster_keywords": ", ".join(keywords.get(label, [])),
                }
            )
        return rows

    @staticmethod
    def _collect_quotations(df: pd.DataFrame, max_per_period: int = 3) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for period in PERIODS:
            for _, row in df[df["period"] == period].head(max_per_period).iterrows():
                out.append({"text": row["text"], "source": row["source"], "period": period})
        return out

    @staticmethod
    def _calculate_confidence(df: pd.DataFrame) -> float:
        n_samples = len(df)
        if n_samples == 0:
            return 0.0
        noise_ratio = float((df["label"] == -1).sum()) / n_samples
        sample_score = 0.5 if n_samples < 10 else (0.7 if n_samples < 50 else 0.9)
        confidence = min(0.95, sample_score * (1 - noise_ratio * 0.3))
        return round(float(confidence), 2)

    @staticmethod
    def _create_empty_result(word: str) -> Dict[str, Any]:
        return {
            "concept": word,
            "periods": PERIODS,
            "definitions": {period: "No related corpus found" for period in PERIODS},
            "period_years": PERIOD_YEARS,
            "scatter_data": [],
            "drift_data": [],
            "cluster_stats": [],
            "quotations": [],
            "confidence_score": 0.0,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "processing_time_ms": 0,
            "model_info": None,
        }


_engine_instance: Optional[AnalysisEngine] = None


def get_engine(force_reload: bool = False) -> AnalysisEngine:
    global _engine_instance
    if _engine_instance is None or force_reload:
        _engine_instance = AnalysisEngine()
    return _engine_instance
