"""
Analysis engine for concept drift analysis.
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

logger = get_logger(__name__)

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # noqa: BLE001
    torch = None
    AutoModel = None
    AutoTokenizer = None

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

DEFAULT_UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2, "metric": "cosine"}
DEFAULT_HDBSCAN_PARAMS = {
    "min_cluster_size": 15,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}
DEFAULT_MODEL_NAME = "roberta-base"
FALLBACK_EMBEDDING_SIZE = 384


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


def _cache_path(word: str, suffix: str) -> Path:
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


def save_vectors_cache(word: str, df: pd.DataFrame) -> None:
    if not settings.enable_cache:
        return
    path = _cache_path(word, "vec.pkl")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_pickle(tmp)
    tmp.replace(path)


def load_vectors_cache(word: str) -> Optional[pd.DataFrame]:
    if not settings.enable_cache:
        return None

    path = _cache_path(word, "vec.pkl")
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


def save_clustering_cache(word: str, params: Dict[str, Any], result: Dict[str, Any]) -> None:
    if not settings.enable_cache:
        return

    path = _cache_path(word, "res.pkl")
    payload = {"params": params, "result": result, "timestamp": time.time()}
    _atomic_pickle_dump(path, payload)


def load_clustering_cache(word: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not settings.enable_cache:
        return None

    path = _cache_path(word, "res.pkl")
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


class AnalysisEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, use_cache: bool = True):
        self.model_name = model_name
        self.use_cache = use_cache and settings.enable_cache
        self.corpus_files: Dict[str, List[Path]] = {}
        self.model = None
        self.tokenizer = None
        self.hidden_size = FALLBACK_EMBEDDING_SIZE

        if torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self._load_model()
        self.load_corpus()

    def _load_model(self) -> None:
        if AutoTokenizer is None or AutoModel is None or torch is None:
            logger.warning("Transformers/Torch unavailable, engine will use hash embeddings fallback")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.hidden_size = int(getattr(self.model.config, "hidden_size", FALLBACK_EMBEDDING_SIZE))
            logger.info("Loaded embedding model '%s' on %s", self.model_name, self.device)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load model '%s', fallback enabled: %s", self.model_name, exc)
            self.model = None
            self.tokenizer = None
            self.hidden_size = FALLBACK_EMBEDDING_SIZE

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
            cached = load_vectors_cache(target_word)
            if cached is not None:
                return cached

        sentences = self._extract_sentences(target_word)
        if not sentences:
            return pd.DataFrame(columns=["text", "period", "source", "vector"])

        rows: List[Dict[str, Any]] = []
        for sentence in sentences:
            try:
                vector = self._get_embedding(sentence.text, target_word)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding failed for one sentence; skipping. reason=%s", exc)
                continue
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
            save_vectors_cache(target_word, df)
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

    def _get_embedding(self, text: str, target_word: str) -> np.ndarray:
        if self.model is None or self.tokenizer is None or torch is None:
            return self._fallback_embedding(text)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_offsets_mapping=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offsets = inputs["offset_mapping"].squeeze().tolist()

        token_indices = self._find_target_token_indices(text, target_word, offsets)
        if not token_indices:
            token_indices = [0]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state.squeeze(0)
        embedding = hidden_states[token_indices].cpu().numpy().mean(axis=0)
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    @staticmethod
    def _fallback_embedding(text: str) -> np.ndarray:
        vec = np.zeros(FALLBACK_EMBEDDING_SIZE, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for word in words:
            index = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % FALLBACK_EMBEDDING_SIZE  # noqa: S324
            vec[index] += 1.0
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    @staticmethod
    def _find_target_token_indices(text: str, target_word: str, offset_mapping: List[Tuple[int, int]]) -> List[int]:
        indices: List[int] = []
        word_lower = target_word.lower()
        text_lower = text.lower()

        for idx, (start, end) in enumerate(offset_mapping):
            if start >= end:
                continue
            token_text = text_lower[start:end]
            if word_lower in token_text:
                indices.append(idx)

        return indices

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

        # 2D projection with robust fallback.
        try:
            if umap is None:
                raise RuntimeError("umap not installed")
            u_params = {**DEFAULT_UMAP_PARAMS, **(umap_params or {})}
            reducer = umap.UMAP(
                n_neighbors=min(int(u_params["n_neighbors"]), max(2, n_samples - 1)),
                min_dist=float(u_params["min_dist"]),
                n_components=2,
                metric=u_params["metric"],
                random_state=42,
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
            h_params = {**DEFAULT_HDBSCAN_PARAMS, **(hdbscan_params or {})}
            min_cluster_size = min(max(2, int(h_params["min_cluster_size"])), n_samples)
            min_samples = min(max(1, int(h_params["min_samples"])), min_cluster_size)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
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
            cached = load_clustering_cache(word, cache_params)
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
            "model_info": {
                "name": self.model_name,
                "hidden_size": str(self.hidden_size),
                "device": str(self.device),
                "fallback_mode": str(self.model is None),
            },
            "algorithm_params": {
                "umap": umap_params or DEFAULT_UMAP_PARAMS,
                "hdbscan": hdbscan_params or DEFAULT_HDBSCAN_PARAMS,
            },
        }

        if self.use_cache:
            save_clustering_cache(word, cache_params, result)

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


def get_engine(model_name: str = DEFAULT_MODEL_NAME, force_reload: bool = False) -> AnalysisEngine:
    global _engine_instance
    if _engine_instance is None or force_reload:
        _engine_instance = AnalysisEngine(model_name=model_name)
    return _engine_instance
