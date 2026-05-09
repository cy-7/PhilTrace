"""
Tests for the ModernBERT encoder and updated analysis engine.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from backend import analysis_engine
from backend.modernbert_encoder import ModernBERTEncoder, FALLBACK_EMBEDDING_SIZE
from backend.pipeline_config import EncoderConfig, PipelineConfig, get_pipeline_config


class TestModernBERTEncoder(unittest.TestCase):
    """Test the ModernBERT encoder module."""

    def test_fallback_embedding_shape(self):
        """When no model is loaded, fallback produces correct shape."""
        vec = ModernBERTEncoder._fallback_embedding("This is a test sentence about force.")
        self.assertEqual(vec.shape, (FALLBACK_EMBEDDING_SIZE,))
        # Should be normalized
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_fallback_embedding_deterministic(self):
        """Fallback embedding is deterministic for same input."""
        text = "Force is a fundamental concept in physics."
        vec1 = ModernBERTEncoder._fallback_embedding(text)
        vec2 = ModernBERTEncoder._fallback_embedding(text)
        np.testing.assert_array_equal(vec1, vec2)

    def test_find_target_token_indices_basic(self):
        """Token index finding works for simple cases."""
        text = "The force of nature is strong."
        # Simulate offset mapping: each word is a token
        offsets = [
            (0, 0),   # [CLS]
            (0, 3),   # The
            (4, 9),   # force
            (10, 12), # of
            (13, 19), # nature
            (20, 22), # is
            (23, 29), # strong
            (29, 30), # .
            (0, 0),   # [SEP]
        ]
        indices = ModernBERTEncoder._find_target_token_indices(text, "force", offsets)
        self.assertEqual(indices, [2])

    def test_find_target_token_indices_subword(self):
        """Token index finding works when word is split into subwords."""
        text = "The forceful argument prevailed."
        # "forceful" split into "force" + "ful"
        offsets = [
            (0, 0),    # [CLS]
            (0, 3),    # The
            (4, 9),    # force  (part of forceful)
            (9, 12),   # ful    (part of forceful)
            (13, 21),  # argument
            (22, 31),  # prevailed
            (31, 32),  # .
            (0, 0),    # [SEP]
        ]
        # Looking for "forceful" - should find tokens that overlap with char span 4-12
        indices = ModernBERTEncoder._find_target_token_indices(text, "forceful", offsets)
        self.assertIn(2, indices)
        self.assertIn(3, indices)

    def test_find_target_token_indices_not_found(self):
        """Returns empty list when target word not in text."""
        text = "The argument prevailed."
        offsets = [
            (0, 0),
            (0, 3),
            (4, 12),
            (13, 22),
            (22, 23),
            (0, 0),
        ]
        indices = ModernBERTEncoder._find_target_token_indices(text, "force", offsets)
        self.assertEqual(indices, [])

    def test_encoder_without_model_uses_fallback(self):
        """Encoder without torch/model uses fallback embeddings."""
        with patch("backend.modernbert_encoder.torch", None):
            config = EncoderConfig(model_name="nonexistent/model")
            encoder = ModernBERTEncoder(config=config)
            self.assertFalse(encoder.is_loaded)

            vec = encoder.encode_single("Force is important.", "force")
            self.assertEqual(vec.shape, (FALLBACK_EMBEDDING_SIZE,))

    def test_compute_embedding_stats(self):
        """Stats computation works correctly."""
        encoder = ModernBERTEncoder.__new__(ModernBERTEncoder)
        vectors = [np.random.randn(384).astype(np.float32) for _ in range(10)]
        stats = encoder.compute_embedding_stats(vectors)
        self.assertEqual(stats["count"], 10)
        self.assertEqual(stats["shape"], (10, 384))
        self.assertIn("norm_mean", stats)
        self.assertIn("vector_std", stats)

    def test_compute_embedding_stats_empty(self):
        """Stats computation handles empty input."""
        encoder = ModernBERTEncoder.__new__(ModernBERTEncoder)
        stats = encoder.compute_embedding_stats([])
        self.assertEqual(stats["count"], 0)


class TestAnalysisEngine(unittest.TestCase):
    """Test the refactored analysis engine."""

    def test_cache_roundtrip_and_corruption_recovery(self):
        """Cache save/load works, and corrupted cache is handled gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            analysis_engine.CACHE_DIR = Path(tmp)

            df = pd.DataFrame(
                [{"text": "reason is central", "period": "Modern", "source": "src", "vector": np.array([0.1, 0.2])}]
            )
            analysis_engine.save_vectors_cache("reason", df, "test-model")
            loaded = analysis_engine.load_vectors_cache("reason", "test-model")
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 1)

            # Corrupt the cache
            bad_path = analysis_engine._cache_path("reason", "vec.pkl", "test-model")
            bad_path.write_bytes(b"not a pickle")
            recovered = analysis_engine.load_vectors_cache("reason", "test-model")
            self.assertIsNone(recovered)

    def test_cache_model_isolation(self):
        """Different models get different cache files."""
        with tempfile.TemporaryDirectory() as tmp:
            analysis_engine.CACHE_DIR = Path(tmp)

            df1 = pd.DataFrame(
                [{"text": "test1", "period": "Ancient", "source": "s1", "vector": np.array([1.0, 0.0])}]
            )
            df2 = pd.DataFrame(
                [{"text": "test2", "period": "Modern", "source": "s2", "vector": np.array([0.0, 1.0])}]
            )

            analysis_engine.save_vectors_cache("force", df1, "model-a")
            analysis_engine.save_vectors_cache("force", df2, "model-b")

            loaded_a = analysis_engine.load_vectors_cache("force", "model-a")
            loaded_b = analysis_engine.load_vectors_cache("force", "model-b")

            self.assertEqual(loaded_a.iloc[0]["text"], "test1")
            self.assertEqual(loaded_b.iloc[0]["text"], "test2")

    def test_cluster_small_sample_fallback(self):
        """Clustering handles very small samples gracefully."""
        with patch("backend.modernbert_encoder.torch", None):
            engine = analysis_engine.AnalysisEngine(use_cache=False)

        df = pd.DataFrame(
            [
                {"text": "a", "period": "Ancient", "source": "s1", "vector": np.array([1.0, 0.0])},
                {"text": "b", "period": "Modern", "source": "s2", "vector": np.array([0.0, 1.0])},
            ]
        )

        clustered = engine.cluster(df)
        self.assertIn("x", clustered.columns)
        self.assertIn("label", clustered.columns)
        self.assertEqual(len(clustered), 2)

    def test_pipeline_config_loads(self):
        """Pipeline config loads without error."""
        config = get_pipeline_config(force_reload=True)
        self.assertIsNotNone(config)
        self.assertEqual(config.encoder.model_name, "answerdotai/ModernBERT-large")
        self.assertEqual(config.encoder.subword_pooling, "mean")
        self.assertEqual(config.umap.n_neighbors, 15)
        self.assertEqual(config.hdbscan.min_cluster_size, 15)


class TestCTFIDFExtractor(unittest.TestCase):
    """Test keyword extraction."""

    def test_basic_extraction(self):
        """CTF-IDF extracts meaningful keywords."""
        clusters = {
            0: ["force and motion in physics", "the force of gravity pulls objects"],
            1: ["justice and fairness in society", "the justice system requires reform"],
        }
        extractor = analysis_engine.CTFIDFExtractor(top_k=3)
        extractor.fit(clusters)

        kw0 = extractor.get_top_keywords(0)
        kw1 = extractor.get_top_keywords(1)

        self.assertTrue(len(kw0) > 0)
        self.assertTrue(len(kw1) > 0)
        # "force" should be in cluster 0 keywords
        self.assertIn("force", kw0)

    def test_noise_cluster_returns_empty(self):
        """Noise cluster (-1) returns no keywords."""
        clusters = {-1: ["random noise text"], 0: ["meaningful content here"]}
        extractor = analysis_engine.CTFIDFExtractor(top_k=3)
        extractor.fit(clusters)
        self.assertEqual(extractor.get_top_keywords(-1), [])


if __name__ == "__main__":
    unittest.main()
