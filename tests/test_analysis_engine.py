import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend import analysis_engine


class AnalysisEngineTests(unittest.TestCase):
    def test_cache_roundtrip_and_corruption_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis_engine.CACHE_DIR = Path(tmp)

            df = pd.DataFrame(
                [{"text": "reason is central", "period": "Modern", "source": "src", "vector": [0.1, 0.2]}]
            )
            analysis_engine.save_vectors_cache("reason", df)
            loaded = analysis_engine.load_vectors_cache("reason")
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded), 1)

            bad_path = analysis_engine._cache_path("reason", "vec.pkl")
            bad_path.write_bytes(b"not a pickle")
            recovered = analysis_engine.load_vectors_cache("reason")
            self.assertIsNone(recovered)

    def test_cluster_small_sample_fallback(self):
        with patch.object(analysis_engine.AnalysisEngine, "_load_model", lambda self: None):
            engine = analysis_engine.AnalysisEngine(use_cache=False)

        df = pd.DataFrame(
            [
                {"text": "a", "period": "Ancient", "source": "s1", "vector": [1.0, 0.0]},
                {"text": "b", "period": "Modern", "source": "s2", "vector": [0.0, 1.0]},
            ]
        )

        clustered = engine.cluster(df)
        self.assertIn("x", clustered.columns)
        self.assertIn("label", clustered.columns)
        self.assertEqual(len(clustered), 2)


if __name__ == "__main__":
    unittest.main()
