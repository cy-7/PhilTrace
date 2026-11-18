"""
Tests for the sense interpreter module.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from backend.sense_interpreter import (
    OllamaInterpreter,
    SenseInterpretation,
    SenseInterpreter,
    create_interpreter,
    save_interpretations,
)
from backend.pipeline_config import InterpretationConfig


class MockInterpreter(SenseInterpreter):
    """Mock interpreter for testing base class logic."""

    def __init__(self, response: str = '{"description": "test", "keywords": ["a", "b"], "tradition": "Aristotle"}'):
        config = InterpretationConfig(enable_cache=False, n_samples=3)
        super().__init__(config)
        self._response = response
        self.call_count = 0

    def _call_llm(self, prompt: str) -> str:
        self.call_count += 1
        return self._response

    def get_model_name(self) -> str:
        return "mock-model"


class TestSenseInterpreterBase(unittest.TestCase):
    """Test the abstract base class logic."""

    def test_interpret_cluster_basic(self):
        """Basic interpretation produces valid result."""
        interpreter = MockInterpreter()
        samples = [
            {"text": "Force is the cause of motion.", "period": "Ancient", "source": "Aristotle_Physics"},
            {"text": "Every force has an equal reaction.", "period": "Modern", "source": "Newton_Principia"},
        ]

        result = interpreter.interpret_cluster(
            target_word="force",
            sense_id=0,
            samples=samples,
            dominant_period="Ancient",
        )

        self.assertIsInstance(result, SenseInterpretation)
        self.assertEqual(result.sense_id, 0)
        self.assertEqual(result.target_word, "force")
        self.assertEqual(result.description, "test")
        self.assertEqual(result.keywords, ["a", "b"])
        self.assertEqual(result.tradition, "Aristotle")
        self.assertFalse(result.is_fallback)

    def test_interpret_cluster_invalid_json_fallback(self):
        """Invalid JSON response still produces a result."""
        interpreter = MockInterpreter(response="This is not JSON at all")
        samples = [{"text": "test", "period": "Ancient", "source": "src"}]

        result = interpreter.interpret_cluster(
            target_word="force",
            sense_id=1,
            samples=samples,
            dominant_period="Ancient",
        )

        # Should use raw response as description
        self.assertIn("This is not JSON", result.description)
        self.assertEqual(result.keywords, [])

    def test_interpret_cluster_llm_failure_uses_fallback(self):
        """When LLM fails completely, fallback interpretation is used."""
        interpreter = MockInterpreter()
        interpreter._response = None  # Will cause _call_with_retry to return None

        # Override _call_llm to raise
        def failing_call(prompt):
            raise RuntimeError("Connection refused")

        interpreter._call_llm = failing_call
        interpreter.config = InterpretationConfig(enable_cache=False, max_retries=1, retry_delay=0)

        samples = [{"text": "Force moves objects.", "period": "Ancient", "source": "src"}]
        result = interpreter.interpret_cluster(
            target_word="force",
            sense_id=0,
            samples=samples,
            dominant_period="Ancient",
        )

        self.assertTrue(result.is_fallback)
        self.assertIn("Fallback", result.description)

    def test_build_prompt_includes_samples(self):
        """Prompt building includes all samples."""
        interpreter = MockInterpreter()
        samples = [
            {"text": "Sample one about force.", "period": "Ancient", "source": "Plato"},
            {"text": "Sample two about force.", "period": "Modern", "source": "Newton"},
        ]

        prompt = interpreter._build_prompt("force", samples, "Ancient")
        self.assertIn("force", prompt)
        self.assertIn("Sample one", prompt)
        self.assertIn("Sample two", prompt)
        self.assertIn("Plato", prompt)
        self.assertIn("Newton", prompt)

    def test_sample_representatives_centroid_nearest(self):
        """Centroid-nearest sampling selects closest vectors."""
        interpreter = MockInterpreter()

        # Create a cluster with clear centroid
        df = pd.DataFrame({
            "text": ["close1", "close2", "far1", "far2"],
            "period": ["Ancient", "Ancient", "Modern", "Modern"],
            "source": ["s1", "s2", "s3", "s4"],
            "vector": [
                np.array([1.0, 0.0]),
                np.array([0.9, 0.1]),
                np.array([-1.0, 0.0]),
                np.array([-0.9, -0.1]),
            ],
        })

        samples = interpreter._sample_representatives(df, "test")
        # Should get 3 samples (config n_samples=3)
        self.assertEqual(len(samples), 3)

    def test_interpret_all_clusters(self):
        """interpret_all_clusters processes all non-noise clusters."""
        interpreter = MockInterpreter()

        df = pd.DataFrame({
            "text": ["a", "b", "c", "d", "noise"],
            "period": ["Ancient", "Ancient", "Modern", "Modern", "Ancient"],
            "source": ["s1", "s2", "s3", "s4", "s5"],
            "vector": [np.zeros(2)] * 5,
            "label": [0, 0, 1, 1, -1],
        })

        results = interpreter.interpret_all_clusters("force", df)
        self.assertEqual(len(results), 2)  # Two clusters (0 and 1), noise excluded
        self.assertEqual(results[0].sense_id, 0)
        self.assertEqual(results[1].sense_id, 1)


class TestSenseInterpreterCache(unittest.TestCase):
    """Test caching behavior."""

    def test_cache_saves_and_loads(self):
        """Cached results are reused on second call."""
        with tempfile.TemporaryDirectory() as tmp:
            interpreter = MockInterpreter()
            interpreter._cache_dir = Path(tmp)
            interpreter.config = InterpretationConfig(enable_cache=True, n_samples=3)

            samples = [{"text": "test", "period": "Ancient", "source": "src"}]

            # First call - should call LLM
            result1 = interpreter.interpret_cluster("force", 0, samples, "Ancient")
            self.assertEqual(interpreter.call_count, 1)

            # Second call - should use cache
            result2 = interpreter.interpret_cluster("force", 0, samples, "Ancient")
            self.assertEqual(interpreter.call_count, 1)  # No additional LLM call
            self.assertEqual(result1.description, result2.description)


class TestCreateInterpreter(unittest.TestCase):
    """Test the factory function."""

    def test_create_ollama(self):
        interpreter = create_interpreter("ollama")
        self.assertIsInstance(interpreter, OllamaInterpreter)

    def test_create_unknown_defaults_to_ollama(self):
        interpreter = create_interpreter("unknown_provider")
        self.assertIsInstance(interpreter, OllamaInterpreter)


class TestSaveInterpretations(unittest.TestCase):
    """Test output persistence."""

    def test_save_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            interpretations = [
                SenseInterpretation(
                    sense_id=0,
                    target_word="force",
                    description="Physical force in Aristotelian physics",
                    keywords=["motion", "cause", "nature"],
                    tradition="Aristotelian physics",
                    period="Ancient",
                    n_samples_used=5,
                    model_used="test-model",
                    timestamp="2025-01-01T00:00:00",
                ),
            ]

            path = save_interpretations("force", interpretations, output_dir=Path(tmp))
            self.assertTrue(path.exists())

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["target_word"], "force")
            self.assertEqual(data["n_senses"], 1)
            self.assertEqual(data["interpretations"][0]["description"], "Physical force in Aristotelian physics")


if __name__ == "__main__":
    unittest.main()
