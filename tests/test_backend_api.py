import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ.setdefault("PHILTRACE_BOOTSTRAP_ENGINE", "false")

from backend.main import app


class DummyEngine:
    def __init__(self):
        self.model = None
        self.corpus_files = {"Ancient": ["x"]}

    def analyze(self, **kwargs):
        return {
            "concept": kwargs["word"],
            "periods": ["Ancient", "Medieval", "Modern", "Contemporary"],
            "definitions": {"Ancient": "a", "Medieval": "b", "Modern": "c", "Contemporary": "d"},
            "period_years": {
                "Ancient": "800 BCE - 500 CE",
                "Medieval": "500 CE - 1500 CE",
                "Modern": "1500 CE - 1900 CE",
                "Contemporary": "1900 CE - Present",
            },
            "scatter_data": [],
            "drift_data": [],
            "cluster_stats": [],
            "quotations": [],
            "confidence_score": 0.8,
            "analysis_timestamp": "2026-01-01T00:00:00",
            "processing_time_ms": 12,
            "model_info": {"name": "mock"},
        }


class BackendApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        with patch("backend.main.get_engine", return_value=DummyEngine()), patch(
            "backend.main.check_ollama_status", return_value={"available": False, "count": 0, "models": []}
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("status", body)
        self.assertIn("extra", body)
        self.assertIn("request_id", body["extra"])

    def test_analyze_success(self):
        with patch("backend.main.get_engine", return_value=DummyEngine()):
            response = self.client.post("/analyze", json={"word": "reason"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["concept"], "reason")

    def test_analyze_validation_error(self):
        response = self.client.post("/analyze", json={"word": ""})
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["error"], "ValidationError")
        self.assertIn("request_id", body)

    def test_chat_simple_service_unavailable(self):
        with patch("backend.main.check_ollama_status", return_value={"available": False, "count": 0, "models": []}):
            response = self.client.post("/chat/simple", json={"message": "Hi", "model": "qwen2:7b"})

        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertIn("request_id", body)

    def test_chat_model_not_available(self):
        with patch(
            "backend.main.check_ollama_status",
            return_value={"available": True, "count": 1, "models": [{"name": "llama3"}]},
        ):
            response = self.client.post("/chat/simple", json={"message": "Hi", "model": "qwen2:7b"})

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["error"], "ModelNotAvailable")


if __name__ == "__main__":
    unittest.main()
