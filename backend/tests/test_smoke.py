import os
import unittest

from fastapi.testclient import TestClient

os.environ["DEBUG"] = "True"

from main import app  # noqa: E402
from src.core.security import create_access_token  # noqa: E402


class BackendSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, base_url="http://localhost")
        cls.auth_headers = {
            "Authorization": f"Bearer {create_access_token({'sub': 'user_1'})}"
        }

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "running")

    def test_health(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_protected_route_requires_auth(self):
        response = self.client.get("/api/v1/organizations/")
        self.assertEqual(response.status_code, 401)

    def test_intelligence_endpoints(self):
        payload = {"document_type": "bank_statement", "document_id": "doc_123"}

        classify = self.client.post(
            "/api/v1/intelligence/documents/classify",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(classify.status_code, 200)
        self.assertEqual(classify.json()["route"], "automated_review")

        capture = self.client.post(
            "/api/v1/intelligence/documents/capture",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(capture.status_code, 200)
        self.assertEqual(capture.json()["status"], "captured")
        self.assertIn("normalized_data", capture.json())

        fraud = self.client.post(
            "/api/v1/intelligence/documents/fraud-detection",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(fraud.status_code, 200)
        self.assertIn("risk_score", fraud.json())

        cash_flow = self.client.post(
            "/api/v1/intelligence/cash-flow-analysis",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(cash_flow.status_code, 200)
        self.assertIn("cash_flow_summary", cash_flow.json())

        income = self.client.post(
            "/api/v1/intelligence/income-calculation",
            json=payload,
            headers=self.auth_headers,
        )
        self.assertEqual(income.status_code, 200)
        self.assertIn("income_summary", income.json())

        decision = self.client.post(
            "/api/v1/intelligence/decisioning",
            json={"document_type": "pay_stub", "income_hint": 16500},
            headers=self.auth_headers,
        )
        self.assertEqual(decision.status_code, 200)
        self.assertEqual(decision.json()["decision"], "approved")


if __name__ == "__main__":
    unittest.main()
