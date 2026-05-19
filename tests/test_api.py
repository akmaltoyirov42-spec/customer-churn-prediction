"""
Basic API tests — these run in CI without a trained model,
so we just check the contract (routes, schema, error handling).
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE = {
    "tenure": 24,
    "MonthlyCharges": 70.0,
    "TotalCharges": 1680.0,
    "Contract_One_year": 1,
    "Contract_Two_year": 0,
    "PaymentMethod_Electronic_check": 1,
    "InternetService_Fiber_optic": 0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_correct_schema():
    mock_result = {"churn_probability": 0.23, "prediction": "no churn"}
    with patch("api.main.predict", return_value=mock_result):
        r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    data = r.json()
    assert "churn_probability" in data
    assert "prediction" in data
    assert data["prediction"] in ("churn", "no churn")


def test_predict_model_not_found_returns_503():
    with patch("api.main.predict", side_effect=FileNotFoundError):
        r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 503


def test_predict_missing_required_field_returns_422():
    r = client.post("/predict", json={"MonthlyCharges": 70.0})
    assert r.status_code == 422
