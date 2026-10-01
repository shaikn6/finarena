from pathlib import Path

import joblib
import pytest
from conftest import APP, H
from fastapi.testclient import TestClient

from finarena.config import Settings
from finarena.main import Registry, create_app
from finarena.models.credit import CreditScorer

ART = Path(__file__).parents[1] / "artifacts" / "credit.joblib"
pytestmark = pytest.mark.skipif(not ART.exists(), reason="run scripts/train_credit.py first")


@pytest.fixture(scope="module")
def client():
    s = Settings(env="dev", api_keys=frozenset({"secret"}), max_batch=4)
    return TestClient(create_app(s, Registry(credit=CreditScorer(joblib.load(ART)))))


def healthy():
    return {**APP, "limit_bal": 300000, "age": 40, "pay_status": [-1] * 6, "bill_amt": [5000] * 6, "pay_amt": [5000] * 6}


def risky():
    return {**APP, "pay_status": [3, 3, 2, 2, 2, 2], "bill_amt": [48000] * 6, "pay_amt": [0] * 6, "limit_bal": 50000}


def test_risky_account_scores_higher_than_healthy_one(client):
    r = client.post("/v1/credit/score", json={"applications": [healthy(), risky()]}, headers=H).json()
    good, bad = r["results"]
    assert bad["probability_of_default"] > good["probability_of_default"]
    assert bad["decision"] == "decline" and good["decision"] == "approve"


def test_explainable_model_returns_plain_language_reason_codes(client):
    r = client.post("/v1/credit/score", json={"applications": [risky()], "model": "explainable"}, headers=H).json()
    item = r["results"][0]
    assert item["model"] == "logistic-regression" and 1 <= len(item["reason_codes"]) <= 3
    assert any("delinquen" in c.lower() or "late" in c.lower() for c in item["reason_codes"])


def test_accurate_model_has_no_reason_codes(client):
    assert client.post("/v1/credit/score", json={"applications": [APP]}, headers=H).json()["results"][0]["reason_codes"] is None


def test_sex_field_is_not_accepted_or_used(client):
    r = client.post("/v1/credit/score", json={"applications": [{**APP, "sex": 1}]}, headers=H)
    assert r.status_code == 200  # unknown fields ignored by schema, never reach the model
    from finarena.models.credit import FEATURES
    assert not any("sex" in f for f in FEATURES)


@pytest.mark.parametrize("patch", [{"age": 10}, {"limit_bal": -5}, {"pay_status": [0] * 5}, {"pay_amt": [-1] + [0] * 5}, {"pay_status": [99] * 6}])
def test_invalid_credit_inputs_rejected(client, patch):
    assert client.post("/v1/credit/score", json={"applications": [{**APP, **patch}]}, headers=H).status_code == 422


def test_batch_limit(client):
    assert client.post("/v1/credit/score", json={"applications": [APP] * 5}, headers=H).status_code == 413
