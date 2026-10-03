from pathlib import Path

import joblib
import pytest
from conftest import APP, H, StubModel, build_client

from finarena.models.concurrency import ModelBusy
from finarena.models.credit import CreditScorer

ART = Path(__file__).parents[1] / "artifacts" / "credit.joblib"
pytestmark = pytest.mark.skipif(not ART.exists(), reason="run scripts/train_credit.py first")


class CountingModel(StubModel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls, self.batch_sizes = 0, []

    def proba(self, texts):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        return super().proba(texts)


class BusyModel:
    name = "llm"

    def proba(self, texts):
        raise ModelBusy("llm is busy; retry shortly")


@pytest.fixture(scope="module")
def scorer():
    return CreditScorer(joblib.load(ART))


def client(scorer=None, fast=None, accurate="default", sentiment=True, max_batch=10):
    return build_client(credit=scorer, fast=fast or CountingModel("fast", (0.9, 0.05, 0.05)), sentiment=sentiment,
                        accurate=CountingModel("llm", (0.1, 0.8, 0.1)) if accurate == "default" else accurate,
                        with_llm=accurate is not None, max_batch=max_batch, max_text_chars=40)


def post(c, items, **extra):
    return c.post("/v1/analyze", json={"items": items, **extra}, headers=H)


def S(text="a"):  # noqa: N802 - short names keep the request tables readable
    return {"task": "sentiment", "text": text}


def C():  # noqa: N802
    return {"task": "credit", "application": APP}


def test_mixed_batch_is_routed_to_specialists_and_returned_in_order(scorer):
    r = post(client(scorer), [S("a"), C(), S("b")])
    assert r.status_code == 200
    out = r.json()["results"]
    assert [x["task"] for x in out] == ["sentiment", "credit", "sentiment"] and all(x["ok"] for x in out)
    assert out[0]["result"]["label"] == "Bearish" and out[1]["result"]["decision"] in ("approve", "decline")


def test_items_of_one_task_share_a_single_model_call(scorer):
    fast = CountingModel("fast", (0.9, 0.05, 0.05))
    post(client(scorer, fast=fast), [S("a"), C(), S("b"), S("c")])
    assert fast.calls == 1 and fast.batch_sizes == [3]


def test_one_bad_item_fails_alone(scorer):
    out = post(client(scorer), [S("a"), S("x" * 41), C()]).json()["results"]
    assert [x["ok"] for x in out] == [True, False, True] and "too long" in out[1]["error"]


def test_missing_credit_model_only_fails_credit_items():
    out = post(client(scorer=None), [S("a"), C()]).json()["results"]
    assert out[0]["ok"] is True and out[1]["ok"] is False and "credit model not loaded" in out[1]["error"]


def test_missing_sentiment_models_only_fail_sentiment_items(scorer):
    out = post(client(scorer, sentiment=False), [S("a"), C()]).json()["results"]
    assert out[0]["ok"] is False and "not loaded" in out[0]["error"] and out[1]["ok"] is True


def test_busy_llm_is_an_item_error_not_a_failed_request(scorer):
    c = client(scorer, fast=CountingModel("fast", (0.4, 0.3, 0.3)), accurate=BusyModel())  # low confidence -> escalates
    r = post(c, [S("a"), C()])
    out = r.json()["results"]
    assert r.status_code == 200 and out[0]["ok"] is False and "busy" in out[0]["error"] and out[1]["ok"] is True


def test_strategy_is_forwarded_and_unavailable_strategy_is_an_item_error(scorer):
    out = post(client(scorer), [S("a")], sentiment_strategy="accurate").json()["results"]
    assert out[0]["result"]["model"] == "llm"
    out = post(client(scorer, accurate=None), [S("a")], sentiment_strategy="accurate").json()["results"]
    assert out[0]["ok"] is False and "not available" in out[0]["error"]


def test_credit_model_choice_is_forwarded(scorer):
    out = post(client(scorer), [C()], credit_model="explainable").json()["results"]
    assert out[0]["result"]["model"] == "logistic-regression"


def test_total_batch_size_is_limited(scorer):
    assert post(client(scorer, max_batch=4), [S()] * 5).status_code == 413


@pytest.mark.parametrize("items", [[], [{"task": "weather", "text": "x"}], [{"task": "sentiment", "text": "   "}],
                                   [{"task": "credit"}], [{"task": "sentiment"}]])
def test_invalid_requests_are_422(scorer, items):
    assert post(client(scorer), items).status_code == 422


def test_requires_api_key(scorer):
    assert client(scorer).post("/v1/analyze", json={"items": [S()]}).status_code == 401




class ExplodingScorer:
    threshold = 0.5

    def score(self, applications, model):
        raise RuntimeError("model file corrupted: /secret/path")


def test_a_crashing_credit_model_fails_only_credit_items_and_leaks_nothing(scorer):
    r = post(client(ExplodingScorer()), [S("a"), C(), S("b")])
    out = r.json()["results"]
    assert r.status_code == 200 and [x["ok"] for x in out] == [True, False, True]
    assert out[1]["error"] == "credit failed" and "secret" not in r.text


def test_busy_llm_inside_analyze_is_counted_in_metrics(scorer):
    c = client(scorer, fast=CountingModel("fast", (0.4, 0.3, 0.3)), accurate=BusyModel())
    post(c, [S("a"), S("b")])
    assert "finarena_model_busy_total 1" in c.get("/metrics", headers=H).text
