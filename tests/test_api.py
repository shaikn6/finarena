import json
from pathlib import Path

import pytest
from conftest import H


def test_health_needs_no_auth(make_client):
    assert make_client().get("/health").json()["status"] == "ok"


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}])
def test_auth_rejects_missing_or_bad_key(make_client, headers):
    r = make_client().post("/v1/sentiment", json={"texts": ["a"]}, headers=headers)
    assert r.status_code == 401


def test_dev_mode_without_keys_allows_requests(make_client):
    assert make_client(keys=()).post("/v1/sentiment", json={"texts": ["a"]}).status_code == 200


def test_auto_keeps_confident_fast_answers(make_client):
    r = make_client(fast=(0.95, 0.03, 0.02)).post("/v1/sentiment", json={"texts": ["a", "b"]}, headers=H).json()
    assert r["escalated_fraction"] == 0 and {x["model"] for x in r["results"]} == {"fast"}
    assert r["results"][0]["label"] == "Bearish"


def test_auto_escalates_low_confidence_to_accurate(make_client):
    r = make_client(fast=(0.4, 0.3, 0.3)).post("/v1/sentiment", json={"texts": ["a"]}, headers=H).json()
    assert r["escalated_fraction"] == 1 and r["results"][0]["model"] == "llm" and r["results"][0]["label"] == "Bullish"


def test_fast_only_deployment_rejects_accurate_strategy(make_client):
    c = make_client(with_llm=False)
    assert c.post("/v1/sentiment", json={"texts": ["a"], "strategy": "accurate"}, headers=H).status_code == 503
    assert c.post("/v1/sentiment", json={"texts": ["a"], "strategy": "fast"}, headers=H).status_code == 200


@pytest.mark.parametrize("body,code", [
    ({"texts": []}, 422), ({"texts": ["  "]}, 422), ({"texts": ["a"], "strategy": "bogus"}, 422),
    ({"texts": ["a"] * 5}, 413), ({"texts": ["x" * 51]}, 413), ({}, 422)])
def test_sentiment_input_validation(make_client, body, code):
    assert make_client().post("/v1/sentiment", json=body, headers=H).status_code == code


def test_probabilities_sum_to_one_and_request_id_echoed(make_client):
    r = make_client().post("/v1/sentiment", json={"texts": ["a"]}, headers={**H, "x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123" and "x-process-time-ms" in r.headers
    assert abs(sum(r.json()["results"][0]["probabilities"].values()) - 1) < 1e-6


def test_models_endpoint_requires_auth(make_client):
    c = make_client()
    assert c.get("/v1/models").status_code == 401 and c.get("/v1/models", headers=H).json() == {"x": 1}


def test_unconfigured_services_return_503_not_500(make_client):
    c = make_client()
    assert c.post("/v1/signature/detect", files={"file": ("a.png", b"x", "image/png")}, headers=H).status_code == 503
    assert c.post("/v1/credit/score", json={"applications": []}, headers=H).status_code in (422, 503)


def test_disabled_signature_endpoint_is_503_even_with_an_empty_body(make_client):
    c = make_client()
    assert c.post("/v1/signature/detect", headers=H).status_code == 503  # Content-Length: 0
    assert c.post("/v1/signature/detect").status_code == 401


def test_bodyless_post_without_content_length_reaches_the_route(make_client):
    """What `curl -X POST` sends: no body, so neither Content-Length nor Transfer-Encoding."""
    import anyio

    sent = []

    async def call():
        scope = dict(type="http", http_version="1.1", method="POST", path="/v1/signature/detect", raw_path=b"/v1/signature/detect",
                     query_string=b"", headers=[(b"x-api-key", b"secret")], client=("testclient", 1), server=("testserver", 80),
                     scheme="http", root_path="")

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await make_client().app(scope, receive, send)

    anyio.run(call)
    assert sent[0]["status"] == 503


def test_ready_reports_loaded_models(make_client):
    assert make_client().get("/ready").json()["models"]["sentiment"] is True


def test_auto_degrades_to_fast_when_llm_unavailable(make_client):
    r = make_client(with_llm=False).post("/v1/sentiment", json={"texts": ["a"]}, headers=H)
    assert r.status_code == 200 and r.json()["results"][0]["model"] == "fast" and r.json()["escalated_fraction"] == 0


def test_ui_is_served_at_root_without_auth(make_client):
    r = make_client().get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "FinArena" in r.text


def test_arena_endpoint_requires_auth_and_reports_missing_results(make_client):
    c = make_client()
    assert c.get("/v1/arena").status_code == 401
    assert c.get("/v1/arena", headers=H).status_code == 404  # stub settings point at a directory without arena.json


@pytest.mark.skipif(not (Path(__file__).parents[1] / "artifacts" / "arena.json").exists(), reason="run scripts/build_arena.py")
def test_arena_endpoint_serves_real_benchmark_results(make_client):
    arena = json.loads((Path(__file__).parents[1] / "artifacts" / "arena.json").read_text())
    r = make_client(arena=arena).get("/v1/arena", headers=H)
    assert r.status_code == 200
    a = r.json()
    assert {"sentiment", "credit", "trading"} <= set(a)
    best = max(a["sentiment"]["models"], key=lambda m: m["accuracy"])
    assert best["accuracy"] > 0.8 and best["detail"]
    assert all(m["accuracy"] < best["accuracy"] for m in a["sentiment"]["models"] if m is not best)
    assert a["sentiment"]["cascade"][-1]["escalated"] == 1.0


def test_model_busy_returns_503_with_retry_after(make_client):
    from finarena.models.concurrency import ModelBusy

    c = make_client()

    class Busy:
        name = "llm"

        def proba(self, texts):
            raise ModelBusy("llm is busy; retry shortly")

    c.app.state.registry.sentiment.accurate = Busy()
    r = c.post("/v1/sentiment", json={"texts": ["a"], "strategy": "accurate"}, headers=H)
    assert r.status_code == 503 and r.headers["retry-after"] == "2"


def test_ready_reports_whether_the_llm_loaded(make_client):
    assert make_client(with_llm=True).get("/ready").json()["models"]["sentiment_llm"] is True
    assert make_client(with_llm=False).get("/ready").json()["models"]["sentiment_llm"] is False


def test_authorization_bearer_is_accepted_like_x_api_key(make_client):
    c = make_client()
    ok = {"authorization": "Bearer secret"}
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers=ok).status_code == 200
    assert c.get("/metrics", headers=ok).status_code == 200  # what a Prometheus `authorization:` block sends


@pytest.mark.parametrize("header", ["Bearer wrong", "Basic secret", "Bearer", "secret"])
def test_wrong_or_malformed_authorization_header_is_401(make_client, header):
    assert make_client().get("/v1/models", headers={"authorization": header}).status_code == 401
