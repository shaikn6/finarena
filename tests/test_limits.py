import pytest
from conftest import H

from finarena.limits import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_burst_up_to_capacity_then_blocked_with_retry_after():
    clock = Clock()
    rl = RateLimiter(60, clock)  # 1 token per second, burst 60
    assert all(rl.check("a")[0] for _ in range(60))
    allowed, retry = rl.check("a")
    assert not allowed and 0 < retry <= 1.0


def test_tokens_refill_over_time():
    clock = Clock()
    rl = RateLimiter(60, clock)
    for _ in range(60):
        rl.check("a")
    assert not rl.check("a")[0]
    clock.t += 2.0
    assert rl.check("a")[0] and rl.check("a")[0] and not rl.check("a")[0]


def test_clients_have_independent_budgets():
    rl = RateLimiter(2, Clock())
    assert rl.check("a")[0] and rl.check("a")[0] and not rl.check("a")[0]
    assert rl.check("b")[0]


def test_zero_disables_limiting():
    rl = RateLimiter(0, Clock())
    assert all(rl.check("a")[0] for _ in range(1000))


def test_memory_is_hard_capped_and_forgets_the_least_recently_seen_client():
    rl = RateLimiter(60, Clock())
    for i in range(RateLimiter.MAX_KEYS + 50):
        rl.check(f"client-{i}")
    assert len(rl._buckets) == RateLimiter.MAX_KEYS
    assert "client-0" not in rl._buckets and f"client-{RateLimiter.MAX_KEYS + 49}" in rl._buckets


def test_a_denied_client_stays_denied_even_while_others_are_evicted():
    rl = RateLimiter(1, Clock())
    rl.check("busy")
    assert not rl.check("busy")[0]
    for i in range(100):
        rl.check(f"other-{i}")
    assert not rl.check("busy")[0]


def test_api_returns_429_with_retry_after_when_budget_is_spent(make_client):
    c = make_client(rate_per_minute=3)
    codes = [c.post("/v1/sentiment", json={"texts": ["a"]}, headers=H).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    r = c.post("/v1/sentiment", json={"texts": ["a"]}, headers=H)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_different_api_keys_do_not_share_a_budget(make_client):
    c = make_client(keys=("k1", "k2"), rate_per_minute=1)
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"x-api-key": "k1"}).status_code == 200
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"x-api-key": "k1"}).status_code == 429
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"x-api-key": "k2"}).status_code == 200


@pytest.mark.parametrize("path", ["/health", "/ready", "/"])
def test_probes_and_ui_are_never_rate_limited(make_client, path):
    c = make_client(rate_per_minute=1)
    assert all(c.get(path).status_code == 200 for _ in range(10))


def test_oversized_body_is_refused_before_it_is_parsed(make_client):
    c = make_client(rate_per_minute=0)  # max_image_bytes=1000 in the fixture -> cap is 1000 + 64 KiB
    r = c.post("/v1/sentiment", content=b"x" * 70_000, headers={**H, "content-type": "application/json"})
    assert r.status_code == 413 and "too large" in r.json()["detail"]


def test_rate_limited_responses_still_carry_a_request_id(make_client):
    c = make_client(rate_per_minute=1)
    c.post("/v1/sentiment", json={"texts": ["a"]}, headers=H)
    r = c.post("/v1/sentiment", json={"texts": ["a"]}, headers={**H, "x-request-id": "trace-7"})
    assert r.status_code == 429 and r.headers["x-request-id"] == "trace-7"


def test_made_up_api_keys_cannot_be_used_to_dodge_the_limit(make_client):
    """An attacker rotating random x-api-key values must be counted together (by IP), not get a fresh budget each time."""
    c = make_client(keys=("real",), rate_per_minute=2)
    codes = [c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"x-api-key": f"fake-{i}"}).status_code for i in range(6)]
    assert codes.count(429) >= 4  # the first two spend the shared IP budget; every later fake key is refused


def test_a_configured_key_keeps_its_own_budget_when_others_are_throttled(make_client):
    c = make_client(keys=("real",), rate_per_minute=2)
    for i in range(4):
        c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"x-api-key": f"fake-{i}"})
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"x-api-key": "real"}).status_code == 200


def test_post_without_content_length_is_refused_not_read_unbounded(make_client):
    r = make_client().post("/v1/sentiment", content=iter([b'{"texts": ["a"]}']), headers=H)  # chunked: no Content-Length
    assert r.status_code == 411


def test_malformed_content_length_is_a_400_not_a_500(make_client):
    r = make_client().post("/v1/sentiment", content=b"{}", headers={**H, "content-length": "abc"})
    assert r.status_code == 400


def test_oversized_probes_spend_rate_limit_tokens(make_client):
    c = make_client(rate_per_minute=2)
    big = b"x" * 70_000
    codes = [c.post("/v1/sentiment", content=big, headers={**H, "content-type": "application/json"}).status_code for _ in range(4)]
    assert codes == [413, 413, 429, 429]


def test_a_valid_bearer_key_gets_its_own_budget_like_x_api_key(make_client):
    c = make_client(keys=("k1", "k2"), rate_per_minute=1)
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"authorization": "Bearer k1"}).status_code == 200
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"authorization": "Bearer k1"}).status_code == 429
    assert c.post("/v1/sentiment", json={"texts": ["a"]}, headers={"x-api-key": "k2"}).status_code == 200
