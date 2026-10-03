import re

from conftest import H

from finarena.metrics import LATENCY_BUCKETS_MS, Metrics


def value(text, line_start):
    m = re.search(rf"^{re.escape(line_start)} ([0-9.e+-]+)$", text, re.M)
    assert m, f"{line_start} not found in:\n{text}"
    return float(m.group(1))


def test_counters_accumulate_and_render_in_prometheus_format():
    m = Metrics()
    m.inc("things_total")
    m.inc("things_total", 2)
    m.inc("labelled_total", kind="a")
    text = m.render()
    assert "# TYPE things_total counter" in text
    assert value(text, "things_total") == 3 and value(text, 'labelled_total{kind="a"}') == 1


def test_histogram_buckets_are_cumulative_and_consistent():
    m = Metrics()
    for ms in (3, 7, 30, 20000):
        m.observe_request("/x", "GET", 200, ms)
    text = m.render()
    assert value(text, 'finarena_request_duration_ms_bucket{path="/x",le="5"}') == 1
    assert value(text, 'finarena_request_duration_ms_bucket{path="/x",le="10"}') == 2
    assert value(text, 'finarena_request_duration_ms_bucket{path="/x",le="50"}') == 3
    assert value(text, f'finarena_request_duration_ms_bucket{{path="/x",le="{LATENCY_BUCKETS_MS[-1]}"}}') == 3
    assert value(text, 'finarena_request_duration_ms_bucket{path="/x",le="+Inf"}') == 4
    assert value(text, 'finarena_request_duration_ms_count{path="/x"}') == 4
    assert value(text, 'finarena_requests_total{path="/x",method="GET",status="200"}') == 4


def test_metrics_endpoint_requires_the_api_key(make_client):
    c = make_client()
    assert c.get("/metrics").status_code == 401
    r = c.get("/metrics", headers=H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")


def test_requests_are_counted_by_route_template_and_status(make_client):
    c = make_client()
    for _ in range(2):
        c.post("/v1/sentiment", json={"texts": ["a"]}, headers=H)
    c.post("/v1/sentiment", json={"texts": []}, headers=H)  # 422
    text = c.get("/metrics", headers=H).text
    assert value(text, 'finarena_requests_total{path="/v1/sentiment",method="POST",status="200"}') == 2
    assert value(text, 'finarena_requests_total{path="/v1/sentiment",method="POST",status="422"}') == 1


def test_unknown_paths_share_one_label_so_cardinality_stays_bounded(make_client):
    c = make_client()
    for i in range(5):
        c.get(f"/no-such-page-{i}")
    text = c.get("/metrics", headers=H).text
    assert value(text, 'finarena_requests_total{path="unrouted",method="GET",status="404"}') == 5
    assert "no-such-page" not in text


def test_escalations_and_rate_limits_are_counted(make_client):
    c = make_client(fast=(0.4, 0.3, 0.3), rate_per_minute=3)  # low confidence -> every item escalates
    for texts in (["a", "b"], ["c"], ["d"], ["e"]):  # the 4th request exceeds the 3/min budget -> 429
        c.post("/v1/sentiment", json={"texts": texts}, headers=H)
    text = c.get("/metrics", headers=H).text
    assert value(text, "finarena_sentiment_items_total") == 4 and value(text, "finarena_sentiment_escalated_total") == 4
    assert value(text, "finarena_rate_limited_total") == 1
