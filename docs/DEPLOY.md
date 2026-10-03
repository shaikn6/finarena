# Deploying FinArena

FinArena is one container with no database, so any host that runs Docker works. This guide is host-agnostic on
purpose. What was tested: the Docker Compose stack below (with its read-only, non-root hardening) and the Prometheus
scrape config in section 5 against a real Prometheus. Adapt the rest to your platform's own config format.

## 1. Pick the image

| Image | Build | Use when |
|---|---|---|
| base | default | fast sentiment and credit scoring are enough |
| LLM | `EXTRAS=[llm]` `PRELOAD_LLM=1` | you want the accurate sentiment path |

Sizes, memory and measured throughput are in the README (Capacity); the accurate path is CPU-bound, so scale it with
replicas. Model quality numbers live in the README and in `GET /v1/models`.

## 2. Configure

```bash
cp .env.example .env
```

`.env.example` explains every setting and how to generate a key. In production the service refuses to start without
API keys. On the LLM image set `FINARENA_REQUIRE_LLM=1` so a failed model load stops the container instead of serving a
degraded service. Clients send the key as `x-api-key: <key>` or `Authorization: Bearer <key>`.

## 3. Run

```bash
docker compose up -d --build
curl localhost:8000/ready          # {"status":"ready", ...}
```

The container is read-only, drops all Linux capabilities, runs as a non-root user and has a health check.

## 4. Put it behind TLS

FinArena speaks plain HTTP. Terminate TLS at your platform's load balancer or a reverse proxy (Caddy, nginx, a cloud
LB) and forward to port 8000. Behind a proxy:

- set `FORWARDED_ALLOW_IPS` to the proxy's address so rate limiting sees real client IPs (API keys work regardless);
- point the platform's health check at `/health` (liveness) and the readiness probe at `/ready`.

## 5. Monitor

`GET /metrics` (needs the API key) exposes Prometheus metrics, including `finarena_requests_total{path,method,status}`,
`finarena_request_duration_ms` (histogram), `finarena_rate_limited_total`, `finarena_model_busy_total` and
`finarena_sentiment_escalated_total` / `finarena_sentiment_items_total`. A Prometheus scrape config that works (tested):

```yaml
scrape_configs:
  - job_name: finarena
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials_file: /etc/prometheus/finarena.key   # or inline `credentials: <key>`
    static_configs:
      - targets: ["finarena:8000"]
```

`/metrics` is deliberately outside the rate limiter (a scraper hits it every few seconds); keep it on an internal
network and do not expose it publicly. Useful alerts: 5xx rate above 1% for 5 minutes; p95 of
`finarena_request_duration_ms` above your target; `rate(finarena_model_busy_total[5m]) > 0` (the LLM is saturated, add
replicas); the escalation fraction (`escalated / items`) drifting away from its usual level (the input mix changed).

## 6. Limits to know before you go live

- Rate limits are per process. With N replicas the effective limit is N times higher unless the gateway also limits.
- No persistence, no per-key quotas, no key rotation UI: keys are an environment variable, so rotating means redeploy.
- The credit model is trained on one public 2005 dataset. Validate it on your own portfolio, and review fair lending,
  before using it for real decisions.
- Keep `/v1/signature/detect` off (the default) unless you accept the AGPL terms of its dependency.

## 7. Rollback

Images are tagged by `IMAGE_TAG` (see `.env.example`); redeploy the previous tag. There is no state to migrate.
