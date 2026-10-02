"""FastAPI application factory. Models are injected so the API can be tested without loading any weights."""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from finarena import __version__
from finarena.auth import require_api_key
from finarena.config import Settings
from finarena.limits import RateLimiter
from finarena.models.concurrency import ModelBusy
from finarena.schemas import CreditRequest, CreditResponse, SentimentRequest, SentimentResponse, SignatureResponse

log = logging.getLogger("finarena")
COST_ASSUMPTION = "threshold minimises expected cost with a missed default costing 5x a wrongly declined good borrower"
IMAGE_TYPES = {"image/jpeg", "image/png"}
BODY_OVERHEAD = 64 * 1024  # headroom over the image limit for multipart framing and JSON bodies
STATIC = Path(__file__).parent / "static"


@dataclass
class Registry:
    sentiment: object = None
    credit: object = None
    signature: object = None
    cards: dict = field(default_factory=dict)
    arena: dict | None = None


def create_app(settings: Settings, registry: Registry) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        log.info("finarena %s starting env=%s auth=%s", __version__, settings.env, bool(settings.api_keys))
        yield

    app = FastAPI(title="FinArena", version=__version__, lifespan=lifespan,
                  description="Fintech model arena: routed sentiment, explainable credit scoring, signature detection.")
    app.state.settings, app.state.registry = settings, registry
    limiter = app.state.limiter = RateLimiter(settings.rate_per_minute)
    body_cap = settings.max_image_bytes + BODY_OVERHEAD

    def client_key(request):
        """A configured API key gets its own budget; anything else (missing or made-up keys) is counted by IP,
        so rotating random key values cannot be used to dodge the limit or to bloat the limiter's memory."""
        key = request.headers.get("x-api-key")
        if key and key in settings.api_keys:
            return f"key:{key}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def reject(request):
        """Early rejection for API calls: the per-client rate limit first (so oversized probes are not free), then the
        body-size cap. The cap trusts the declared Content-Length, which the HTTP server enforces, so POSTs that do
        not declare one (chunked uploads) are refused rather than read without a bound."""
        if not request.url.path.startswith("/v1/"):
            return None
        allowed, retry = limiter.check(client_key(request))
        if not allowed:
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": str(max(1, round(retry)))})
        declared = request.headers.get("content-length")
        if declared is None:
            return JSONResponse({"detail": "Content-Length required"}, status_code=411) if request.method == "POST" else None
        if not declared.isdigit():
            return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        if int(declared) > body_cap:
            return JSONResponse({"detail": f"request body too large (max {body_cap} bytes)"}, status_code=413)
        return None

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t = time.perf_counter()
        try:
            response = reject(request)
            if response is None:
                response = await call_next(request)
        except Exception:
            log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t) * 1e3
        response.headers["x-request-id"], response.headers["x-process-time-ms"] = rid, f"{ms:.1f}"
        log.info(json.dumps(dict(rid=rid, method=request.method, path=request.url.path, status=response.status_code, ms=round(ms, 1))))
        return response

    def check_batch(n):
        if n > settings.max_batch:
            raise HTTPException(413, f"batch too large (max {settings.max_batch})")

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/v1/arena", dependencies=[Depends(require_api_key)])
    def arena():
        if registry.arena is None:
            raise HTTPException(404, "arena results not built (run scripts/build_arena.py)")
        return registry.arena

    @app.exception_handler(ModelBusy)
    async def model_busy(request: Request, exc: ModelBusy):
        return JSONResponse({"detail": str(exc)}, status_code=503, headers={"Retry-After": "2"})

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    def ready():
        loaded = {"sentiment": registry.sentiment is not None, "credit": registry.credit is not None,
                  "signature": registry.signature is not None,
                  "sentiment_llm": getattr(registry.sentiment, "accurate", None) is not None}
        if not (loaded["sentiment"] or loaded["credit"]) or (settings.require_llm and not loaded["sentiment_llm"]):
            raise HTTPException(503, detail=dict(status="not ready", models=loaded))
        return {"status": "ready", "models": loaded}

    @app.get("/v1/models", dependencies=[Depends(require_api_key)])
    def models():
        return registry.cards

    @app.post("/v1/sentiment", response_model=SentimentResponse, dependencies=[Depends(require_api_key)])
    def sentiment(req: SentimentRequest):
        if registry.sentiment is None:
            raise HTTPException(503, "sentiment models not loaded")
        check_batch(len(req.texts))
        if any(len(t) > settings.max_text_chars for t in req.texts):
            raise HTTPException(413, f"text too long (max {settings.max_text_chars} chars)")
        try:
            results = registry.sentiment.predict(req.texts, req.strategy)
        except LookupError as e:
            raise HTTPException(503, str(e)) from e
        return SentimentResponse(results=results, strategy=req.strategy,
                                 escalated_fraction=sum(r["escalated"] for r in results) / len(results))

    @app.post("/v1/credit/score", response_model=CreditResponse, dependencies=[Depends(require_api_key)])
    def credit(req: CreditRequest):
        if registry.credit is None:
            raise HTTPException(503, "credit model not loaded")
        check_batch(len(req.applications))
        return CreditResponse(results=registry.credit.score(req.applications, req.model),
                              threshold=registry.credit.threshold, cost_assumption=COST_ASSUMPTION)

    @app.post("/v1/signature/detect", response_model=SignatureResponse, dependencies=[Depends(require_api_key)])
    async def signature(file: UploadFile = File(...)):
        if registry.signature is None:
            raise HTTPException(503, "signature detector not enabled on this deployment")
        if file.content_type not in IMAGE_TYPES:
            raise HTTPException(415, "image must be JPEG or PNG")
        data = await file.read(settings.max_image_bytes + 1)
        if len(data) > settings.max_image_bytes:
            raise HTTPException(413, f"image too large (max {settings.max_image_bytes} bytes)")
        try:
            dets, w, h = await run_in_threadpool(registry.signature.detect, data)
        except (OSError, ValueError) as e:
            raise HTTPException(422, f"could not decode image: {e}") from e
        return SignatureResponse(detections=dets, image_width=w, image_height=h)

    return app
