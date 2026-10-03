import hmac

from fastapi import HTTPException, Request


def presented_key(request: Request):
    """The API key a client sent: `x-api-key`, or `Authorization: Bearer <key>` (what Prometheus and most HTTP clients use)."""
    key = request.headers.get("x-api-key")
    if key:
        return key
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def require_api_key(request: Request):
    """Constant-time API-key check. Auth is disabled only when settings hold no keys (dev mode)."""
    keys = request.app.state.settings.api_keys
    if not keys:
        return
    key = presented_key(request)
    if not key or not any(hmac.compare_digest(key, k) for k in keys):
        raise HTTPException(status_code=401, detail="invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
