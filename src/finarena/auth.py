import hmac

from fastapi import Header, HTTPException, Request


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)):
    """Constant-time API-key check. Auth is disabled only when settings hold no keys (dev mode)."""
    keys = request.app.state.settings.api_keys
    if not keys:
        return
    if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(status_code=401, detail="invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"})
