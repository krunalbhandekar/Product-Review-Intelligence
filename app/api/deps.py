import hmac

from fastapi import Header, HTTPException, status

from app.config import get_settings
from app.core.logging import get_logger

_log = get_logger("auth")


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected = settings.api_key or ""
    if not expected:
        _log.error("auth.server_misconfigured.api_key_unset")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="server misconfigured: API_KEY not set",
        )
    # Constant-time compare to avoid leaking the key via timing side channel.
    if not x_api_key or not hmac.compare_digest(
        x_api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        # Debug-only diagnostics: log presence (never the value) so devs can
        # tell "header missing" from "header wrong" without exposing the key.
        if settings.app.DEBUG:
            _log.warning(
                "auth.rejected",
                header_present=bool(x_api_key),
                reason="missing_header" if not x_api_key else "value_mismatch",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-API-Key",
        )
