"""Simple API-key authentication router and dependency."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Dependency ───────────────────────────────────────────────────────────────


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    """FastAPI dependency that enforces API-key auth when configured.

    If ``settings.API_KEY`` is ``None`` (default), authentication is
    bypassed entirely.  Otherwise the caller must supply a matching
    ``X-API-Key`` header.
    """
    if settings.API_KEY is None:
        # Auth is disabled – allow all requests
        return None

    if x_api_key is None or x_api_key != settings.API_KEY:
        logger.warning("Rejected request – invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key",
        )
    return x_api_key


# ── Schemas ──────────────────────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    api_key: str


class VerifyResponse(BaseModel):
    valid: bool
    message: str


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/verify", response_model=VerifyResponse)
async def verify_key(
    body: VerifyRequest,
    settings: Settings = Depends(get_settings),
) -> VerifyResponse:
    """Verify whether an API key is valid.

    Returns ``{"valid": true}`` when the key matches or when auth is
    disabled entirely.
    """
    if settings.API_KEY is None:
        return VerifyResponse(valid=True, message="Auth is disabled; all keys accepted.")

    if body.api_key == settings.API_KEY:
        return VerifyResponse(valid=True, message="API key is valid.")

    return VerifyResponse(valid=False, message="API key is invalid.")
