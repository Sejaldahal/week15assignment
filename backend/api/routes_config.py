from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/config")
async def get_config(request: Request) -> dict:
    settings = request.app.state.settings
    return settings.safe_public_config()
