"""Current date/time utility tool."""
from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel

SCHEMA = {
    "name": "current_datetime",
    "description": "Get the current date and time in UTC.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class TimeToolInput(BaseModel):
    pass


def execute(args: dict) -> dict:
    TimeToolInput.model_validate(args or {})
    now = datetime.now(timezone.utc)
    return {"utc_iso": now.isoformat(), "unix_timestamp": int(now.timestamp())}
