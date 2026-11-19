from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorDetails(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
