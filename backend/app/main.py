import os
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.middleware import RequestBodyLimitMiddleware
from app.ml.linear_regression import DatasetValueOutOfRangeError, NumericalDivergenceError

LOCAL_CORS_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def get_allowed_origins() -> list[str]:
    environment = os.getenv("APP_ENV", "development").strip().lower()
    configured = os.getenv("CORS_ALLOWED_ORIGINS")

    if not configured or not configured.strip():
        if environment == "production":
            raise RuntimeError("CORS_ALLOWED_ORIGINS must be set in production.")
        return list(LOCAL_CORS_ORIGINS)

    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if not origins:
        if environment == "production":
            raise RuntimeError("CORS_ALLOWED_ORIGINS must contain at least one origin in production.")
        return list(LOCAL_CORS_ORIGINS)

    for origin in origins:
        parsed = urlsplit(origin)
        if "*" in origin:
            raise RuntimeError("Wildcard CORS origins are not allowed.")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            raise RuntimeError(f"Invalid CORS origin: {origin!r}")
        if parsed.username is not None or parsed.password is not None:
            raise RuntimeError("CORS origins must not contain credentials.")

    return origins


def create_app() -> FastAPI:
    application = FastAPI(title="ML Playground API", version="0.1.0")
    application.add_middleware(RequestBodyLimitMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    application.include_router(api_router)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in error["loc"] if part != "body"),
                "reason": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "VALIDATION_ERROR", "message": "Request validation failed.", "details": {"fields": details}}},
        )

    @application.exception_handler(NumericalDivergenceError)
    async def divergence_error_handler(request: Request, exc: NumericalDivergenceError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "NUMERICAL_DIVERGENCE",
                    "message": "Gradient descent produced a non-finite intermediate value.",
                    "details": {"step": exc.step, "value": exc.value},
                }
            },
        )

    @application.exception_handler(DatasetValueOutOfRangeError)
    async def dataset_value_out_of_range_handler(request: Request, exc: DatasetValueOutOfRangeError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "DATASET_VALUE_OUT_OF_RANGE",
                    "message": "Generated dataset values must have absolute magnitude at most 1,000,000.",
                    "details": {
                        "field": f"dataset.{exc.field}",
                        "index": exc.index,
                        "actual": exc.actual,
                        "max_abs": exc.maximum,
                    },
                }
            },
        )

    return application


app = create_app()
