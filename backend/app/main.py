import os
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router

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
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
    )
    application.include_router(api_router)
    return application


app = create_app()
