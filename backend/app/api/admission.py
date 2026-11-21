"""Single-process admission for buffered JSON experiment routes."""

import asyncio
from threading import Lock

from anyio import CancelScope
from fastapi import APIRouter, Request
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse, Response

from app.schemas.errors import ErrorResponse


_slot = Lock()


class ExperimentRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def finish_work(request: Request) -> Response:
            # RequestBodyLimitMiddleware has finished receiving the bounded upload.
            if not _slot.acquire(blocking=False):
                return JSONResponse(
                    {"error": {
                        "code": "EXPERIMENT_BUSY",
                        "message": "The server is busy running another experiment. Wait a second and retry.",
                        "details": {},
                    }},
                    status_code=503,
                    headers={"Retry-After": "1"},
                )
            try:
                work = asyncio.create_task(handle(request))
                try:
                    return await asyncio.shield(work)
                except asyncio.CancelledError:
                    # Cancelling the HTTP task cannot stop FastAPI's worker thread.
                    # Join through repeated cancellation before unwinding the slot.
                    with CancelScope(shield=True):
                        while not work.done():
                            try:
                                await asyncio.shield(work)
                            except asyncio.CancelledError:
                                continue
                    if not work.cancelled():
                        work.result()
                    raise
            finally:
                # Successful JSON responses are serialized. ASGI send happens later.
                _slot.release()

        return finish_work


def experiment_router() -> APIRouter:
    return APIRouter(route_class=ExperimentRoute, responses={
        503: {"model": ErrorResponse, "description": "Another experiment is running.",
              "headers": {"Retry-After": {"schema": {"type": "integer"}, "description": "Seconds before retry."}}},
    })
