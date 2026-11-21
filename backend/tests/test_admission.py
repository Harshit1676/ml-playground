import asyncio
import json
import socket
from threading import Event

import httpx
import pytest
import uvicorn
from fastapi import routing
from uvicorn.protocols.http.h11_impl import H11Protocol

from app.api.routes import experiments
from app.main import create_app
from app.ml.linear_regression import NumericalDivergenceError


BASE = "/api/v1/experiments/"
LINEAR = BASE + "linear-regression"
BODY = {"dataset": {"kind": "points", "x": [0, 1], "y": [0, 1]}, "seed": 0,
        "parameters": {"initial_slope": 0, "initial_intercept": 0, "steps": 1, "learning_rate": 0.1}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def blocked_fit(monkeypatch, anyio_backend):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    finish = Event()
    original = experiments.run_linear_regression_experiment
    calls = []

    def blocked(request):
        calls.append(request)
        loop.call_soon_threadsafe(entered.set)
        assert finish.wait(10), "Test did not release the worker"
        return original(request)

    monkeypatch.setattr(experiments, "run_linear_regression_experiment", blocked)
    try:
        yield entered, finish, calls
    finally:
        finish.set()


@pytest.fixture
async def client(anyio_backend):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(), raise_app_exceptions=False),
                                 base_url="http://local") as http:
        yield http


@pytest.mark.anyio
async def test_shared_slot_health_preflight_and_cancelled_request(client, blocked_fit):
    entered, finish, calls = blocked_fit
    first = asyncio.create_task(client.post(LINEAR, json=BODY))
    await asyncio.wait_for(entered.wait(), 3)
    first.cancel()
    await asyncio.sleep(0)  # Deliver cancellation while the worker is held by the event.
    first.cancel()
    origin = {"Origin": "http://localhost:5173"}
    competitors = {
        "linear-regression": BODY,
        "logistic-regression": {**BODY, "threshold": 0.5, "parameters": {
            "initial_coefficient": 0, "initial_intercept": 0, "steps": 1, "learning_rate": 0.1}},
        "k-means": {"dataset": {"kind": "points", "points": [[0, 0], [1, 1]]},
                    "parameters": {"k": 2, "max_iterations": 2}, "seed": 0},
    }
    for name, payload in competitors.items():
        response = await client.post(BASE + name, json=payload, headers=origin)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "EXPERIMENT_BUSY"
        assert "retry" in response.json()["error"]["message"].lower()
        assert response.headers["retry-after"] == "1"
        assert response.headers["access-control-allow-origin"] == origin["Origin"]
        assert "retry-after" in response.headers["access-control-expose-headers"].lower()
    assert (await client.get("/api/v1/health")).json() == {"status": "ok"}
    assert (await client.options(LINEAR, headers={**origin, "Access-Control-Request-Method": "POST"})).status_code == 200
    assert (await client.get("/openapi.json")).json()["paths"][LINEAR]["post"]["responses"]["503"]
    assert len(calls) == 1
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await client.post(LINEAR, json=BODY)).status_code == 200


@pytest.mark.anyio
async def test_real_socket_disconnect_during_fit(client, blocked_fit):
    entered, finish, _ = blocked_fit
    lost, finished = asyncio.Event(), asyncio.Event()
    app = create_app()

    class ObserveDisconnect(H11Protocol):
        def connection_lost(self, exc):
            super().connection_lost(exc)
            lost.set()

    async def observe_end(scope, receive, send):
        try:
            await app(scope, receive, send)
        finally:
            finished.set()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        server = uvicorn.Server(uvicorn.Config(observe_end, http=ObserveDisconnect, lifespan="off",
                                                log_level="error", access_log=False))
        serving = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            _, writer = await asyncio.open_connection(*sock.getsockname())
            body = json.dumps(BODY).encode()
            writer.write(f"POST {LINEAR} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
            await asyncio.wait_for(entered.wait(), 3)
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(lost.wait(), 3)
            assert (await client.post(LINEAR, json=BODY)).status_code == 503
            finish.set()
            await asyncio.wait_for(finished.wait(), 3)
            assert (await client.post(LINEAR, json=BODY)).status_code == 200
        finally:
            finish.set()
            server.should_exit = True
            await asyncio.wait_for(serving, 5)


@pytest.mark.anyio
@pytest.mark.parametrize("failure,status", [(RuntimeError(), 500), (NumericalDivergenceError(1, "loss"), 422)])
async def test_failure_releases_slot(client, monkeypatch, failure, status):
    def fail(request):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(experiments, "run_linear_regression_experiment", fail)
        assert (await client.post(LINEAR, json=BODY)).status_code == status
    assert (await client.post(LINEAR, json={})).status_code == 422
    assert (await client.post(LINEAR, json=BODY)).status_code == 200


@pytest.mark.anyio
async def test_incomplete_upload_leaves_health_and_experiment_available(client):
    uploading = asyncio.Event()

    async def upload():
        yield b'{"dataset":'
        uploading.set()
        await asyncio.Event().wait()
        yield b""

    first = asyncio.create_task(client.post(LINEAR, content=upload()))
    await asyncio.wait_for(uploading.wait(), 3)
    assert (await client.get("/api/v1/health")).json() == {"status": "ok"}
    assert (await client.post(LINEAR, json=BODY)).status_code == 200
    assert not first.done()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert (await client.post(LINEAR, json=BODY)).status_code == 200


@pytest.mark.anyio
async def test_slot_held_until_response_serialization_finishes(client, monkeypatch):
    serializing, release = asyncio.Event(), asyncio.Event()
    serialize = routing.serialize_response

    async def paused_serializer(**kwargs):
        serializing.set()
        await release.wait()
        return await serialize(**kwargs)

    monkeypatch.setattr(routing, "serialize_response", paused_serializer)
    first = asyncio.create_task(client.post(LINEAR, json=BODY))
    try:
        await asyncio.wait_for(serializing.wait(), 3)
        assert (await client.post(LINEAR, json=BODY)).status_code == 503
    finally:
        release.set()
    assert (await first).status_code == 200
    assert (await client.post(LINEAR, json=BODY)).status_code == 200


@pytest.mark.anyio
async def test_delayed_response_send_leaves_slot_available_even_if_send_fails(client):
    sending, release = asyncio.Event(), asyncio.Event()
    app = create_app()

    async def receive():
        return {"type": "http.request", "body": json.dumps(BODY).encode()}

    async def send(message):
        if message["type"] == "http.response.body":
            assert "predictions" in json.loads(message["body"])
            sending.set()
            await release.wait()
            raise OSError("Connection closed during response")

    scope = {"type": "http", "method": "POST", "scheme": "http", "path": LINEAR, "query_string": b"",
             "headers": [(b"content-type", b"application/json")], "http_version": "1.1"}
    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(sending.wait(), 3)
    assert (await client.post(LINEAR, json=BODY)).status_code == 200
    assert not task.done()
    release.set()
    with pytest.raises(OSError):
        await task
    assert (await client.post(LINEAR, json=BODY)).status_code == 200
