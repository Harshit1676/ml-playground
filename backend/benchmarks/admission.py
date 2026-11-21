"""Local-only staging benchmark. Run with a client Python containing httpx and psutil.

Usage: python backend/benchmarks/admission.py <backend-venv-python>
The separate Uvicorn process uses the supplied interpreter and an ephemeral loopback port.
"""

import asyncio
from functools import wraps
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


def serve():
    import uvicorn
    from app.api.routes import experiments, kmeans, logistic_regression
    from app.main import app

    active = peak = 0
    lock = threading.Lock()

    def measure(service):
        @wraps(service)
        def measured(request):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                return service(request)
            finally:
                with lock:
                    active -= 1
        return measured

    experiments.run_linear_regression_experiment = measure(experiments.run_linear_regression_experiment)
    logistic_regression.run_logistic_regression_experiment = measure(logistic_regression.run_logistic_regression_experiment)
    kmeans.run_kmeans_experiment = measure(kmeans.run_kmeans_experiment)

    async def instrumented(scope, receive, send):
        async def measured_send(message):
            if message["type"] == "http.response.start":
                with lock:
                    message["headers"].extend([(b"x-bench-active", str(active).encode()),
                                               (b"x-bench-peak", str(peak).encode())])
            await send(message)
        await app(scope, receive, measured_send)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        print(os.getpid(), sock.getsockname()[1], flush=True)
        server = uvicorn.Server(uvicorn.Config(instrumented, access_log=False, log_level="error"))
        server.run(sockets=[sock])


def fixture(lab, maximum):
    n, steps = (1000, 500) if maximum else (250, 100)
    x = [i / (n - 1) * 2 - 1 for i in range(n)]
    params = dict(initial_intercept=0, learning_rate=0.05, steps=steps)
    if lab == "k-means":
        points = [[i / (n-1) * 999999.123456789, i / (n-1) * -38231.2345678901] for i in range(n)]
        return dict(dataset=dict(kind="points", points=points), seed=3,
                    parameters=dict(k=16 if maximum else 3, max_iterations=100))
    if lab == "linear-regression":
        params["initial_slope"] = 0
        y = [v * 0.5 for v in x]
    else:
        params["initial_coefficient"] = 0
        y = [int(v > 0) for v in x]
    request = dict(dataset=dict(kind="points", x=x, y=y), parameters=params, seed=3)
    if lab == "logistic-regression":
        request["threshold"] = 0.5
    return request


def percentile(values, percent):
    if values:
        return round(sorted(values)[math.ceil(len(values) * percent / 100) - 1], 2)


async def run(client, worker):
    labs = ("linear-regression", "logistic-regression", "k-means")
    for lab in (*labs, "mixed"):
        selected = labs if lab == "mixed" else (lab,)
        for maximum in (False, True):
            bodies = {name: json.dumps(fixture(name, maximum)).encode() for name in selected}

            async def post(index=0):
                name = selected[index % len(selected)]
                start = time.perf_counter()
                response = await client.post("/api/v1/experiments/" + name, content=bodies[name],
                                             headers={"content-type": "application/json"})
                latency = (time.perf_counter() - start) * 1000
                if response.status_code == 503:
                    assert response.json()["error"]["code"] == "EXPERIMENT_BUSY"
                else:
                    response.raise_for_status()
                    if maximum and name == "k-means":
                        assert len(response.json()["training"]["snapshots"]) == 101
                return response.status_code, latency, int(response.headers["x-bench-peak"])

            for index in range(4 * len(selected)):
                assert (await post(index))[0] == 200
            for concurrency in (1, 2, 4):
                rows, health, loaded_health, rss = [], [], [], []

                async def monitor():
                    while True:
                        start = time.perf_counter()
                        r = await client.get("/api/v1/health")
                        latency = (time.perf_counter() - start) * 1000
                        r.raise_for_status()
                        health.append(latency)
                        if int(r.headers["x-bench-active"]):
                            loaded_health.append(latency)
                        rss.append(worker.memory_info().rss)
                        await asyncio.sleep(0.025)

                before = worker.cpu_times()
                monitor_task = asyncio.create_task(monitor())
                try:
                    batches = 100 if maximum and concurrency == 1 and lab != "mixed" else 12
                    for batch in range(batches):
                        rows.extend(await asyncio.gather(*(post(batch + i) for i in range(concurrency))))
                finally:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except asyncio.CancelledError:
                        pass
                after = worker.cpu_times()
                completed = [lat for status, lat, _ in rows if status == 200]
                rejected = [lat for status, lat, _ in rows if status == 503]
                assert (await post())[0] == 200
                print(json.dumps(dict(lab=lab, maximum=maximum, concurrency=concurrency,
                    admitted=len(completed), rejected=len(rejected), peak=max(r[2] for r in rows),
                    complete_ms=[percentile(completed, p) for p in (50,95,99,100)],
                    reject_p95=percentile(rejected,95), health_p95=percentile(health,95),
                    active_health_n=len(loaded_health), active_health_p95=percentile(loaded_health,95),
                    cpu_s=round(after.user+after.system-before.user-before.system,3),
                    rss_mib=round(max(rss)/2**20,2),
                    lifetime_peak_mib=round(worker.memory_info().peak_wset/2**20,2) if os.name == "nt" else None,
                    recovered=True)), flush=True)


async def main(interpreter):
    import httpx
    import psutil

    script = Path(__file__).resolve()
    env = {**os.environ, "PYTHONPATH": str(script.parents[1])}
    proc = subprocess.Popen([interpreter, str(script), "--worker"], env=env, stdout=subprocess.PIPE, text=True)
    try:
        pid, port = map(int, proc.stdout.readline().split())
        worker = psutil.Process(pid)
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30, trust_env=False) as client:
            await run(client, worker)
    finally:
        if proc.poll() is None:
            # A Windows venv launcher may have spawned the actual worker as a child.
            for child in psutil.Process(proc.pid).children(recursive=True):
                child.terminate()
            proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        serve()
    else:
        asyncio.run(main(sys.argv[1]))
