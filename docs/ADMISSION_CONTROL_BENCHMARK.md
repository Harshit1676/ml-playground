# Local experiment admission benchmark

One Uvicorn process completed all 24 workload groups with a measured peak of one active experiment service call. Every immediate recovery POST succeeded. Windows reported a lifetime peak working set of 66.34 MiB. This trial ran on a developer machine. The target host's CPU allocation, memory tier, network and ingress deadlines still need measurement.

## Method

Windows build 26300, Intel Core i5-12500H, 16 logical CPUs, 15.69 GiB RAM. Python 3.11.9, FastAPI 0.165.0, Starlette 1.10.0, Uvicorn 0.54.0, NumPy 2.4.6 and Pydantic 2.14.3. The load client used HTTPX 0.28.1 and psutil 7.2.2 in a separate process. It reached the staging worker over loopback HTTP. Host power settings and other OS activity were uncontrolled.

Run from the repository root with a client interpreter containing HTTPX and psutil:

```powershell
python backend/benchmarks/admission.py backend/.venv/Scripts/python.exe
```

Representative regressions used 250 inline observations and 100 updates, maximum regressions used 1,000 and 500. K-Means used 250 points and three clusters or 1,000 points and 16 clusters, with seed 3 and a 100-update limit. The client verified every completed maximum K-Means response contained 101 snapshots. The mixed batches rotated which algorithm was submitted first. All fixtures are deterministic.

Each individual workload had four warmups. Each maximum serial series then had 100 measured calls. Other series used 12 batches. Latency spans the HTTP request through receipt of the entire response. The percentiles use nearest rank, so a p95 based on 12 completions is the largest observation. The instrumented worker counted concurrent service calls and exposed counts only to the local benchmark client. The peak stayed at one across the run. Warmups and post-series recovery calls are excluded from the table.

## Original trial with the slot held across network transfer

Times are milliseconds. The health client waited 25 ms between probes. The active-health subset includes probes whose response headers were sent while a service call was running. A dash means there were no samples in that category. The broader health column also captures delays during validation and serialization. CPU is total worker user plus system seconds for the series, including health. RSS is the highest sampled worker working set for that series, excluding the load client.

| Lab | Workload | Arrivals per batch | 200 / 503 | Completed p50 / p95 / p99 / max | Busy p95 | Health p95 | Active health count / p95 | CPU s | RSS MiB |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| linear-regression | Representative | 1 | 12/0 | 10.4 / 12.3 / 12.3 / 12.3 | — | 3.62 | 0 / — | 0.125 | 59.03 |
| linear-regression | Representative | 2 | 12/12 | 12.11 / 21.58 / 21.58 / 21.58 | 11.09 | 11.54 | 0 / — | 0.125 | 59.12 |
| linear-regression | Representative | 4 | 12/36 | 16.95 / 22.65 / 22.65 / 22.65 | 21.04 | 12.71 | 0 / — | 0.094 | 59.26 |
| linear-regression | Maximum | 1 | 100/0 | 41.7 / 58.71 / 99.99 / 110.7 | — | 5.05 | 70 / 6.09 | 3.922 | 60.21 |
| linear-regression | Maximum | 2 | 12/12 | 45.23 / 58.47 / 58.47 / 58.47 | 5.83 | 4.99 | 11 / 7.11 | 0.469 | 60.22 |
| linear-regression | Maximum | 4 | 12/36 | 46.37 / 56.69 / 56.69 / 56.69 | 22.69 | 10.2 | 11 / 10.2 | 0.547 | 60.57 |
| logistic-regression | Representative | 1 | 12/0 | 26.52 / 33.97 / 33.97 / 33.97 | — | 3.61 | 0 / — | 0.312 | 60.69 |
| logistic-regression | Representative | 2 | 12/12 | 26.74 / 33.09 / 33.09 / 33.09 | 23.12 | 13.71 | 1 / 13.71 | 0.281 | 60.7 |
| logistic-regression | Representative | 4 | 12/36 | 29.73 / 37.27 / 37.27 / 37.27 | 36.15 | 27.39 | 0 / — | 0.328 | 60.71 |
| logistic-regression | Maximum | 1 | 100/0 | 148.01 / 185.49 / 224.28 / 244.31 | — | 4.03 | 405 / 4.04 | 14.516 | 60.77 |
| logistic-regression | Maximum | 2 | 12/12 | 169.59 / 211.13 / 211.13 / 211.13 | 8.14 | 4.6 | 59 / 4.56 | 2.062 | 60.78 |
| logistic-regression | Maximum | 4 | 12/36 | 213.08 / 236.15 / 236.15 / 236.15 | 11.27 | 6.49 | 69 / 5.09 | 2.375 | 60.78 |
| k-means | Representative | 1 | 12/0 | 12.47 / 13.25 / 13.25 / 13.25 | — | 5.27 | 0 / — | 0.109 | 62.81 |
| k-means | Representative | 2 | 12/12 | 13.66 / 16.37 / 16.37 / 16.37 | 6.86 | 6.48 | 5 / 6.48 | 0.141 | 62.82 |
| k-means | Representative | 4 | 12/36 | 13.42 / 19.24 / 19.24 / 19.24 | 11.12 | 7.16 | 7 / 7.16 | 0.125 | 62.82 |
| k-means | Maximum | 1 | 100/0 | 229.47 / 313.71 / 361.7 / 371.74 | — | 20.53 | 529 / 5.37 | 21.938 | 65.33 |
| k-means | Maximum | 2 | 12/12 | 227.92 / 234.56 / 234.56 / 234.56 | 8.52 | 20.94 | 62 / 4.77 | 2.703 | 65.3 |
| k-means | Maximum | 4 | 12/36 | 230.47 / 271.38 / 271.38 / 271.38 | 39.57 | 20.51 | 66 / 5.69 | 2.719 | 65.43 |
| mixed | Representative | 1 | 12/0 | 14.74 / 38.13 / 38.13 / 38.13 | — | 8.49 | 0 / — | 0.234 | 63.42 |
| mixed | Representative | 2 | 12/12 | 15.68 / 40.65 / 40.65 / 40.65 | 36.91 | 22.57 | 1 / 16.65 | 0.25 | 63.45 |
| mixed | Representative | 4 | 12/36 | 18.52 / 43.72 / 43.72 / 43.72 | 41.67 | 21.17 | 3 / 6.79 | 0.328 | 63.53 |
| mixed | Maximum | 1 | 12/0 | 211.4 / 229.59 / 229.59 / 229.59 | — | 19.91 | 48 / 4.68 | 1.953 | 65.22 |
| mixed | Maximum | 2 | 12/12 | 209.43 / 234.31 / 234.31 / 234.31 | 6.9 | 6.41 | 49 / 4.23 | 1.906 | 66.21 |
| mixed | Maximum | 4 | 12/36 | 209.56 / 231.68 / 231.68 / 231.68 | 14.02 | 11.59 | 51 / 4.72 | 1.953 | 65.6 |

The contention tests independently held a fit behind an event while health and preflight completed and the server rejected a second POST before reading its body. This distinguishes admission behavior from OS or HTTP scheduling delay. These measurements do not set production quotas, demonstrate multiworker fairness or establish a hosting memory requirement.

## Repeat with admission inside the experiment route

The original table above measured the earlier middleware boundary. The repeat used the same fixtures, harness, runtime and sample counts after moving admission between the buffered upload and the prepared JSON response. Across 24 groups the worker completed 552 measured experiments, rejected 384, and peaked at one active service call. Every immediate recovery POST succeeded. Its lifetime peak working set was 66.23 MiB. The timing difference between runs includes ordinary machine variability.

| Lab | Workload | Arrivals per batch | 200 / 503 | Completed p50 / p95 / p99 / max | Busy p95 | Health p95 | Active health count / p95 | CPU s | RSS MiB |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| linear-regression | Representative | 1 | 12/0 | 11.64 / 15.11 / 15.11 / 15.11 | — | 5.24 | 0 / — | 0.109 | 58.96 |
| linear-regression | Representative | 2 | 12/12 | 12.58 / 14.59 / 14.59 / 14.59 | 13.76 | 4.36 | 0 / — | 0.141 | 58.99 |
| linear-regression | Representative | 4 | 12/36 | 15.54 / 21.75 / 21.75 / 21.75 | 20.13 | 11.77 | 0 / — | 0.062 | 59.23 |
| linear-regression | Maximum | 1 | 100/0 | 41.47 / 52.01 / 64.52 / 66.01 | — | 4.71 | 61 / 5 | 3.812 | 60.33 |
| linear-regression | Maximum | 2 | 12/12 | 41.96 / 50.76 / 50.76 / 50.76 | 5.41 | 4.56 | 8 / 4.56 | 0.5 | 60.34 |
| linear-regression | Maximum | 4 | 12/36 | 47.6 / 52.83 / 52.83 / 52.83 | 12.02 | 6.49 | 8 / 5.12 | 0.5 | 60.57 |
| logistic-regression | Representative | 1 | 12/0 | 27.77 / 32.48 / 32.48 / 32.48 | — | 3.69 | 0 / — | 0.312 | 60.7 |
| logistic-regression | Representative | 2 | 12/12 | 28.74 / 33.66 / 33.66 / 33.66 | 22.08 | 19.5 | 0 / — | 0.328 | 60.7 |
| logistic-regression | Representative | 4 | 12/36 | 31.59 / 38.76 / 38.76 / 38.76 | 35.61 | 32.43 | 1 / 16.88 | 0.344 | 60.7 |
| logistic-regression | Maximum | 1 | 100/0 | 146.41 / 169.31 / 174.78 / 176.95 | — | 3.83 | 390 / 3.83 | 14 | 60.75 |
| logistic-regression | Maximum | 2 | 12/12 | 143.04 / 164.9 / 164.9 / 164.9 | 6.64 | 3.45 | 51 / 3.45 | 1.641 | 60.75 |
| logistic-regression | Maximum | 4 | 12/36 | 139.36 / 171.17 / 171.17 / 171.17 | 10.4 | 5.13 | 47 / 3.65 | 1.609 | 60.87 |
| k-means | Representative | 1 | 12/0 | 6.83 / 18.81 / 18.81 / 18.81 | — | 6.69 | 0 / — | 0.094 | 63.05 |
| k-means | Representative | 2 | 12/12 | 12.2 / 14.17 / 14.17 / 14.17 | 5.44 | 4.63 | 2 / 4.08 | 0.125 | 63.05 |
| k-means | Representative | 4 | 12/36 | 12.49 / 16.54 / 16.54 / 16.54 | 7.91 | 7.06 | 3 / 7.06 | 0.125 | 63.09 |
| k-means | Maximum | 1 | 100/0 | 170.85 / 205.96 / 216.88 / 221.26 | — | 16.07 | 380 / 3.78 | 16.578 | 65.67 |
| k-means | Maximum | 2 | 12/12 | 170.2 / 203.09 / 203.09 / 203.09 | 7.07 | 16.12 | 48 / 4.18 | 2.016 | 65.77 |
| k-means | Maximum | 4 | 12/36 | 169.91 / 244.02 / 244.02 / 244.02 | 14.86 | 15.42 | 54 / 5.24 | 2.016 | 65.62 |
| mixed | Representative | 1 | 12/0 | 10.19 / 29.05 / 29.05 / 29.05 | — | 4.04 | 0 / — | 0.172 | 64.12 |
| mixed | Representative | 2 | 12/12 | 12.91 / 32.52 / 32.52 / 32.52 | 10.53 | 5.58 | 0 / — | 0.219 | 64.12 |
| mixed | Representative | 4 | 12/36 | 19.14 / 34.39 / 34.39 / 34.39 | 32.55 | 14.04 | 1 / 4.81 | 0.234 | 64.18 |
| mixed | Maximum | 1 | 12/0 | 153.29 / 203 / 203 / 203 | — | 16.91 | 30 / 4.03 | 1.422 | 65.54 |
| mixed | Maximum | 2 | 12/12 | 142.69 / 186.57 / 186.57 / 186.57 | 5.92 | 7.06 | 34 / 5.64 | 1.328 | 65.04 |
| mixed | Maximum | 4 | 12/36 | 147.59 / 171.37 / 171.37 / 171.37 | 8.75 | 14.2 | 38 / 4.54 | 1.344 | 65.84 |
