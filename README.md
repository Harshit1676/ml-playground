# ML Playground

ML Playground is an interactive platform for beginner-to-intermediate learners to explore foundational machine learning through data, controls, visualizations, and explanations.

## MVP

Algorithms ship in this order: Linear Regression, Logistic Regression, K-Nearest Neighbors (KNN), then K-Means. Learners adjust datasets and parameters, inspect mathematically grounded results, and follow learning-step calculations where supported. Experiments are reproducible with explicit random seeds.

The frontend owns controls, interaction, visualization, and explanation presentation. ML computation runs in a separate Python engine behind a stateless FastAPI service. MVP excludes authentication, a database, Redis, Celery, and an AI chatbot.

## Final stack

- **Frontend:** React, Vite, TypeScript, Tailwind CSS, Plotly.js
- **Backend:** Python 3.11+, FastAPI, Pydantic, NumPy, scikit-learn
- **Testing:** Vitest, React Testing Library, pytest, and API integration tests
- **Deployment:** Static frontend hosting and a separate FastAPI service

## Monorepo

```text
frontend/   React + Vite app, controls, UI state, visualization, explanations
backend/    FastAPI routes/schemas/services and Python ML engine
docs/       Product, architecture, and development documentation
```

The ML engine handles dataset generation, training, predictions, metrics, and learning-step calculations; it does not run in React. See [the PRD](docs/PRD.md), [the architecture](docs/ARCHITECTURE.md), and [the development log](docs/DEVELOPMENT_LOG.md). Contributor workflow is in [AGENTS.md](AGENTS.md).

## Local development (Windows PowerShell)

Use two PowerShell terminals from the repository root.

Use Node.js **24.22.3 LTS**, pinned in the repository-root `.nvmrc` and constrained by `frontend/package.json`. This patch release is within the installed Vite, Vitest, and jsdom engine ranges. With nvm-windows, run `nvm use 24.22.3`; otherwise install Node.js 24.22.3 using the official Node.js installer. The current workstation's Node.js 25.2.0 is outside the supported range and may emit engine warnings; CI uses the pinned LTS version.

### Backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Copy the example values into the process environment as needed.
$env:APP_ENV = "development"
$env:CORS_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The health endpoint is `http://localhost:8000/api/v1/health`. Interactive API documentation is at `http://localhost:8000/docs`.

To run backend tests in a separate terminal:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

### Frontend

```powershell
cd frontend
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm ci
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://localhost:5173`). The page checks the backend health endpoint and offers a retry when it cannot connect. Frontend checks:

```powershell
npm test
npm run build
```

`frontend/.env.example` sets `VITE_API_BASE_URL`; Vite exposes this value to the browser bundle, so only public configuration belongs there. Do not put secrets in `VITE_*` variables.

`backend/.env.example` documents backend configuration, but the application reads operating-system environment variables directly and does not automatically load `.env` files. The PowerShell assignments above set them for the current terminal. In production, configure `APP_ENV=production` and `CORS_ALLOWED_ORIGINS` in the backend hosting platform's environment settings. Production startup fails if that origin list is missing or contains `*`.

## CI

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`. It uses the Node version in `.nvmrc`, runs `npm ci`, `npm test`, and `npm run build` from `frontend/`, then installs `backend/requirements.txt` with Python 3.11 and runs pytest. These checks need no repository secrets. This is CI only; no deployment workflow is configured.

## Production deployment configuration

No hosting provider has been selected and nothing has been deployed.

- **Static frontend:** run `npm run build` from `frontend/`; upload the generated `frontend/dist/` directory to a static host. Set `VITE_API_BASE_URL` to the public FastAPI base URL in the frontend host's build-time environment. It is embedded in public assets and must not contain secrets.
- **FastAPI service:** install `backend/requirements.txt`, set `APP_ENV=production`, and set `CORS_ALLOWED_ORIGINS` to a comma-separated list of exact frontend origins, for example `https://playground.example.com`. Set these as backend service environment variables; do not use wildcard origins. Start with `python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"`, where the platform supplies `PORT`.
- **Future write routes:** when adding POST endpoints, add `POST` to the CORS `allow_methods` list and test the intended preflight headers and production origins. Keep only required methods enabled.

### Deployment smoke-test checklist

1. Confirm the static host serves the frontend build output and the configured `VITE_API_BASE_URL` points to the deployed API.
2. Confirm the backend starts on the platform-provided port with `APP_ENV=production` and explicit `CORS_ALLOWED_ORIGINS`.
3. Request `/api/v1/health` directly and confirm HTTP 200 with `{"status":"ok"}`.
4. Load the frontend from its real origin and confirm it reports the backend as connected.
5. Confirm an allowed-origin preflight succeeds and an unlisted origin receives no `Access-Control-Allow-Origin` header.
6. Verify no secrets are present in frontend build assets or client-visible `VITE_*` settings.

These instructions describe the deployment contract, not a provider-specific release setup. Production domain, TLS, DNS, monitoring, and platform-specific health-check configuration remain open.

## Important files and dependencies

- `frontend/src/App.tsx` renders the responsive landing page and loading/connected/failed connection states.
- `frontend/src/lib/api/health.ts` provides the typed health client, validates the response shape, and uses `VITE_API_BASE_URL` (defaulting to `http://localhost:8000`).
- `frontend/src/styles.css` imports Tailwind CSS and contains the small amount of custom responsive page styling.
- `frontend/vite.config.ts` configures the React and Tailwind Vite plugins and Vitest's jsdom environment.
- `frontend/src/App.test.tsx` and `frontend/src/lib/api/health.test.ts` cover UI states and API request/response behavior.
- `backend/app/main.py` creates FastAPI and reads `APP_ENV`/`CORS_ALLOWED_ORIGINS` from the process environment; it defaults to Vite localhost origins in development and requires explicit non-wildcard origins in production. `backend/app/api/routes/health.py` defines the route and Pydantic response model.
- `backend/app/api/router.py` mounts versioned routes under `/api/v1`; `backend/tests/test_health.py` covers the health response and CORS.
- `.nvmrc` pins Node.js 24.22.3 for local use and GitHub Actions; `.github/workflows/ci.yml` runs frontend and backend checks without secrets.
- `frontend/package.json` declares React, Vite, TypeScript, Tailwind CSS, Vitest, and React Testing Library. Tailwind's Vite plugin integrates utility generation; Vitest runs the tests, React Testing Library renders components in jsdom, and pytest plus HTTPX exercises FastAPI in-process.
- `backend/requirements.txt` bounds FastAPI/Pydantic for the API, Uvicorn for local serving, and HTTPX/pytest for tests.

## Linear Regression API (Milestone 2)

`POST /api/v1/experiments/linear-regression` accepts either inline points (`x` and `y` arrays) or a `synthetic_linear` dataset and returns the effective dataset, seed, initial configuration, fitted slope/intercept, predictions, MSE, completed update count, and loss history. The endpoint is stateless: to continue training, send the same data/configuration with the previous model parameters as the next initial values and `steps: 1`.

Example request (the three point fixture used by tests):

```json
{
  "dataset": {"kind": "points", "x": [-1, 0, 1], "y": [-1, 1, 3]},
  "parameters": {"initial_slope": 0, "initial_intercept": 0, "learning_rate": 0.1, "steps": 1},
  "seed": 42
}
```

The response contains `algorithm: "linear_regression"`, `dataset`, `seed`, `parameters`, `model`, `predictions.y_hat`, `metrics.mse`, and `training` with `steps_completed` and `loss_history`. Here, `MSE = mean((slope*x + intercept - y)^2)`. Each batch step calculates `d_slope = 2*mean((prediction-y)*x)` and `d_intercept = 2*mean(prediction-y)` from the same pre-update parameters, then subtracts `learning_rate * gradient` from both parameters simultaneously. Loss history starts with the initial MSE and includes each post-update MSE, so it always contains `steps + 1` finite values. With the example fixture, one update yields slope `4/15`, intercept `0.2`, and MSE approximately `2.64296296296`.

Example response for the fixture above:

```json
{
  "algorithm": "linear_regression",
  "seed": 42,
  "dataset": {"x": [-1.0, 0.0, 1.0], "y": [-1.0, 1.0, 3.0]},
  "parameters": {"initial_slope": 0.0, "initial_intercept": 0.0, "learning_rate": 0.1, "steps": 1},
  "model": {"slope": 0.26666666666666666, "intercept": 0.2},
  "predictions": {"y_hat": [-0.06666666666666665, 0.2, 0.4666666666666667]},
  "metrics": {"mse": 2.642962962962963},
  "training": {"steps_completed": 1, "loss_history": [3.6666666666666665, 2.642962962962963]}
}
```

Inline data accepts 1–1,000 observations; generated data accepts 2–1,000. Data and initial parameters are finite and bounded to absolute value `1e6`; synthetic noise standard deviation is 0–1,000; learning rate is `(0, 1]`; steps are 0–500; seed is an unsigned 32-bit integer; and request bodies are limited to 1 MiB. Invalid input returns HTTP 422 with `error.code: "VALIDATION_ERROR"`; oversized bodies return HTTP 413 with `REQUEST_TOO_LARGE`. Non-finite intermediate predictions, residuals, gradients, candidate parameters, or losses return HTTP 422 with `NUMERICAL_DIVERGENCE`, the first failing step and field, and no partial result. A finite but increasing loss is returned as calculated; the API does not silently rescale data or alter the requested learning rate.

## Backend module responsibilities

- `backend/app/api/routes/experiments.py` exposes the thin versioned HTTP route.
- `backend/app/schemas/linear_regression.py` defines discriminated input schemas, bounds, and response consistency validation; `schemas/errors.py` describes structured API errors.
- `backend/app/services/linear_regression.py` resolves inline/generated datasets and assembles the API response without retaining state.
- `backend/app/ml/linear_regression.py` is the FastAPI-independent NumPy engine: seeded data generation, predictions, MSE, gradients, updates, and divergence detection.
- `backend/app/middleware.py` enforces the 1 MiB write-body limit before JSON parsing.
- `backend/tests/test_linear_regression.py` covers mathematical fixtures, determinism, continuation, validation, numerical safety, API behavior, and a scikit-learn comparison.

NumPy supplies explicit float64 vector calculations and makes each educational gradient step inspectable. scikit-learn's `LinearRegression` is used only as a converged-fit reference in tests; for its seeded 100-point fixture, both slope and intercept are compared with absolute tolerance `1e-8`. It does not replace the requested batch-gradient-descent trace. New backend runtime dependencies are NumPy and scikit-learn; FastAPI/Pydantic provide HTTP/schema validation, while pytest/HTTPX provide route tests. The health check and Milestone 1 frontend remain in place. Plotly visualization of this endpoint is deferred to a later frontend milestone.
