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

The frontend tooling currently supports Node.js `^22.22.2`, `^24.15.0`, or `>=26.0.0`; use an active LTS release. This environment has Node.js 25.2.0, which is outside the declared jsdom/Vitest engine ranges and emits install warnings, although the tests and build have run successfully here.

### Backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

## Important files and dependencies

- `frontend/src/App.tsx` renders the responsive landing page and loading/connected/failed connection states.
- `frontend/src/lib/api/health.ts` provides the typed health client, validates the response shape, and uses `VITE_API_BASE_URL` (defaulting to `http://localhost:8000`).
- `frontend/src/styles.css` imports Tailwind CSS and contains the small amount of custom responsive page styling.
- `frontend/vite.config.ts` configures the React and Tailwind Vite plugins and Vitest's jsdom environment.
- `frontend/src/App.test.tsx` and `frontend/src/lib/api/health.test.ts` cover UI states and API request/response behavior.
- `backend/app/main.py` creates FastAPI and restricts local CORS to the Vite development origins; `backend/app/api/routes/health.py` defines the route and Pydantic response model.
- `backend/app/api/router.py` mounts versioned routes under `/api/v1`; `backend/tests/test_health.py` covers the health response and CORS.
- `frontend/package.json` declares React, Vite, TypeScript, Tailwind CSS, Vitest, and React Testing Library. Tailwind's Vite plugin integrates utility generation; Vitest runs the tests, React Testing Library renders components in jsdom, and pytest plus HTTPX exercises FastAPI in-process.
- `backend/requirements.txt` bounds FastAPI/Pydantic for the API, Uvicorn for local serving, and HTTPX/pytest for tests.

This milestone contains only the health check and landing page. It does not include Plotly, algorithms, persistence, authentication, or background work.
